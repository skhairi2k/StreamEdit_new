# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        return "h100" in device_name or "hopper" in device_name
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

# FLASH_ATTN_3_AVAILABLE = False

import warnings

from torch.nn.attention import SDPBackend, sdpa_kernel

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def _sdpa_attention_with_bias(
    q,
    k,
    v,
    attn_bias,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    dtype=torch.bfloat16,
    q_lens=None,
    k_lens=None,
    window_size=(-1, -1),
):
    """Attention with a per-head additive bias on the softmax logits.

    FlashAttention accepts no additive bias, so this is the only path that can
    carry one. Consequence for any experiment that biases logits: the zero-bias
    control must ALSO come through here, because zeros-bias SDPA is not bitwise
    equal to ``attn_mask=None`` SDPA (measured 2.4e-4 at production shapes).
    Comparing a biased arm against the FlashAttention baseline would confound
    the bias with the change of kernel.

    ``attn_bias`` must already be in SDPA layout -- broadcastable to
    ``[B, Nh, Lq, Lk]``, where the head axis is dim **1**, not dim 2 as in
    q/k/v. It is deliberately never transposed here, and a head axis of 1 is
    rejected rather than broadcast: silently spreading one head's bias across
    every head is the failure mode this signature exists to prevent.
    """
    if q_lens is not None or k_lens is not None:
        raise NotImplementedError(
            "attn_bias does not support the variable-length (q_lens/k_lens) "
            "path: a dense bias cannot express per-sequence packing."
        )
    if window_size != (-1, -1):
        raise NotImplementedError(
            f"attn_bias does not support sliding-window attention; got "
            f"window_size={window_size}."
        )

    out_dtype = q.dtype
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)
    if q_scale is not None:
        q = q * q_scale

    n_heads, lq, lk = q.shape[1], q.shape[2], k.shape[2]
    if attn_bias.dim() != 4:
        raise ValueError(
            f"attn_bias must be 4-D [B, Nh, Lq|1, Lk]; got shape "
            f"{tuple(attn_bias.shape)}."
        )
    if attn_bias.shape[1] != n_heads:
        raise ValueError(
            f"attn_bias head axis (dim 1) is {attn_bias.shape[1]}, expected "
            f"{n_heads}. A bias of shape [*, 1, *, *] is refused on purpose: "
            f"it would broadcast one head's bias to all {n_heads} heads. Build "
            f"the bias in SDPA layout with an explicit row per head."
        )
    if attn_bias.shape[2] not in (1, lq):
        raise ValueError(
            f"attn_bias query axis (dim 2) is {attn_bias.shape[2]}, expected "
            f"1 or {lq}."
        )
    if attn_bias.shape[3] != lk:
        raise ValueError(
            f"attn_bias key axis (dim 3) is {attn_bias.shape[3]}, expected "
            f"{lk} (the concatenated key length)."
        )

    # The cutlass backend requires the bias dtype to equal q's; fp32 raises.
    bias = attn_bias.to(device=q.device, dtype=q.dtype)

    # Pinning the backend makes a future shape change RAISE instead of silently
    # dropping to the math backend, which materializes the full logit matrix
    # (~3.15 GB per copy at production shapes) and runs ~25x slower.
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, is_causal=causal, dropout_p=dropout_p,
            scale=softmax_scale)

    return out.transpose(1, 2).contiguous().type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
    attn_bias=None,
):
    if attn_bias is not None:
        return _sdpa_attention_with_bias(
            q=q,
            k=k,
            v=v,
            attn_bias=attn_bias,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            dtype=dtype,
            q_lens=q_lens,
            k_lens=k_lens,
            window_size=window_size,
        )

    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
