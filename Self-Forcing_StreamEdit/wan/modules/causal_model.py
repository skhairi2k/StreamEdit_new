from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import os
import torch.distributed as dist

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        assert seq_len == x.size(1), 'seq_len=%d, x.size(1)=%d' % (seq_len, x.size(1))

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


def causal_rope_apply_multi_chunk(x, grid_sizes, freqs, query, start_frame=0):
    """
    Args:
        x: Tensor of shape [b, m*f, n, c]
        grid_sizes: Tensor of shape [b, 3], each row is (f, h, w)
        freqs: Rotary embeddings, split into 3 parts: [frame_freqs, h_freqs, w_freqs]
        start_frame: Starting index for frame-wise rotary frequencies

    Returns:
        Tensor of shape [b, m*f, n, c] with rotary embedding applied
    """
    n, c = x.size(2), x.size(3) // 2

    # Split freqs into 3 parts
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    last_chunk_start_frame = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        total_frames = x.size(1) // (h * w)  # m*f
        seq_len = total_frames * h * w       # total positions to apply rope
        q_frame = query.size(1) // (h * w)

        assert seq_len == x.size(1), 'seq_len=%d, x.size(1)=%d' % (seq_len, x.size(1))
        last_chunk_start_frame.append(total_frames - q_frame)

        # View input as complex
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))

        # Prepare frequency components
        frame_freq = freqs[0][start_frame : start_frame + total_frames]              # [m*f, c1]
        frame_freq = frame_freq.view(total_frames, 1, 1, -1).expand(total_frames, h, w, -1)

        h_freq = freqs[1][:h].view(1, h, 1, -1).expand(total_frames, h, w, -1)
        w_freq = freqs[2][:w].view(1, 1, w, -1).expand(total_frames, h, w, -1)

        freqs_i = torch.cat([frame_freq, h_freq, w_freq], dim=-1).reshape(seq_len, 1, -1)

        # Apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)

        # Append untouched part (e.g., padding)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        output.append(x_i)

    return torch.stack(output).type_as(x), last_chunk_start_frame


#✨ R32: set R32_TRACE=1 to log the observed segment lengths at every biased
# capture. Used by the smoke gate to confirm the source-current segment really
# is absent in the first half of the denoising steps and present after.
_R32_TRACE = os.environ.get("R32_TRACE", "") == "1"


def build_segment_logit_bias(prev_head_bias, len_prev, total_keys, device, dtype):
    """Per-head additive logit bias over the target branch's key segments.

    Returns ``[1, Nh, 1, total_keys]`` -- already in SDPA layout, where the head
    axis is dim 1. ``attention(attn_bias=...)`` refuses any other head axis
    precisely so this cannot be silently broadcast across heads.

    ``prev_head_bias`` is ``[Nh]``: the bias on the FIRST key segment (the
    target's previous frames). Every later segment -- the optional
    source-current frame and the target's current frame -- is left at 0.
    One scalar per head is enough for the whole past-vs-current axis because
    softmax is shift-invariant per query row, so only differences between
    segments change the attention distribution.
    """
    n_heads = prev_head_bias.shape[0]
    bias = torch.zeros(1, n_heads, 1, total_keys, device=device, dtype=dtype)
    # .view(-1, 1) is load-bearing: the destination slice is [Nh, len_prev], and a
    # bare [Nh] would try to broadcast along the KEY axis instead of the head axis
    # (and simply raise for len_prev != Nh). len_prev == 0 at the first block --
    # there is no past yet -- and this stays a well-formed no-op there.
    bias[0, :, 0, :len_prev] = prev_head_bias.to(device=device, dtype=dtype).view(-1, 1)
    return bias


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

        elif kv_cache == 'local_self_attn':
            # for inference, only focus on the current chunk
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=0).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=0).type_as(v)

            x = attention(roped_query, roped_key, v)

        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_end = current_start + q.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = q.shape[1]
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
            
            local_start_index = local_end_index - num_new_tokens
            attn_seq_slice = slice(max(0, local_end_index - self.max_attention_size), local_end_index)

            if sink_tokens == 0:
                kv_cache["k"][:, local_start_index:local_end_index] = k
                kv_cache["v"][:, local_start_index:local_end_index] = v

                # apply rope for all cached k (except sink tokens)
                select_key = kv_cache["k"][:, attn_seq_slice]
                roped_key, last_chunk_start_frame = causal_rope_apply_multi_chunk(
                    select_key[:, sink_tokens: ], grid_sizes, freqs, q, start_frame=0
                )
                attn_key = torch.cat([select_key[:, : sink_tokens], roped_key.type_as(v)], dim=1)
                # relatively apply rope for query (batched)
                roped_query = torch.cat([
                    causal_rope_apply(
                        q[b_idx].unsqueeze(0), grid_sizes[b_idx].unsqueeze(0), freqs, 
                        start_frame=last_chunk_start_frame[b_idx]
                    ).type_as(v) for b_idx in range(len(last_chunk_start_frame))
                ], dim=0)
                # select value for attn
                attn_value = kv_cache["v"][:, attn_seq_slice]
            else:
                current_start_frame = current_start // frame_seqlen
                roped_query = causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
                roped_key = causal_rope_apply(
                    k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
                attn_key = kv_cache["k"][:, attn_seq_slice]
                attn_value = kv_cache["v"][:, attn_seq_slice]

            # ====================================================================================================
            # generation or editing
            # ====================================================================================================
            
            if kv_cache.get("trg_fg_mask", None) is None or kv_cache.get("current_src_fg_mask", None) is None:
                x = attention(roped_query, attn_key, attn_value)
            else:
                # init
                src_query, trg_query = roped_query.chunk(2, dim=0)
                src_key, trg_key = attn_key.chunk(2, dim=0)
                src_value, trg_value = attn_value.chunk(2, dim=0)
                shared_dict = kv_cache['shared_dict']
                #✨ R10: blending off -> blender_rate=1 turns every blend below into
                # the identity (pure target Q/K). The bg source-KV injection further
                # down is deliberately KEPT: it is the source-anchoring channel, and
                # without it the target branch decouples from the source video.
                blend_off = shared_dict.get('blend_off', False)
                #✨ R20: when the pipeline installed a scheduled rate for this
                # denoising step, use it verbatim; otherwise evaluate Eq. 4 exactly
                # as before. `blender_rate` absent or None => bit-for-bit baseline.
                if blend_off:
                    blender_rate = 1.0
                elif shared_dict.get('blender_rate') is not None:
                    blender_rate = shared_dict['blender_rate']
                else:
                    blender_rate = 1 - shared_dict['current_timestep_next'] ** shared_dict['blend_power']

                #✨ R26: optional SPATIAL blender rate. The pipeline installs, per
                # denoising step, a CACHE-ALIGNED field for the previous chunks'
                # tokens plus a [B, Lq] field for the current chunk -- the same split
                # `trg_fg_mask` / `current_src_fg_mask` already uses. Cache alignment
                # is what makes the field correct under KV-cache eviction and across
                # chunk boundaries: the previous-token field is sliced by exactly the
                # same `attn_seq_slice` as `b_trg_attn_fg_mask` below, so no position
                # arithmetic is needed anywhere.
                #
                # Both fields carry their rate AND its complement in float32. The
                # scalar path multiplies a bfloat16 tensor by a PYTHON FLOAT, which
                # PyTorch evaluates at float32 compute precision; holding the rate in
                # bfloat16 instead would round 0.360000 -> 0.359375 (0.17%) and
                # compound over 15 steps x 30 blocks, breaking the tau_bg == tau_fg
                # control. `rate_tok is None` => every expression below is the
                # untouched scalar path, bit for bit.
                if blend_off:
                    rate_tok = comp_tok = cur_rate_tok = cur_comp_tok = None
                else:
                    rate_tok = kv_cache.get('blend_rate_tok', None)          # [B, L_cache_size]
                    comp_tok = kv_cache.get('blend_comp_tok', None)          # [B, L_cache_size]
                    cur_rate_tok = kv_cache.get('current_blend_rate', None)  # [B, Lq]
                    cur_comp_tok = kv_cache.get('current_blend_comp', None)  # [B, Lq]
                    if rate_tok is not None and (
                            comp_tok is None or cur_rate_tok is None or cur_comp_tok is None):
                        raise RuntimeError(
                            "R26: a spatial blend rate was installed without its "
                            "complement or its current-chunk half; all four of "
                            "blend_rate_tok / blend_comp_tok / current_blend_rate / "
                            "current_blend_comp must be set together."
                        )

                #✨ R10: persistent visual-prompt bank (set by the pipeline at setup).
                # Unroped anchor K/V, re-roped per block so it is always "present".
                vp_kv = getattr(self, 'vp_kv', None)
                vp_gate = getattr(self, 'vp_gate', None)
                if vp_kv is not None and sink_tokens != 0:
                    raise RuntimeError(
                        "R10 persistent VP injection assumes the sink_tokens==0 path "
                        "(unroped cached keys + last_chunk_start_frame); got "
                        f"sink_tokens={sink_tokens}."
                    )

                #✨ R32: per-head additive logit bias re-balancing the target
                # branch's key segments along the past-vs-current axis. A [Nh]
                # row stamped by the pipeline; absent => the baseline call below
                # is reached untouched.
                seg_bias = getattr(self, 'seg_bias', None)
                if seg_bias is not None:
                    if vp_kv is not None:
                        raise RuntimeError(
                            "R32 segment bias and R10 VP injection are not "
                            "validated together -- R32 is deliberately "
                            "anchor-free. Stamp one or the other, not both."
                        )
                    if sink_tokens != 0:
                        raise RuntimeError(
                            "R32 segment bias assumes the sink_tokens==0 path "
                            f"(see the VP guard above); got sink_tokens={sink_tokens}."
                        )

                src_prev_key = src_key[:, : -num_new_tokens]
                src_prev_value = src_value[:, : -num_new_tokens]
                src_current_key = src_key[:, -num_new_tokens: ]
                src_current_value = src_value[:, -num_new_tokens: ]
                trg_prev_key = trg_key[:, : -num_new_tokens]
                trg_prev_value = trg_value[:, : -num_new_tokens]
                trg_current_key = trg_key[:, -num_new_tokens: ]
                trg_current_value = trg_value[:, -num_new_tokens: ]

                x_list = [
                    attention(src_query, src_key, src_value)   # source
                ]
                for b_idx in range(b // 2):
                    b_key_list = []
                    b_value_list = []

                    #✨ masked-blended previous kv
                    b_trg_fg_mask = kv_cache["trg_fg_mask"][b_idx]                              # [L_cache_size, ]
                    b_trg_attn_fg_mask = b_trg_fg_mask[attn_seq_slice]                          # [L_attn_size, ]
                    b_trg_prev_fg_mask = b_trg_attn_fg_mask[: -num_new_tokens]
                    b_trg_prev_fg_key = trg_prev_key[b_idx]
                    #✨ R26: the per-token rate rides EXACTLY the slice above, so it is
                    # aligned with the cache by construction. Scalar path => Python floats.
                    if rate_tok is None:
                        br_prev, bc_prev = blender_rate, 1 - blender_rate
                        br_cur, bc_cur = blender_rate, 1 - blender_rate
                    else:
                        br_prev = rate_tok[b_idx][attn_seq_slice][: -num_new_tokens].view(-1, 1, 1)
                        bc_prev = comp_tok[b_idx][attn_seq_slice][: -num_new_tokens].view(-1, 1, 1)
                        br_cur = cur_rate_tok[b_idx].view(-1, 1, 1)
                        bc_cur = cur_comp_tok[b_idx].view(-1, 1, 1)
                    if not blend_off:
                        # (a no-op when blender_rate==1, but skipped explicitly to avoid
                        # the in-place masked write on a view of the cache)
                        #
                        #✨ R26: `.to(dtype)` rounds each PRODUCT back to the key dtype
                        # before the add, matching the scalar path's rounding structure
                        # exactly. On the scalar path the products are already in that
                        # dtype, so every `.to` below is a no-op and the expression is
                        # bit-for-bit what it was.
                        _dt = b_trg_prev_fg_key.dtype
                        _rp = br_prev if rate_tok is None else br_prev[b_trg_prev_fg_mask]
                        _cp = bc_prev if rate_tok is None else bc_prev[b_trg_prev_fg_mask]
                        b_trg_prev_fg_key[b_trg_prev_fg_mask] = \
                            (b_trg_prev_fg_key[b_trg_prev_fg_mask] * _rp).to(_dt) \
                            + (src_prev_key[b_idx, b_trg_prev_fg_mask] * _cp).to(_dt)
                    b_trg_prev_fg_value = trg_prev_value[b_idx]
                    b_key_list.append(b_trg_prev_fg_key)
                    b_value_list.append(b_trg_prev_fg_value)

                    #✨ current source condition
                    b_src_current_fg_mask = kv_cache["current_src_fg_mask"][b_idx]              # [Lq, ]
                    b_src_current_bg_mask = ~b_src_current_fg_mask                              # [Lq, ]
                    #✨ R31 probe: `src_kv_full` injects the WHOLE source K/V -- every
                    # token, foreground included -- at EVERY denoising step, instead of
                    # the default background-only injection over the second half. It is a
                    # DIAGNOSTIC (maximum source anchoring through the KV channel with no
                    # spatial selectivity anywhere), not a candidate setting.
                    # `.get(..., False)` => absent key reproduces the original branch
                    # bit-for-bit, so every other task/arm is untouched.
                    if kv_cache['shared_dict'].get('src_kv_full', False):
                        b_key_list.append(src_current_key[b_idx])                           # [Lq, Nh, Dk]
                        b_value_list.append(src_current_value[b_idx])                       # [Lq, Nh, Dv]
                    # t^inj=0.5
                    elif kv_cache['shared_dict']['current_timestep_index'] > kv_cache['shared_dict']['total_timestep'] // 2:
                        b_src_current_bg_key = src_current_key[b_idx][b_src_current_bg_mask]        # [L_bg, Nh, Dk]
                        b_src_current_bg_value = src_current_value[b_idx][b_src_current_bg_mask]    # [L_bg, Nh, Dv]
                        b_key_list.append(b_src_current_bg_key)
                        b_value_list.append(b_src_current_bg_value)

                    #✨ masked-blended current target condition
                    b_trg_current_key = trg_current_key[b_idx]                                  # [Lq, Nh, Dk]
                    b_trg_current_value = trg_current_value[b_idx]                              # [Lq, Nh, Dk]
                    #✨ R26: current-chunk half of the field (see br_prev above).
                    b_trg_current_key = (b_trg_current_key * br_cur).to(b_trg_current_key.dtype) \
                        + (src_current_key[b_idx] * bc_cur).to(b_trg_current_key.dtype)
                    b_key_list.append(b_trg_current_key)
                    b_value_list.append(b_trg_current_value)

                    # store and concatenate key and value
                    #✨ R32: read the segment lengths off the ACTUAL entries
                    # rather than recomputing them. The source-current segment
                    # is data-dependent -- its length is the background-mask
                    # popcount, it is absent entirely in the first half of the
                    # denoising steps, and it is not a multiple of the frame
                    # token count -- so any recomputation here would drift.
                    seg_lens = [t.shape[0] for t in b_key_list]
                    b_trg_key = torch.cat(b_key_list, dim=0)
                    b_trg_value = torch.cat(b_value_list, dim=0)

                    #✨ query blending
                    b_query = (trg_query[b_idx] * br_cur).to(trg_query.dtype) \
                        + (src_query[b_idx] * bc_cur).to(trg_query.dtype)

                    if vp_kv is None and seg_bias is None:
                        x_list.append(
                            attention(b_query.unsqueeze(0), b_trg_key.unsqueeze(0), b_trg_value.unsqueeze(0))
                        )
                    elif vp_kv is None:
                        #✨ R32: bias the first segment (target's previous
                        # frames) against the current-frame segments, per head.
                        if sum(seg_lens) != b_trg_key.shape[0]:
                            raise RuntimeError(
                                f"R32: segment lengths {seg_lens} sum to "
                                f"{sum(seg_lens)} but the concatenated key set "
                                f"is {b_trg_key.shape[0]} long -- the bias "
                                f"would be misaligned."
                            )
                        seg_logit_bias = build_segment_logit_bias(
                            seg_bias, seg_lens[0], b_trg_key.shape[0],
                            device=b_trg_key.device, dtype=b_query.dtype,
                        )
                        if _R32_TRACE:
                            print(
                                f"[r32] layer={getattr(self, 'layer_idx', -1)} "
                                f"step={kv_cache['shared_dict'].get('current_timestep_index')} "
                                f"seg_lens={seg_lens} Lk={b_trg_key.shape[0]}",
                                flush=True,
                            )
                        x_list.append(
                            attention(
                                b_query.unsqueeze(0), b_trg_key.unsqueeze(0),
                                b_trg_value.unsqueeze(0), attn_bias=seg_logit_bias,
                            )
                        )
                    else:
                        #✨ R10: re-rope the anchor bank to the CURRENT chunk's start frame so
                        # the visual prompt carries zero relative temporal offset from the
                        # queries -- it reads as "present" in every block and never fades.
                        dual_idx = b // 2 + b_idx
                        vp_k, vp_v = vp_kv
                        vp_grid = grid_sizes[dual_idx].clone().unsqueeze(0)
                        vp_grid[0, 0] = 1  # the bank holds exactly one latent frame
                        vp_k_roped = causal_rope_apply(
                            vp_k, vp_grid, freqs, start_frame=last_chunk_start_frame[dual_idx]
                        ).type_as(v)[0]

                        #✨ R10: `gate` is a per-head bool [1, 1, Nh, 1] -- True = this head
                        # attends to the VP bank, False = this head is left at baseline.
                        # `gate is None` is NOT the same as an all-False gate: None means
                        # "no gating requested", i.e. the `all` arm, where every head sees
                        # the VP. An all-False gate means "gate every head off", which
                        # reduces to the `none` arm.
                        #
                        #   gate       need_vp  need_base  attention calls  meaning
                        #   None       True     False      x_vp only        arm `all` (ungated)
                        #   all True   True     False      x_vp only        arm `all` (explicit)
                        #   all False  False    True       x_base only      degenerates to `none`
                        #   mixed      True     True       BOTH             arms `spatial`/`temporal`
                        #
                        # Both flags are therefore active exactly when the gate is mixed --
                        # the normal case for the routed arms (R19 gates: spatial 117/360,
                        # temporal 49/360, and their count-matched random controls). The
                        # single-call rows are short-circuits: whenever every head in THIS
                        # layer wants the same key set the other call would be dead work,
                        # which is why e.g. the `temporal` arm takes one call at layers
                        # holding no temporal head.
                        gate = None if vp_gate is None else vp_gate.to(b_query.device).view(1, 1, -1, 1)
                        need_base = gate is not None and not bool(gate.all())
                        need_vp = gate is None or bool(gate.any())

                        # Per-head gating needs two attention calls: gated heads see a longer
                        # key set than ungated ones. Softmax is per-head, so this split is
                        # exact, not an approximation.
                        #
                        # x_vp:   every head attends [b_trg_key ; vp_k_roped] (VP appended).
                        # x_base: every head attends b_trg_key alone (no VP).
                        # Both are full [1, Lq, Nh, Dv] outputs -- each call computes ALL
                        # heads, including the ones whose result will be discarded. That
                        # redundancy is the price of the exactness: attention cannot be
                        # asked for a per-head key set in one call.
                        x_vp = attention(
                            b_query.unsqueeze(0),
                            torch.cat([b_trg_key, vp_k_roped], dim=0).unsqueeze(0),
                            torch.cat([b_trg_value, vp_v[0]], dim=0).unsqueeze(0),
                        ) if need_vp else None
                        x_base = attention(
                            b_query.unsqueeze(0), b_trg_key.unsqueeze(0), b_trg_value.unsqueeze(0)
                        ) if need_base else None

                        # When both ran, keep each head's output from the call that matches
                        # its gate: `gate` broadcasts [1,1,Nh,1] over [1,Lq,Nh,Dv], so the
                        # selection is per head and never mixes the two -- no blending, no
                        # averaging. A head's output depends only on its own key set (softmax
                        # normalises per head), so the scattered result is bit-identical to
                        # what a hypothetical per-head-key-set attention would return.
                        # Cost when both are active: 2x target-branch self-attention.
                        if x_base is None:
                            x_list.append(x_vp)
                        elif x_vp is None:
                            x_list.append(x_base)
                        else:
                            x_list.append(torch.where(gate, x_vp, x_base))
                x = torch.cat(x_list, dim=0)

            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)
            kv_cache["sink_tokens"] = sink_tokens
            kv_cache["num_new_tokens"] = num_new_tokens

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start)

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # debug
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                               padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
