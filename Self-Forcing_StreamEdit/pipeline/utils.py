import math
import os
import numpy as np
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw
import torch


# -------------------- blend-rate schedules (R20) --------------------
#✨ R20: alternatives to the StreamGVE Eq. 4 blender rate.
#
# SIGN CONVENTION -- read this before adding a schedule. These functions return the
# SOURCE-ANCHORING WEIGHT s(p) in [0, 1]; the caller sets
#
#     blender_rate = 1 - s(p)
#
# so s=1 is "pure source" and s=0 is "pure target". That is the opposite polarity
# to `blender_rate` itself (causal_model.py: blender_rate=1 => pure target), and
# it is the polarity in which the paper's schedule is monotonically DECREASING:
# Eq. 4 gives s = t_{i+1}**rho, falling 0.87 -> 0.00 over the 15 steps, i.e. anchor
# hard at high noise and release as the sample resolves. Every schedule below keeps
# that direction; `cos_half`/`cos_third` merely release earlier (by p = 1/k) and
# then stay released.
def _schedule_blend_rate(sched: str, step_idx: int, num_steps: int) -> float:
    """Source-anchoring weight s(p) in [0, 1]; caller sets blender_rate = 1 - s(p)."""
    p = 0.0 if num_steps <= 1 else step_idx / (num_steps - 1)
    if sched == "cos_full":
        return 0.5 * (1.0 + math.cos(math.pi * p))
    if sched in ("cos_half", "cos_third"):
        k = {"cos_half": 2, "cos_third": 3}[sched]
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, k * p)))
    if sched == "const":
        return 1.0
    if sched == "zero":
        return 0.0
    raise ValueError(f"unknown blend schedule: {sched!r}")


# -------------------- tokenizer helper --------------------
def find_phrase_token_indices(tokenizer, prompt: str | List[str], phrase: str | List[str],
                              add_special_tokens_prompt=True,
                              add_special_tokens_phrase=False) -> List[int]:
    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(phrase, str):
        phrase = [phrase]
    ret_list = []
    for pr, ph in zip(prompt, phrase):
        ret_list.append(
            _find_phrase_token_indices_single(
                tokenizer, pr, ph, add_special_tokens_prompt, add_special_tokens_phrase
            )
        )
    return ret_list

def _find_phrase_token_indices_single(tokenizer, prompt: str, phrase: str,
                              add_special_tokens_prompt=True,
                              add_special_tokens_phrase=False) -> List[int]:
    if not phrase:
        return []
    enc = tokenizer(prompt, padding=False, truncation=False,
                    add_special_tokens=add_special_tokens_prompt,
                    return_attention_mask=False, return_tensors=None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    def _clean(ids): return [i for i in ids if i not in {pad_id, eos_id, None}]
    ids = _clean(enc["input_ids"])
    phrase_ids = _clean(tokenizer(phrase, add_special_tokens=add_special_tokens_phrase)["input_ids"])
    if len(phrase_ids) == 0 or len(ids) < len(phrase_ids):
        return []
    # Case-insensitive match: keep original ids for indexing (so returned indices
    # stay aligned with the attention maps), but compare on lowercased token strings.
    ids_str = [t.lower() for t in tokenizer.convert_ids_to_tokens(ids)]
    phrase_str = [t.lower() for t in tokenizer.convert_ids_to_tokens(phrase_ids)]
    for i in range(0, len(ids_str) - len(phrase_str) + 1):
        if ids_str[i:i+len(phrase_str)] == phrase_str:
            return list(range(i, i+len(phrase_str)))
    return [] 


# -------------------- R26: spatial (per-token) release exponent --------------------
#✨ R26: StreamGVE Eq. 4 gives every token the same source-anchoring weight
# W^src(t_i) = t_i ** rho, with rho == `blend_power` a single scalar. R26 makes rho
# SPATIAL: edit-region tokens get a large exponent (release the source fast),
# background tokens a small one (stay anchored). The field is hard binary and driven
# by the grounding-mask union M_f dumped by pass 1 -- no dilation, no feathering, no
# percentile rescale, because the input is already a binary mask and not a continuous
# saliency field.
def build_rho_field_from_mask(mask: torch.Tensor, tau_bg: float, tau_fg: float) -> torch.Tensor:
    """Two-level exponent field ``tau(p) = tau_bg + (tau_fg - tau_bg) * M(p)``.

    Args:
        mask: bool tensor, any shape (typically ``[B, L]`` of latent tokens).
        tau_bg: exponent applied where ``mask`` is False.
        tau_fg: exponent applied where ``mask`` is True.

    Returns:
        float32 tensor shaped like ``mask``.
    """
    if mask.dtype != torch.bool:
        raise TypeError(f"R26: expected a bool mask, got {mask.dtype}")
    # `tau < 0` would make W^src = t ** tau blow up as t -> 0 (and raise
    # ZeroDivisionError on the final t == 0 step); it is never a schedule.
    if tau_bg < 0.0 or tau_fg < 0.0:
        raise ValueError(f"R26: tau must be >= 0, got tau_bg={tau_bg}, tau_fg={tau_fg}")
    return tau_bg + (tau_fg - tau_bg) * mask.to(torch.float32)


def blender_rate_from_rho(rho: torch.Tensor, timestep_next: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tensor twin of the scalar ``blender_rate = 1 - t ** blend_power``.

    Returns ``(rate, one_minus_rate)`` -- BOTH, on purpose. ``causal_model.py`` blends
    with ``trg * rate + src * (1 - rate)`` where, on the untouched scalar path, both
    factors are Python floats evaluated in float64 and only then rounded to the kernel's
    float32 compute type. Recovering ``1 - rate`` from an already-float32 ``rate`` would
    round twice and can land one ulp away, so the complement is computed in float64 here
    and rounded once, exactly like the scalar path. That is what lets the
    ``tau_bg == tau_fg`` control reproduce the global schedule bit-for-bit.

    ✨ R31 2026-09-15: VECTORIZED. This used to loop `for r in torch.unique(rho)`,
    doing a full-array compare plus two boolean-masked writes PER DISTINCT EXPONENT.
    That is cheap for R26's two-level field (2 exponents) but catastrophic for R31's
    CONTINUOUS one, which has up to 21 * 1560 = 32,760: MEASURED 2699 ms/call on CPU and
    475 ms on CUDA, versus 1.1 ms / 0.3 ms vectorized (2488x / 1540x). At ~210 calls per
    clip that is ~2.4 h of pure GPU overhead across 4 arms x 22 clips, versus 6 s.

    Verified bit-identical to the loop over the real 15-step ``t_next`` grid x R26's
    exponents {0,1,2,3,4,6,8,10,20,50} + 5000 random continuous + 2000 spanning R31's
    [2, 50]: 0 / 105,150 differences in BOTH returned tensors, on CPU and on CUDA.

    ⚠️ THE COMPLEMENT IS NOT ``w_src``. It is ``1 - (1 - w_src)``, evaluated in
    float64 and rounded ONCE -- because that is literally what the scalar path computes
    (``causal_model.py:365`` builds ``blender_rate = 1 - t ** blend_power``, then
    ``:452`` takes ``1 - blender_rate``). The two differ once ``w_src`` falls below
    ~4e-9, since ``1 - w_src`` discards its low bits, and ``w_src = t ** rho`` with rho
    up to 50 is routinely ~1e-15. Measured: substituting ``w_src`` changes 33.1% of comp
    values (34,821 / 105,150) and would silently break the ``tau_bg == tau_fg`` control
    this function exists to preserve. Do not "simplify" it back.
    """
    t = float(timestep_next)
    if t < 0.0:
        raise ValueError(f"R26: timestep_next must be >= 0, got {t}")
    # `tau < 0` would make W^src = t ** tau blow up as t -> 0. Checked on the whole
    # tensor at once; the loop used to check each distinct value.
    if bool((rho < 0.0).any()):
        bad = float(rho[rho < 0.0].reshape(-1)[0])
        raise ValueError(f"R26: rho field holds a negative exponent {bad}")
    w_src = torch.pow(torch.as_tensor(t, dtype=torch.float64, device=rho.device),
                      rho.double())
    # tau == 0 spelled out rather than left to `0.0 ** 0.0 == 1.0`: at tau = 0 the
    # source weight is 1 at EVERY step INCLUDING the final t = 0, so those tokens
    # stay pinned to the source for the whole rollout and never release. That is an
    # extreme ORACLE arm (see the R26 plan), not a schedule -- expect seam/freeze
    # artefacts, and read a good background metric there with suspicion.
    one = torch.ones_like(w_src)
    w_src = torch.where(rho == 0.0, one, w_src)
    rate = (one - w_src).float()
    comp = (one - (one - w_src)).float()       # NOT w_src -- see the warning above
    return rate, comp
