"""Backbone dispatch for the FiVE-Bench drivers (R45): Self-Forcing (``sf``) or LongLive (``ll``).

Shared, not task-prefixed: ``run_fivebench.py``, ``r31_stage1.py`` and ``r31_stage3.py`` all
load the editing pipeline through here, so ``--backbone`` means the same thing in each.

``sf`` is the path every pre-R45 run took, moved here unchanged: the build's own
``inference_edit_streamedit.load_pipe`` (which already casts to bf16 and places the modules)
and ``pipeline.rollout_inference``. ``ll`` mirrors ``LongLive_StreamEdit/
inference_edit_streamedit.py``: its ``load_pipe`` (config grid, flow shift, generator
checkpoint, LoRA) followed by the same post-load steps that script's ``__main__`` runs, and
``pipeline.inference`` over the whole clip -- LongLive's sink + local-window KV cache bounds
memory, so there is no rollout and no window offset.

Both builds import as top-level packages (``pipeline``, ``utils``, ``wan``) and read configs
and weights through relative paths, so only ONE backbone can be active per process:
``activate`` puts its root on ``sys.path`` and makes it the cwd.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Optional, Tuple

import torch

_REPO = Path(__file__).resolve().parent.parent

BACKBONES = {
    "sf": dict(root=_REPO / "Self-Forcing_StreamEdit",
               config_path="configs/self_forcing_dmd.yaml"),
    "ll": dict(root=_REPO / "LongLive_StreamEdit",
               config_path="configs/longlive_inference.yaml"),
}

# SF-only kwargs of `rollout_inference` that LL's `inference` does not implement. Passing a
# non-None value on LL is refused rather than dropped: a silently ignored option would still
# produce a full set of renders that is not the arm it claims to be.
_SF_ONLY = ("vp_latent", "union_dump_path", "split_dump_path", "union_mask_path",
            "tau_bg", "tau_fg", "seg_bias_table", "vp_head_gate", "src_kv_full")
# `attn_record` (R48's opt-in per-token cross-attention recorder) is NOT in this list any more: R55 ported it to LongLive's `inference`,
# so it is forwarded on both backbones (None by default => no extra op anywhere).


def add_backbone_args(parser) -> None:
    """``--backbone`` plus per-backbone ``--sf_root`` / ``--config_path`` (None => default)."""
    parser.add_argument("--backbone", choices=sorted(BACKBONES), default="sf",
                        help="sf = Self-Forcing (every pre-R45 run); ll = LongLive (R45).")
    parser.add_argument("--sf_root", type=str, default=None,
                        help="Path to the backbone build (default: the --backbone's own).")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Pipeline config, relative to the build (default: the "
                             "--backbone's own).")


def resolve_defaults(args) -> None:
    """Fill ``args.sf_root`` / ``args.config_path`` left at None from ``BACKBONES``.

    For ``sf`` these are exactly the pre-R45 defaults, so existing callers are unchanged.
    """
    spec = BACKBONES[args.backbone]
    if args.sf_root is None:
        args.sf_root = str(spec["root"])
    if args.config_path is None:
        args.config_path = spec["config_path"]


def activate(backbone: str, root: Optional[Path] = None) -> types.SimpleNamespace:
    """Put the backbone build on ``sys.path``, chdir into it, and return its helpers.

    Returns a namespace with ``root`` and the build's own ``find_closest_num_frame``,
    ``read_json``, ``find_phrase_token_indices`` and ``set_seed`` -- imported from the build so
    each backbone uses its own copy (LL's trigger matcher is the case-insensitive port).
    """
    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}; expected one of {sorted(BACKBONES)}")
    root = Path(root or BACKBONES[backbone]["root"]).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{backbone} build not found at {root}")
    sys.path.insert(0, str(root))
    os.chdir(root)

    from inference_edit_streamedit import find_closest_num_frame, read_json
    from pipeline.utils import find_phrase_token_indices
    from utils.misc import set_seed
    return types.SimpleNamespace(root=root, find_closest_num_frame=find_closest_num_frame,
                                 read_json=read_json,
                                 find_phrase_token_indices=find_phrase_token_indices,
                                 set_seed=set_seed)


def load_backbone(backbone: str, args) -> Tuple[torch.nn.Module, bool, torch.device]:
    """Load the editing pipeline; ``activate(backbone)`` must have run first.

    ``args`` needs ``config_path``, ``step`` and ``flow_shift`` (both backbones), plus
    ``checkpoint_path`` / ``use_ema`` / ``sink_size`` for ``sf``. On ``ll`` the generator and
    LoRA checkpoints come from the config (``generator_ckpt`` / ``lora_ckpt``), as in LL's own
    driver, and its ``load_pipe`` seeds from ``config.seed``; the FiVE drivers re-seed per pair
    with ``--seed`` anyway.
    """
    from inference_edit_streamedit import load_pipe
    pipeline, low_memory, device, _local_rank = load_pipe(args)
    if backbone == "ll":
        # LongLive_StreamEdit/inference_edit_streamedit.py __main__, verbatim order. SF's
        # load_pipe does the equivalent internally.
        from utils.memory import DynamicSwapInstaller
        pipeline = pipeline.to(dtype=torch.bfloat16)
        if low_memory:
            DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
        pipeline.generator.to(device=device)
        pipeline.vae.to(device=device)
    return pipeline, low_memory, device


def encode_vp(backbone: str, pipeline, transform, src_frame0, anchor_img,
              device) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Encode the paper-SS4.5 visual prompt: source first frame, THEN the anchor.

    The encode order and the single trailing ``clear_cache`` are load bearing (the VAE
    carries temporal-conv state between calls; see run_fivebench.py). Returns
    ``(src_latent, trg_latent, flags)`` where ``flags`` holds the two first-frame kwargs for
    ``edit_call``: ``sf`` caches ONE independent frame, ``ll`` repeats it x3 as one block
    (``triple_first_frame``, LongLive's default).
    """
    def _enc(img):
        return pipeline.vae.encode_to_latent(
            transform(img).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
        ).to(device=device, dtype=torch.bfloat16)

    src_lat = _enc(src_frame0)
    trg_lat = _enc(anchor_img)
    pipeline.vae.model.clear_cache()
    if backbone == "ll":
        src_lat = src_lat.repeat_interleave(3, dim=1)      # [B, F, C, H, W]
        trg_lat = trg_lat.repeat_interleave(3, dim=1)
        return src_lat, trg_lat, dict(independent_first_frame=False, triple_first_frame=True)
    return src_lat, trg_lat, dict(independent_first_frame=True, triple_first_frame=False)


def edit_call(backbone: str, pipeline, *, rollout_chunk_size: int = 21,
              rollout_overlap_block_num: int = 1, **kw):
    """One edit, same return contract on both backbones.

    ``kw`` are ``rollout_inference`` kwargs. ``sf`` forwards everything (plus the two rollout
    sizes); ``ll`` forwards the subset its ``inference`` implements -- including
    ``blend_sched``, ``rho_frames`` and ``step_dump`` (R45 port) -- and refuses any SF-only
    option that is set.
    """
    if backbone == "sf":
        return pipeline.rollout_inference(rollout_chunk_size=rollout_chunk_size,
                                          rollout_overlap_block_num=rollout_overlap_block_num,
                                          **kw)
    set_sf_only = [k for k in _SF_ONLY if kw.get(k) is not None and kw.get(k) is not False]
    if set_sf_only:
        raise ValueError(f"backbone 'll' does not implement {set_sf_only}; these are "
                         "Self-Forcing-only options.")
    for k in _SF_ONLY:
        kw.pop(k, None)
    return pipeline.inference(**kw)
