"""R48 stage 1 (Wan-Edit): the 17-step high-editability draft + per-token cross-attention recording.

Draft: ``sample_steps 17``, shift 5, ``skip_timesteps 3`` (start sigma 0.958), seeded noise, text only --
14 active steps, 4 model calls each (cond_src, cond_tgt, uncond_src, uncond_tgt; n_avg 1).

Attention (read-only forward-pre-hooks, Wan-Edit's own code untouched): on the first 20 of the 30 blocks, for the
draft steps whose sigma lies in [0.55, 0.9] (steps 7..13 of the 17-grid, asserted), recompute q/k with the
block's own layers and the softmax over all 512 padded text tokens (scale 1/sqrt(d), as flash_attention does),
then record, mean over heads:
  * source branch -> one map per SOURCE trigger token, target branch -> one map per TARGET trigger token;
  * each branch's total mass on the real (non-pad) prompt tokens (Eq. 7's "other tokens" denominator).
Everything is averaged over those 20 blocks and the window steps (divided inline) and written in the SAME npz
schema r31_stage1.py --attn_out_dir writes for StreamEdit, so r48_field.compute_E never sees a host flag.

Outputs:
  {out_attn}/edit{T}/{video}.npz            src_tokens/trg_tokens [nT, F, 30, 52], src_prompt/trg_prompt [F, 30, 52],
                                            idx_src, idx_trg, n_prompt_src, n_prompt_trg, steps_t, video_name, edit_type
  {out_frames}/step00/edit{T}/{video}/00000.png ...   the draft video (edit_video layout)
usage: python evaluation/r48_wan_draft.py --cases evaluation/r48_cases.json --out_attn DIR --out_frames DIR
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r48_wan_common as C  # noqa: E402

SAMPLE_STEPS, SKIP, SHIFT = 17, 3, 5.0
SIGMA_LO, SIGMA_HI = 0.55, 0.9
N_BLOCKS = 20          # first 20 cross-attention blocks (StreamEdit's own mask_layers = range(20); better localised than all 30)


def sigma_grid(n_steps: int = SAMPLE_STEPS, shift: float = SHIFT) -> np.ndarray:
    """The host scheduler's sigma at every step index: shift*t/(1+(shift-1)t), t = (N-i)/N."""
    t = (n_steps - np.arange(n_steps)) / n_steps
    return shift * t / (1 + (shift - 1) * t)


def window_steps() -> list:
    sg = sigma_grid()
    win = [i for i in range(SKIP, SAMPLE_STEPS) if SIGMA_LO <= sg[i] <= SIGMA_HI]
    assert win == list(range(7, 14)), f"attention window drifted from steps 7..13: {win}"
    return win


class Recorder:
    """Counts model calls (branch/step from the call index) and accumulates window attention."""

    def __init__(self, model, idx_src, idx_trg, n_src, n_trg, f_lat, win, n_blocks=N_BLOCKS):
        self.model, self.win = model, set(win)
        self.idx = {"src": idx_src, "tgt": idx_trg}
        self.nprompt = {"src": n_src, "tgt": n_trg}
        self.denom = n_blocks * len(win)           # first n_blocks blocks only (default: the first 20)
        grid = (f_lat,) + C.GRID
        self.tok = {b: np.zeros((len(self.idx[b]),) + grid, np.float32) for b in ("src", "tgt")}
        self.prompt = {b: np.zeros(grid, np.float32) for b in ("src", "tgt")}
        self.calls, self.branch, self.step = 0, None, None
        self.handles = [model.register_forward_pre_hook(self._on_model)]
        for i, blk in enumerate(model.blocks):
            if i < n_blocks:
                self.handles.append(blk.cross_attn.register_forward_pre_hook(self._make_hook()))

    def _on_model(self, module, args):
        c = self.calls
        self.calls += 1
        self.branch = ["src", "tgt", None, None][c % 4]
        self.step = SKIP + c // 4

    def _make_hook(self):
        @torch.no_grad()
        def hook(module, args):
            if self.branch is None or self.step not in self.win:
                return
            x, context = args[0], args[1]
            n, d = module.num_heads, module.head_dim
            q = module.norm_q(module.q(x)).float().view(-1, n, d)               # [L, n, d]
            k = module.norm_k(module.k(context)).float().view(-1, n, d)         # [512, n, d]
            idx, npr = self.idx[self.branch], self.nprompt[self.branch]
            pt, pp = [], []
            for s in range(0, q.shape[0], 4096):
                p = torch.softmax(torch.einsum("lnd,snd->nls", q[s:s + 4096], k) * d ** -0.5, dim=-1)
                pt.append(p[..., idx].mean(0))                                  # [l, nT]
                pp.append(p[..., :npr].sum(-1).mean(0))                         # [l]
            tok = torch.cat(pt).cpu().numpy().T.reshape(self.tok[self.branch].shape)
            pro = torch.cat(pp).cpu().numpy().reshape(self.prompt[self.branch].shape)
            self.tok[self.branch] += tok / self.denom
            self.prompt[self.branch] += pro / self.denom
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--anno_dir", default=None,
                    help="annotation folder with edit{T}_FiVE.json (default: the released FiVE edit_prompt)")
    ap.add_argument("--out_attn", required=True)
    ap.add_argument("--out_frames", required=True)
    a = ap.parse_args()
    out_attn, out_frames = Path(a.out_attn).expanduser(), Path(a.out_frames).expanduser()

    win = window_steps()
    cases = C.load_cases(a.cases, a.anno_dir)
    cli = C.load_cli()
    se = C.load_se_utils()
    pipe, d = C.build_pipeline(cli)
    tokenizer = pipe.text_encoder.tokenizer.tokenizer
    f_lat = C.n_latent_frames()
    n_fail = 0
    for c in cases:
        vid, t = c["video_name"], c["edit_type"]
        npz = out_attn / f"edit{t}" / f"{vid}.npz"
        if npz.exists():
            print(f"[r48_wan_draft] skip existing {t}/{vid}", flush=True)
            continue
        try:
            idx_src = se.find_phrase_token_indices(tokenizer, c["source_prompt"], c["source_object"])[0]
            idx_trg = se.find_phrase_token_indices(tokenizer, c["target_prompt"], c["target_object"])[0]
            if not idx_src or not idx_trg:
                raise ValueError(f"trigger phrase not found (src {idx_src}, trg {idx_trg})")
            n_src = len(tokenizer(c["source_prompt"], add_special_tokens=True)["input_ids"])
            n_trg = len(tokenizer(c["target_prompt"], add_special_tokens=True)["input_ids"])
            video = cli.load_frames(str(C.FIVE / "videos" / f"{vid}.mp4"))
            f_lat = (min(C.FRAME_NUM, video.shape[2]) - 1) // 4 + 1      # 11 for 41 frames, fewer for the 7 short videos
            rec = Recorder(pipe.model, idx_src, idx_trg, n_src, n_trg, f_lat, win)
            try:
                out = pipe.edit(
                    video, c["source_prompt"], c["target_prompt"], size=cli.SIZE_CONFIGS[C.SIZE],
                    frame_num=min(C.FRAME_NUM, video.shape[2]), shift=SHIFT, sample_solver=d.sample_solver,
                    sampling_steps=SAMPLE_STEPS, guide_scale=d.sample_guide_scale,
                    tgt_guide_scale=d.tgt_guide_scale, skip_timesteps=SKIP, n_avg=1, seed_noise=True,
                    seed=0, offload_model=False)
            finally:
                rec.close()
            assert rec.calls == 4 * (SAMPLE_STEPS - SKIP), f"unexpected model-call count {rec.calls}"
            C.save_frames(out, out_frames / "step00" / f"edit{t}" / vid)
            npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                npz, src_tokens=rec.tok["src"], trg_tokens=rec.tok["tgt"],
                src_prompt=rec.prompt["src"], trg_prompt=rec.prompt["tgt"],
                idx_src=np.array(idx_src, dtype=np.int64), idx_trg=np.array(idx_trg, dtype=np.int64),
                n_prompt_src=np.int32(n_src), n_prompt_trg=np.int32(n_trg),
                steps_t=sigma_grid().astype(np.float32), video_name=np.str_(vid), edit_type=np.int32(t))
            print(f"[r48_wan_draft] ok {t}/{vid}: tokens src {idx_src} trg {idx_trg}, calls {rec.calls}", flush=True)
        except Exception as e:                                   # noqa: BLE001 -- report per clip, keep going
            n_fail += 1
            print(f"[r48_wan_draft] FAILED {t}/{vid}: {type(e).__name__}: {e}", flush=True)
        torch.cuda.empty_cache()
    print(f"[r48_wan_draft] attn npz total: {sum(1 for _ in out_attn.glob('edit*/*.npz'))}  failures: {n_fail}")
    sys.exit(int(n_fail > 0))


if __name__ == "__main__":
    main()
