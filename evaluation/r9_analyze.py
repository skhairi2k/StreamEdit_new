"""R9 -- aggregate head-taxonomy profiling scalars into tau, margins, and figures.

Consumes the per-video ``r9_scalars.npz`` files written by
``evaluation/r9_head_profiler.py`` and produces:

  * ``{out_tau}``                       -- tau [30,12] bool (True = temporal),
                                           majority vote over last-3 profile steps
                                           at the last block, then across videos
  * ``{out_tau%.pt}_per_block.pt``      -- per-block tau table [n_blocks,30,12]
  * ``{out_csv}/r9_head_margins.csv``   -- per-head errors, margins, flip stats
  * ``{out_fig}/r9_layer_hist.pdf``     -- per-layer head-type counts
  * ``{out_fig}/r9_margin_dist.pdf``    -- margin histogram (pooled + per video)
  * ``{out_fig}/r9_stability.pdf``      -- step / video / block flip rates
  * ``{out_fig}/r9_ramp.pdf``           -- margin vs block (reach ramp)
  * ``{out_fig}/r9_maps/*.png``         -- raw attention panels for 6 post-hoc
                                           picked (layer, head) pairs

Margin convention: m = (err_spat - err_temp) / (err_spat + err_temp), so
m > 0 means the temporal (same-position stripe) key-set reconstructs the head
better => the head is classified temporal.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAST3 = slice(-3, None)          # profile steps {12,13,14} within the stored axis
AMBIG_BAND = 0.05                # provisional |margin| band shown in figures;
                                 # the binding 3-way decision is R12's


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in_root", type=str, required=True,
                   help="Root holding {case}/r9_scalars.npz")
    p.add_argument("--out_csv", type=str, default="evaluation/csv")
    p.add_argument("--out_fig", type=str, default="evaluation/figures")
    p.add_argument("--out_tau", type=str, default="evaluation/r9_tau.pt")
    return p.parse_args()


def load_runs(in_root: Path):
    runs = []
    for npz_path in sorted(in_root.glob("*/r9_scalars.npz")):
        d = np.load(npz_path, allow_pickle=False)
        meta = json.loads(bytes(d["meta"]).decode())
        runs.append(dict(
            case=meta["case_id"], meta=meta,
            err_spat=d["err_spat"], err_temp=d["err_temp"],   # [B, S, L, H]
            maps=d["maps"] if "maps" in d.files else None,
            map_rows=d["map_rows"], map_hw=tuple(d["map_hw"]),
            svis=d["svis"],
        ))
    if not runs:
        raise SystemExit(f"[r9_analyze] no r9_scalars.npz under {in_root}")
    return runs


def margins_of(run) -> np.ndarray:
    """Signed margin per (block, step, layer, head); >0 = temporal."""
    es, et = run["err_spat"], run["err_temp"]
    return (es - et) / (es + et + 1e-12)


def main() -> None:
    args = parse_args()
    in_root = Path(args.in_root).expanduser().resolve()
    out_csv = Path(args.out_csv); out_csv.mkdir(parents=True, exist_ok=True)
    out_fig = Path(args.out_fig); out_fig.mkdir(parents=True, exist_ok=True)
    out_tau = Path(args.out_tau)

    runs = load_runs(in_root)
    n_videos = len(runs)
    L, H = runs[0]["err_spat"].shape[2:]
    n_blocks = max(r["err_spat"].shape[0] for r in runs)
    print(f"[r9_analyze] {n_videos} videos, layers={L}, heads={H}, "
          f"max blocks={n_blocks}")

    # ---- per-video classification point: last block x last-3 steps ----
    # cls_margin[v, L, H]: mean margin at the classification point
    cls_margin = np.stack([
        margins_of(r)[r["err_spat"].shape[0] - 1, LAST3].mean(axis=0) for r in runs
    ])                                                        # [V, L, H]
    pooled_margin = cls_margin.mean(axis=0)                   # [L, H]
    votes = (cls_margin > 0)
    tau = votes.sum(axis=0) > n_videos / 2                    # [L, H] True=temporal

    # ---- per-block tau (video-majority at each block, last-3 steps) ----
    per_block_votes = np.full((n_videos, n_blocks, L, H), np.nan)
    for v, r in enumerate(runs):
        b = r["err_spat"].shape[0]
        per_block_votes[v, :b] = margins_of(r)[:, LAST3].mean(axis=1)
    tau_per_block = np.nanmean(per_block_votes > 0, axis=0) > 0.5   # [B, L, H]

    # ---- stability ----
    # steps: consensus sign at {0,7} vs last-3 consensus, per video, then mean
    step_flip = np.zeros((L, H))
    for v, r in enumerate(runs):
        m = margins_of(r)[r["err_spat"].shape[0] - 1]         # [S, L, H]
        early = (m[:2].mean(axis=0) > 0)
        late = (m[LAST3].mean(axis=0) > 0)
        step_flip += (early != late)
    step_flip /= n_videos
    # videos: pairwise disagreement of the classification-point sign
    pair_flips = []
    for i in range(n_videos):
        for j in range(i + 1, n_videos):
            pair_flips.append((votes[i] != votes[j]).astype(float))
    video_flip = np.mean(pair_flips, axis=0) if pair_flips else np.zeros((L, H))
    # blocks: per-block sign vs last-block sign (video-pooled margins)
    block_margin = np.nanmean(per_block_votes, axis=0)        # [B, L, H]
    block_flip = np.mean(
        (block_margin > 0) != (block_margin[-1] > 0)[None], axis=(1, 2))  # [B]

    # ---- outputs: tau tensors ----
    import torch
    torch.save(torch.from_numpy(tau), out_tau)
    per_block_path = out_tau.with_name(out_tau.stem + "_per_block.pt")
    torch.save(torch.from_numpy(tau_per_block), per_block_path)

    # ---- CSV ----
    csv_path = out_csv / "r9_head_margins.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = (["layer", "head", "err_spat", "err_temp", "margin", "tau_temporal",
                   "step_flip", "video_flip"]
                  + [f"margin_block_{b}" for b in range(n_blocks)])
        w.writerow(header)
        es_mean = np.stack([r["err_spat"][r["err_spat"].shape[0] - 1, LAST3].mean(axis=0)
                            for r in runs]).mean(axis=0)
        et_mean = np.stack([r["err_temp"][r["err_temp"].shape[0] - 1, LAST3].mean(axis=0)
                            for r in runs]).mean(axis=0)
        for l in range(L):
            for h in range(H):
                w.writerow([l, h, f"{es_mean[l,h]:.6g}", f"{et_mean[l,h]:.6g}",
                            f"{pooled_margin[l,h]:.4f}", int(tau[l, h]),
                            f"{step_flip[l,h]:.3f}", f"{video_flip[l,h]:.3f}"]
                           + [f"{block_margin[b,l,h]:.4f}" for b in range(n_blocks)])

    # ---- figures ----
    ambig = np.abs(pooled_margin) < AMBIG_BAND

    fig, ax = plt.subplots(figsize=(9, 3.2))
    temp_c = (tau & ~ambig).sum(axis=1)
    spat_c = (~tau & ~ambig).sum(axis=1)
    amb_c = ambig.sum(axis=1)
    x = np.arange(L)
    ax.bar(x, spat_c, label="spatial")
    ax.bar(x, temp_c, bottom=spat_c, label="temporal")
    ax.bar(x, amb_c, bottom=spat_c + temp_c, label=f"|m|<{AMBIG_BAND} (provisional)")
    ax.set_xlabel("layer"); ax.set_ylabel("#heads"); ax.legend(fontsize=8)
    ax.set_title("R9: head types per layer (last block, last-3 steps, video majority)")
    fig.tight_layout(); fig.savefig(out_fig / "r9_layer_hist.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.linspace(-1, 1, 61)
    ax.hist(pooled_margin.ravel(), bins=bins, alpha=0.85, label="pooled")
    for v, r in enumerate(runs):
        ax.hist(cls_margin[v].ravel(), bins=bins, histtype="step",
                linewidth=0.8, label=r["case"])
    ax.axvline(0, color="k", linewidth=0.6)
    ax.set_xlabel("margin (>0 = temporal)"); ax.set_ylabel("#heads")
    ax.legend(fontsize=6); ax.set_title("R9: margin distribution")
    fig.tight_layout(); fig.savefig(out_fig / "r9_margin_dist.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].hist(step_flip.ravel(), bins=20)
    axes[0].set_title(f"step flip (mean {step_flip.mean():.3f})")
    axes[1].hist(video_flip.ravel(), bins=20)
    axes[1].set_title(f"video flip (mean {video_flip.mean():.3f})")
    axes[2].plot(np.arange(n_blocks), block_flip, marker="o")
    axes[2].set_title("block-vs-last flip rate"); axes[2].set_xlabel("block")
    for a in axes[:2]:
        a.set_xlabel("flip rate"); a.set_ylabel("#heads")
    fig.tight_layout(); fig.savefig(out_fig / "r9_stability.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    sel_t = tau & ~ambig
    sel_s = ~tau & ~ambig
    for sel, name in ((sel_t, "temporal"), (sel_s, "spatial"), (ambig, "ambiguous")):
        if sel.any():
            ax.plot(np.arange(n_blocks), block_margin[:, sel].mean(axis=1),
                    marker="o", label=f"{name} (n={int(sel.sum())})")
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xlabel("block (reach ramps 3→21 frames)"); ax.set_ylabel("mean margin")
    ax.legend(fontsize=8); ax.set_title("R9: margin vs block (ramp)")
    fig.tight_layout(); fig.savefig(out_fig / "r9_ramp.pdf"); plt.close(fig)

    # ---- raw map panels: 6 post-hoc picks ----
    maps_run = next((r for r in runs if r["maps"] is not None), None)
    if maps_run is not None:
        maps_dir = out_fig / "r9_maps"; maps_dir.mkdir(exist_ok=True)
        flat = pooled_margin.ravel()
        order = np.argsort(flat)
        picks = (list(order[-2:])                                   # strongest temporal
                 + list(order[:2])                                  # strongest spatial
                 + list(np.argsort(np.abs(flat))[:2]))              # most ambiguous
        hl, wl = maps_run["map_hw"]
        for flat_idx in picks:
            l, h = divmod(int(flat_idx), H)
            rows_probs = maps_run["maps"][l, h].astype(np.float32)  # [P, S_vis]
            n_rows, s_vis = rows_probs.shape
            f_vis = s_vis // (hl * wl)
            fig, axes = plt.subplots(n_rows, 1, figsize=(min(24, f_vis * 1.15), 1.4 * n_rows))
            axes = np.atleast_1d(axes)
            for ri, axr in enumerate(axes):
                grid = rows_probs[ri].reshape(f_vis, hl, wl)
                strip = np.concatenate(list(grid), axis=1)          # frames side by side
                axr.imshow(strip, cmap="viridis", aspect="auto",
                           vmax=np.percentile(rows_probs[ri], 99.9))
                axr.set_ylabel(f"row {maps_run['map_rows'][ri]}", fontsize=6)
                axr.set_xticks([]); axr.set_yticks([])
            kind = "temporal" if flat[flat_idx] > 0 else "spatial"
            if abs(flat[flat_idx]) < AMBIG_BAND:
                kind = "ambiguous"
            axes[0].set_title(
                f"L{l} H{h}  margin={flat[flat_idx]:+.3f} ({kind})  "
                f"[{maps_run['case']}, last block/step; frames left→right]",
                fontsize=8)
            fig.tight_layout()
            fig.savefig(maps_dir / f"r9_map_L{l:02d}_H{h:02d}.png", dpi=160)
            plt.close(fig)
        print(f"[r9_analyze] map panels -> {maps_dir}")

    # ---- console verdict aids ----
    print(f"[r9_analyze] tau: {int(tau.sum())}/{L*H} temporal, "
          f"{int((~tau).sum())} spatial; |m|<{AMBIG_BAND}: {int(ambig.sum())}")
    print(f"[r9_analyze] mean flip rates: step={step_flip.mean():.3f} "
          f"video={video_flip.mean():.3f}; block flips: "
          f"{np.array2string(block_flip, precision=3)}")
    print(f"[r9_analyze] wrote {csv_path}, {out_tau}, {per_block_path}, "
          f"figures -> {out_fig}")


if __name__ == "__main__":
    main()
