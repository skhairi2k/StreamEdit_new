"""R10 analysis driver: run the per-frame evaluator once per arm, then summarise.

Deliberately contains no metric maths of its own -- everything routes through
evaluation/fivebench/evaluate.py --per_frame, which uses the unmodified
metrics_calculator. That is what keeps R10's numbers comparable to R1/R7.

Three axes, per arm x clip:
  MOTION              motion_fidelity_score           (whole-clip, frame_idx=-1)
  EDIT PERSISTENCE    clip_similarity_target_image    (per-frame -> mean + slope)
  BACKGROUND PRESERV. psnr/lpips/mse/ssim_unedit_part,
                      structure_distance              (per-frame -> mean + slope)

CAUTION -- `motion_fidelity_score` here is WHOLE-FRAME (video_masks=None, a 55x55
tracking grid over the full image). Every arm injects source background KV, so
the background is pinned to the source and tracks near perfectly no matter what
the gate does; the foreground signal is diluted to nothing (`all` vs `none`
scored 0.775 vs 0.773, p=0.45). R10's predictions are all about the EDIT PART,
so the arm-discriminating metrics are the masked variants. The default list below
is kept only for backward compatibility with R1/R7 call sites --
`slurm_scripts/five_bench/r10_eval.sh` passes `--metrics` explicitly and is the
only thing that should be used to score R10. Do not read a null on whole-frame
motion_fidelity_score as evidence about heads.

The slope (per-frame OLS fit vs frame index) is the fading signal claim 4
predicts: temporal-only should drift away from the target prompt over the clip
while spatial-only holds.

Outputs:
  evaluation/csv/r10_arms.csv              one row per (arm, video)
  evaluation/csv/r10_per_frame.csv         concatenated long-format rows
  evaluation/figures/r10_fading_curves.pdf one panel per clip, one line per arm
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIVEBENCH = _REPO_ROOT / "evaluation" / "fivebench"

# Metrics summarised with a per-frame trend line in r10_arms.csv.
_CURVE_METRIC = "clip_similarity_target_image"

# R7 §4.5 enters as a reference arm, not a controlled ablation: it writes the
# anchor into kv_cache_trg (so the VP fades as the window slides) AND runs with
# bridge blending on, whereas the R10 arms run blend_off. It is the decaying
# baseline the persistent bank is argued against; read gaps accordingly.
_R7_ARM = "r7_vp"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_root", type=str, required=True,
                   help="root holding {arm}/edit{T}/{video_name}/ frame dirs")
    p.add_argument("--arms", nargs="+", default=["none", "all", "spatial", "temporal"])
    p.add_argument("--r7_root", type=str, default=None,
                   help="R7 §4.5 output root, scored as a fifth reference arm named "
                        "'r7_vp' ({r7_root}/edit{T}/{video}, i.e. --tgt_layout edit_video). "
                        "This is the FADING VP baseline the persistent bank is argued "
                        "against -- without it the curves have no decaying reference. "
                        "NOT a controlled ablation: R7 also runs with bridge blending ON, "
                        "so an r7_vp-vs-all gap conflates persistence with blend_off.")
    p.add_argument("--src_image_folder", type=str, required=True,
                   help="FiVE-Bench root (holds images/ and bmasks/)")
    p.add_argument("--annotation_dir", type=str, required=True,
                   help="dir holding editN_FiVE.json")
    p.add_argument("--cases_json", type=str, default=str(_REPO_ROOT / "evaluation" / "cases.json"))
    p.add_argument("--frame_stride", type=int, default=2)
    p.add_argument("--metrics", nargs="+", default=[
        "structure_distance",
        "psnr_unedit_part",
        "lpips_unedit_part",
        "mse_unedit_part",
        "ssim_unedit_part",
        "clip_similarity_target_image",
        "clip_similarity_target_image_edit_part",
        "motion_fidelity_score",
    ])
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--out_fig", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--skip_eval", action="store_true",
                   help="reuse existing per-arm CSVs instead of re-running the evaluator")
    return p.parse_args()


def arm_targets(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    """(arm_name, target_root, tgt_layout) for every arm to score.

    Both the R10 arms and R7 §4.5 use {root}/edit{T}/{video}. R10 needs the
    edit{T} level because 0011_lucia is scored under two edit types, which a flat
    {root}/{video} layout cannot address separately.

    Each --arms entry may be written as ``dir`` or ``dir=key``. The directory is
    what exists on disk; the key is what names the CSV column. They have to be
    separable because r10_vp_arms.py forces the directory to equal the GATE name,
    so the persistent-bank condition is called ``all`` whether blending is on or
    off -- two different experiments, one name, and a silent collision the moment
    R10 and R10b are put in one table. Score them as
    ``all=all_blendoff`` and ``all=all_blendon``.

    A ``dir`` containing a separator is treated as a path in its own right rather
    than a child of --in_root, so arms living under unrelated roots (baseline,
    r7_visual_prompting, r10b_blend_on) can be scored in one invocation.
    """
    # Absolute: evaluate.py runs with cwd=evaluation/fivebench.
    in_root = Path(args.in_root).expanduser().resolve()
    targets = []
    for spec in args.arms:
        d, _, key = spec.partition("=")
        root = Path(d).expanduser().resolve() if "/" in d else in_root / d
        targets.append((key or d, str(root), "edit_video"))
    if args.r7_root:
        targets.append((_R7_ARM, str(Path(args.r7_root).expanduser().resolve()), "edit_video"))
    return targets


def run_arm(args: argparse.Namespace, arm: str, tgt_root: str, layout: str, out_csv: Path) -> Path:
    """Invoke evaluate.py --per_frame for one arm; return the CSV it wrote."""
    annotation_files = sorted(Path(args.annotation_dir).expanduser().resolve().glob("edit*_FiVE.json"))
    if not annotation_files:
        raise SystemExit(f"[r10] no edit*_FiVE.json under {args.annotation_dir}")

    result_path = out_csv / f"r10_{arm}.csv"
    cmd = [
        sys.executable, "evaluate.py",
        "--per_frame",
        "--config_path", "config.yaml",
        "--src_image_folder", str(Path(args.src_image_folder).expanduser().resolve()),
        "--annotation_mapping_files", *[str(f) for f in annotation_files],
        "--tgt_methods", tgt_root,
        "--tgt_layout", layout,
        "--tgt_key", arm,
        # NB: every path handed to evaluate.py must be absolute -- it runs with
        # cwd=evaluation/fivebench, so repo-relative paths would not resolve.
        "--cases_json", str(Path(args.cases_json).expanduser().resolve()),
        "--frame_stride", str(args.frame_stride),
        "--metrics", *args.metrics,
        "--device", args.device,
        "--result_path", str(result_path.resolve()),
    ]
    print(f"[r10] arm={arm}: {' '.join(cmd)}", flush=True)
    # cwd=fivebench so metrics_calculator and config.yaml resolve as they do for
    # evaluate.py in the R2/R5 slurm scripts.
    subprocess.run(cmd, cwd=_FIVEBENCH, check=True)

    # evaluate.py appends _frame_stride{N} to --result_path, then _per_frame.
    written = result_path.with_name(
        result_path.name.replace(".csv", f"_frame_stride{args.frame_stride}_per_frame.csv")
    )
    if not written.exists():
        raise SystemExit(f"[r10] evaluator produced no per-frame CSV for arm {arm}: expected {written}")
    return written


def slope(frame_idx: np.ndarray, values: np.ndarray) -> float:
    """OLS slope per 100 frames; nan if fewer than 2 finite points."""
    ok = np.isfinite(values)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(frame_idx[ok], values[ok], 1)[0] * 100.0)


def add_clip_key(df: pd.DataFrame) -> pd.DataFrame:
    """Key clips by (video, edit type), not video alone.

    0011_lucia appears in the panel twice -- edit2 and edit5 -- as two different
    edits of one source video. Grouping on video_name alone would silently pool
    them into a single row and a single curve.
    """
    df = df.copy()
    df["clip"] = df["video_name"].astype(str) + "@edit" + df["editing_type_id"].astype(str)
    return df


def common_clips(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict every arm to the clips present in ALL arms.

    r7_vp is scored from R7's full-bench render, so it covers cases.json entries
    the R10 arms never rendered (0034_cows, 0045_butterfly -- kept in cases.json
    for R9). Left unchecked, a pooled arm-vs-arm mean would compare different clip
    sets. Drop loudly rather than silently averaging over a ragged panel.
    """
    per_arm = {arm: set(g["clip"]) for arm, g in df.groupby("method")}
    shared = set.intersection(*per_arm.values()) if per_arm else set()
    for arm, clips in sorted(per_arm.items()):
        extra = sorted(clips - shared)
        if extra:
            print(f"[r10] WARNING: dropping {len(extra)} clip(s) scored only for "
                  f"'{arm}': {extra} -- not rendered by every arm")
    if not shared:
        raise SystemExit("[r10] no clip is present in every arm; nothing comparable to summarise")
    print(f"[r10] comparing {len(shared)} clips shared by all {len(per_arm)} arms")
    return df[df["clip"].isin(shared)]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (method, clip): whole-clip values + per-frame mean/slope."""
    df = df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    per_frame = df[df["frame_idx"] >= 0]
    whole_clip = df[df["frame_idx"] < 0]

    rows = []
    for (method, clip), g in per_frame.groupby(["method", "clip"], sort=False):
        row = {"arm": method, "clip": clip,
               "video_name": g["video_name"].iloc[0],
               "editing_type_id": g["editing_type_id"].iloc[0],
               "n_frames_scored": g["frame_idx"].nunique()}
        for metric, gm in g.groupby("metric"):
            gm = gm.sort_values("frame_idx")
            row[f"{metric}_mean"] = gm["value"].mean()
            row[f"{metric}_slope_per100f"] = slope(
                gm["frame_idx"].to_numpy(dtype=float), gm["value"].to_numpy(dtype=float)
            )
        wc = whole_clip[(whole_clip["method"] == method) & (whole_clip["clip"] == clip)]
        for metric, gm in wc.groupby("metric"):
            row[metric] = gm["value"].mean()
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["clip", "arm"]).reset_index(drop=True)


def plot_fading_curves(df: pd.DataFrame, arms: list[str], out_pdf: Path) -> None:
    """One panel per clip; CLIP-to-target vs frame index, one line per arm."""
    curve = df[(df["metric"] == _CURVE_METRIC) & (df["frame_idx"] >= 0)].copy()
    curve["value"] = pd.to_numeric(curve["value"], errors="coerce")
    if curve.empty:
        print(f"[r10] WARNING: no {_CURVE_METRIC} rows -- skipping fading curves")
        return

    videos = sorted(curve["clip"].unique())
    ncols = 3
    nrows = int(np.ceil(len(videos) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 3.2 * nrows), squeeze=False)

    for ax, video in zip(axes.ravel(), videos):
        g = curve[curve["clip"] == video]
        for arm in arms:
            ga = g[g["method"] == arm].sort_values("frame_idx")
            if ga.empty:
                continue
            # Dashed + grey: a reference baseline, not one of the gated arms.
            style = dict(linestyle="--", color="0.45") if arm == _R7_ARM else {}
            ax.plot(ga["frame_idx"], ga["value"], marker="o", markersize=2.5,
                    linewidth=1.2, label=arm, **style)
        ax.set_title(video, fontsize=9)
        ax.set_xlabel("frame")
        ax.set_ylabel("CLIP-to-target")
        ax.grid(alpha=0.25, linewidth=0.5)
    for ax in axes.ravel()[len(videos):]:
        ax.axis("off")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle("R10 — edit persistence over the clip (higher = edit still present)", fontsize=11)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[r10] fading curves -> {out_pdf}")


def main() -> None:
    args = parse_args()
    out_csv = Path(args.out_csv)
    out_csv.mkdir(parents=True, exist_ok=True)

    targets = arm_targets(args)
    frames = []
    for arm, tgt_root, layout in targets:
        if args.skip_eval:
            path = out_csv / f"r10_{arm}_frame_stride{args.frame_stride}_per_frame.csv"
            if not path.exists():
                raise SystemExit(f"[r10] --skip_eval but {path} is missing")
        else:
            path = run_arm(args, arm, tgt_root, layout, out_csv)
        frames.append(pd.read_csv(path))

    df = pd.concat(frames, ignore_index=True)
    long_path = out_csv / "r10_per_frame.csv"
    df.to_csv(long_path, index=False)
    print(f"[r10] per-frame rows -> {long_path} ({len(df)} rows, all clips as scored)")

    # Everything downstream compares arms, so it runs on the shared panel only.
    df = common_clips(add_clip_key(df))
    summary = summarise(df)
    summary_path = out_csv / "r10_arms.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[r10] summary -> {summary_path} ({len(summary)} arm x clip rows)")

    plot_fading_curves(df, [a for a, _, _ in targets], Path(args.out_fig) / "r10_fading_curves.pdf")


if __name__ == "__main__":
    main()
