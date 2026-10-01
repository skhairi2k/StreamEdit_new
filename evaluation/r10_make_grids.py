"""R10 qualitative grids: one comparison figure per clip.

Rows are src / none / all / spatial / temporal / R7-§4.5, columns are frames
[0, T/3, 2T/3, T-1]. This is the paper's motivation figure: if the routing
thesis holds, the `spatial` row shows motion returning while the edit persists,
and the `temporal` row shows a static clip whose appearance drifts.

The R7-§4.5 row is free (already rendered by R7) and is skipped per clip when
that output is missing, rather than failing the whole figure.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]

_ARM_ROWS = ["none", "all", "spatial", "temporal"]
# Rows holding ONE reference image rather than a temporal sequence. They are
# excluded from the shared column count (a 1-frame row would otherwise collapse
# every grid to a single column) and rendered only in column 0.
_SINGLE_ROWS = {"anchor"}
_ROW_LABELS = {
    "src": "source",
    "anchor": "appearance anchor",
    "none": "no VP",
    "all": "VP: all heads",
    "spatial": "VP: spatial only",
    "temporal": "VP: temporal only",
    "rand_spatial": "VP: 117 random (ctrl)",
    "rand_temporal": "VP: 49 random (ctrl)",
    "r7": "VP baseline (R7 §4.5)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_root", type=str, required=True,
                   help="root holding {arm}/edit{T}/{video}/ frames")
    p.add_argument("--r7_root", type=str, default=None,
                   help="R7 output root; rows resolved as {r7_root}/edit{T}/{video}/")
    p.add_argument("--src_root", type=str,
                   default="~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark/images",
                   help="source frame dirs, {src_root}/{video}/")
    p.add_argument("--anchor_root", type=str,
                   default="/projects/dataggen/outputs/five_bench/anchors",
                   help="Qwen-edited anchors, resolved as {root}/edit{T}/{video}.png -- "
                        "the same files r10_vp_arms.py injects. Pass '' to omit the row.")
    p.add_argument("--cases_json", type=str, default=str(_REPO_ROOT / "evaluation" / "cases.json"))
    p.add_argument("--arms", nargs="+", default=_ARM_ROWS)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def list_frames(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted([p for p in d.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])


def pick_indices(n: int) -> list[int]:
    """[0, T/3, 2T/3, T-1] clamped to the available frame count."""
    if n == 0:
        return []
    return sorted({0, n // 3, (2 * n) // 3, n - 1})


def main() -> None:
    args = parse_args()
    in_root = Path(args.in_root)
    src_root = Path(args.src_root).expanduser()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = json.loads(Path(args.cases_json).expanduser().read_text())

    # Drive off cases.json rather than globbing the arm dirs: the panel holds one
    # video under two edit types (0011_lucia), so (video, edit_type) -- not video --
    # identifies a clip, and a glob cannot recover the edit type.
    # Only clips actually rendered are plotted; cases.json is a shared dev file and
    # holds R9-only entries R10 never ran. evaluate.py also writes resized copies to
    # {clip}_resize/ inside the arm dirs; those are eval scratch, not clips.
    rendered = []
    for case in cases:
        rel = Path(f"edit{case['edit_type']}") / case["video_name"]
        if any((in_root / arm / rel).is_dir() for arm in args.arms):
            rendered.append((case, rel))
    if not rendered:
        raise SystemExit(f"[r10-grids] no rendered clips under {in_root}")

    for case, rel in rendered:
        video = case["video_name"]
        # Filename must carry the edit type, or the two lucia grids overwrite each other.
        clip_key = f"{video}_edit{case['edit_type']}"
        rows: list[tuple[str, list[Path]]] = []

        src_frames = list_frames(src_root / video)
        if src_frames:
            rows.append(("src", src_frames))

        # The anchor is the appearance target every VP arm is trying to hold, so
        # it belongs beside the source rather than buried under the arms. It is a
        # single Qwen-edited frame, not a clip.
        if args.anchor_root:
            anchor = (Path(args.anchor_root).expanduser()
                      / f"edit{case['edit_type']}" / f"{video}.png")
            if anchor.exists():
                rows.append(("anchor", [anchor]))
            else:
                print(f"[r10-grids] {clip_key}: no anchor at {anchor} -- row skipped")

        # The VP baseline sits directly under the source so every R10 arm below it
        # is read against both references at once: the unedited clip, and the
        # existing visual-prompting recipe these arms are meant to improve on.
        if args.r7_root:
            r7_dir = Path(args.r7_root) / rel
            r7_frames = list_frames(r7_dir)
            if r7_frames:
                rows.append(("r7", r7_frames))
            else:
                print(f"[r10-grids] {clip_key}: no R7 §4.5 output at {r7_dir} -- row skipped")

        for arm in args.arms:
            frames = list_frames(in_root / arm / rel)
            if frames:
                rows.append((arm, frames))
            else:
                print(f"[r10-grids] {clip_key}: arm '{arm}' missing -- row skipped")

        if not rows:
            print(f"[r10-grids] {clip_key}: nothing to plot -- skipped")
            continue

        # Column positions come from the shortest SEQUENCE row so every row has the
        # frame: arms truncate at different lengths than the source clip (4n+1
        # chunking). Single-image rows are excluded or they would force one column.
        seq_rows = [(k, f) for k, f in rows if k not in _SINGLE_ROWS]
        if not seq_rows:
            print(f"[r10-grids] {clip_key}: no sequence rows -- skipped")
            continue
        n_common = min(len(f) for _, f in seq_rows)
        cols = pick_indices(n_common)
        # Frame numbers go on the first row that actually has frames.
        title_row = next(r for r, (k, _) in enumerate(rows) if k not in _SINGLE_ROWS)

        fig, axes = plt.subplots(
            len(rows), len(cols), figsize=(2.6 * len(cols), 1.8 * len(rows)), squeeze=False
        )
        for r, (key, frames) in enumerate(rows):
            def label(ax):
                ax.set_ylabel(_ROW_LABELS.get(key, key), fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=42)

            if key in _SINGLE_ROWS:
                ax = axes[r][0]
                ax.imshow(Image.open(frames[0]).convert("RGB"))
                ax.set_xticks([])
                ax.set_yticks([])
                label(ax)
                for c in range(1, len(cols)):      # one image, not a clip
                    axes[r][c].axis("off")
                continue

            for c, idx in enumerate(cols):
                ax = axes[r][c]
                ax.imshow(Image.open(frames[idx]).convert("RGB"))
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    label(ax)
                if r == title_row:
                    ax.set_title(f"frame {idx}", fontsize=8)

        trg = case.get("trg_prompt", "")
        fig.suptitle(f"{clip_key} — {trg[:90]}", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = out_dir / f"{clip_key}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[r10-grids] {out_path}")


if __name__ == "__main__":
    main()
