"""Shared helpers of the R57 unified metric suite (stdlib only, so every env of the suite can import it).

* pair list: the (edit_type, video_name) pairs of an --edit_json_dir (FiVE-Bench-Correct), optionally restricted by a
  --cases file ([{"video_name": ..., "edit_type": ...}]), with the fields the metrics read;
* frames: the harness's frame convention (evaluation/fivebench/evaluate.py): sorted png/jpg/jpeg, every
  ``--frame_stride``-th frame, source and render zipped positionally after truncating to the shorter, render resized to the
  source's size by the caller;
* result CSV with resume: one row per pair, rewritten atomically after every pair, pairs already scored are skipped on a
  rerun (SLURM jobs get interrupted);
* a small CLI used by the launcher / job scripts:
    counts       six integers, the pair count per edit type (the row-count guards of the job scripts)
    missing      pairs whose render has no frames (and no .mp4) under --root
    clipd_cases  the cases file r33_clip_directional.py reads with --cases (it needs prompts, which --cases files of the
                 suite do not carry)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

EDIT_TYPES = (1, 2, 3, 4, 5, 6)
FIELDS_REQUIRED = ("video_name", "editing_type_id", "instruction", "source_object", "target_object",
                   "source_prompt", "target_prompt")
DEFAULT_DATA_ROOT = "~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"


# ----------------------------------------------------------------------------- pairs ----------

def load_pairs(edit_json_dir: str, cases_path: Optional[str] = None) -> List[Dict[str, object]]:
    """Pairs in annotation-file order. ``file_id`` is the index inside edit{T}_FiVE.json, which is the ``file_id`` the
    FiVE harness writes into its per-type CSVs (the harness rows carry no video name)."""
    wanted = None
    if cases_path:
        wanted = {(str(c["video_name"]), int(c["edit_type"]))
                  for c in json.loads(Path(cases_path).expanduser().read_text())}
    pairs: List[Dict[str, object]] = []
    found = set()
    for t in EDIT_TYPES:
        path = Path(edit_json_dir).expanduser() / f"edit{t}_FiVE.json"
        for file_id, item in enumerate(json.loads(path.read_text())):
            for k in FIELDS_REQUIRED:
                if k not in item:
                    raise KeyError(f"{path}: entry {file_id} ({item.get('video_name')}) has no '{k}'")
            if int(item["editing_type_id"]) != t:
                raise ValueError(f"{path}: entry {file_id} has editing_type_id {item['editing_type_id']}, expected {t}")
            key = (str(item["video_name"]), t)
            if wanted is not None and key not in wanted:
                continue
            found.add(key)
            pairs.append({"edit_type": t, "video_name": item["video_name"], "file_id": file_id,
                          "instruction": item["instruction"], "source_object": item["source_object"],
                          "target_object": item["target_object"], "source_prompt": item["source_prompt"],
                          "target_prompt": item["target_prompt"]})
    if wanted is not None and wanted != found:
        raise ValueError(f"cases not found in {edit_json_dir}: {sorted(wanted - found)}")
    if not pairs:
        raise ValueError(f"no pair selected from {edit_json_dir} (cases: {cases_path})")
    return pairs


def counts_per_type(pairs: Sequence[Dict[str, object]]) -> List[int]:
    return [sum(1 for p in pairs if p["edit_type"] == t) for t in EDIT_TYPES]


# ----------------------------------------------------------------------------- frames ---------

def list_images(directory: str) -> List[str]:
    """Same as evaluate.py's list_images (sorted png/jpg/jpeg)."""
    out: List[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        out.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(out)


def render_dir(root: str, layout: str, edit_type: int, video_name: str) -> str:
    root = os.path.expanduser(root).rstrip("/")
    if layout == "edit_video":
        return os.path.join(root, f"edit{edit_type}", video_name)
    if layout == "flat":
        return os.path.join(root, video_name)
    raise ValueError(f"layout '{layout}' not supported by the suite (edit_video | flat)")


def source_dir(data_root: str, video_name: str) -> str:
    return os.path.join(os.path.expanduser(data_root), "images", video_name)


def ensure_render_frames(directory: str) -> List[str]:
    """Frames of a render directory; when only ``<directory>.mp4`` exists they are extracted next to it with the exact
    ffmpeg call evaluate.py uses (``%05d.jpg``), so the suite scores the frames the harness scored."""
    frames = list_images(directory)
    if frames or not os.path.exists(directory.rstrip("/") + ".mp4"):
        return frames
    os.makedirs(directory, exist_ok=True)
    subprocess.run(["ffmpeg", "-i", directory.rstrip("/") + ".mp4", os.path.join(directory, "%05d.jpg")],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return list_images(directory)


def paired_frames(data_root: str, root: str, layout: str, pair: Dict[str, object],
                  frame_stride: int) -> Tuple[List[str], List[str]]:
    """(source frame paths, render frame paths), strided, truncated to the shorter, positionally paired."""
    src = list_images(source_dir(data_root, str(pair["video_name"])))[::frame_stride]
    tgt = ensure_render_frames(render_dir(root, layout, int(pair["edit_type"]), str(pair["video_name"])))[::frame_stride]
    if not src:
        raise FileNotFoundError(f"no source frames under {source_dir(data_root, str(pair['video_name']))}")
    if not tgt:
        raise FileNotFoundError("no render frames under "
                                f"{render_dir(root, layout, int(pair['edit_type']), str(pair['video_name']))}")
    n = min(len(src), len(tgt))
    return src[:n], tgt[:n]


# ----------------------------------------------------------------------------- results ---------

def finite(x: object) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


class ResultStore:
    """Per-pair result CSV with resume. ``metric_cols`` must all be finite for a row to count as done."""

    def __init__(self, path: str, id_cols: Sequence[str], metric_cols: Sequence[str]):
        self.path = Path(path)
        self.fields = list(id_cols) + list(metric_cols)
        self.metric_cols = list(metric_cols)
        self.rows: Dict[Tuple[int, str], Dict[str, object]] = {}
        if self.path.exists():
            for r in csv.DictReader(self.path.open()):
                if all(finite(r.get(c)) for c in self.metric_cols):
                    self.rows[(int(r["edit_type"]), r["video_name"])] = r

    def done(self, pair: Dict[str, object]) -> bool:
        return (int(pair["edit_type"]), str(pair["video_name"])) in self.rows

    def add(self, row: Dict[str, object]) -> None:
        self.rows[(int(row["edit_type"]), str(row["video_name"]))] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=self.fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted(self.rows.values(), key=lambda r: (int(r["edit_type"]), str(r["video_name"]))))
        os.replace(tmp, self.path)

    def nan_pairs(self) -> List[Tuple[int, str]]:
        return [k for k, r in self.rows.items() if not all(finite(r.get(c)) for c in self.metric_cols)]


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", required=True, help="method render root")
    p.add_argument("--layout", default="edit_video", choices=("edit_video", "flat"))
    p.add_argument("--edit_json_dir", default="evaluation/five_bench_correct")
    p.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--cases", default=None, help="optional subset: [{video_name, edit_type}]")
    p.add_argument("--out", required=True, help="per-pair CSV")


def finish(store: ResultStore, expected: int, name: str) -> int:
    """Exit code of a scoring script: 1 on any missing / NaN pair, after the CSV was written."""
    bad = [k for k in store.nan_pairs()]
    missing = expected - len(store.rows)
    if bad or missing:
        print(f"[{name}] FAILED: {missing} pair(s) not scored, {len(bad)} with NaN -- {store.path}")
        return 1
    print(f"[{name}] done: {len(store.rows)} pairs, all metrics finite -- {store.path}")
    return 0


# ----------------------------------------------------------------------------- CLI -------------

def _cli(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("counts", "missing", "clipd_cases"):
        s = sub.add_parser(name)
        s.add_argument("--edit_json_dir", default="evaluation/five_bench_correct")
        s.add_argument("--cases", default=None)
        if name == "missing":
            s.add_argument("--root", required=True)
            s.add_argument("--layout", default="edit_video", choices=("edit_video", "flat"))
        if name == "clipd_cases":
            s.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
            s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    pairs = load_pairs(a.edit_json_dir, a.cases)
    if a.cmd == "counts":
        print(" ".join(str(n) for n in counts_per_type(pairs)))
    elif a.cmd == "missing":
        for q in pairs:
            d = render_dir(a.root, a.layout, int(q["edit_type"]), str(q["video_name"]))
            if not list_images(d) and not os.path.exists(d.rstrip("/") + ".mp4"):
                print(f"edit{q['edit_type']}/{q['video_name']}")
    else:  # clipd_cases
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from r33_clip_directional import fullbench_cases  # lazy: pulls torch, only this subcommand needs it
        keep = {(str(q["video_name"]), int(q["edit_type"])) for q in pairs}
        cases = [c for c in fullbench_cases(Path(a.data_root).expanduser(), Path(a.edit_json_dir).expanduser())
                 if (c["video_name"], int(c["edit_type"])) in keep]
        if len(cases) != len(pairs):
            raise SystemExit(f"clipd_cases: {len(cases)} cases for {len(pairs)} pairs")
        Path(a.out).write_text(json.dumps(cases, indent=1))
        print(f"[suite_common] {len(cases)} CLIP-D cases -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
