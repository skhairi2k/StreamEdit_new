"""Shared helpers of the R57 unified metric suite (stdlib only, so every env of the suite can import it).

* pair list: the (edit_type, video_name) pairs of an --edit_json_dir (FiVE-Bench-Correct), optionally restricted by a
  --cases file ([{"video_name": ..., "edit_type": ...}]), with the fields the metrics read;
* frames: the harness's frame convention (evaluation/fivebench/evaluate.py): sorted png/jpg/jpeg, every
  ``--frame_stride``-th frame, source and render zipped positionally after truncating to the shorter, render resized to the
  source's size by the caller;
* result CSV with resume: one row per pair, rewritten atomically after every pair, pairs already scored are skipped on a
  rerun (SLURM jobs get interrupted);
* a small CLI used by the launcher / job scripts:
    counts           six integers, the pair count per edit type (the row-count guards of the job scripts)
    missing          pairs whose render has no frames (and no .mp4) under --root
    clipd_cases      the cases file r33_clip_directional.py reads with --cases (it needs prompts, which --cases files of the
                     suite do not carry)
    harness_missing  pairs without a row in a harness stem's CSVs (cleans cut-short rows first); prints the count, writes a
                     cases JSON of them -- the resume of evaluate.py
    harness_append   append a temporary stem's rows to the main stem, delete the temporary files
    keyed_complete   exit 0 when a keyed suite CSV holds every pair with finite values (stage skip, merge decision)
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


SRC_WINDOWS = ("full", "render")


def add_src_window_arg(p: argparse.ArgumentParser) -> None:
    """--src_window for the scorers that read the source video themselves (PSC, CF)."""
    p.add_argument("--src_window", default="full", choices=SRC_WINDOWS,
                   help="source frames used: full = the whole source clip (default); render = its first len(render) "
                        "frames, the window the harness, CLIP-D and EditReward use (they truncate to the shorter)")


def source_window(src: Sequence[str], tgt: Sequence[str], mode: str) -> List[str]:
    """Source frame paths of a pair: all of them (full) or the first len(tgt) (render)."""
    if mode not in SRC_WINDOWS:
        raise ValueError(f"src_window '{mode}' not in {SRC_WINDOWS}")
    return list(src[:len(tgt)] if mode == "render" else src)


def check_window(store: ResultStore, mode: str, name: str) -> bool:
    """A resumed CSV must have been scored with the same --src_window (rows written before the column existed = full)."""
    other = sorted({str(r.get("src_window") or "full") for r in store.rows.values()} - {mode})
    if other:
        print(f"Error: {name}: {store.path} holds rows scored with --src_window {', '.join(other)}, this run uses {mode}; "
              f"use another --tag (or --out) instead of mixing windows in one table", flush=True)
        return False
    return True


def finish(store: ResultStore, expected: int, name: str) -> int:
    """Exit code of a scoring script: 1 on any missing / NaN pair, after the CSV was written."""
    bad = [k for k in store.nan_pairs()]
    missing = expected - len(store.rows)
    if bad or missing:
        print(f"[{name}] FAILED: {missing} pair(s) not scored, {len(bad)} with NaN -- {store.path}")
        return 1
    print(f"[{name}] done: {len(store.rows)} pairs, all metrics finite -- {store.path}")
    return 0


# ----------------------------------------------------------------------------- resume ----------
# evaluate.py (FiVE harness, FiVE-Acc) truncates its CSVs when it starts, but appends one row per video as soon as that video
# is scored. The suite resumes it from the outside, without modifying it: list the pairs whose row is missing, run
# evaluate.py on those only (--cases_json) under a temporary stem, then append the new rows to the main CSVs.

def harness_csv(csv_dir: str, stem: str, edit_type: int) -> Path:
    return Path(csv_dir) / f"edit{edit_type}_FiVE_{stem}_frame_stride8.csv"


def harness_missing(csv_dir: str, stem: str, pairs: Sequence[Dict[str, object]],
                    allow_nan: Sequence[str] = ()) -> List[Dict[str, object]]:
    """Pairs without a usable row in the stem's per-type CSVs. Cleans those CSVs first: a row cut short when a job was killed
    (wrong field count or a non-integer file_id), a row of a pair not expected here, a repeated file_id, and a row with a NaN
    cell (a metric that raised for that video; columns in allow_nan excepted) are dropped, so a relaunch scores them again."""
    missing: List[Dict[str, object]] = []
    for t in EDIT_TYPES:
        want = {int(q["file_id"]): q for q in pairs if int(q["edit_type"]) == t}
        if not want:
            continue
        path = harness_csv(csv_dir, stem, t)
        have = set()
        if path.exists():
            rows = list(csv.reader(path.open()))
            if rows:
                header, kept, dropped = rows[0], [rows[0]], 0
                nan_ok = {i for i, h in enumerate(header) if h.split("|")[-1] in set(allow_nan)}
                for r in rows[1:]:
                    try:
                        fid = int(r[0])
                    except (ValueError, IndexError):
                        dropped += 1
                        continue
                    if len(r) != len(header) or fid not in want or fid in have:
                        dropped += 1
                        continue
                    if any(not finite(v) for i, v in enumerate(r[1:], start=1) if i not in nan_ok):
                        dropped += 1
                        continue
                    have.add(fid)
                    kept.append(r)
                if dropped:
                    tmp = path.with_suffix(".csv.tmp")
                    with tmp.open("w", newline="") as fh:
                        csv.writer(fh).writerows(kept)
                    os.replace(tmp, path)
                    print(f"[suite_common] {path.name}: dropped {dropped} cut-short / unexpected / repeated / NaN row(s)")
        missing += [q for fid, q in sorted(want.items()) if fid not in have]
    return missing


def harness_append(csv_dir: str, stem: str, part_stem: str) -> int:
    """Append the rows of part_stem's per-type CSVs to stem's (same header required), then delete every part_stem file."""
    n = 0
    for t in EDIT_TYPES:
        part = harness_csv(csv_dir, part_stem, t)
        if not part.exists():
            continue
        rows = list(csv.reader(part.open()))
        if not rows:
            continue
        main = harness_csv(csv_dir, stem, t)
        if main.exists() and main.stat().st_size > 0:
            header = next(csv.reader(main.open()))
            if header != rows[0]:
                raise SystemExit(f"harness_append: header of {part.name} differs from {main.name}")
            with main.open("a", newline="") as fh:
                csv.writer(fh).writerows(rows[1:])
        else:
            with main.open("w", newline="") as fh:
                csv.writer(fh).writerows(rows)
        n += len(rows) - 1
    for f in Path(csv_dir).glob(f"*{part_stem}*"):
        f.unlink()
    return n


def keyed_complete(path: str, cols: Sequence[str], pairs: Sequence[Dict[str, object]], method: str = "") -> bool:
    """A keyed suite CSV (CLIP-D, EditReward, PSC, CF) holds every expected pair once, with finite values in cols."""
    if not Path(path).exists():
        return False
    expected = {(int(q["edit_type"]), str(q["video_name"])) for q in pairs}
    got = {}
    for r in csv.DictReader(Path(path).open()):
        if method and r.get("method") != method:
            continue
        got[(int(r["edit_type"]), r["video_name"])] = r
    return set(got) == expected and all(finite(r.get(c)) for r in got.values() for c in cols)


# ----------------------------------------------------------------------------- CLI -------------

def _cli(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("counts", "missing", "clipd_cases", "harness_missing", "harness_append", "keyed_complete"):
        s = sub.add_parser(name)
        s.add_argument("--edit_json_dir", default="evaluation/five_bench_correct")
        s.add_argument("--cases", default=None)
        if name == "missing":
            s.add_argument("--root", required=True)
            s.add_argument("--layout", default="edit_video", choices=("edit_video", "flat"))
        if name == "clipd_cases":
            s.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
            s.add_argument("--out", required=True)
        if name in ("harness_missing", "harness_append"):
            s.add_argument("--csv_dir", default="evaluation/csv")
            s.add_argument("--stem", required=True)
        if name == "harness_missing":
            s.add_argument("--out", required=True, help="cases JSON of the missing pairs (written even when empty)")
            s.add_argument("--allow_nan", nargs="*", default=[], metavar="COL")
        if name == "harness_append":
            s.add_argument("--part_stem", required=True)
        if name == "keyed_complete":
            s.add_argument("--csv", required=True)
            s.add_argument("--cols", required=True, help="comma-separated metric columns that must be finite")
            s.add_argument("--method", default="")
    a = p.parse_args(argv)
    pairs = load_pairs(a.edit_json_dir, a.cases)
    if a.cmd == "counts":
        print(" ".join(str(n) for n in counts_per_type(pairs)))
    elif a.cmd == "missing":
        for q in pairs:
            d = render_dir(a.root, a.layout, int(q["edit_type"]), str(q["video_name"]))
            if not list_images(d) and not os.path.exists(d.rstrip("/") + ".mp4"):
                print(f"edit{q['edit_type']}/{q['video_name']}")
    elif a.cmd == "harness_missing":
        miss = harness_missing(a.csv_dir, a.stem, pairs, a.allow_nan)
        Path(a.out).write_text(json.dumps([{"video_name": q["video_name"], "edit_type": q["edit_type"]} for q in miss]))
        print(len(miss))
    elif a.cmd == "harness_append":
        print(f"[suite_common] {harness_append(a.csv_dir, a.stem, a.part_stem)} row(s) appended to {a.stem}")
    elif a.cmd == "keyed_complete":
        return 0 if keyed_complete(a.csv, a.cols.split(","), pairs, a.method) else 1
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
