"""R48: the list of renders to score, in one place (r48_eval.sh / r48_fiveacc.sh / r48_clipd.sh / r48_score.py).

A "unit" is one render directory: a StreamEdit arm or a Wan-Edit config. Order is fixed so an sbatch array index always
means the same unit:  StreamEdit arms, then the 9 Wan-Edit baselines (r48_wan_sched.py --list order), then Wan-Edit arms.
  subset: 4 se arms + 9 baselines + 4 wan arms = 17 units
  full  : the `se` lines of evaluation/r48_selected_arms.txt + 9 baselines + its `wan` lines
Output (TSV, one per line):  base <TAB> dir <TAB> host <TAB> config
  base = "{host}_{config with '.' -> 'p'}_{mode}"  (a dot in a CSV stem would confuse evaluate.py's result-path suffixing)
  dir  = a valid `--tgt_layout edit_video` root: {dir}/edit{T}/{video}/*.png
usage: python evaluation/r48_units.py --mode subset [--count | --index N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r48_wan_sched as SCH  # noqa: E402

D = Path.home() / "Data/dataggen/outputs/five_bench"
SELECTED = HERE / "r48_selected_arms.txt"


def selected_arms() -> List[Tuple[str, str]]:
    if not SELECTED.exists() or not SELECTED.read_text().strip():
        raise SystemExit(f"[r48_units] {SELECTED} missing or empty -- write the passing (host, arm) pairs at gate-decision")
    out = []
    for line in SELECTED.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            host, arm = line.split()[:2]
            if host not in ("se", "wan") or arm not in SCH.ARMS:
                raise SystemExit(f"[r48_units] bad line in {SELECTED}: {line!r}")
            out.append((host, arm))
    return out


def units(mode: str) -> List[Tuple[str, Path, str, str]]:
    if mode == "subset":
        sel = [("se", a) for a in SCH.ARMS] + [("wan", a) for a in SCH.ARMS]
    else:
        sel = selected_arms()
    out = []
    for host, arm in sel:
        if host == "se":
            out.append((f"se_{arm}_{mode}", D / "r48_se_arms" / f"r48_{arm}_{mode}" / "step14", "se", arm))
    for cfg in SCH.configs():
        out.append((f"wan_{cfg.replace('.', 'p')}_{mode}", D / "r48_wan_arms" / f"{cfg}_{mode}", "wan", cfg))
    for host, arm in sel:
        if host == "wan":
            out.append((f"wan_adapt_{arm}_{mode}", D / "r48_wan_arms" / f"adapt_{arm}_{mode}", "wan", f"adapt_{arm}"))
    return out


_NEED = {}


def _need(video_name: str) -> int:
    if video_name not in _NEED:
        import r48_wan_common as C
        _NEED[video_name] = C.expected_frames(C.source_frame_count(video_name))
    return _NEED[video_name]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("subset", "full"), required=True)
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--missing", default=None, metavar="DIR",
                    help="print one line per clip of --cases that has no complete render (>= the expected frame count, 41 or fewer for the 7 short videos) under DIR/edit{T}/{video}/")
    ap.add_argument("--cases", default=None)
    a = ap.parse_args()
    if a.missing is not None:
        import json
        for c in json.loads(Path(a.cases).read_text()):
            d = Path(a.missing) / f"edit{int(c['edit_type'])}" / c["video_name"]
            # complete = at least the frames a Wan-Edit render of this source has (41; fewer for the 7 short videos)
            if not d.is_dir() or len(list(d.glob("*.png"))) < _need(c["video_name"]):
                print(f"{int(c['edit_type'])}/{c['video_name']}")
        return 0
    u = units(a.mode)
    if a.count:
        print(len(u))
        return 0
    rows = u if a.index is None else [u[a.index]] if 0 <= a.index < len(u) else []
    for base, d, host, cfg in rows:
        print(f"{base}\t{d}\t{host}\t{cfg}")
    return 0 if rows or a.index is None else 1


if __name__ == "__main__":
    sys.exit(main())
