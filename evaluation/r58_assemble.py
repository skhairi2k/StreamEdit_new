"""R58 -- assemble one 415-clip tree per Wan-Edit config from two render folders (ON JEAN ZAY, five-bench python).

For each of the 12 configs, ``$D/r58_wan_correct/{config}/edit{T}/{video}`` becomes a SYMLINK to the clip folder of

  * the new render   $D/r58_wan_arms/{config}_full/...   if the pair is in the config's regen set
                     (r58_cases_base.json for the 11 baselines, r58_cases_maxs5.json for max_s5), else
  * the R49 render   $D/r48_wan_arms/{config}_full/...   (native_*)  or  $D/r49_wan_arms/{config}_full/...  (matched59_*, max_s5)

over the 415 pairs of ``--manifest``. Wan-Edit is text-only, so a pair whose inputs Correct did not change renders exactly as in R49
(evaluation/r58_repro_check.py verified bit-identical renders). The evaluation then reads one ordinary tree per config.

Safety: only symlinks inside ``--out_root`` are created or replaced; a link target is never written to. An existing entry that is not a
symlink, or a symlink pointing outside r58_wan_arms / r48_wan_arms / r49_wan_arms, is a CONFLICT: left untouched, exit 1. Link targets are
the RESOLVED absolute paths (the $SCRATCH folders), so the tree stays valid for ``rsync -aL`` from the lab. A clip is linked only if its
target holds >= the expected frame count (41, fewer for the 7 short videos), checked through the link. Re-running is idempotent.

Writes evaluation/csv/r58_assembly.csv (config, case_id, source new|reused, n_png) only for a clean assembly. Exit 1 on a conflict, a missing/incomplete clip, or a
count other than 415 per config.
usage: python evaluation/r58_assemble.py [--check]        (--check: change nothing, report what would be linked)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

D = Path.home() / "Data/dataggen/outputs/five_bench"
NATIVE = ("native_s0.95", "native_s0.92", "native_s0.85", "native_s0.75", "native_s0.6", "native_s0.4", "native_s0.978")
MATCHED = ("matched59_s0.95", "matched59_s0.92", "matched59_s0.85", "matched59_s0.978")
CONFIGS = NATIVE + MATCHED + ("max_s5",)
N_PAIRS = 415


def reused_folder(config: str) -> str:
    return "r48_wan_arms" if config.startswith("native_") else "r49_wan_arms"


def case_ids(path: Path) -> Set[str]:
    return {c["case_id"] for c in json.loads(path.read_text())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=REPO / "evaluation/r43_anchor_manifest_correct.json")
    ap.add_argument("--cases_base", type=Path, default=REPO / "evaluation/r58_cases_base.json")
    ap.add_argument("--cases_maxs5", type=Path, default=REPO / "evaluation/r58_cases_maxs5.json")
    ap.add_argument("--out_root", type=Path, default=D / "r58_wan_correct")
    ap.add_argument("--out_csv", type=Path, default=REPO / "evaluation/csv/r58_assembly.csv")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=CONFIGS)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    import r48_wan_common as C      # expected_frames / source_frame_count (torch, cv2: the five-bench env)

    manifest: List[dict] = json.loads(a.manifest.read_text())
    if len(manifest) != N_PAIRS:
        print(f"[r58_assemble] FATAL: {a.manifest} lists {len(manifest)} pairs, expected {N_PAIRS}")
        return 1
    regen = {"base": case_ids(a.cases_base), "maxs5": case_ids(a.cases_maxs5)}
    allowed = [(D / f).resolve() for f in ("r58_wan_arms", "r48_wan_arms", "r49_wan_arms")]
    out_root = a.out_root
    need: Dict[str, int] = {}
    rows, problems, n_link = [], [], 0
    per_config: Dict[str, int] = {c: 0 for c in a.configs}

    for config in a.configs:
        regen_set = regen["maxs5" if config == "max_s5" else "base"]
        for m in manifest:
            cid, t, vid = m["case_id"], int(m["edit_type"]), m["video_name"]
            is_new = cid in regen_set
            src_root = D / ("r58_wan_arms" if is_new else reused_folder(config)) / f"{config}_full"
            target = (src_root / f"edit{t}" / vid)
            if vid not in need:
                need[vid] = C.expected_frames(C.source_frame_count(vid))
            n_png = len(list(target.glob("*.png"))) if target.is_dir() else 0
            if n_png < need[vid]:
                problems.append(f"{config} {cid}: {'new' if is_new else 'reused'} render has {n_png}/{need[vid]} frames at {target}")
                continue
            target = target.resolve()
            link = out_root / config / f"edit{t}" / vid
            if link.is_symlink():
                cur = Path(os.readlink(link))
                if not any(p == cur.resolve() or p in cur.resolve().parents for p in allowed):
                    problems.append(f"CONFLICT {link}: symlink to {cur}, outside r58_wan_arms / r48_wan_arms / r49_wan_arms (left untouched)")
                    continue
            elif link.exists():
                problems.append(f"CONFLICT {link}: exists and is not a symlink (left untouched)")
                continue
            if not a.check:
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.is_symlink() and Path(os.readlink(link)) == target:
                    pass                                               # identical link: kept
                else:
                    if link.is_symlink():
                        link.unlink()                                  # one of our own links, pointing elsewhere: replaced
                    link.symlink_to(target, target_is_directory=True)
                    n_link += 1
            per_config[config] += 1
            rows.append([config, cid, "new" if is_new else "reused", n_png])

    for config in a.configs:
        n_new = sum(1 for r in rows if r[0] == config and r[2] == "new")
        print(f"[r58_assemble] {config:16s} {per_config[config]:3d}/{N_PAIRS} clips ({n_new} new, {per_config[config] - n_new} reused)")
        if per_config[config] != N_PAIRS:
            problems.append(f"{config}: {per_config[config]} clips linked, expected {N_PAIRS}")
    if not a.check and not problems:          # the CSV is the fetch's manifest: written only for a clean assembly
        a.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(a.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["config", "case_id", "source", "n_png"])
            w.writerows(rows)
    for p in problems:
        print(f"[r58_assemble] PROBLEM: {p}")
    print(f"[r58_assemble] {'CHECK' if a.check else 'links created/replaced: %d' % n_link}; {len(rows)} (config, pair) entries"
          f"{' -> ' + str(a.out_csv) if not a.check and not problems else ''}")
    print(f"[r58_assemble] failures: {len(problems)}")
    return int(len(problems) > 0)


if __name__ == "__main__":
    sys.exit(main())
