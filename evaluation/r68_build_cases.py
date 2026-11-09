"""R68 -- the 225 FiVE-Bench-Correct pairs whose Wan-Edit inputs equal the released FiVE ones.

R68 renders Wan-Edit's demand ablations A1-A3 on all 415 Correct pairs. Their field is built as R58's A0
(max_s5) was assembled: the 190 pairs of ``r58_cases_maxs5.json`` (prompt or trigger object changed) from
R58's Correct draft / divergence, the other pairs from R49's official-FiVE draft / divergence, which are
valid for them only because their Wan-Edit inputs did not change. This script writes that complement:

  evaluation/r68_cases_reuse225.json   entries of --manifest NOT in --maxs5, manifest order, same dict format

Checks (exit 1, nothing written, on any failure):
  * |maxs5| = 190, |complement| = 225, maxs5 is a subset of the manifest (so maxs5 + complement = manifest);
  * every complement pair has source_prompt, target_prompt, source_object, target_object in
    --correct_dir/edit{T}_FiVE.json equal to the released --official_dir/edit{T}_FiVE.json.
usage: python evaluation/r68_build_cases.py
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parent.parent
FIELDS = ("source_prompt", "target_prompt", "source_object", "target_object")  # every Wan-Edit input of a pair
EXPECT_MAXS5, EXPECT_REUSE = 190, 225


def load_by_video(directory: Path, edit_type: int) -> Dict[str, dict]:
    entries = json.loads((directory / f"edit{edit_type}_FiVE.json").read_text())
    return {e["video_name"]: e for e in entries}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=REPO / "evaluation/r43_anchor_manifest_correct.json")
    ap.add_argument("--maxs5", type=Path, default=REPO / "evaluation/r58_cases_maxs5.json")
    ap.add_argument("--correct_dir", type=Path, default=REPO / "evaluation/five_bench_correct")
    ap.add_argument("--official_dir", type=Path,
                    default=Path.home() / "Data/FiVE-Fine-Grained-Video-Editing-Benchmark/edit_prompt")
    ap.add_argument("--out", type=Path, default=REPO / "evaluation/r68_cases_reuse225.json")
    a = ap.parse_args()

    manifest: List[dict] = json.loads(a.manifest.read_text())
    maxs5: List[dict] = json.loads(a.maxs5.read_text())
    manifest_ids = [m["case_id"] for m in manifest]
    maxs5_ids = {m["case_id"] for m in maxs5}
    errors: List[str] = []
    if len(manifest_ids) != len(set(manifest_ids)):
        errors.append("the manifest holds duplicate case_ids")
    if len(maxs5_ids) != EXPECT_MAXS5:
        errors.append(f"{a.maxs5.name}: {len(maxs5_ids)} pairs, expected {EXPECT_MAXS5}")
    outside = sorted(maxs5_ids - set(manifest_ids))
    if outside:
        errors.append(f"{len(outside)} pair(s) of {a.maxs5.name} not in the manifest, e.g. {outside[:3]}")
    reuse = [m for m in manifest if m["case_id"] not in maxs5_ids]
    if len(reuse) != EXPECT_REUSE:
        errors.append(f"complement: {len(reuse)} pairs, expected {EXPECT_REUSE}")

    # the condition under which R49's inputs are valid: every Wan-Edit input field unchanged
    correct = {t: load_by_video(a.correct_dir, t) for t in range(1, 7)}
    official = {t: load_by_video(a.official_dir, t) for t in range(1, 7)}
    for m in reuse:
        t, vid = int(m["edit_type"]), m["video_name"]
        c, o = correct[t].get(vid), official[t].get(vid)
        if c is None or o is None:
            errors.append(f"{m['case_id']}: missing from the {'Correct' if c is None else 'released'} edit{t}_FiVE.json")
            continue
        changed = [f for f in FIELDS if c[f] != o[f]]
        if changed:
            errors.append(f"{m['case_id']}: Wan-Edit inputs changed in Correct ({', '.join(changed)})")

    if errors:
        for e in errors[:20]:
            print(f"[r68_build_cases] ERROR {e}")
        print(f"[r68_build_cases] {len(errors)} error(s) -- nothing written")
        return 1
    a.out.write_text(json.dumps(reuse, indent=2) + "\n")
    per_type = collections.Counter(int(m["edit_type"]) for m in reuse)
    print(f"[r68_build_cases] manifest {len(manifest_ids)} = maxs5 {len(maxs5_ids)} + reuse {len(reuse)}; "
          f"reuse per type: {' '.join(f'edit{t}={per_type.get(t, 0)}' for t in range(1, 7))}")
    print(f"[r68_build_cases] all {len(reuse)} reuse pairs have unchanged {', '.join(FIELDS)} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
