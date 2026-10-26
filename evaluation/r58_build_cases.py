"""R58 -- regeneration sets of Wan-Edit on FiVE-Bench-Correct.

Wan-Edit is text-only, so a Correct pair whose Wan-Edit inputs equal the released FiVE ones renders
exactly as in R49 and its R49 render is reused. Compares ``--correct_dir/edit{T}_FiVE.json`` with the
released ``--official_dir/edit{T}_FiVE.json`` per pair and writes, in the r7 anchor-manifest format
(the entries of ``--manifest``, restricted to the set):

  r58_cases_base.json   pairs whose ``source_prompt`` / ``target_prompt`` changed      (the 11 baseline configs)
  r58_cases_maxs5.json  pairs whose prompt OR ``source_object`` / ``target_object`` changed
                        (max_s5: the draft's attention support reads the trigger tokens)
  r58_cases_repro.json  the first pair of edit1 and of edit4 outside the max_s5 set (reproducibility gate)
  csv/r58_regen_sets.csv  case_id, changed fields, in_base, in_maxs5, in_repro

Exit 1 (nothing written) if the counts differ from 47 / 190 / 2, a Correct pair is missing from the
released files, or the Correct annotations and the manifest do not hold the same pairs.
usage: python evaluation/r58_build_cases.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parent.parent
PROMPT_FIELDS = ("source_prompt", "target_prompt")
OBJECT_FIELDS = ("source_object", "target_object")
EXPECT_BASE, EXPECT_MAXS5, EXPECT_REPRO = 47, 190, 2
REPRO_TYPES = (1, 4)


def load_by_video(directory: Path, edit_type: int) -> Dict[str, dict]:
    entries = json.loads((directory / f"edit{edit_type}_FiVE.json").read_text())
    return {e["video_name"]: e for e in entries}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--correct_dir", type=Path, default=REPO / "evaluation/five_bench_correct")
    ap.add_argument("--official_dir", type=Path,
                    default=Path.home() / "Data/FiVE-Fine-Grained-Video-Editing-Benchmark/edit_prompt")
    ap.add_argument("--manifest", type=Path, default=REPO / "evaluation/r43_anchor_manifest_correct.json")
    ap.add_argument("--out_dir", type=Path, default=REPO / "evaluation")
    ap.add_argument("--csv_dir", type=Path, default=REPO / "evaluation/csv")
    a = ap.parse_args()

    manifest: List[dict] = json.loads(a.manifest.read_text())
    errors: List[str] = []
    rows = []
    manifest_ids = {m["case_id"] for m in manifest}
    correct_ids = set()
    for t in range(1, 7):
        correct = load_by_video(a.correct_dir, t)
        official = load_by_video(a.official_dir, t)
        for vid, c in sorted(correct.items()):
            case_id = f"edit{t}/{vid}"
            correct_ids.add(case_id)
            o = official.get(vid)
            if o is None:
                errors.append(f"{case_id}: missing from the released {a.official_dir}/edit{t}_FiVE.json")
                continue
            changed = [f for f in PROMPT_FIELDS + OBJECT_FIELDS if c[f] != o[f]]
            rows.append(dict(case_id=case_id, edit_type=t, changed=changed,
                             in_base=any(f in PROMPT_FIELDS for f in changed),
                             in_maxs5=bool(changed)))
    if correct_ids != manifest_ids:
        errors.append(f"annotations and manifest differ: {len(correct_ids - manifest_ids)} only in annotations "
                      f"{sorted(correct_ids - manifest_ids)[:3]}, {len(manifest_ids - correct_ids)} only in manifest "
                      f"{sorted(manifest_ids - correct_ids)[:3]}")

    repro = []
    for t in REPRO_TYPES:
        first = next((r for r in rows if r["edit_type"] == t and not r["in_maxs5"]), None)
        if first is None:
            errors.append(f"no unchanged pair in edit{t} for the reproducibility gate")
        else:
            repro.append(first["case_id"])
    for r in rows:
        r["in_repro"] = r["case_id"] in repro

    n_base = sum(r["in_base"] for r in rows)
    n_maxs5 = sum(r["in_maxs5"] for r in rows)
    print(f"[r58_build_cases] {len(rows)} Correct pairs: base {n_base} (expect {EXPECT_BASE}), "
          f"max_s5 {n_maxs5} (expect {EXPECT_MAXS5}), repro {len(repro)} (expect {EXPECT_REPRO}): {repro}")
    if (n_base, n_maxs5, len(repro)) != (EXPECT_BASE, EXPECT_MAXS5, EXPECT_REPRO):
        errors.append("counts differ from the expected 47 / 190 / 2")
    if errors:
        for e in errors:
            print(f"[r58_build_cases] ERROR: {e}")
        print(f"[r58_build_cases] failures: {len(errors)}")
        return 1

    by_id = {m["case_id"]: m for m in manifest}
    sets = {"base": [r["case_id"] for r in rows if r["in_base"]],
            "maxs5": [r["case_id"] for r in rows if r["in_maxs5"]],
            "repro": repro}
    a.csv_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in sets.items():
        path = a.out_dir / f"r58_cases_{name}.json"
        path.write_text(json.dumps([by_id[i] for i in ids], indent=1) + "\n")
        print(f"[r58_build_cases] {path}: {len(ids)} pairs")
    with open(a.csv_dir / "r58_regen_sets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "changed_fields", "in_base", "in_maxs5", "in_repro"])
        for r in rows:
            w.writerow([r["case_id"], "+".join(r["changed"]), int(r["in_base"]), int(r["in_maxs5"]), int(r["in_repro"])])
    print(f"[r58_build_cases] {a.csv_dir / 'r58_regen_sets.csv'}: {len(rows)} rows")
    print("[r58_build_cases] failures: 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
