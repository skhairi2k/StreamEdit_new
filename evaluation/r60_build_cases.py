"""R60 -- regeneration set of StreamEdit text-only on FiVE-Bench-Correct.

A text-only StreamEdit render (``--vp_mode novp``) reads ``source_prompt`` / ``target_prompt`` and, in
every arm, the trigger words ``source_object`` / ``target_object`` (cross-attention grounding + boosting).
A Correct pair whose four fields equal the released FiVE ones renders exactly as in R44, so its R44
render is reused. Compares ``--correct_dir/edit{T}_FiVE.json`` with the released
``--official_dir/edit{T}_FiVE.json`` per pair and writes, in the r7 anchor-manifest format (the entries
of ``--manifest``, restricted to the set):

  r60_cases_regen.json  pairs whose prompt OR trigger word changed (the 12 R44 uniform-rho configs)
  r60_cases_repro.json  the first unchanged pair of edit1 and of edit4 (reproducibility gate)
  csv/r60_regen_sets.csv  case_id, edit_type, video_name, changed fields, regen

Exit 1 (nothing written) if the counts differ from 190 (15/32/54/70/9/10 per type) / 2, a Correct pair
is missing from the released files, or the Correct annotations and the manifest do not hold the same pairs.
usage: python evaluation/r60_build_cases.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parent.parent
FIELDS = ("source_prompt", "target_prompt", "source_object", "target_object")
EXPECT_REGEN = {1: 15, 2: 32, 3: 54, 4: 70, 5: 9, 6: 10}   # 190, checked 2026-10-26
EXPECT_REPRO = 2
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
            changed = [f for f in FIELDS if c[f] != o[f]]
            rows.append(dict(case_id=case_id, edit_type=t, video_name=vid, changed=changed, regen=bool(changed)))
    if correct_ids != manifest_ids:
        errors.append(f"annotations and manifest differ: {len(correct_ids - manifest_ids)} only in annotations "
                      f"{sorted(correct_ids - manifest_ids)[:3]}, {len(manifest_ids - correct_ids)} only in manifest "
                      f"{sorted(manifest_ids - correct_ids)[:3]}")

    repro = []
    for t in REPRO_TYPES:
        first = next((r for r in rows if r["edit_type"] == t and not r["regen"]), None)
        if first is None:
            errors.append(f"no unchanged pair in edit{t} for the reproducibility gate")
        else:
            repro.append(first["case_id"])

    per_type = {t: sum(r["regen"] for r in rows if r["edit_type"] == t) for t in range(1, 7)}
    n_regen = sum(per_type.values())
    print(f"[r60_build_cases] {len(rows)} Correct pairs: regen {n_regen} (expect {sum(EXPECT_REGEN.values())}), "
          f"per type {per_type} (expect {EXPECT_REGEN}), repro {len(repro)} (expect {EXPECT_REPRO}): {repro}")
    if per_type != EXPECT_REGEN or len(repro) != EXPECT_REPRO:
        errors.append("counts differ from the expected 190 (15/32/54/70/9/10) / 2")
    if errors:
        for e in errors:
            print(f"[r60_build_cases] ERROR: {e}")
        print(f"[r60_build_cases] failures: {len(errors)}")
        return 1

    by_id = {m["case_id"]: m for m in manifest}
    sets = {"regen": [r["case_id"] for r in rows if r["regen"]], "repro": repro}
    a.csv_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in sets.items():
        path = a.out_dir / f"r60_cases_{name}.json"
        path.write_text(json.dumps([by_id[i] for i in ids], indent=1) + "\n")
        print(f"[r60_build_cases] {path}: {len(ids)} pairs")
    with open(a.csv_dir / "r60_regen_sets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "edit_type", "video_name", "changed_fields", "regen"])
        for r in rows:
            w.writerow([r["case_id"], r["edit_type"], r["video_name"], "+".join(r["changed"]), int(r["regen"])])
    print(f"[r60_build_cases] {a.csv_dir / 'r60_regen_sets.csv'}: {len(rows)} rows")
    print("[r60_build_cases] failures: 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
