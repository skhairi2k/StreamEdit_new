"""R59: prompt provenance for the FiVE-Bench-Correct runs.

Two roles in one file:

  * ``write_sidecar`` -- imported by run_fivebench.py / r31_stage1.py / r31_stage3.py. Records, right before
    the render loop, the exact strings the pipeline will receive (source/target prompt, source/target trigger
    word) and the anchor path of every pair, plus the sha256 of the edit-prompt JSON they were read from.
  * the CLI -- run after a render inside every R59 job: every recorded pair must equal the FiVE-Bench-Correct
    entry of that video, the anchors must come from the Correct anchor root, and the video set must be exactly
    the Correct set. Exit 1 on any mismatch, so a Slurm ``afterok`` chain stops instead of rendering on the
    official prompts (nothing downstream of a render could tell which JSON it was read from).

Usage:
    python evaluation/r59_check_prompts.py --sidecar 'PATH/_prompts_edit*.json' [--sidecar ...] \
        --edit_json_dir evaluation/five_bench_correct --anchor_dir ~/Data/dataggen/outputs/five_bench/anchors_r43_correct \
        --expect_n 415
"""
import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

PROMPT_KEYS = ("source_prompt", "target_prompt", "source_object", "target_object")
DEFAULT_OFFICIAL_DIR = "~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark/edit_prompt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sidecar(path, edit_json, anchor_dir, edit_type: int, entries) -> None:
    """Record what the render loop is about to use. ``entries`` are the very dicts passed to the pipeline.

    anchor_dir is None for text-only runs (recorded as null)."""
    edit_json = Path(edit_json).expanduser().resolve()
    anchor_dir = Path(anchor_dir).expanduser().resolve() if anchor_dir is not None else None
    cases = {}
    for e in entries:
        case = {k: e[k] for k in PROMPT_KEYS}
        case["anchor"] = (str(anchor_dir / f"edit{edit_type}" / f"{e['video_name']}.png")
                          if anchor_dir is not None else None)
        cases[e["video_name"]] = case
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "edit_type": int(edit_type),
        "edit_json": str(edit_json),
        "edit_json_sha256": _sha256(edit_json),
        "anchor_dir": str(anchor_dir) if anchor_dir is not None else None,
        "cases": cases,
    }, indent=1, ensure_ascii=False))


def _load_entries(json_dir: Path, edit_type: int) -> dict:
    return {e["video_name"]: e for e in json.loads((json_dir / f"edit{edit_type}_FiVE.json").read_text())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sidecar", action="append", required=True,
                    help="sidecar file or glob; repeatable. One sidecar per edit type is expected.")
    ap.add_argument("--edit_json_dir", required=True, help="FiVE-Bench-Correct JSON directory")
    ap.add_argument("--anchor_dir", required=True, help="Correct anchor root (anchors_r43_correct)")
    ap.add_argument("--official_dir", default=DEFAULT_OFFICIAL_DIR,
                    help="official FiVE edit_prompt dir, used only to count how many pairs differ (sanity)")
    ap.add_argument("--expect_n", type=int, required=True, help="total pairs expected over all sidecars")
    args = ap.parse_args()

    correct_dir = Path(args.edit_json_dir).expanduser().resolve()
    official_dir = Path(args.official_dir).expanduser().resolve()
    anchor_dir = Path(args.anchor_dir).expanduser().resolve()

    files = sorted({f for pat in args.sidecar for f in glob.glob(str(Path(pat).expanduser()))})
    if not files:
        print(f"[r59_check_prompts] FAIL: no sidecar matches {args.sidecar}")
        sys.exit(1)

    problems, n_pairs, n_differ, seen_types = [], 0, 0, []
    for f in files:
        sc = json.loads(Path(f).read_text())
        t = int(sc["edit_type"])
        if t in seen_types:
            problems.append(f"{f}: edit{t} appears in two sidecars")
        seen_types.append(t)
        correct = _load_entries(correct_dir, t)
        official = _load_entries(official_dir, t) if official_dir.exists() else {}

        want_sha = _sha256(correct_dir / f"edit{t}_FiVE.json")
        if sc["edit_json_sha256"] != want_sha:
            problems.append(f"{f}: read {sc['edit_json']} (sha {sc['edit_json_sha256'][:10]}), "
                            f"not the Correct edit{t}_FiVE.json (sha {want_sha[:10]})")
        got, want = set(sc["cases"]), set(correct)
        if got != want:
            problems.append(f"{f}: video set differs from Correct edit{t}: missing {sorted(want - got)[:5]}, "
                            f"extra {sorted(got - want)[:5]}")
        for v, c in sc["cases"].items():
            ref = correct.get(v)
            if ref is None:
                continue
            for k in PROMPT_KEYS:
                if c[k] != ref[k]:
                    problems.append(f"{f}: edit{t}/{v}.{k}: {c[k]!r} != Correct {ref[k]!r}")
            exp_anchor = anchor_dir / f"edit{t}" / f"{v}.png"
            if c["anchor"] is None or Path(c["anchor"]).expanduser().resolve() != exp_anchor:
                problems.append(f"{f}: edit{t}/{v}.anchor: {c['anchor']!r} != {exp_anchor}")
            if v in official and any(c[k] != official[v][k] for k in PROMPT_KEYS):
                n_differ += 1
        n_pairs += len(sc["cases"])

    if n_pairs != args.expect_n:
        problems.append(f"{n_pairs} pairs over {len(files)} sidecar(s), expected {args.expect_n}")
    # Only a warning: the sha256 check above already proves which file was read, and one edit type alone
    # (the per-type guard in r59_infer.sh) may legitimately have no corrected pair.
    if official_dir.exists() and n_differ == 0 and not problems:
        print("[r59_check_prompts] WARNING: 0 pairs differ from the official prompts in these sidecars")

    if problems:
        print(f"[r59_check_prompts] FAIL: {len(problems)} problem(s)")
        for p in problems[:30]:
            print(f"[r59_check_prompts]   {p}")
        sys.exit(1)
    print(f"[r59_check_prompts] OK {n_pairs}/{args.expect_n}, {n_differ} differ from official "
          f"(edit types {sorted(seen_types)})")


if __name__ == "__main__":
    main()
