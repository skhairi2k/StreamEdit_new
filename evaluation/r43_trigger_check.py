"""R43 -- exact trigger-token check of every source/target object, FiVE as released vs Correct.

StreamEdit passes ``source_object`` / ``target_object`` to the pipeline as trigger words, and
``find_phrase_token_indices`` locates them as a contiguous, case-insensitive token run in
their prompt with the umt5-xxl tokenizer (``run_fivebench.py`` reaches it through
``pipeline.text_encoder.tokenizer.tokenizer``). An object it cannot find gives an empty
trigger, i.e. no cross-attention foreground. This script runs that exact matcher -- the
function is loaded from ``Self-Forcing_StreamEdit/pipeline/utils.py`` by file path, so the
heavy ``pipeline`` package ``__init__`` is never imported -- on every pair of both tables.

Writes one row per (table, case_id): ``src_tokens`` / ``tgt_tokens`` (matched token count,
0 = empty trigger) and, for Correct rows, ``object_changed`` (a ``*_object_orig`` key exists).
Exit 1 if any FiVE-Bench-Correct pair has an empty trigger other than the intended empties
(``source_object == ""`` on edit5, ``target_object == ""`` on edit6); FiVE-as-released counts
are informational.

Example
-------
    python evaluation/r43_trigger_check.py \\
        --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \\
        --correct_dir evaluation/five_bench_correct \\
        --tokenizer Self-Forcing_StreamEdit/wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl \\
        --out_csv evaluation/csv/r43_trigger_check.csv
"""

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

EDIT_TYPES = range(1, 7)
MATCHER_FILE = Path("Self-Forcing_StreamEdit/pipeline/utils.py")
INTENDED_EMPTY = {5: "src", 6: "tgt"}   # edit5: added object (no source); edit6: removed object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", type=str,
                        default="~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark")
    parser.add_argument("--correct_dir", type=str, default="evaluation/five_bench_correct")
    parser.add_argument("--tokenizer", type=str,
                        default="Self-Forcing_StreamEdit/wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl")
    parser.add_argument("--out_csv", type=str, default="evaluation/csv/r43_trigger_check.csv")
    return parser.parse_args()


def load_matcher():
    """``find_phrase_token_indices`` from the inference code, without the package __init__."""
    if not MATCHER_FILE.is_file():
        sys.exit(f"[r43_trigger_check] missing {MATCHER_FILE} (run from the repo root)")
    spec = importlib.util.spec_from_file_location("streamedit_pipeline_utils", MATCHER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.find_phrase_token_indices


def load_table(root: Path) -> dict[str, dict]:
    pairs = {}
    for t in EDIT_TYPES:
        path = root / f"edit{t}_FiVE.json"
        if not path.is_file():
            sys.exit(f"[r43_trigger_check] missing annotation file: {path}")
        for e in json.loads(path.read_text()):
            pairs[f"edit{t}/{e['video_name']}"] = e
    return pairs


def main() -> None:
    args = parse_args()
    find_phrase_token_indices = load_matcher()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tables = {"five": load_table(Path(args.data_root).expanduser() / "edit_prompt"),
              "correct": load_table(Path(args.correct_dir))}

    rows, empty = [], {name: Counter() for name in tables}
    unintended = []
    for name, pairs in tables.items():
        for cid, e in pairs.items():
            t = int(cid[4])
            n_src = len(find_phrase_token_indices(tokenizer, e["source_prompt"], e["source_object"])[0])
            n_tgt = len(find_phrase_token_indices(tokenizer, e["target_prompt"], e["target_object"])[0])
            for side, n in (("src", n_src), ("tgt", n_tgt)):
                if n == 0:
                    empty[name][(t, side)] += 1
                    obj = e["source_object" if side == "src" else "target_object"]
                    intended = INTENDED_EMPTY.get(t) == side and obj == ""
                    if name == "correct" and not intended:
                        unintended.append(f"{cid} {side}: object {obj!r} has no token match")
            changed = ("source_object_orig" in e or "target_object_orig" in e) if name == "correct" else ""
            rows.append({"table": name, "case_id": cid, "edit_type": t,
                         "source_object": e["source_object"], "target_object": e["target_object"],
                         "src_tokens": n_src, "tgt_tokens": n_tgt, "object_changed": changed})

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for name, pairs in tables.items():
        detail = ", ".join(f"edit{t} {side}={n}" for (t, side), n in sorted(empty[name].items())) or "none"
        print(f"[r43_trigger_check] {name}: {len(pairs)} pairs, empty triggers: {detail}")
    print(f"[r43_trigger_check] {len(rows)} rows -> {out}")
    if unintended:
        print("\n".join(unintended))
        sys.exit(f"[r43_trigger_check] {len(unintended)} unintended empty trigger(s) in FiVE-Bench-Correct")
    print("[r43_trigger_check] PASS: the only empty triggers in FiVE-Bench-Correct are the "
          "edit5 source / edit6 target conventions")


if __name__ == "__main__":
    main()
