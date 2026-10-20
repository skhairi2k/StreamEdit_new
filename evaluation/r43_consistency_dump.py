"""R43 -- dump every FiVE-Bench-Correct pair for the instruction <-> prompts <-> objects read.

Reads the six FINAL ``--correct_dir/edit{T}_FiVE.json`` files written by
``r43_build_correct.py`` and writes one CSV row per pair with the fields the
``consistency-audit`` step reads, plus heuristic flags. Flags only order the reading; every
row is read. Stdlib only.

Word handling: lower-cased, surrounding punctuation and a trailing 's stripped, naive plural
strip; articles, prepositions, pronouns and the edit verbs change / replace / turn / make /
add / remove / convert are ignored. "Span" means a verbatim, case-insensitive, word-bounded
match (the rule of the inference's trigger matcher).

Columns:
  * ``instruction_changed`` / ``target_changed`` / ``object_changed`` -- the build kept an
    ``_orig`` value for the instruction / target prompt / an object;
  * ``prompt_added`` / ``prompt_removed`` -- word diff target vs source prompt (the edit the
    prompt pair implies);
  * ``instr_missing_in_target`` -- instruction words found in neither ``target_prompt`` nor
    ``target_object`` (the instruction asks for something the target does not show);
  * ``target_extra_unasked`` -- added prompt words absent from the instruction (a possible
    extra edit);
  * ``tgt_obj_in_instr`` / ``tgt_obj_in_target`` (types 1-5);
  * ``src_obj_in_source`` / ``src_obj_in_instr`` (types 1-4 and 6; edit5 has no source object);
  * ``added_obj_in_source`` (type 5: the added object is already in ``source_prompt``);
  * ``removed_obj_in_target`` (type 6: ``source_object`` still in ``target_prompt``);
  * ``verb_vs_type`` -- type 5 instruction without add / insert / place, type 6 without
    remove / erase / delete.

Example
-------
    python evaluation/r43_consistency_dump.py \\
        --correct_dir evaluation/five_bench_correct \\
        --out_csv evaluation/csv/r43_consistency_dump.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

EDIT_TYPES = range(1, 7)
IGNORED = {
    # articles / pronouns / conjunctions
    "a", "an", "the", "this", "that", "it", "its", "his", "her", "their", "they", "them", "he",
    "she", "him", "and", "or",
    # prepositions
    "of", "in", "on", "at", "to", "by", "with", "from", "into", "onto", "for", "as", "over",
    "under", "near", "along", "across", "through", "around", "behind", "beside", "out", "up",
    # edit verbs
    "change", "replace", "turn", "make", "add", "remove", "convert",
}
ADD_VERBS = {"add", "insert", "place"}
REMOVE_VERBS = {"remove", "erase", "delete"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--correct_dir", type=str, default="evaluation/five_bench_correct")
    parser.add_argument("--out_csv", type=str, default="evaluation/csv/r43_consistency_dump.csv")
    return parser.parse_args()


def norm(token: str) -> str:
    tok = token.strip(".,;:!?\"'()").lower()
    if tok.endswith("'s"):
        tok = tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        tok = tok[:-1]
    return tok


def raw_words(text: str) -> list[str]:
    return [w for w in (norm(t) for t in text.split()) if w]


def content(text: str) -> list[str]:
    return list(dict.fromkeys(w for w in raw_words(text) if w not in IGNORED))


def is_span(obj: str, text: str) -> bool:
    obj = obj.strip()
    return bool(obj) and re.search(r"(?<!\w)" + re.escape(obj) + r"(?!\w)", text, re.I) is not None


def flag(value) -> str:
    """'' for not applicable, else True / False."""
    return "" if value is None else str(bool(value))


def row_for(t: int, e: dict) -> dict:
    instr, src_p, tgt_p = e["instruction"], e["source_prompt"], e["target_prompt"]
    src_o, tgt_o = e["source_object"], e["target_object"]
    instr_w = set(content(instr))
    src_w, tgt_w = set(content(src_p)), set(content(tgt_p))
    added = [w for w in content(tgt_p) if w not in src_w]
    removed = [w for w in content(src_p) if w not in tgt_w]
    shown = tgt_w | set(content(tgt_o))
    verbs = set(raw_words(instr))
    if t == 5:
        verb_ok = bool(verbs & ADD_VERBS)
    elif t == 6:
        verb_ok = bool(verbs & REMOVE_VERBS)
    else:
        verb_ok = None
    return {
        "case_id": f"edit{t}/{e['video_name']}", "edit_type": t,
        "instruction": instr, "source_prompt": src_p, "target_prompt": tgt_p,
        "source_object": src_o, "target_object": tgt_o,
        "instruction_changed": str("instruction_orig" in e),
        "target_changed": str("target_prompt_orig" in e),
        "object_changed": str("source_object_orig" in e or "target_object_orig" in e),
        "prompt_added": " ".join(added),
        "prompt_removed": " ".join(removed),
        "instr_missing_in_target": " ".join(w for w in content(instr) if w not in shown),
        "target_extra_unasked": " ".join(w for w in added if w not in instr_w),
        "tgt_obj_in_instr": flag(set(content(tgt_o)) <= instr_w) if t <= 5 else "",
        "tgt_obj_in_target": flag(is_span(tgt_o, tgt_p)) if t <= 5 else "",
        "src_obj_in_source": flag(is_span(src_o, src_p)) if t != 5 else "",
        "src_obj_in_instr": flag(set(content(src_o)) <= instr_w) if t != 5 else "",
        "added_obj_in_source": flag(is_span(tgt_o, src_p)) if t == 5 else "",
        "removed_obj_in_target": flag(is_span(src_o, tgt_p)) if t == 6 else "",
        "verb_vs_type": "" if verb_ok is None else ("" if verb_ok else "verb missing"),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.correct_dir)
    rows = []
    for t in EDIT_TYPES:
        path = root / f"edit{t}_FiVE.json"
        if not path.is_file():
            sys.exit(f"[r43_consistency_dump] missing {path} (run build-correct first)")
        rows += [row_for(t, e) for e in json.loads(path.read_text())]

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def count(key, bad):
        return sum(r[key] == bad for r in rows)
    print(f"[r43_consistency_dump] {len(rows)} pairs -> {out}")
    print("[r43_consistency_dump] flags: "
          f"instr_missing_in_target={sum(bool(r['instr_missing_in_target']) for r in rows)}, "
          f"target_extra_unasked={sum(bool(r['target_extra_unasked']) for r in rows)}, "
          f"tgt_obj_in_instr=False:{count('tgt_obj_in_instr', 'False')}, "
          f"tgt_obj_in_target=False:{count('tgt_obj_in_target', 'False')}, "
          f"src_obj_in_source=False:{count('src_obj_in_source', 'False')}, "
          f"src_obj_in_instr=False:{count('src_obj_in_instr', 'False')}, "
          f"added_obj_in_source=True:{count('added_obj_in_source', 'True')}, "
          f"removed_obj_in_target=True:{count('removed_obj_in_target', 'True')}, "
          f"verb_vs_type:{sum(bool(r['verb_vs_type']) for r in rows)}")


if __name__ == "__main__":
    main()
