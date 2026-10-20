"""R43 -- build FiVE-Bench-Correct: corrected copies of the six FiVE annotation files.

Reads the released ``edit_prompt/edit{T}_FiVE.json`` files (never modified), applies the
per-pair corrections in ``--corrections`` and the approved single-edit instructions
(``instruction_v2`` in ``--instructions``, for EVERY non-excluded pair), and writes full copies
to ``--out_dir/edit{T}_FiVE.json`` (same filename, so ``evaluate.py`` still derives the edit
type from the ``editT`` prefix). Every changed field keeps its released value as
``<field>_orig``; excluded pairs are dropped. A diff CSV lists one row per changed field.

Validation (non-zero exit, nothing written on any failure):
  * corrections and the issues CSV cover the same pairs; every pair has an instruction_v2;
  * span check -- each word a correction removes from a target prompt lies inside one of the
    pair's ``changed_spans``;
  * copy check -- each word a correction inserts into a target prompt or a question exists in
    the pair's own FiVE fields (or ``template_from``'s), except the typo-fix exception (a
    replacement within edit distance 2 of the word it replaces, or a <-> an:
    ``L1-spelling``) and the grammar exception (is <-> are: ``L1-grammar``); reported per row
    in ``inserted_from``;
  * no ``must_not_contain`` word survives in the corrected target prompt (word boundary);
  * no edit6 target prompt keeps a negated mention;
  * object-span guard -- every non-empty object is a verbatim, case-insensitive, word-bounded
    span of its own (corrected) prompt (the rule of the inference's
    ``find_phrase_token_indices``); ``source_object`` is empty on every edit5 pair and only
    there, ``target_object`` on every edit6 pair and only there;
  * suffix guard -- the keep-pose suffix never appears in ``instruction``, ``target_prompt``
    or a question;
  * every uncorrected pair is identical to the released entry apart from ``instruction``.

With ``--manifest_dir``, also writes three anchor manifests in R7's schema: the correct
manifest (every non-excluded pair, instruction_v2 + keep-pose suffix for types 1-4), the
smoke manifest (26 smoke cases taken from it) and the repro manifest (3 R7 pairs verbatim).

Example
-------
    python evaluation/r43_build_correct.py \\
        --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \\
        --corrections evaluation/r43_corrections.json \\
        --instructions evaluation/r43_instructions.json \\
        --issues evaluation/csv/r43_five_prompt_issues.csv \\
        --r7_manifest evaluation/r7_anchor_manifest.json \\
        --out_dir evaluation/five_bench_correct \\
        --diff_csv evaluation/csv/r43_correct_diff.csv \\
        --manifest_dir evaluation
"""

import argparse
import copy
import csv
import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

EDIT_TYPES = range(1, 7)
CORRECTABLE = ("target_prompt", "source_yes_no_question", "target_yes_no_question",
               "multiple_choice_question", "source_object", "target_object")
COPY_CHECKED = ("target_prompt", "source_yes_no_question", "target_yes_no_question",
                "multiple_choice_question")
QUESTION_FIELDS = ("source_yes_no_question", "target_yes_no_question", "multiple_choice_question")
VOCAB_FIELDS = ("source_prompt", "target_prompt", "source_object", "target_object", "instruction",
                "source_yes_no_question", "target_yes_no_question", "multiple_choice_question")
CORRECTION_KEYS = {"case_id", "level", "changed_spans", "must_not_contain", "template_from",
                   "exclude", "note", *CORRECTABLE}
OBJECT_PROMPT = {"source_object": "source_prompt", "target_object": "target_prompt"}
NEGATION = re.compile(r"\bwithout\b|\bnot wearing\b|\bnot accompanied\b", re.I)
KEEP_POSE_SUFFIX = (" Keep the coarse position, size, facing direction and coarse pose of the "
                    "original object.")
SUFFIX_TYPES = (1, 2, 3, 4)
ANCHOR_BASE = "~/Data/dataggen/outputs/five_bench"
SMOKE_TYPES = (1, 2)
SMOKE_VIDEOS = ("0012_goat", "0045_butterfly", "0059_drift-straight", "0081_A_horse",
                "0098_A_squirrel", "0074_rhino", "0005_breakdance", "0036_camel",
                "0003_ocean-birds", "0022_skate-jump", "0030_rallye", "0062_elephant",
                "0018_breakdance-flare")
REPRO_CASES = ("edit1/0001_bus", "edit2/0081_A_horse", "edit5/0007_guitar-violin")
MAX_TYPO_DISTANCE = 2
ARTICLES = {"a", "an"}
AGREEMENT = {"is", "are"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", type=str,
                        default="~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark")
    parser.add_argument("--corrections", type=str, default="evaluation/r43_corrections.json")
    parser.add_argument("--instructions", type=str, default="evaluation/r43_instructions.json")
    parser.add_argument("--issues", type=str, default="evaluation/csv/r43_five_prompt_issues.csv")
    parser.add_argument("--r7_manifest", type=str, default="evaluation/r7_anchor_manifest.json")
    parser.add_argument("--out_dir", type=str, default="evaluation/five_bench_correct")
    parser.add_argument("--diff_csv", type=str, default="evaluation/csv/r43_correct_diff.csv")
    parser.add_argument("--manifest_dir", type=str, default=None,
                        help="write the correct / smoke / repro anchor manifests here")
    return parser.parse_args()


def words(text: str) -> list[str]:
    """Whitespace tokens, lower-cased, stripped of surrounding punctuation and a trailing 's."""
    out = []
    for tok in text.split():
        tok = tok.strip(".,;:!?\"'()").lower()
        if tok.endswith("'s"):
            tok = tok[:-2]
        if tok:
            out.append(tok)
    return out


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (small words only)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def in_vocab(word: str, vocab: set[str]) -> bool:
    return word in vocab or word + "s" in vocab or (word.endswith("s") and word[:-1] in vocab)


def removed_words(original: str, corrected: str) -> list[str]:
    """Original words that the correction deleted or replaced."""
    a, b = words(original), words(corrected)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return [w for tag, i1, i2, _, _ in sm.get_opcodes() if tag in ("replace", "delete")
            for w in a[i1:i2]]


def copy_check(original: str, corrected: str, vocab: set[str],
               template_vocab: set[str]) -> tuple[list[str], list[str]]:
    """Classify every inserted word. Returns (provenance tags, offending words)."""
    a, b = words(original), words(corrected)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    tags, bad = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag not in ("replace", "insert"):
            continue
        replaced = a[i1:i2]
        for w in b[j1:j2]:
            if in_vocab(w, vocab):
                tags.append("pair")
            elif in_vocab(w, template_vocab):
                tags.append("template")
            elif w in ARTICLES and set(replaced) & ARTICLES:
                tags.append(f"L1-spelling:{w}")
            elif w in AGREEMENT and set(replaced) & AGREEMENT:
                tags.append(f"L1-grammar:{w}")
            elif len(w) >= 3 and any(edit_distance(w, r) <= MAX_TYPO_DISTANCE for r in replaced):
                tags.append(f"L1-spelling:{w}")
            else:
                bad.append(w)
    return tags, bad


def is_span(obj: str, prompt: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(obj.strip()) + r"(?!\w)", prompt, re.I) is not None


def pair_vocab(entry: dict) -> set[str]:
    return {w for f in VOCAB_FIELDS for w in words(str(entry.get(f) or ""))}


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser()
    corrections = json.loads(Path(args.corrections).read_text())
    by_id = {c["case_id"]: c for c in corrections}
    if len(by_id) != len(corrections):
        sys.exit("[r43_build_correct] duplicate case_id in corrections")
    instr_v2 = {c["case_id"]: c["instruction_v2"]
                for c in json.loads(Path(args.instructions).read_text())}
    issue_groups, issue_levels = defaultdict(list), {}
    with open(args.issues, newline="") as f:
        for r in csv.DictReader(f):
            issue_groups[r["case_id"]].append(r["group"].split("_")[0])
            issue_levels[r["case_id"]] = max(issue_levels.get(r["case_id"], "L1"), r["level"])

    failures = []
    if set(by_id) != set(issue_groups):
        failures.append(f"corrections != issue pairs: missing={sorted(set(issue_groups) - set(by_id))} "
                        f"extra={sorted(set(by_id) - set(issue_groups))}")
    for cid, corr in by_id.items():
        unknown = set(corr) - CORRECTION_KEYS
        if unknown:
            failures.append(f"{cid}: unknown correction keys {sorted(unknown)}")

    released = {}
    for t in EDIT_TYPES:
        for entry in json.loads((data_root / "edit_prompt" / f"edit{t}_FiVE.json").read_text()):
            released[f"edit{t}/{entry['video_name']}"] = entry
    missing_v2 = sorted(set(released) - set(instr_v2))
    if missing_v2:
        failures.append(f"pairs without instruction_v2: {missing_v2}")

    diff_rows, outputs = [], {t: [] for t in EDIT_TYPES}
    for cid, entry in released.items():
        t = int(cid[4])
        corr = by_id.get(cid, {})
        groups = "+".join(sorted(set(issue_groups.get(cid, []))))
        level = corr.get("level", issue_levels.get(cid, ""))
        if corr.get("exclude"):
            diff_rows.append([cid, level, "(pair)", "", "EXCLUDED", "", groups, corr.get("note", "")])
            continue

        vocab = pair_vocab(entry)
        template = corr.get("template_from")
        template_vocab = pair_vocab(released[template]) if template else set()
        span_words = {w for s in corr.get("changed_spans", []) for w in words(s)}
        new = {}
        for key, value in entry.items():
            if key == "instruction":
                v2 = instr_v2.get(cid, value)
                if v2 != value:
                    new["instruction_orig"] = value
                    diff_rows.append([cid, level, "instruction", value, v2,
                                      "instruction_v2 (exempt)", groups, corr.get("note", "")])
                new["instruction"] = v2
            elif key in CORRECTABLE and key in corr and corr[key] != value:
                fixed = corr[key]
                provenance = ""
                if key == "target_prompt":
                    stray = [w for w in removed_words(value, fixed) if w not in span_words]
                    if stray:
                        failures.append(f"{cid} target_prompt: edits outside changed_spans: {stray}")
                if key in COPY_CHECKED:
                    tags, bad = copy_check(value, fixed, vocab, template_vocab)
                    if bad:
                        failures.append(f"{cid} {key}: inserted words not in the pair's FiVE "
                                        f"fields: {bad}")
                    provenance = ",".join(sorted(Counter(tags)))
                new[f"{key}_orig"] = value
                new[key] = fixed
                diff_rows.append([cid, level, key, value, fixed, provenance, groups,
                                  corr.get("note", "")])
            else:
                new[key] = value

        if not corr:
            rest = {k: v for k, v in new.items() if k not in ("instruction", "instruction_orig")}
            if rest != {k: v for k, v in entry.items() if k != "instruction"}:
                failures.append(f"{cid}: uncorrected pair differs from the released entry")

        tp = new["target_prompt"]
        for w in corr.get("must_not_contain", []):
            if re.search(rf"\b{re.escape(w.strip())}\b", tp, re.I):
                failures.append(f"{cid}: target_prompt still contains {w!r}")
        if t == 6 and NEGATION.search(tp):
            failures.append(f"{cid}: edit6 target_prompt keeps a negated mention")

        for obj_key, prompt_key in OBJECT_PROMPT.items():
            obj = new[obj_key].strip()
            must_be_empty = (t == 5 and obj_key == "source_object") or \
                            (t == 6 and obj_key == "target_object")
            if must_be_empty and obj:
                failures.append(f"{cid}: {obj_key} must be empty on edit{t}, got {obj!r}")
            elif not must_be_empty and not obj:
                failures.append(f"{cid}: {obj_key} is empty (allowed only for "
                                f"{'edit5 source' if obj_key == 'source_object' else 'edit6 target'})")
            elif obj and not is_span(obj, new[prompt_key]):
                failures.append(f"{cid}: {obj_key} {obj!r} is not a span of {prompt_key}")

        for key in ("instruction", "target_prompt", *QUESTION_FIELDS):
            if KEEP_POSE_SUFFIX.strip() in str(new.get(key) or ""):
                failures.append(f"{cid}: keep-pose suffix leaked into {key}")
        outputs[t].append(new)

    unknown_ids = sorted(set(by_id) - set(released))
    if unknown_ids:
        failures.append(f"corrections for unknown pairs: {unknown_ids}")

    manifests = {}
    if args.manifest_dir:
        correct = [{"case_id": f"edit{t}/{e['video_name']}", "edit_type": t,
                    "video_name": e["video_name"],
                    "src_first_frame": f"images/{e['video_name']}/00001.jpg",
                    "first_frame_edit_prompt": e["instruction"] + (KEEP_POSE_SUFFIX if t in SUFFIX_TYPES else ""),
                    "anchor_image": f"{ANCHOR_BASE}/anchors_r43_correct/edit{t}/{e['video_name']}.png"}
                   for t in EDIT_TYPES for e in outputs[t]]
        smoke = [dict(m, anchor_image=f"{ANCHOR_BASE}/anchor_r43_smoke/{m['case_id']}.png")
                 for m in correct if m["edit_type"] in SMOKE_TYPES and m["video_name"] in SMOKE_VIDEOS]
        if len(smoke) != len(SMOKE_TYPES) * len(SMOKE_VIDEOS):
            failures.append(f"smoke manifest has {len(smoke)} cases, expected "
                            f"{len(SMOKE_TYPES) * len(SMOKE_VIDEOS)}")
        r7 = {m["case_id"]: m for m in json.loads(Path(args.r7_manifest).read_text())}
        missing_repro = [c for c in REPRO_CASES if c not in r7]
        if missing_repro:
            failures.append(f"repro cases missing from the R7 manifest: {missing_repro}")
        repro = [dict(r7[c], anchor_image=f"{ANCHOR_BASE}/anchor_r43_smoke/repro/{c}.png")
                 for c in REPRO_CASES if c in r7]
        manifests = {"r43_anchor_manifest_correct.json": correct,
                     "r43_anchor_manifest_smoke.json": smoke,
                     "r43_anchor_manifest_repro.json": repro}

    if failures:
        print("\n".join(failures))
        sys.exit(f"[r43_build_correct] {len(failures)} validation failure(s); nothing written")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for t, entries in outputs.items():
        (out_dir / f"edit{t}_FiVE.json").write_text(json.dumps(entries, indent=4, ensure_ascii=False))
    Path(args.diff_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.diff_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "level", "field", "original", "corrected", "inserted_from",
                    "issue_groups", "note"])
        w.writerows(diff_rows)
    for name, rows in manifests.items():
        (Path(args.manifest_dir) / name).write_text(json.dumps(rows, indent=1, ensure_ascii=False))

    n_excl = sum(1 for c in corrections if c.get("exclude"))
    per_type = {t: len(v) for t, v in outputs.items()}
    print(f"[r43_build_correct] {len(by_id) - n_excl} pairs corrected, {n_excl} excluded, "
          f"{sum(per_type.values())} pairs written {per_type} -> {out_dir}; "
          f"{len(diff_rows)} diff rows -> {args.diff_csv}")
    for name, rows in manifests.items():
        print(f"[r43_build_correct] {len(rows)} cases -> {Path(args.manifest_dir) / name}")


if __name__ == "__main__":
    main()
