"""R43 -- dump every FiVE-Bench pair's prompts and FiVE-Acc questions with automatic flags.

Reads the released ``edit_prompt/edit{T}_FiVE.json`` files (read-only) and writes one CSV row
per pair (419 rows) for the manual ``audit`` step. Multi-line fields (the multiple-choice
question) have their newlines written as `` | `` so every pair stays on one line; all other
characters, including trailing spaces, are kept verbatim.

Automatic flags (hints that order the manual read, not verdicts; empty = not raised):
  * ``polarity_suspect`` -- the target question names the source object, or the source
    question names the target object. "Names" = contains a content word that is in that
    object but not in the other one (so the shared head noun of an attribute edit, e.g.
    "horse" in white horse -> black horse, does not count). Written as ``tgt_q:<words>`` /
    ``src_q:<words>``. FiVE's template often names both objects ("Is the bear a panda?"),
    so hits where the question names none of its OWN object's distinguishing words (the
    likely swaps) are tagged `` [own object missing]``.
  * ``unknown_words`` -- question content words found in none of the pair's
    ``source_prompt``, ``target_prompt``, ``source_object``, ``target_object``. Written per
    question as ``src_q:`` / ``tgt_q:`` / ``mcq:``. MCQ distractor options land here by design.
    Invisible characters (e.g. zero-width spaces) are shown escaped (``\\u200b``).
  * ``negation`` -- ``not`` / ``no`` / ``without`` (or an ``n't`` contraction) in a question;
    a bare ``yes`` / ``no`` MCQ option line does not count. Lists the questions it was found in.
  * ``empty_field`` -- dumped fields that are empty or whitespace-only.
  * ``mcq_n_options`` / ``mcq_b_names_target`` -- number of ``a)``/``b)``/... option lines, and
    whether option b) contains every content word of ``target_object``.

Object checks -- presence:
  * ``src_obj_in_prompt`` / ``tgt_obj_in_prompt`` -- the object is a verbatim, case-insensitive,
    word-bounded span of its own prompt (stdlib stand-in for the inference's
    ``find_phrase_token_indices``; the exact token check is R43's ``trigger-check``). Empty
    when the object itself is empty.
  * ``src_obj_ctx`` / ``tgt_obj_ctx`` -- the matched span in brackets with up to 3 prompt words
    on each side (``a [hang] glider is soaring``), so truncated or wrong spans are visible.
  * ``obj_same`` -- source and target objects have the same content words (a no-op).
  * ``obj_negated`` -- ``not`` / ``no`` / ``without`` inside an object (``src`` / ``tgt``).

Object checks -- choice, on the content-word diff of the two prompts:
  * ``diff_removed`` / ``diff_added`` -- source-only / target-only prompt words, i.e. the edit
    the prompt pair implies.
  * ``obj_unedited_words`` -- object words in neither diff side and not shared by both objects
    (the shared head noun of an attribute edit is accepted; the "rider" of
    ``a horse with a rider`` -> ``A camel`` is flagged). Written as ``src:`` / ``tgt:``.
  * ``diff_uncovered`` -- removed words absent from ``source_object`` and added words absent
    from ``target_object``. Prompt leftovers also land here, so it is a hint.
  * ``obj_vs_type`` -- deviation from the FiVE-Bench-Correct object conventions: edit5
    ``source_object`` must be empty (added object), edit6 ``target_object`` must be empty
    (removed object), types 1-4 need both objects.

Content words: lower-cased, surrounding punctuation stripped, naive plural strip (trailing
"s"), stop words and question-template words removed.

Example
-------
    python evaluation/r43_audit_dump.py \\
        --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \\
        --out_csv evaluation/csv/r43_question_dump.csv
"""

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

EDIT_TYPES = range(1, 7)
QUESTION_FIELDS = {"src_q": "source_yes_no_question",
                   "tgt_q": "target_yes_no_question",
                   "mcq": "multiple_choice_question"}
DUMP_FIELDS = ("source_object", "target_object", "instruction", "source_prompt",
               "target_prompt", "source_yes_no_question", "target_yes_no_question",
               "multiple_choice_question")
STOP_WORDS = {
    # articles, pronouns, conjunctions, prepositions
    "a", "an", "the", "this", "that", "these", "those", "it", "its", "his", "her", "their",
    "they", "he", "she", "him", "them", "and", "or", "of", "in", "on", "at", "to", "by",
    "with", "from", "into", "onto", "for", "as", "up", "down", "over", "under", "near",
    "along", "across", "through", "around", "behind", "beside", "alongside",
    # auxiliaries / question words
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "has", "have",
    "had", "can", "there", "what", "which", "who", "whom", "where", "how", "any", "some",
    # negation (reported by the negation flag instead)
    "not", "no", "without", "yes",
    # FiVE question templates
    "image", "picture", "video", "frame", "scene", "object", "present", "visible", "shown",
    "edited", "other", "please", "specify",
}
NEGATION = re.compile(r"\b(not|no|without)\b|n't\b", re.I)
YES_NO_OPTION = re.compile(r"^\s*[a-z]\)\s*(yes|no)\s*$", re.I)
OPTION_LINE = re.compile(r"^\s*([a-z])\)\s*(.*)$", re.I)
OPTION_LETTER = re.compile(r"[a-z]\)", re.I)
OWN_MISSING = " [own object missing]"
PUNCT = ".,;:!?\"'()[]{}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", type=str,
                        default="~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark")
    parser.add_argument("--out_csv", type=str, default="evaluation/csv/r43_question_dump.csv")
    return parser.parse_args()


def show(token: str) -> str:
    """Escape invisible format characters (zero-width spaces etc.) so they are readable."""
    return "".join(f"\\u{ord(ch):04x}" if unicodedata.category(ch) == "Cf" else ch
                   for ch in token)


def normalize(token: str) -> str:
    """Lower-case, strip surrounding punctuation and possessive, naive plural strip."""
    tok = token.strip(PUNCT).lower()
    if tok.endswith("'s"):
        tok = tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        tok = tok[:-1]
    return tok


def content_words(text: str) -> list[str]:
    """Content words of ``text`` in order (duplicates kept), stop words removed."""
    out = []
    for raw in text.split():
        if OPTION_LETTER.fullmatch(raw):
            continue
        tok = normalize(raw)
        if tok and tok not in STOP_WORDS:
            out.append(tok)
    return out


def mcq_options(mcq: str) -> list[tuple[str, str]]:
    """``(letter, text)`` for every option line of a multiple-choice question."""
    opts = []
    for line in mcq.split("\n"):
        m = OPTION_LINE.match(line)
        if m:
            opts.append((m.group(1).lower(), m.group(2)))
    return opts


def flag_polarity(item: dict) -> str:
    src = set(content_words(item["source_object"]))
    tgt = set(content_words(item["target_object"]))
    only_src, only_tgt = src - tgt, tgt - src
    tgt_q_words = set(content_words(item["target_yes_no_question"]))
    src_q_words = set(content_words(item["source_yes_no_question"]))
    hits = []
    if only_src & tgt_q_words:
        hit = "tgt_q:" + " ".join(sorted(only_src & tgt_q_words))
        hits.append(hit + ("" if only_tgt & tgt_q_words else OWN_MISSING))
    if only_tgt & src_q_words:
        hit = "src_q:" + " ".join(sorted(only_tgt & src_q_words))
        hits.append(hit + ("" if only_src & src_q_words else OWN_MISSING))
    return "; ".join(hits)


def flag_unknown(item: dict) -> str:
    known = set()
    for field in ("source_prompt", "target_prompt", "source_object", "target_object"):
        known.update(content_words(item[field]))
    hits = []
    for tag, field in QUESTION_FIELDS.items():
        unknown = []
        for w in content_words(item[field]):
            if w not in known and show(w) not in unknown:
                unknown.append(show(w))
        if unknown:
            hits.append(f"{tag}:" + " ".join(unknown))
    return "; ".join(hits)


def flag_negation(item: dict) -> str:
    hits = []
    for tag, field in QUESTION_FIELDS.items():
        lines = item[field].split("\n")
        if tag == "mcq":
            lines = [ln for ln in lines if not YES_NO_OPTION.match(ln)]
        if any(NEGATION.search(ln) for ln in lines):
            hits.append(tag)
    return " ".join(hits)


def flag_empty(item: dict) -> str:
    return " ".join(f for f in DUMP_FIELDS if not str(item.get(f) or "").strip())


def find_span(obj: str, prompt: str) -> re.Match | None:
    """Verbatim, case-insensitive, word-bounded match of ``obj`` in ``prompt``."""
    obj = obj.strip()
    if not obj:
        return None
    return re.search(r"(?<!\w)" + re.escape(obj) + r"(?!\w)", prompt, re.I)


def span_context(match: re.Match | None, prompt: str, n_words: int = 3) -> str:
    """``before [span] after`` with up to ``n_words`` prompt words on each side."""
    if match is None:
        return ""
    before = prompt[:match.start()].split()[-n_words:]
    after = prompt[match.end():].split()[:n_words]
    return " ".join(before + [f"[{match.group(0)}]"] + after)


def unique(words: list[str]) -> list[str]:
    return list(dict.fromkeys(words))


def object_flags(edit_type: int, item: dict) -> dict:
    src_obj, tgt_obj = item["source_object"], item["target_object"]
    src_prompt, tgt_prompt = item["source_prompt"], item["target_prompt"]
    src_m, tgt_m = find_span(src_obj, src_prompt), find_span(tgt_obj, tgt_prompt)

    src_words, tgt_words = content_words(src_obj), content_words(tgt_obj)
    src_prompt_words = content_words(src_prompt)
    tgt_prompt_words = content_words(tgt_prompt)
    removed = unique([w for w in src_prompt_words if w not in set(tgt_prompt_words)])
    added = unique([w for w in tgt_prompt_words if w not in set(src_prompt_words)])
    edited, shared = set(removed) | set(added), set(src_words) & set(tgt_words)

    unedited = []
    for tag, words in (("src", src_words), ("tgt", tgt_words)):
        hits = unique([w for w in words if w not in edited and w not in shared])
        if hits:
            unedited.append(f"{tag}:" + " ".join(hits))
    uncovered = []
    if [w for w in removed if w not in set(src_words)]:
        uncovered.append("removed:" + " ".join(w for w in removed if w not in set(src_words)))
    if [w for w in added if w not in set(tgt_words)]:
        uncovered.append("added:" + " ".join(w for w in added if w not in set(tgt_words)))

    vs_type = []
    if edit_type == 5 and src_obj.strip():
        vs_type.append("edit5 source_object not empty")
    if edit_type == 6 and tgt_obj.strip():
        vs_type.append("edit6 target_object not empty")
    if edit_type in (1, 2, 3, 4):
        vs_type += [f"{name} empty" for name, obj in (("source_object", src_obj),
                                                     ("target_object", tgt_obj))
                    if not obj.strip()]

    negated = [tag for tag, obj in (("src", src_obj), ("tgt", tgt_obj)) if NEGATION.search(obj)]
    return {
        "src_obj_in_prompt": "" if not src_obj.strip() else str(src_m is not None),
        "tgt_obj_in_prompt": "" if not tgt_obj.strip() else str(tgt_m is not None),
        "src_obj_ctx": span_context(src_m, src_prompt),
        "tgt_obj_ctx": span_context(tgt_m, tgt_prompt),
        "obj_same": str(bool(src_words) and src_words == tgt_words),
        "obj_negated": " ".join(negated),
        "diff_removed": " ".join(removed),
        "diff_added": " ".join(added),
        "obj_unedited_words": "; ".join(unedited),
        "diff_uncovered": "; ".join(uncovered),
        "obj_vs_type": "; ".join(vs_type),
    }


def dump_row(edit_type: int, item: dict) -> dict:
    opts = mcq_options(item["multiple_choice_question"])
    b_text = dict(opts).get("b")
    tgt_words = set(content_words(item["target_object"]))
    if b_text is None:
        b_names_target = ""
    else:
        b_names_target = str(bool(tgt_words) and tgt_words <= set(content_words(b_text)))
    row = {"case_id": f"edit{edit_type}/{item['video_name']}", "edit_type": edit_type}
    for field in DUMP_FIELDS:
        row[field] = str(item.get(field) or "").replace("\n", " | ")
    row.update({
        "polarity_suspect": flag_polarity(item),
        "unknown_words": flag_unknown(item),
        "negation": flag_negation(item),
        "empty_field": flag_empty(item),
        "mcq_n_options": len(opts),
        "mcq_b_names_target": b_names_target,
    })
    row.update(object_flags(edit_type, item))
    return row


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser()
    rows = []
    for t in EDIT_TYPES:
        path = data_root / "edit_prompt" / f"edit{t}_FiVE.json"
        if not path.is_file():
            sys.exit(f"[r43_audit_dump] missing annotation file: {path}")
        for item in json.loads(path.read_text()):
            missing = [f for f in ("video_name",) + DUMP_FIELDS if f not in item]
            if missing:
                sys.exit(f"[r43_audit_dump] edit{t}/{item.get('video_name')}: "
                         f"missing keys {missing}")
            rows.append(dump_row(t, item))

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    n_flag = {k: sum(bool(r[k]) for r in rows)
              for k in ("polarity_suspect", "unknown_words", "negation", "empty_field")}
    n_own_missing = sum(OWN_MISSING in r["polarity_suspect"] for r in rows)
    n_mcq_b = sum(r["mcq_b_names_target"] == "False" for r in rows)
    print(f"[r43_audit_dump] {len(rows)} pairs -> {out}")
    print("[r43_audit_dump] flagged pairs: "
          + ", ".join(f"{k}={v}" for k, v in n_flag.items())
          + f" (polarity own-object-missing={n_own_missing})"
          + f", mcq_b_names_target=False: {n_mcq_b}")
    n_obj = {k: sum(r[k] not in ("", "True") for r in rows)
             for k in ("src_obj_in_prompt", "tgt_obj_in_prompt")}
    n_obj.update({k: sum(bool(r[k]) and r[k] != "False" for r in rows)
                  for k in ("obj_same", "obj_negated", "obj_unedited_words",
                            "diff_uncovered", "obj_vs_type")})
    print("[r43_audit_dump] object flags: "
          + ", ".join(f"{k}={v}" for k, v in n_obj.items())
          + " (in_prompt counts = objects NOT found)")


if __name__ == "__main__":
    main()
