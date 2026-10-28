"""R62 step wan-assemble (ON JEAN ZAY): one 415-clip tree of R51's Wan-Edit A4 (nosupp_s5) renders on FiVE-Bench-Correct.

The A4 videos live in two folders: the pairs whose Wan-Edit inputs changed in Correct (R58's evaluation/r58_cases_maxs5.json, 190 pairs) were
rendered by R62 into r62_wan_arms; every other pair has the same prompts and trigger words as on official FiVE, so R51's render is reused
(r51_wan_arms). The evaluation reads ONE root, so for each of the 415 pairs of the Correct manifest this builds a symlink

    $D/r62_wan_correct/nosupp_s5/edit{T}/{video}  ->  $D/r62_wan_arms/nosupp_s5_full/edit{T}/{video}   (pair in the regen set)
                                                   or $D/r51_wan_arms/nosupp_s5_full/edit{T}/{video}   (otherwise)

and checks that the target is a complete render (>= the frames Wan-Edit produces for that source: 41, fewer for the 7 short videos; the same
criterion as r48_units.py --missing). Nothing is copied and nothing is ever written into a render folder: only symlinks are created inside the
output root, which must be named .../r62_wan_correct/nosupp_s5. An existing entry that is not a symlink, or a symlink to anything other than the
intended target, is a CONFLICT: it is left untouched and the run exits 1. Re-running is idempotent (an identical link is kept).
Writes evaluation/csv/r62_wan_assembly.csv (case_id, source new|reused, n_png, need); jz/r62_fetch_renders.sh compares clip PNG counts to n_png.
Exit 1 unless 415 clips (190 new + 225 reused), 0 missing / incomplete, 0 conflicts.

usage (Jean Zay): $WORK/pyenv/five-bench/python evaluation/r62_assemble.py [--check]     (--check: report only, create nothing)
       python evaluation/r62_assemble.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
D = Path.home() / "Data/dataggen/outputs/five_bench"
CONFIG = "nosupp_s5"
EXPECT_TOTAL, EXPECT_NEW = 415, 190


def need_frames(video_name: str) -> int:
    """PNGs a complete Wan-Edit render of this source has (r48_units.py --missing's criterion)."""
    sys.path.insert(0, str(HERE))
    import r48_wan_common as C  # noqa: E402  (imports torch: only needed on the render platform)
    return C.expected_frames(C.source_frame_count(video_name))


def load_pairs(path: Path) -> List[Tuple[int, str]]:
    return [(int(c["edit_type"]), c["video_name"]) for c in json.loads(path.read_text())]


def assemble(cases: List[Tuple[int, str]], regen: set, new_root: Path, reuse_root: Path, out_root: Path,
             need: Callable[[str], int], check_only: bool, expect_total: Optional[int], expect_new: Optional[int]
             ) -> Tuple[int, List[Dict[str, object]]]:
    """Returns (failures, rows). Failures count missing / incomplete clips, conflicts and count mismatches."""
    rows: List[Dict[str, object]] = []
    failures = 0
    real_new, real_reuse = os.path.realpath(new_root), os.path.realpath(reuse_root)
    for t, vid in cases:
        is_new = (t, vid) in regen
        src = (new_root if is_new else reuse_root) / f"edit{t}" / vid
        link = out_root / f"edit{t}" / vid
        n_png = len(list(src.glob("*.png"))) if src.is_dir() else 0
        n_need = need(vid)
        row = {"case_id": f"edit{t}/{vid}", "source": "new" if is_new else "reused", "n_png": n_png, "need": n_need}
        rows.append(row)
        if n_png < n_need:
            print(f"[r62_assemble] {'MISSING' if n_png == 0 else 'INCOMPLETE'} {row['case_id']} ({row['source']}): {n_png}/{n_need} PNGs under {src}")
            failures += 1
            continue
        if os.path.lexists(link):
            if not os.path.islink(link):
                print(f"[r62_assemble] CONFLICT {link} exists and is not a symlink -- left untouched")
                failures += 1
            elif os.path.realpath(link) != os.path.realpath(src):
                where = os.path.realpath(link)
                other = "the other source root" if where.startswith((real_new, real_reuse)) else "somewhere else"
                print(f"[r62_assemble] CONFLICT {link} points to {other} ({where}), expected {os.path.realpath(src)} -- left untouched")
                failures += 1
            continue                                           # identical link: kept
        if not check_only:
            link.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(src, link)
            if len(list(link.glob("*.png"))) != n_png:         # read back through the link
                print(f"[r62_assemble] FAILED {link}: {len(list(link.glob('*.png')))} PNGs through the link, {n_png} at the target")
                failures += 1
    n_new = sum(1 for r in rows if r["source"] == "new")
    n_reused = len(rows) - n_new
    if expect_total is not None and len(rows) != expect_total:
        print(f"[r62_assemble] COUNT: {len(rows)} clips, expected {expect_total}")
        failures += 1
    if expect_new is not None and n_new != expect_new:
        print(f"[r62_assemble] COUNT: {n_new} new clips, expected {expect_new}")
        failures += 1
    print(f"[r62_assemble] {CONFIG}: {len(rows)} clips = {n_new} new + {n_reused} reused; bad {failures}")
    return failures, rows


def selftest() -> int:
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        ok = ok and cond

    with tempfile.TemporaryDirectory(prefix="r62_assemble_") as tmp:
        tmp_p = Path(tmp)
        new, old, out = tmp_p / "r62_wan_arms/nosupp_s5_full", tmp_p / "r51_wan_arms/nosupp_s5_full", tmp_p / "r62_wan_correct/nosupp_s5"
        for root, t, vid, n in ((new, 1, "a", 3), (old, 2, "b", 3), (old, 5, "c", 2)):   # "c" is a short video: needs only 2 frames
            d = root / f"edit{t}" / vid
            d.mkdir(parents=True)
            for i in range(n):
                (d / f"{i:05d}.png").write_bytes(b"x")
        cases = [(1, "a"), (2, "b"), (5, "c")]
        regen = {(1, "a")}
        need = lambda v: 2 if v == "c" else 3                                              # noqa: E731
        f, rows = assemble(cases, regen, new, old, out, need, False, 3, 1)
        check("clean run: 0 failures, 3 links", f == 0 and len(list(out.glob("edit*/*"))) == 3)
        check("sources: a new, b/c reused", [r["source"] for r in rows] == ["new", "reused", "reused"])
        check("links point at the right roots", os.path.realpath(out / "edit1/a") == os.path.realpath(new / "edit1/a")
              and os.path.realpath(out / "edit2/b") == os.path.realpath(old / "edit2/b"))
        f, _ = assemble(cases, regen, new, old, out, need, False, 3, 1)
        check("re-run is idempotent", f == 0)
        f, _ = assemble(cases, regen, new, old, out, need, False, 3, 2)
        check("wrong expected new-count fails", f == 1)
        (new / "edit1/a/00002.png").unlink()
        f, _ = assemble(cases, regen, new, old, out, need, False, 3, 1)
        check("incomplete render fails", f == 1)
        (new / "edit1/a/00002.png").write_bytes(b"x")
        os.remove(out / "edit2/b")
        os.symlink(new / "edit1/a", out / "edit2/b")                                       # link into the OTHER root
        f, _ = assemble(cases, regen, new, old, out, need, False, 3, 1)
        check("link into the other root is a conflict and is left untouched", f == 1 and os.path.realpath(out / "edit2/b") == os.path.realpath(new / "edit1/a"))
        os.remove(out / "edit2/b")
        (out / "edit2/b").mkdir()
        f, _ = assemble(cases, regen, new, old, out, need, False, 3, 1)
        check("real directory in the output root is a conflict", f == 1 and (out / "edit2/b").is_dir() and not os.path.islink(out / "edit2/b"))
        (out / "edit2/b").rmdir()
        f, _ = assemble([(1, "a"), (2, "zz")], regen, new, old, out, need, False, None, None)
        check("missing clip fails", f == 1)
        out2 = tmp_p / "elsewhere/r62_wan_correct/nosupp_s5"
        f, _ = assemble(cases, regen, new, old, out2, need, True, 3, 1)
        check("--check creates nothing", f == 0 and not out2.exists())
    print("[r62_assemble] selftest", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check", action="store_true", help="report only, create no link and no CSV")
    ap.add_argument("--cases", default=str(HERE / "r43_anchor_manifest_correct.json"))
    ap.add_argument("--regen", default=str(HERE / "r58_cases_maxs5.json"), help="pairs rendered by R62 (R58's max_s5 regeneration set)")
    ap.add_argument("--new_root", default=str(D / "r62_wan_arms" / f"{CONFIG}_full"))
    ap.add_argument("--reuse_root", default=str(D / "r51_wan_arms" / f"{CONFIG}_full"))
    ap.add_argument("--out_root", default=str(D / "r62_wan_correct" / CONFIG))
    ap.add_argument("--csv", default=str(HERE / "csv" / "r62_wan_assembly.csv"))
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    out_root = Path(a.out_root).expanduser()
    if out_root.name != CONFIG or out_root.parent.name != "r62_wan_correct":
        print(f"[r62_assemble] FATAL: refusing output root {out_root} (must end with r62_wan_correct/{CONFIG}: links are only ever created there)")
        return 1
    new_root, reuse_root = Path(a.new_root).expanduser(), Path(a.reuse_root).expanduser()
    for r in (new_root, reuse_root):
        if not r.is_dir():
            print(f"[r62_assemble] FATAL: {r} is not a directory (render not finished, or purged from $SCRATCH?)")
            return 1
    cases, regen = load_pairs(Path(a.cases)), set(load_pairs(Path(a.regen)))
    if not regen <= set(cases):
        print(f"[r62_assemble] FATAL: {len(regen - set(cases))} regeneration pair(s) are not in the Correct manifest")
        return 1
    print(f"[r62_assemble] {'CHECK' if a.check else 'ASSEMBLE'} {len(cases)} pairs ({len(regen)} new): new={new_root} reuse={reuse_root} -> {out_root}")
    failures, rows = assemble(cases, regen, new_root, reuse_root, out_root, need_frames, a.check, EXPECT_TOTAL, EXPECT_NEW)
    if not a.check and rows:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["case_id", "source", "n_png", "need"])
            w.writeheader()
            w.writerows(rows)
        print(f"[r62_assemble] wrote {a.csv} ({len(rows)} rows)")
    print(f"[r62_assemble] failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
