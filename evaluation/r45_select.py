"""R45 method selection: the single source of truth for which routing combos run.

The routing config is not committed yet (metric {DINO, LPIPS} x regime {unblended,
tau = 0.9} x draft N {5, 15}), so all 8 combos are built and every R45 array is derived
from ONE file, ``evaluation/r45_method.json``, written at the plan's ``method-gate`` step:

    python evaluation/r45_select.py --set --draft_n 5 \\
        --combos ll_dino_n5_unblended,ll_dino_n5_tgt09,ll_lpips_n5_unblended,ll_lpips_n5_tgt09

Read side (each prints one line for an sbatch command or a script):

    r45_select.py stage1           sbatch --array list of the stage-1 RUNS the selection needs
    r45_select.py stage2|stage3    sbatch --array list of the selected COMBOS
    r45_select.py sweepM           the matched-compute step count, 15 + draft_n
    r45_select.py eval             0-(k-1) over the eval method table
    r45_select.py eval --table     the table itself: index <TAB> method <TAB> frame root
    r45_select.py check <id> <stage>   exit 1 if array task <id> of <stage> is not selected
    r45_select.py framecheck       per-clip frame counts of every rendered method vs R36
                                   r36_rho2_vp (SF); exit 1 on any mismatch

Missing json => every read command exits with "run method-gate first".
``R45_METHOD_JSON`` overrides the json path (tests only).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
METHOD_JSON = Path(os.environ.get("R45_METHOD_JSON", REPO / "evaluation" / "r45_method.json"))
FIVE = Path("~/Data/dataggen/outputs/five_bench").expanduser()
R36_REF = FIVE / "r36_rho_sweep" / "r36_rho2_vp"

# Array index order is fixed (plan Decisions): DINO before LPIPS, unblended before tau=0.9,
# N=15 before N=5. Shell scripts carry the same tables; never reorder.
COMBOS = [
    "ll_dino_n15_unblended", "ll_dino_n15_tgt09", "ll_lpips_n15_unblended", "ll_lpips_n15_tgt09",
    "ll_dino_n5_unblended", "ll_dino_n5_tgt09", "ll_lpips_n5_unblended", "ll_lpips_n5_tgt09",
]
RUNS = ["n15_unblended", "n15_tgt09", "n5_unblended", "n5_tgt09"]
RHOS = [2, 3, 4, 6, 8, 10, 20, 50]
STAGES = ("stage1", "stage2", "stage3", "sweepM", "eval")


def combo_n(combo: str) -> int:
    return int(combo.split("_")[2][1:])            # "ll_dino_n15_unblended" -> 15


def combo_run(combo: str) -> str:
    return "_".join(combo.split("_")[2:])          # -> "n15_unblended"


def load() -> dict:
    if not METHOD_JSON.exists():
        raise SystemExit(f"[r45_select] {METHOD_JSON} not found -- run method-gate first "
                         "(r45_select.py --set ...).")
    m = json.loads(METHOD_JSON.read_text())
    bad = [c for c in m["combos"] if c not in COMBOS]
    if bad:
        raise SystemExit(f"[r45_select] {METHOD_JSON} names unknown combos {bad}")
    return m


def selected_runs(m: dict) -> list:
    return sorted({RUNS.index(combo_run(c)) for c in m["combos"]})


def method_table(m: dict) -> list:
    """[(method, frame_root)] in eval-array order: sweep N=15, sweep matched N, routed arms."""
    rows = [(f"ll_rho{r}_n15_vp", FIVE / "r45_sweep" / "n15" / f"ll_rho{r}_n15_vp") for r in RHOS]
    n = m["matched_n"]
    rows += [(f"ll_rho{r}_n{n}_vp", FIVE / "r45_sweep" / f"n{n}" / f"ll_rho{r}_n{n}_vp")
             for r in RHOS]
    rows += [(f"r45_{c}", FIVE / "r45_arms" / f"r45_{c}" / "step14")
             for c in sorted(m["combos"], key=COMBOS.index)]
    return rows


def array_list(ids) -> str:
    return ",".join(str(i) for i in ids)


def cmd_set(a) -> None:
    combos = [c.strip() for c in a.combos.split(",") if c.strip()]
    bad = [c for c in combos if c not in COMBOS]
    if bad or not combos:
        raise SystemExit(f"[r45_select] unknown or empty combos {bad or combos}; "
                         f"choose from {COMBOS}")
    if len(set(combos)) != len(combos):
        raise SystemExit("[r45_select] duplicate combos")
    off = [c for c in combos if combo_n(c) != a.draft_n]
    if off and not a.allow_mixed:
        raise SystemExit(f"[r45_select] combos {off} do not use draft N={a.draft_n}; pass "
                         "--allow_mixed if a mixed-N selection is really intended.")
    m = dict(draft_n=a.draft_n, combos=sorted(combos, key=COMBOS.index),
             matched_n=15 + a.draft_n, allow_mixed=bool(off),
             set_at=_dt.datetime.now().isoformat(timespec="seconds"))
    METHOD_JSON.write_text(json.dumps(m, indent=2) + "\n")
    print(f"[r45_select] wrote {METHOD_JSON}: draft_n={a.draft_n} matched_n={m['matched_n']} "
          f"combos={m['combos']}")


def cmd_check(m: dict, task_id: int, stage: str) -> None:
    if stage == "stage1":
        ok = task_id in selected_runs(m)
        what = RUNS[task_id] if 0 <= task_id < len(RUNS) else "?"
    elif stage in ("stage2", "stage3"):
        what = COMBOS[task_id] if 0 <= task_id < len(COMBOS) else "?"
        ok = what in m["combos"]
    elif stage == "sweepM":
        ok, what = 0 <= task_id < len(RHOS), f"rho index {task_id}"
    else:                                           # eval
        rows = method_table(m)
        ok = 0 <= task_id < len(rows)
        what = rows[task_id][0] if ok else "?"
    if not ok:
        raise SystemExit(f"[r45_select] {stage} array task {task_id} ({what}) is not in the "
                         f"method-gate selection {m['combos']} -- refusing to run it.")
    print(f"[r45_select] {stage} task {task_id} = {what}: selected")


def cmd_framecheck(m: dict) -> None:
    if not R36_REF.is_dir():
        raise SystemExit(f"[r45_select] R36 reference {R36_REF} missing")
    bad = 0
    for method, root in method_table(m):
        if not root.is_dir():
            print(f"MISSING  {method}: {root}")
            bad += 1
            continue
        n_clips = n_off = 0
        for tdir in sorted(R36_REF.glob("edit[1-6]")):
            for vdir in sorted(p for p in tdir.iterdir() if p.is_dir() and not p.name.endswith("_resize")):
                n_clips += 1
                ref = len(list(vdir.glob("*.png")))
                got = len(list((root / tdir.name / vdir.name).glob("*.png")))
                if got != ref:
                    n_off += 1
                    if n_off <= 3:
                        print(f"  {method} {tdir.name}/{vdir.name}: {got} frames, R36 has {ref}")
        print(f"{'OK      ' if n_off == 0 else 'MISMATCH'} {method}: {n_clips - n_off}/{n_clips} clips match R36")
        bad += n_off > 0
    raise SystemExit(1 if bad else 0)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if "--set" in argv:
        p = argparse.ArgumentParser(description="write evaluation/r45_method.json")
        p.add_argument("--set", action="store_true", required=True)
        p.add_argument("--draft_n", type=int, choices=(5, 15), required=True)
        p.add_argument("--combos", required=True, help="comma-separated, from: " + ", ".join(COMBOS))
        p.add_argument("--allow_mixed", action="store_true")
        cmd_set(p.parse_args(argv))
        return
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for s in ("stage1", "stage2", "stage3", "sweepM"):
        sub.add_parser(s)
    e = sub.add_parser("eval")
    e.add_argument("--table", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("task_id", type=int)
    c.add_argument("stage", choices=STAGES)
    sub.add_parser("framecheck")
    a = p.parse_args(argv)
    m = load()
    if a.cmd == "stage1":
        print(array_list(selected_runs(m)))
    elif a.cmd in ("stage2", "stage3"):
        print(array_list(COMBOS.index(x) for x in m["combos"]))
    elif a.cmd == "sweepM":
        print(m["matched_n"])
    elif a.cmd == "eval":
        rows = method_table(m)
        if a.table:
            for i, (method, root) in enumerate(rows):
                print(f"{i}\t{method}\t{root}")
        else:
            print(f"0-{len(rows) - 1}")
    elif a.cmd == "check":
        cmd_check(m, a.task_id, a.stage)
    else:
        cmd_framecheck(m)


if __name__ == "__main__":
    main()
