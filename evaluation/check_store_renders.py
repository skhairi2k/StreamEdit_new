"""Check that every eval setting of R58-R66 has a complete archive copy on /store, clip by clip.

Run ON THE LAB FRONTAL (gpu-gw): /store is mounted there only.

    ~/anaconda3/envs/five-bench/bin/python -I evaluation/check_store_renders.py            # every batch file r58..r66
    ~/anaconda3/envs/five-bench/bin/python -I evaluation/check_store_renders.py r60 r62    # some tasks only

The settings are read from evaluation/eval_suite/batches/<task>.txt (`NAME ROOT`, ROOT relative to the /scratch root
base), the same files the evaluation uses. For each setting and each of its clips (edit{T}/{video}), the PNG count of the
/store copy is compared with the PNG count of the /scratch copy (the one the evaluation reads, checked 2026-11-04:
415 clips, 0 links). Where the /store copy lives:

  r58_wan_correct/*, r62_wan_correct/*, r66_contextflow/*   /store/.../jz_five_bench/<ROOT> (Jean Zay renders)
  r60_correct/<config>      no merged tree on /store: per clip, the source given by evaluation/csv/r60_merge_renders.csv
                            (new -> five_bench/r60_rho_sweep/r60_<config>_novp, or r60_se_arms/r60_max_novp_correct/step14
                            for max_novp; reused -> five_bench/r44_rho_sweep/r44_<config>_novp)
  r62_wan_correct/nosupp_s5 if jz_five_bench/r62_wan_correct is not there yet (R62 store-copy pending): per clip from
                            evaluation/csv/r62_wan_assembly.csv (new -> jz_five_bench/r62_wan_arms/nosupp_s5_full,
                            reused -> jz_five_bench/r51_wan_arms/nosupp_s5_full)
  everything else           /store/.../five_bench/<ROOT>

A clip fails if its /store directory is missing, holds a broken link, or has a PNG count different from /scratch (and
from the CSV's n_png where there is one). Read-only. Prints one line per setting and ends with '[check_store] failures: N'.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
BATCHES = REPO / "evaluation/eval_suite/batches"
SCRATCH = Path("/scratch/PERSO/skhairi/five_bench")
STORE = Path("/store/PERSO/skhairi/Outputs/five_bench")
STORE_JZ = Path("/store/PERSO/skhairi/Outputs/jz_five_bench")
JZ_ROOTS = ("r58_wan_correct/", "r62_wan_correct/", "r66_contextflow/")
TASKS = ("r58", "r59", "r60", "r61", "r62", "r63", "r64", "r65", "r66")


def read_batch(task: str) -> List[Tuple[str, str]]:
    out = []
    for line in (BATCHES / f"{task}.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, root = line.split()
            out.append((name, root))
    return out


def png_count(d: Path) -> Tuple[Optional[int], int]:
    """(number of PNG files, number of broken entries); (None, 0) if the directory is missing."""
    try:
        names = os.listdir(d)
    except (FileNotFoundError, NotADirectoryError):
        return None, 0
    n = broken = 0
    for x in names:
        if x.endswith(".png"):
            if os.path.isfile(d / x):   # follows links; a dangling link is not a file
                n += 1
            else:
                broken += 1
    return n, broken


def clips(root: Path) -> List[str]:
    return sorted(f"{e.name}/{v.name}" for e in os.scandir(root) if e.is_dir() and e.name.startswith("edit")
                  for v in os.scandir(e.path) if v.is_dir())


def csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def store_map(root: str) -> Tuple[Dict[str, Path], Dict[str, int], str]:
    """case_id -> /store clip dir, case_id -> expected n_png from a CSV (may be empty), and a label of the source."""
    if root.startswith("r60_correct/"):
        cfg = root.split("/", 1)[1]
        rows = [r for r in csv_rows(REPO / "evaluation/csv/r60_merge_renders.csv") if r["config"] == cfg]
        m, n = {}, {}
        for r in rows:
            if cfg == "max_novp":
                m[r["case_id"]] = STORE / "r60_se_arms/r60_max_novp_correct/step14" / r["case_id"]
            elif r["source"] == "new":
                m[r["case_id"]] = STORE / f"r60_rho_sweep/r60_{cfg}_novp" / r["case_id"]
            else:
                m[r["case_id"]] = STORE / f"r44_rho_sweep/r44_{cfg}_novp" / r["case_id"]
            n[r["case_id"]] = int(r["n_png"])
        return m, n, "r60_merge_renders.csv (new + reused)"
    if root.startswith("r62_wan_correct/") and not (STORE_JZ / root).is_dir():
        rows = csv_rows(REPO / "evaluation/csv/r62_wan_assembly.csv")
        m = {r["case_id"]: STORE_JZ / ("r62_wan_arms" if r["source"] == "new" else "r51_wan_arms") / "nosupp_s5_full" / r["case_id"]
             for r in rows}
        return m, {r["case_id"]: int(r["n_png"]) for r in rows}, "r62_wan_assembly.csv (r62_wan_arms + r51_wan_arms; store-copy pending)"
    base = (STORE_JZ if root.startswith(JZ_ROOTS) else STORE) / root
    return {}, {}, str(base)


def check(task: str, name: str, root: str) -> Tuple[bool, str]:
    sroot = SCRATCH / root
    if not sroot.is_dir():
        return True, f"{task} {name:<30} SKIP  not staged on /scratch ({sroot})"
    cases = clips(sroot)
    mapping, expect, label = store_map(root)
    if not mapping:
        mapping = {c: Path(label) / c for c in cases}
    bad: List[str] = []
    n_store = 0
    for c in cases:
        n_s, _ = png_count(sroot / c)
        n_t, broken = png_count(mapping[c]) if c in mapping else (None, 0)
        if n_t is None:
            bad.append(f"{c}: missing on /store")
            continue
        n_store += 1
        if broken or n_t != n_s or (c in expect and expect[c] != n_t):
            bad.append(f"{c}: /store {n_t} PNG ({broken} broken) vs /scratch {n_s}" + (f" vs CSV {expect[c]}" if c in expect else ""))
    ok = not bad and len(cases) == 415
    line = f"{task} {name:<30} {'OK  ' if ok else 'FAIL'}  clips scratch {len(cases)} store {n_store}  bad {len(bad)}  [{label}]"
    if bad:
        line += "".join(f"\n      {b}" for b in bad[:5]) + (f"\n      ... {len(bad) - 5} more" if len(bad) > 5 else "")
    return ok, line


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="*", default=list(TASKS), help="tasks to check (default: r58..r66)")
    ap.add_argument("--workers", type=int, default=8, help="settings checked in parallel (NFS listing is the cost)")
    a = ap.parse_args()
    if not STORE.is_dir():
        print(f"[check_store] FATAL: {STORE} not visible on {os.uname().nodename} -- run this on the lab frontal (gpu-gw)")
        return 1
    jobs = [(t, n, r) for t in a.tasks for n, r in read_batch(t)]
    print(f"[check_store] {len(jobs)} settings from {', '.join(a.tasks)}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(lambda j: check(*j), jobs))
    for _, line in results:
        print(line)
    failures = sum(not ok for ok, _ in results)
    print(f"[check_store] failures: {failures}")
    return int(failures > 0)


if __name__ == "__main__":
    sys.exit(main())
