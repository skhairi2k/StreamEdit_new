"""R49: the renders to score, by group (shared by r49_eval_all.sh and r49_score.py).

--mode subset (default): the 18 clips of r49_cases.json
  main: se_max (StreamEdit, r49_se_arms/r49_max_subset/step14), wan_max_s9, wan_max_s5 (r49_wan_arms)
  base: the 9 Wan-Edit global baselines (native_s0.95 .. native_s0.4, matched_s0.95 .. matched_s0.85) and native_s0.978, all in
        r48_wan_arms/{config}_subset (17 clips from R48 + the new clip rendered by r49_newclip_baselines.sh), plus the reference arm
        (R48's dino_SD): se_r48_dino_SD (r48_se_arms/r48_dino_SD_subset/step14) and wan_r48_dino_SD (r48_wan_arms/adapt_dino_SD_subset)
  m59 : the 59-active-step matched Wan-Edit curve, matched59_s0.95 / s0.92 / s0.85 (r49_wan_arms)
  wan_pair: wan_max_s5 (r49_wan_arms/max_s5_subset) and the official Wan-Edit baseline wan_native_s0.92 (r48_wan_arms/native_s0.92_subset):
        the two renders of R50's Jean Zay reproduction (jz/r50_repro_chain.sh); same unit names and metric stems as the lab's main / base
--mode full: the 419 pairs, groups split by host (each host runs on its own platform: StreamEdit lab L40S, Wan-Edit Jean Zay H100)
  se  : se_max (r49_se_arms/r49_max_full/step14)
  wan : wan_max_s5 (r49_wan_arms/max_s5_full)
  base: the native sweep, N = 50 (r48_wan_arms/{config}_full): the 6 native_s0.95 .. native_s0.4 (native_s0.92 = the official Wan-Edit
        baseline) + native_s0.978 last (index 6, rendered by r48_wan_render.sh explore_full); matched-55 dropped (it serves max_s9 only),
        no R48 reference arm
  m59 : matched59_s0.95 / s0.92 / s0.85 / s0.978 (r49_wan_arms/matched59_s{sg}_full; 0.978 last, like the render index)
Output (TSV, one per line):  base <TAB> dir <TAB> host <TAB> config
  base = "{host}_{config with '.' -> 'p'}" (+ "_full" in full mode); metric stems are r49_{base} / r49_fiveacc_{base}
usage: python evaluation/r49_units.py --group main [--mode subset|full] [--count | --index N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r48_wan_sched as SCH  # noqa: E402

D = Path.home() / "Data/dataggen/outputs/five_bench"
R48_WAN = D / "r48_wan_arms"
GROUPS = {
    "main": [("se", "max", D / "r49_se_arms" / "r49_max_subset" / "step14"),
             ("wan", "max_s9", D / "r49_wan_arms" / "max_s9_subset"),
             ("wan", "max_s5", D / "r49_wan_arms" / "max_s5_subset")],
    "base": [("wan", cfg, R48_WAN / f"{cfg}_subset") for cfg in list(SCH.configs()) + ["native_s0.978"]] +
            [("se", "r48_dino_SD", D / "r48_se_arms" / "r48_dino_SD_subset" / "step14"),
             ("wan", "r48_dino_SD", R48_WAN / "adapt_dino_SD_subset")],
    "m59": [("wan", f"matched59_s{sg}", D / "r49_wan_arms" / f"matched59_s{sg}_subset") for sg in ("0.95", "0.92", "0.85")],
    "wan_pair": [("wan", "max_s5", D / "r49_wan_arms" / "max_s5_subset"),
                 ("wan", "native_s0.92", R48_WAN / "native_s0.92_subset")],
}
GROUPS_FULL = {
    "se": [("se", "max", D / "r49_se_arms" / "r49_max_full" / "step14")],
    "wan": [("wan", "max_s5", D / "r49_wan_arms" / "max_s5_full")],
    "base": [("wan", cfg, R48_WAN / f"{cfg}_full") for cfg in list(SCH.configs())[:6] + ["native_s0.978"]],
    "m59": [("wan", f"matched59_s{sg}", D / "r49_wan_arms" / f"matched59_s{sg}_full") for sg in ("0.95", "0.92", "0.85", "0.978")],
}
MODES = {"subset": GROUPS, "full": GROUPS_FULL}


def units(group: str, mode: str = "subset") -> List[Tuple[str, Path, str, str]]:
    suffix = "_full" if mode == "full" else ""
    return [(f"{host}_{cfg.replace('.', 'p')}{suffix}", d, host, cfg) for host, cfg, d in MODES[mode][group]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, help="subset: main | base | m59 | wan_pair; full: se | wan | base | m59")
    ap.add_argument("--mode", choices=sorted(MODES), default="subset")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--index", type=int, default=None)
    a = ap.parse_args()
    if a.group not in MODES[a.mode]:
        ap.error(f"--group {a.group!r} is not a {a.mode} group (choose from {', '.join(sorted(MODES[a.mode]))})")
    u = units(a.group, a.mode)
    if a.count:
        print(len(u))
        return 0
    if a.index is not None:
        if not 0 <= a.index < len(u):
            return 1
        u = [u[a.index]]
    for base, d, host, cfg in u:
        print(f"{base}\t{d}\t{host}\t{cfg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
