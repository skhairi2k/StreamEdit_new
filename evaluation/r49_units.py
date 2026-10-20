"""R49: the renders to score, by group (shared by r49_eval_all.sh and r49_score.py). All on the 18 clips of r49_cases.json.

  main: se_max (StreamEdit, r49_se_arms/r49_max_subset/step14), wan_max_s9, wan_max_s5 (r49_wan_arms)
  base: the 9 Wan-Edit global baselines (native_s0.95 .. native_s0.4, matched_s0.95 .. matched_s0.85) and native_s0.978, all in
        r48_wan_arms/{config}_subset (17 clips from R48 + the new clip rendered by r49_newclip_baselines.sh), plus the reference arm
        (R48's dino_SD): se_r48_dino_SD (r48_se_arms/r48_dino_SD_subset/step14) and wan_r48_dino_SD (r48_wan_arms/adapt_dino_SD_subset)
  m59 : the 59-active-step matched Wan-Edit curve, matched59_s0.95 / s0.92 / s0.85 (r49_wan_arms)
Output (TSV, one per line):  base <TAB> dir <TAB> host <TAB> config
  base = "{host}_{config with '.' -> 'p'}"; metric stems are r49_{base} / r49_fiveacc_{base}
usage: python evaluation/r49_units.py --group main [--count | --index N]
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
}


def units(group: str) -> List[Tuple[str, Path, str, str]]:
    return [(f"{host}_{cfg.replace('.', 'p')}", d, host, cfg) for host, cfg, d in GROUPS[group]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=sorted(GROUPS), required=True)
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--index", type=int, default=None)
    a = ap.parse_args()
    u = units(a.group)
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
