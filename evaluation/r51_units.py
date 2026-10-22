"""R51: the renders to score, by group (shared by r51_eval.sh, r51_motion.sh and r51_score.py). Full bench (419 pairs) only.

  se    : the 4 StreamEdit ablation arms (r51_se_arms/r51_{arm}_full/step14): nodiv, dino, lpips, nosupp      (lab L40S)
  wan   : the Wan-Edit no-support arm nosupp_s5 (r51_wan_arms/nosupp_s5_full)                                (Jean Zay H100)
  motion: motion fidelity units, StreamEdit only -- the 4 arms (index 0-3), A0 = R49 se_max full (index 4) and the official
          rho = 2, N = 15 render r36_rho2_vp (index 5, last: it lives on /store, so r51_se_chain.sh pins that task to node39)
Output (TSV, one per line):  base <TAB> dir <TAB> host <TAB> config
  base = "{host}_{config with '.' -> 'p'}_full"; metric stems are r51_{base} / r51_fiveacc_{base} / r51_motion_{base}
usage: python evaluation/r51_units.py --group se [--count | --index N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

D = Path.home() / "Data/dataggen/outputs/five_bench"
SE_ARMS = ("nodiv", "dino", "lpips", "nosupp")
GROUPS = {
    "se": [("se", arm, D / "r51_se_arms" / f"r51_{arm}_full" / "step14") for arm in SE_ARMS],
    "wan": [("wan", "nosupp_s5", D / "r51_wan_arms" / "nosupp_s5_full")],
    "motion": [("se", arm, D / "r51_se_arms" / f"r51_{arm}_full" / "step14") for arm in SE_ARMS] +
              [("se", "max", D / "r49_se_arms" / "r49_max_full" / "step14"),
               ("se", "r36_rho2_vp", D / "r36_rho_sweep" / "r36_rho2_vp")],
}


def units(group: str) -> List[Tuple[str, Path, str, str]]:
    return [(f"{host}_{cfg.replace('.', 'p')}_full", d, host, cfg) for host, cfg, d in GROUPS[group]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=sorted(GROUPS))
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
