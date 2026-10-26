"""R51: the renders to score, by group (shared by r51_eval.sh, r51_motion.sh and r51_score.py). Full bench (419 pairs) only.

  se    : the 4 StreamEdit ablation arms (r51_se_arms/r51_{arm}_full/step14): nodiv, dino, lpips, nosupp      (lab L40S)
  wan   : the Wan-Edit no-support arm nosupp_s5 (r51_wan_arms/nosupp_s5_full)                                (Jean Zay H100)
  motion: motion fidelity units, StreamEdit only -- the 4 arms (index 0-3), A0 = R49 se_max full (index 4) and the official
          rho = 2, N = 15 render r36_rho2_vp (index 5, last: it lives on /store, so r51_se_chain.sh pins that task to node39)
  r52_se / r52_motion: R52's B1 arm (se_max_linrho_full, r52_se_arms/r52_max_linrho_full/step14), stem prefix r52
  r53_se   : R53's text-only se_max (se_max_novp_full, r53_se_arms/r53_max_novp_full/step14), stem prefix r53
  r53_sweep: the 4 extra text-only N=20 uniform-rho renders rho 3, 6, 8, 20 (r53_rho_sweep/r53_n20_rho{R}_novp), prefix r53
  r56_se / r56_motion: R56's se_max with a 15-step draft (se_max_n15_full, r56_se_arms/r56_max_n15_full/step14), stem prefix r56
  r55_se / r55_motion: R55's se_max on LongLive (se_max_ll_full, r55_ll_arms/r55_max_ll_full/step14), stem prefix r55; r55_motion index 1 (last) is the
                       official LongLive rho=2, N=15 render r45_sweep/n15/ll_rho2_n15_vp (on /store, node39)
  r54_se / r54_motion: R54's draft-free arm (se_warp_max_full, r54_se_arms/r54_warp_max_full/step14), stem prefix r54
  r53_motion: se_max_novp (index 0) and the text-only official rho=2, N=15 render r44_n15_rho2_novp (index 1, last: on /store, node39), prefix r53
Output (TSV, one per line):  base <TAB> dir <TAB> host <TAB> config [<TAB> stem_prefix]
  base = "{host}_{config with '.' -> 'p'}_full"; metric stems are {prefix}_{base} / {prefix}_fiveacc_{base} / {prefix}_motion_{base}
  The 5th column (stem_prefix) is printed ONLY for the groups listed in STEM_PREFIX (R52 onward). The R51 groups keep their original four columns
  byte for byte, and r51_eval.sh / r51_motion.sh read a missing prefix as `r51` -- so jobs queued before R52 existed see no change.
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
    "r52_se": [("se", "max_linrho", D / "r52_se_arms" / "r52_max_linrho_full" / "step14")],
    "r52_motion": [("se", "max_linrho", D / "r52_se_arms" / "r52_max_linrho_full" / "step14")],
    "r53_se": [("se", "max_novp", D / "r53_se_arms" / "r53_max_novp_full" / "step14")],
    "r53_sweep": [("se", f"n20_rho{rho}_novp", D / "r53_rho_sweep" / f"r53_n20_rho{rho}_novp") for rho in (3, 6, 8, 20)],
    "r53_motion": [("se", "max_novp", D / "r53_se_arms" / "r53_max_novp_full" / "step14"),
                   ("se", "n15_rho2_novp", D / "r44_rho_sweep" / "r44_n15_rho2_novp")],
    "r54_se": [("se", "warp_max", D / "r54_se_arms" / "r54_warp_max_full" / "step14")],
    "r54_motion": [("se", "warp_max", D / "r54_se_arms" / "r54_warp_max_full" / "step14")],
    "r55_se": [("se", "max_ll", D / "r55_ll_arms" / "r55_max_ll_full" / "step14")],
    "r55_motion": [("se", "max_ll", D / "r55_ll_arms" / "r55_max_ll_full" / "step14"),
                   ("se", "ll_rho2_n15_vp", D / "r45_sweep" / "n15" / "ll_rho2_n15_vp")],
    "r56_se": [("se", "max_n15", D / "r56_se_arms" / "r56_max_n15_full" / "step14")],
    "r56_motion": [("se", "max_n15", D / "r56_se_arms" / "r56_max_n15_full" / "step14")],
}
STEM_PREFIX = {"r52_se": "r52", "r52_motion": "r52", "r53_se": "r53", "r53_sweep": "r53", "r53_motion": "r53",
               "r54_se": "r54", "r54_motion": "r54", "r55_se": "r55", "r55_motion": "r55",
               "r56_se": "r56", "r56_motion": "r56"}   # groups that print the 5th column


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
        print(f"{base}\t{d}\t{host}\t{cfg}" + (f"\t{STEM_PREFIX[a.group]}" if a.group in STEM_PREFIX else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
