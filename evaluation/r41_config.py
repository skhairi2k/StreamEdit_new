"""R41: read ``slurm_scripts/five_bench/r41_config.sh`` from Python.

The bash file is the single source of DRAFT_N / COMBOS / THETAS for every R41 script; this
module only runs it (``--json``) so the Python side never keeps a second copy of those
settings.

``load(draft_n)`` requires ``draft_n`` to equal the configured DRAFT_N, so a Python step can
never read one N's data while the Slurm side was configured for the other.
``load(draft_n, allow_unset=True)`` skips that check -- used ONLY by
``r41_fg_stats.py --precheck``, which runs at gate-draft to compare both N before one is set.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

REPO = Path(__file__).resolve().parent.parent
# R41_CONFIG / R41_TEST_D: synthetic tests ONLY, same meaning as in the bash scripts.
CONFIG_SH = Path(os.environ.get("R41_CONFIG",
                                REPO / "slurm_scripts" / "five_bench" / "r41_config.sh"))
EXPECT: Dict[int, int] = {1: 100, 2: 100, 3: 100, 4: 100, 5: 9, 6: 10}
N_CLIPS: int = sum(EXPECT.values())            # 419


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(CONFIG_SH), *args], capture_output=True, text=True)


def configured_draft_n() -> int:
    """DRAFT_N as set in r41_config.sh; SystemExit (with the config's message) if unset."""
    r = _run("--draft_n")
    if r.returncode != 0:
        raise SystemExit(r.stderr.strip() or "[r41] r41_config.sh --draft_n failed")
    return int(r.stdout.strip())


def load(draft_n: int, allow_unset: bool = False) -> Dict[str, Any]:
    """Resolved R41 config for ``draft_n`` (roots, combos, thetas, arms, upstream names)."""
    if draft_n not in (5, 15):
        raise SystemExit(f"[r41] --draft_n {draft_n}: must be 5 or 15")
    if not allow_unset:
        configured = configured_draft_n()
        if configured != draft_n:
            raise SystemExit(f"[r41] --draft_n {draft_n} but r41_config.sh has "
                             f"DRAFT_N={configured} -- refusing to mix the two N")
    r = _run("--json", str(draft_n))
    if r.returncode != 0:
        raise SystemExit(r.stderr.strip() or "[r41] r41_config.sh --json failed")
    cfg = json.loads(r.stdout)
    for k in ("div_abs_root", "cont_rho_root", "cont_arms_root", "rho_root", "arms_root"):
        cfg[k] = Path(cfg[k])
    cfg["cont_clipd_csv"] = REPO / cfg["cont_clipd_csv"]
    print(f"[r41] DRAFT_N={draft_n} fields={cfg['div_abs_root']} "
          f"continuous={cfg['cont_prefix']}_* combos={' '.join(cfg['combos'])} "
          f"thetas={cfg['thetas']} arms={len(cfg['arms'])}", flush=True)
    return cfg
