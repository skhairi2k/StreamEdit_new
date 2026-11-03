"""R60 -- merge the new and the reused renders into one 415-pair tree per config, FiVE-Bench-Correct.

Builds ``--out_root/{config}/edit{T}/{video}`` for the 17 configs over the 415 pairs of ``--manifest``:

  new     $D/r60_rho_sweep/r60_{config}_novp/edit{T}/{video}      pairs in ``--regen`` (190), and ALL 415 pairs of
                                                                  n20_rho{3,6,8,20} (R44 never rendered them)
  max     $D/r60_se_arms/r60_max_novp_correct/step14/edit{T}/{video}   all 415 pairs of max_novp
  reused  $D/r44_rho_sweep/r44_{config}_novp/edit{T}/{video}      the 225 pairs whose prompt and trigger words are unchanged
                                                                  (n15 all rho, n20 rho 2/4/10/50; gate: r60_repro_check.py)

A clip that resolves under /home is symlinked; one that resolves under /store (not mounted on every node: run on nodecpu12,
node39 or gpu-gw) is copied, through a temporary directory renamed on success. The PNG count of every merged clip must
equal its source's. Idempotent: a clip already merged and identical is left alone, a different one is an error.
Writes ``--csv`` (config, case_id, source new|reused, storage link|copy, n_png). Exit 1 on a missing or incomplete clip,
or a count other than the manifest's per config.
usage: python evaluation/r60_merge_renders.py [--configs n15_rho2 ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
D = Path.home() / "Data/dataggen/outputs/five_bench"
RHOS = (2, 3, 4, 6, 8, 10, 20, 50)
CONFIGS = tuple(f"n{n}_rho{r}" for n in (15, 20) for r in RHOS) + ("max_novp",)
NEW_ON_ALL = {f"n20_rho{r}" for r in (3, 6, 8, 20)} | {"max_novp"}   # no R44 render exists / different arm
STORE_PREFIX = "/store"


def source_dir(cfg: str, new: bool, t: int, vid: str, a: argparse.Namespace) -> Path:
    if cfg == "max_novp":
        return a.arms_root / "r60_max_novp_correct" / "step14" / f"edit{t}" / vid
    if new:
        return a.new_root / f"r60_{cfg}_novp" / f"edit{t}" / vid
    return a.r44_root / f"r44_{cfg}_novp" / f"edit{t}" / vid


def n_png(path: Path) -> int:
    return sum(1 for _ in path.glob("*.png"))


def merge_clip(src: Path, dst: Path) -> Tuple[str, int]:
    """Link or copy ``src`` to ``dst``; returns (storage, n_png). Raises ValueError on a missing / incomplete / different clip."""
    real = src.resolve()
    n_src = n_png(real) if real.is_dir() else 0
    if n_src == 0:
        raise ValueError(f"no PNG in {src} (resolves to {real})")
    storage = "copy" if str(real).startswith(STORE_PREFIX) else "link"
    if dst.is_symlink() or dst.exists():
        if storage == "link":
            if not dst.is_symlink() or dst.resolve() != real:
                raise ValueError(f"{dst} exists and is not a link to {real}")
        elif dst.is_symlink() or n_png(dst) != n_src:
            raise ValueError(f"{dst} exists with {n_png(dst) if not dst.is_symlink() else 'a link and'} != {n_src} PNG of {real}")
        return storage, n_src
    dst.parent.mkdir(parents=True, exist_ok=True)
    if storage == "link":
        dst.symlink_to(real, target_is_directory=True)
    else:
        tmp = dst.with_name(dst.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(real, tmp)
        if n_png(tmp) != n_src:
            raise ValueError(f"copy of {real} has {n_png(tmp)} PNG, source {n_src}")
        tmp.rename(dst)
    return storage, n_src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=REPO / "evaluation/r43_anchor_manifest_correct.json")
    ap.add_argument("--regen", type=Path, default=REPO / "evaluation/r60_cases_regen.json")
    ap.add_argument("--new_root", type=Path, default=D / "r60_rho_sweep")
    ap.add_argument("--arms_root", type=Path, default=D / "r60_se_arms")
    ap.add_argument("--r44_root", type=Path, default=D / "r44_rho_sweep")
    ap.add_argument("--out_root", type=Path, default=D / "r60_correct")
    ap.add_argument("--csv", type=Path, default=REPO / "evaluation/csv/r60_merge_renders.csv")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=CONFIGS)
    a = ap.parse_args()

    manifest = json.loads(a.manifest.read_text())
    regen = {c["case_id"] for c in json.loads(a.regen.read_text())}
    if not regen <= {m["case_id"] for m in manifest}:
        print("[r60_merge_renders] ERROR: --regen holds a pair that is not in --manifest")
        return 1
    print(f"[r60_merge_renders] {len(manifest)} pairs ({len(regen)} regenerated), configs: {len(a.configs)}, out {a.out_root}")

    rows: List[tuple] = []
    errors: List[str] = []
    counts: Dict[str, Dict[str, int]] = {c: defaultdict(int) for c in a.configs}
    frames_by_case: Dict[str, set] = defaultdict(set)
    for cfg in a.configs:
        for m in manifest:
            t, vid, case_id = int(m["edit_type"]), m["video_name"], m["case_id"]
            new = cfg in NEW_ON_ALL or case_id in regen
            try:
                storage, n = merge_clip(source_dir(cfg, new, t, vid, a), a.out_root / cfg / f"edit{t}" / vid)
            except (ValueError, OSError) as e:
                errors.append(f"{cfg} {case_id}: {e}")
                continue
            rows.append((cfg, case_id, "new" if new else "reused", storage, n))
            counts[cfg]["new" if new else "reused"] += 1
            counts[cfg][storage] += 1
            frames_by_case[case_id].add(n)
        n_ok = counts[cfg]["new"] + counts[cfg]["reused"]
        print(f"[r60_merge_renders] {cfg}: {n_ok}/{len(manifest)} clips (new {counts[cfg]['new']}, reused {counts[cfg]['reused']}; "
              f"link {counts[cfg]['link']}, copy {counts[cfg]['copy']})")
        if n_ok != len(manifest):
            errors.append(f"{cfg}: {n_ok} clips merged, expected {len(manifest)}")

    # informational: the frame count is a property of the video, so it should not vary across configs
    odd = sorted(c for c, s in frames_by_case.items() if len(s) > 1)
    if odd:
        print(f"[r60_merge_renders] WARNING: {len(odd)} pairs have a different PNG count across configs, e.g. {odd[:3]}")

    a.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(a.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "case_id", "source", "storage", "n_png"])
        w.writerows(rows)
    print(f"[r60_merge_renders] {a.csv}: {len(rows)} rows")
    for e in errors[:30]:
        print(f"[r60_merge_renders] ERROR: {e}")
    print(f"[r60_merge_renders] failures: {len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
