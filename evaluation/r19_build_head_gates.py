"""Build R10's per-head VP-injection gates from the R19 labels.

Replaces ``r10_build_head_gates.py``, whose gates came from the superseded
margin table (spatial and temporal key sets of 1560 vs 18 keys, where a head
with no specialisation at all scores margin -0.97). Those gates must not be
used; see the supersede note on the R9 plan.

Emits the same key names ``r10_vp_arms.py`` already consumes -- ``all``,
``spatial``, ``temporal`` -- so the driver needs no change, plus stable-core
variants and full provenance.

WHICH HEADS SHOULD RECEIVE THE VISUAL PROMPT
--------------------------------------------
The editing thesis is that spatial heads anchor appearance and temporal heads
carry motion, so the edited first frame belongs in the SPATIAL heads: change
what things look like without freezing how they move. MIXED and DENSE heads are
deliberately left at baseline -- DENSE means neither key set reconstructed the
head, so there is nothing to route on, and forcing it to a side is exactly the
error the four-way scheme exists to avoid.

THE ARMS ARE NOT SIZE-MATCHED, AND THAT MATTERS
-----------------------------------------------
``spatial`` and ``temporal`` differ in count (117 vs 49 at the default
thresholds), so a difference between those arms confounds head TYPE with head
COUNT. ``rand_spatial``/``rand_temporal`` are count-matched random draws at a
fixed seed: if a random set of the same size reproduces the effect, the labels
are not doing the work. Run them -- the control is what makes the result mean
anything.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

LABELS = ("SPATIAL", "TEMPORAL", "MIXED", "DENSE")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tau", type=Path, default=Path("evaluation/r19_tau_flat.pt"),
                   help="written by r19_analyze.py with --tau_route/--tau_dense set")
    p.add_argument("--out", type=Path, default=Path("evaluation/r19_head_gates.pt"))
    p.add_argument("--max_wrong_arm", type=float, default=0.0,
                   help="core variants keep only heads whose WRONG-ARM rate is at "
                        "or below this on all three axes (video, step, block). "
                        "Default 0.0 = never observed in the opposite arm.")
    p.add_argument("--labels_csv", type=Path, default=None,
                   help="default: evaluation/csv/r19_head_labels_{shape}.csv")
    p.add_argument("--seed", type=int, default=0, help="for the count-matched random arms")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.tau.exists():
        raise SystemExit(
            f"[gates] {args.tau} not found. Run:\n"
            "  python evaluation/r19_analyze.py --tau_route <x> --tau_dense <y>")
    tau = torch.load(args.tau, weights_only=False)
    shape = tau["thresholds"]["shape"]
    labels = tau["labels"]                                   # nested list [L][H]
    lab = [[str(x) for x in row] for row in labels]
    n_layers, n_heads = len(lab), len(lab[0])

    def mask_of(name):
        return torch.tensor([[lab[i][j] == name for j in range(n_heads)]
                             for i in range(n_layers)], dtype=torch.bool)

    gates = {
        "all": torch.ones((n_layers, n_heads), dtype=torch.bool),
        "spatial": mask_of("SPATIAL"),
        "temporal": mask_of("TEMPORAL"),
        "mixed": mask_of("MIXED"),
        "dense": mask_of("DENSE"),
    }

    # ---- stability filter -> *_core variants -------------------------------
    # Filter on the WRONG-ARM rate, not the any-label flip rate. A head moving to
    # DENSE is an abstain, not a contradiction -- it never lands in the arm meant
    # for its opposite, which is the only way gating can misfire. Filtering on
    # any-label flips instead conflates the two and, because TEMPORAL heads
    # abstain far more often (29.3% of videos vs 8.2%), it discarded 41 of the 49
    # temporal heads for a failure mode that cannot occur.
    csv_path = args.labels_csv or Path(f"evaluation/csv/r19_head_labels_{shape}.csv")
    stable = torch.ones((n_layers, n_heads), dtype=torch.bool)
    if csv_path.exists():
        import csv as _csv
        cols = ("wrong_arm_video", "wrong_arm_step", "wrong_arm_block")
        with csv_path.open() as fh:
            rows = list(_csv.DictReader(fh))
        if not all(c in rows[0] for c in cols):
            raise SystemExit(
                f"[gates] {csv_path} predates the wrong-arm columns. Re-run:\n"
                "  python evaluation/r19_analyze.py --tau_route 0.15 --tau_dense 0.28")
        for r in rows:
            vals = [float(r[c]) for c in cols]
            # NaN = the head has no opposite arm (MIXED/DENSE); it is excluded by
            # the label mask anyway, so it must not be treated as a failure here.
            if any(v == v and v > args.max_wrong_arm for v in vals):
                stable[int(r["layer"]), int(r["head"])] = False
        for nm in ("spatial", "temporal"):
            gates[f"{nm}_core"] = gates[nm] & stable
    else:
        print(f"[gates] WARNING: {csv_path} missing -- no *_core variants emitted")

    # ---- count-matched random controls ------------------------------------
    # The arms differ in size, so a spatial-vs-temporal gap confounds type with
    # count. These draw the SAME number of heads at random.
    g = torch.Generator().manual_seed(args.seed)
    for nm in ("spatial", "temporal"):
        k = int(gates[nm].sum())
        pick = torch.randperm(n_layers * n_heads, generator=g)[:k]
        m = torch.zeros(n_layers * n_heads, dtype=torch.bool)
        m[pick] = True
        gates[f"rand_{nm}"] = m.view(n_layers, n_heads)

    payload = {
        **gates,
        "margin_flat": tau["margin_flat"],
        "margin_disk": tau["margin_disk"],
        "best_rel": tau["best_rel"],
        "labels": labels,
        "thresholds": tau["thresholds"],
        "max_wrong_arm": args.max_wrong_arm,
        "seed": args.seed,
        "source_tau": str(args.tau),
        "cases": tau["cases"],
        "provenance": ("R19 equal-budget disjoint key sets; supersedes "
                       "r10_head_gates.pt, which came from the 1560-vs-18 margins"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)

    total = n_layers * n_heads
    print(f"[gates] from {args.tau}  (shape={shape}, "
          f"tau_route={tau['thresholds']['tau_route']}, "
          f"tau_dense={tau['thresholds']['tau_dense']}, {len(tau['cases'])} videos)")
    for k in ("all", "spatial", "temporal", "mixed", "dense",
              "spatial_core", "temporal_core", "rand_spatial", "rand_temporal"):
        if k in gates:
            n = int(gates[k].sum())
            print(f"[gates]   {k:<14} {n:4d}/{total}  ({n / total:5.1%})")
    print(f"[gates] wrote {args.out}")


if __name__ == "__main__":
    main()
