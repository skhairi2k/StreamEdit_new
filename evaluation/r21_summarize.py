"""Join R21's per-arm FiVE CSVs into one comparison table.

Each arm is read against the reference matching ITS VP mode -- R7 (`r21_ref_vp`)
for `vp`, the full-bench `paper_pvp` render for `pvp` -- because a schedule's
effect is only meaningful against the Eq.4 baseline under the same anchoring
regime. `ref_novp` (R1) is carried as the un-anchored floor.

Two reference CONFIGURATIONS are emitted where available (see daily.md 2026-08-20):
  jpg  -- model fed videos/*.mp4, scored against images/*.jpg. This is FiVE-Bench's
          default AND what the published baselines (e.g. Wan-Edit) use, so it is the
          configuration to quote alongside published numbers.
  mp4  -- scored against the same mp4 frames the model was fed. Reproduces StreamGVE
          Table 1 on 10/10 reported metrics, but no published baseline uses it.
Only source-reading metrics differ between the two; target-only metrics are identical.
"""
import argparse, csv, glob, os, re
from collections import OrderedDict

REF_OF = {"vp": "ref_vp", "pvp": "paper_pvp", "novp": "ref_novp"}
REFS = set(REF_OF.values())
PCT_LOWER_IS_BETTER = {"structure_distance", "lpips_unedit_part", "mse_unedit_part",
                       "niqe_target_image"}


def load(path):
    rows = list(csv.reader(open(path)))
    return OrderedDict(zip([c.split("|")[-1] for c in rows[0]], rows[1]))


def vp_mode(arm):
    if arm.endswith("_pvp") or arm == "paper_pvp":
        return "pvp"
    if arm.endswith("_novp") or arm == "ref_novp":
        return "novp"
    return "vp"


def collect(pattern, strip):
    """{arm: {metric: value}} from CSVs matching `pattern`."""
    out = {}
    for f in sorted(glob.glob(pattern)):
        arm = re.sub(strip, "", os.path.basename(f))
        out.setdefault(arm, {}).update(load(f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default="evaluation/csv")
    ap.add_argument("-o", "--out", default="evaluation/csv/r21_blend_full.csv")
    a = ap.parse_args()
    d = a.csv_dir

    configs = {
        # jpg: the 16-metric run (job 937950) is already the official configuration.
        "jpg": collect(f"{d}/r21_*_avg.csv", r"^r21_|_avg\.csv$"),
        # mp4: preservation (951202 t0) + the 3 source-dependent metrics (951499),
        # merged per arm; target-only metrics are filled from the jpg run below.
        "mp4": collect(f"{d}/r21bg_mp4_*_avg.csv", r"^r21bg_mp4_|_avg\.csv$"),
    }
    for arm, vals in collect(f"{d}/r21mf_mp4_*_avg.csv", r"^r21mf_mp4_|_avg\.csv$").items():
        configs["mp4"].setdefault(arm, {}).update(vals)
    # target-only metrics are reference-independent -> carry them across
    for arm, vals in configs["mp4"].items():
        for m, v in configs["jpg"].get(arm, {}).items():
            vals.setdefault(m, v)

    rows = []
    for cfg, arms in configs.items():
        for arm in sorted(arms):
            if arm == "file_id":
                continue
            mode = vp_mode(arm)
            ref = arms.get(REF_OF[mode], {})
            for metric, val in arms[arm].items():
                if metric == "file_id":
                    continue
                try:
                    v = float(val)
                except ValueError:
                    continue
                rv = ref.get(metric)
                try:
                    rv = float(rv)
                except (TypeError, ValueError):
                    rv = None
                delta = None if rv is None else v - rv
                pct = None if not rv else delta / abs(rv) * 100.0
                better = ""
                if delta is not None and arm not in REFS:
                    lower = metric in PCT_LOWER_IS_BETTER
                    better = "better" if ((delta < 0) == lower and delta != 0) else \
                             ("worse" if delta != 0 else "same")
                rows.append([cfg, arm, mode, "ref" if arm in REFS else "arm", metric,
                             f"{v:.4f}",
                             "" if rv is None else f"{rv:.4f}",
                             "" if delta is None else f"{delta:+.4f}",
                             "" if pct is None else f"{pct:+.2f}", better])

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reference_config", "arm", "vp_mode", "kind", "metric",
                    "value", "ref_value", "delta", "delta_pct", "direction"])
        w.writerows(rows)
    n_arms = len({r[1] for r in rows})
    print(f"[r21_summarize] {len(rows)} rows, {n_arms} arms, "
          f"configs={sorted(configs)} -> {a.out}")


if __name__ == "__main__":
    main()
