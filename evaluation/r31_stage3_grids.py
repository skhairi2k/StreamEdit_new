"""R31 stage 3: qualitative grids over the SPATIALLY-GATED renders.

Two independent grid types, selected by ``--which``:

``--which tau``  (per clip, per arm: divergence vs. the exponent it routes)
    Two rows per arm -- ``m`` (the normalised [0,1] divergence field) directly above
    ``tau`` (the exponent r31_rho_map.py actually mapped it to) -- over the SAME latent
    tokens, so the reader can see the mapping act rather than take it on faith. ``tau``'s
    colorbar is annotated with ``tau(m=0.5)``, computed FROM THE NPZ'S OWN STORED MAPPING
    METADATA (``mapping``/``thresh``/``tau_min``/``tau_max``/``step``/``flow_shift``), not
    hardcoded -- a hardcoded "m=0.5 -> tau~=4.46" (the budget_linear calibration's own
    number) would silently mislabel any other mapping run through this script, which is
    exactly the bug this script's ``linear_threshold`` support was built to avoid
    repeating (fixed 2026-09-15, see ``_tau_at_m_half``).

``--which arms``  (per clip: source vs. all four arms' rendered output)
    One row per divergence arm (``lpips``, ``dino_patch``, ``normals``, ``latent``) plus a
    ``source`` row, sharing the same K evenly-subsampled pixel frames, read directly from
    stage 3's ``step14`` (fully denoised) render tree. Purely qualitative -- no numbers, just
    "does the routed blend look reasonable" side by side across arms.

Both grid types default to the CALIBRATED (``budget_linear``) run's output paths
(``r31_div``, ``r31_rho``, ``r31_arms``), but every root is a flag specifically so the
EXPLORATORY ``linear_threshold`` run's separate output trees (``r31_rho_lin_t0.4``,
``r31_arms_lin04``) can be pointed at without touching this script or the calibrated
figures -- see ``--rho_root`` / ``--out_root`` / ``--out_dir``.

This script runs no model. Every value and frame already exists on disk, so it is
local/CPU and needs no GPU allocation.

RECREATED 2026-09-15 after this file was found deleted from disk by an external process
(root cause unknown, no destructive command in any tracked shell/terminal history) --
recreated from conversation context. The calibrated run's already-built grids
(``r31_tau_grids/``, ``r31_arm_grids/``) and the exploratory tau grids
(``r31_tau_grids_lin_t0.4/``) were untouched by the deletion and are unaffected by this
recreation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r31_rho_map import (  # noqa: E402
    budget_linear_tau,
    linear_threshold_tau,
)
from r30_b_map import t_next_schedule  # noqa: E402

ARMS: Sequence[str] = ("lpips", "dino_patch", "normals", "latent")


# --------------------------------------------------------------------------- shared ----

def load_cases(cases_path: Path) -> List[dict]:
    return json.loads(cases_path.expanduser().read_text())


def evenly_spaced(n: int, k: int) -> List[int]:
    """K evenly-spaced indices into ``range(n)``, endpoints included, deduplicated."""
    if k <= 0 or k >= n:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))


def _tau_at_m_half(npz: np.lib.npyio.NpzFile) -> float:
    """``tau(m=0.5)`` for THIS clip's rho field, from its own stored mapping metadata.

    Added 2026-09-15 alongside ``--mapping linear_threshold``: this script used to print
    a hardcoded "m=0.5 -> tau~=4.46", which is only true for the calibrated
    ``budget_linear`` run at ``step=15, flow_shift=1.0, tau_min=2, tau_max=50`` -- silently
    wrong for any other mapping or schedule. Recomputes from whatever the npz actually
    recorded, with a fallback for pre-2026-09-15 npz files that predate this metadata
    (recomputed under the OLD hardcoded assumption, budget_linear at the R30-calibrated
    endpoints, since that is the only mapping that ever existed before this date).
    """
    if "tau_at_m_half" in npz.files:
        return float(npz["tau_at_m_half"])

    mapping = str(npz["mapping"]) if "mapping" in npz.files else "budget_linear"
    tau_max = float(npz["tau_max"]) if "tau_max" in npz.files else 50.0
    if mapping == "linear_threshold":
        thresh = float(npz["thresh"]) if "thresh" in npz.files else 0.4
        return float(linear_threshold_tau(np.array([0.5]), thresh, tau_max)[0])

    tau_min = float(npz["tau_min"]) if "tau_min" in npz.files else 2.0
    step = int(npz["step"]) if "step" in npz.files else 15
    flow_shift = float(npz["flow_shift"]) if "flow_shift" in npz.files else 1.0
    t_next = t_next_schedule(step, flow_shift)
    return float(budget_linear_tau(np.array([0.5]), t_next, tau_min, tau_max)[0])


# --------------------------------------------------------------------------- tau grid --

def build_tau_grid(div_root: Path, rho_root: Path, cases: List[dict], out_dir: Path,
                   n_frames: int, cmap_m: str, cmap_tau: str, dpi: int,
                   arms: Sequence[str] = ARMS) -> Tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for c in cases:
        name, T = c["video_name"], int(c["edit_type"])
        try:
            m_fields: Dict[str, np.ndarray] = {}
            tau_fields: Dict[str, np.ndarray] = {}
            spans: Dict[str, Tuple[float, float]] = {}
            # How `m` was produced, per arm. Read from the npz rather than assumed:
            # r31_div writes per-clip min/max, r31_div_abs writes a fixed absolute range.
            # Mislabelling which one a figure shows is the same class of bug as the
            # hardcoded tau(m=0.5) annotation fixed on 2026-09-15.
            norms: Dict[str, Tuple[str, float, float, float, float]] = {}
            tau_min_seen = 0.0
            tau_max_seen = 50.0
            tau_half = None
            missing: List[str] = []
            for arm in arms:
                mf = div_root / arm / f"edit{T}" / f"{name}.npz"
                rf = rho_root / arm / f"edit{T}" / f"{name}.npz"
                if not mf.exists() or not rf.exists():
                    missing.append(arm)
                    continue
                zm = np.load(mf)
                zr = np.load(rf)
                gh, gw = int(zm["grid_h"]), int(zm["grid_w"])
                m_fields[arm] = zm["m"].reshape(-1, gh, gw)
                tau_fields[arm] = zr["rho"].reshape(-1, gh, gw)
                spans[arm] = (float(zm["d_min"]), float(zm["d_max"]))
                norms[arm] = (
                    str(zm["norm_kind"]) if "norm_kind" in zm.files else "per_clip",
                    float(zm["div_min"]) if "div_min" in zm.files else float("nan"),
                    float(zm["div_max"]) if "div_max" in zm.files else float("nan"),
                    float(zm["frac_clipped"]) if "frac_clipped" in zm.files else 0.0,
                    # Per-frame offset removal, if this tree was built with --detrend.
                    float(zm["detrend_p"]) if "detrend_p" in zm.files else float("nan"),
                )
                tau_max_seen = float(zr["tau_max"]) if "tau_max" in zr.files else tau_max_seen
                # tau_min matters: budget_linear's floor is 2 (NOT 0) -- the colorbar
                # and imshow vmin must match, or the whole [2,50] range gets silently
                # compressed into a [0,50] display range and every color reads wrong.
                # linear_threshold forces this to 0 by construction (see r31_rho_map.py).
                if "mapping" in zr.files and str(zr["mapping"]) == "linear_threshold":
                    tau_min_seen = 0.0
                elif "tau_min" in zr.files:
                    tau_min_seen = float(zr["tau_min"])
                if tau_half is None:
                    tau_half = _tau_at_m_half(zr)
            if not m_fields:
                raise FileNotFoundError(f"no rho/m npz for {name} under {div_root}/{rho_root}")
            if tau_half is None:
                tau_half = 0.0

            n_lat = next(iter(m_fields.values())).shape[0]
            lat_idx = evenly_spaced(n_lat, n_frames)
            n_cols = len(lat_idx)

            n_rows = 2 * len(arms)
            fig_h = 0.60 * n_rows
            # Header/footer geometry. The historical 4-arm figure (n_rows == 8) keeps its
            # literal fractions so its PNGs stay byte-identical; a NARROWER arm set keeps
            # the same ABSOLUTE header height (0.48" -- what 0.90 buys at fig_h = 4.8)
            # rather than the same fraction, which at 4 rows leaves only 0.24" and drives
            # the two-line suptitle straight through the f0/f2/... column labels.
            if n_rows == 8:
                top_frac, bottom_frac, title_y = 0.90, 0.02, 0.985
                cax_bot, cax_h = 0.15, 0.65
            else:
                top_frac = 1.0 - 0.48 / fig_h
                bottom_frac = 0.096 / fig_h
                title_y = 1.0 - 0.072 / fig_h
                # Same share of the plotting band the 8-row figure gives its colorbars
                # (0.15..0.80 inside 0.02..0.90), so they stay centred at any arm count.
                band = top_frac - bottom_frac
                cax_bot = bottom_frac + 0.1477 * band
                cax_h = 0.7386 * band
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(max(1.0, 0.75 * n_cols), fig_h),
                                     squeeze=False)
            for ax in axes.ravel():
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)

            def label(r: int, text: str) -> None:
                axes[r][0].set_ylabel(text, fontsize=4.6, rotation=0, ha="right",
                                      va="center", labelpad=32)

            im_m = im_tau = None
            for r, arm in enumerate(arms):
                row_m, row_tau = axes[2 * r], axes[2 * r + 1]
                if arm in missing:
                    for ax in list(row_m) + list(row_tau):
                        ax.set_facecolor("0.92")
                    label(2 * r, f"{arm}\n(missing)")
                    continue
                mf, tf = m_fields[arm], tau_fields[arm]
                lo, hi = spans[arm]
                for j, f in enumerate(lat_idx):
                    k = min(f, mf.shape[0] - 1)
                    im_m = row_m[j].imshow(mf[k], cmap=cmap_m, vmin=0.0, vmax=1.0,
                                           interpolation="nearest", aspect="auto")
                    im_tau = row_tau[j].imshow(tf[k], cmap=cmap_tau, vmin=tau_min_seen,
                                               vmax=tau_max_seen,
                                               interpolation="nearest", aspect="auto")
                    if r == 0:
                        row_m[j].set_title(f"f{f}", fontsize=4.6, pad=1.5)
                kind, dlo, dhi, clipped, dt_p = norms[arm]
                dt_tag = "" if dt_p != dt_p else f", \u2212p{dt_p:g}/frame"
                if kind == "absolute":
                    # The clipped fraction belongs ON the figure: it is how much of this
                    # arm's field was saturated to m=1 (i.e. to tau_max) by the cap, and
                    # therefore how much within-region structure the cap discarded here.
                    scale_note = (f"m = d/{dhi:.3g} (ABS{dt_tag})\n"
                                  f"d=[{lo:.3g},{hi:.3g}] clip {clipped:.1%}")
                else:
                    # Historical wording, kept VERBATIM so the calibrated and
                    # linear_threshold figures still reproduce byte-for-byte.
                    scale_note = f"m [0,1]\nd=[{lo:.3g},{hi:.3g}]"
                label(2 * r, f"{arm}\n{scale_note}")
                label(2 * r + 1, f"{arm}\ntau [{tau_min_seen:g},{tau_max_seen:g}]")

            fig.subplots_adjust(left=0.13, right=0.90, top=top_frac, bottom=bottom_frac,
                                wspace=0.03, hspace=0.10)
            # "near the floor" / "near the ceiling" framing makes the nonlinearity of
            # budget_linear legible at a glance -- tau(0.5) sitting at just 5% of the way
            # from tau_min to tau_max (see r31_rho_map.py's self-test) is easy to miss
            # if the annotation only states the absolute number.
            tau_range = max(tau_max_seen - tau_min_seen, 1e-9)
            half_frac = (tau_half - tau_min_seen) / tau_range
            position_note = (", i.e. near the floor" if half_frac < 0.15 else
                             ", i.e. near the ceiling" if half_frac > 0.85 else "")
            linearity_note = "monotone in m" if abs(half_frac - 0.5) < 0.05 else \
                "monotone in m but strongly NONLINEAR"
            kinds = {v[0] for v in norms.values()}
            dts = {v[4] for v in norms.values()}
            dt_suffix = ("" if len(dts) != 1 or (lambda x: x != x)(next(iter(dts)))
                         else f", per-frame p{next(iter(dts)):g} offset removed")
            regime = ("ABSOLUTE-normalised m (fixed div range, shared across clips"
                      + dt_suffix + ")"
                      if kinds == {"absolute"} else
                      # Verbatim historical phrasing for the per-clip trees.
                      "divergence m (fixed [0,1])" if kinds == {"per_clip"} else
                      "MIXED normalisation across arms — read each row's label")
            fig.suptitle(
                f"R31 {name} (edit{T}) — {regime} paired with the tau "
                f"it routes (absolute [{tau_min_seen:g},{tau_max_seen:g}])\n"
                f"{n_lat} latent frames, {n_cols} shown — tau is {linearity_note}: "
                f"m=0.5 → tau≈{tau_half:.2g}{position_note}", fontsize=7, y=title_y)

            if im_m is not None:
                cax_m = fig.add_axes([0.905, cax_bot, 0.010, cax_h])
                cb_m = fig.colorbar(im_m, cax=cax_m)
                cb_m.set_label("m (fixed [0,1])", fontsize=4.6)
                cb_m.ax.tick_params(labelsize=4)
            if im_tau is not None:
                cax_t = fig.add_axes([0.945, cax_bot, 0.010, cax_h])
                cb_t = fig.colorbar(im_tau, cax=cax_t)
                cb_t.set_label(f"tau (absolute [{tau_min_seen:g},{tau_max_seen:g}])",
                               fontsize=4.6)
                cb_t.ax.tick_params(labelsize=4)
                # halfline position is FRACTIONAL along [tau_min, tau_max], the axis the
                # colorbar itself is drawn over -- not a fraction of tau_max alone, which
                # would be wrong whenever tau_min != 0 (budget_linear's floor is 2).
                cb_t.ax.axhline(half_frac, color="white", ls="--", lw=0.7)
                cb_t.ax.text(1.6, half_frac, "m=0.5", fontsize=3.6, color="0.2",
                            va="center", ha="left", transform=cb_t.ax.transAxes)

            out = out_dir / f"edit{T}_{name}_tau.png"
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            note = f" (missing: {','.join(missing)})" if missing else ""
            print(f"[ok] {out.name}: {n_rows}x{n_cols}{note}", flush=True)
            n_ok += 1
        except Exception as ex:
            print(f"[ERROR] {name} edit{T}: {type(ex).__name__}: {ex}", flush=True)

    return n_ok, len(cases)


# ------------------------------------------------------------------------- arms grid ---

def load_source_frames(data_root: Path, name: str) -> List[Image.Image]:
    from diffusers.utils import load_video
    return [f.convert("RGB") for f in load_video(str(data_root / "videos" / f"{name}.mp4"))]


def load_arm_frames(out_root: Path, method: str, T: int, name: str) -> List[Image.Image]:
    d = out_root / method / "step14" / f"edit{T}" / name
    files = sorted(d.glob("*.png"), key=lambda p: int(p.stem))
    if not files:
        raise FileNotFoundError(f"no rendered frames under {d}")
    return [Image.open(f).convert("RGB") for f in files]


def build_arm_grid(out_root: Path, data_root: Path, cases: List[dict], out_dir: Path,
                   method_suffix: str, n_frames: int, dpi: int) -> Tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for c in cases:
        name, T = c["video_name"], int(c["edit_type"])
        try:
            src = load_source_frames(data_root, name)
            arm_frames: Dict[str, List[Image.Image]] = {}
            missing: List[str] = []
            for arm in ARMS:
                method = f"r31_{arm}{method_suffix}"
                try:
                    arm_frames[arm] = load_arm_frames(out_root, method, T, name)
                except FileNotFoundError:
                    missing.append(arm)

            n_have = min([len(src)] + [len(f) for f in arm_frames.values()])
            idx = evenly_spaced(n_have, n_frames)
            n_cols = len(idx)

            n_rows = 1 + len(ARMS)
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(1.5 * n_cols, 1.5 * n_rows), squeeze=False)
            for ax in axes.ravel():
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)

            def label(r: int, text: str) -> None:
                axes[r][0].set_ylabel(text, fontsize=7, rotation=0, ha="right",
                                      va="center", labelpad=28)

            for j, pi in enumerate(idx):
                axes[0][j].imshow(src[pi])
                axes[0][j].set_title(f"frame {pi}", fontsize=6.5, pad=2)
            label(0, "source")

            for r, arm in enumerate(ARMS):
                row = axes[1 + r]
                if arm in missing:
                    for ax in row:
                        ax.set_facecolor("0.92")
                    label(1 + r, f"{arm}\n(missing)")
                    continue
                frames = arm_frames[arm]
                for j, pi in enumerate(idx):
                    row[j].imshow(frames[min(pi, len(frames) - 1)])
                label(1 + r, arm)

            fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.01,
                                wspace=0.02, hspace=0.04)
            fig.suptitle(
                f"R31 {name} (edit{T}) — source vs the four divergence-gated stage-3 "
                f"renders (step14); {n_have} frames rendered, {n_cols} shown",
                fontsize=8, y=0.995)

            out = out_dir / f"edit{T}_{name}_arms.png"
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            note = f" (missing: {','.join(missing)})" if missing else ""
            print(f"[ok] {out.name}: {n_rows}x{n_cols}{note}", flush=True)
            n_ok += 1
        except Exception as ex:
            print(f"[ERROR] {name} edit{T}: {type(ex).__name__}: {ex}", flush=True)

    return n_ok, len(cases)


# --------------------------------------------------------------------------- main ------

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--which", choices=("tau", "arms"), required=True)
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"))
    p.add_argument("--data_root", type=Path,
                   default=Path("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark"))
    p.add_argument("--dpi", type=int, default=110)

    # --which tau
    p.add_argument("--div_root", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r31_div"))
    p.add_argument("--rho_root", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r31_rho"),
                   help="Point at r31_rho_lin_t0.4 for the linear_threshold run.")
    p.add_argument("--tau_out_dir", type=Path,
                   default=Path("evaluation/figures/r31_tau_grids"))
    p.add_argument("--tau_n_frames", type=int, default=12)
    p.add_argument("--cmap_m", type=str, default="viridis")
    p.add_argument("--cmap_tau", type=str, default="inferno")
    p.add_argument("--arms", nargs="+", default=list(ARMS),
                   help="Which divergence arms get a row pair. Default: all four, i.e. "
                        "the historical figure. Narrow it (e.g. --arms lpips dino_patch) "
                        "for a tree that only holds some arms.")

    # --which arms
    p.add_argument("--out_root", type=Path,
                   default=Path("/projects/dataggen/outputs/five_bench/r31_arms"),
                   help="Point at r31_arms_lin04 for the linear_threshold run.")
    p.add_argument("--method_suffix", type=str, default="",
                   help="Appended to 'r31_{arm}' to form the render method name under "
                        "--out_root, e.g. '_lin04' for the linear_threshold run.")
    p.add_argument("--arms_out_dir", type=Path,
                   default=Path("evaluation/figures/r31_arm_grids"))
    p.add_argument("--arms_n_frames", type=int, default=8)

    args = p.parse_args(argv)
    cases = load_cases(args.cases)
    data_root = args.data_root.expanduser().resolve()

    if args.which == "tau":
        n_ok, n_total = build_tau_grid(
            args.div_root.expanduser().resolve(), args.rho_root.expanduser().resolve(),
            cases, args.tau_out_dir.expanduser().resolve(), args.tau_n_frames,
            args.cmap_m, args.cmap_tau, args.dpi, args.arms)
        print(f"[r31_stage3_grids] tau: {n_ok}/{n_total} pages -> {args.tau_out_dir}")
    else:
        n_ok, n_total = build_arm_grid(
            args.out_root.expanduser().resolve(), data_root, cases,
            args.arms_out_dir.expanduser().resolve(), args.method_suffix,
            args.arms_n_frames, args.dpi)
        print(f"[r31_stage3_grids] arms: {n_ok}/{n_total} pages -> {args.arms_out_dir}")

    return 0 if n_ok == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
