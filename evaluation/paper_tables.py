r"""Write the main-paper LaTeX tables from the R57 suite tables on FiVE-Bench-Correct.

Reads the R57 suite's per-clip tables ``evaluation/csv/suite_<stem>.csv`` (plain per-clip means over the 415 pairs,
cross-checked against ``suite_<stem>_overall.csv``; never ``_avg.csv``) and writes:

* ``tab_main_t2v.tex`` / ``tab_main_i2v.tex`` -- one table per editor family (supervisor, 2026-11-09), StreamEdit-
  style columns, no colour scale; best and second best of each table in bold / underline (ranked on the displayed,
  rounded values, ties share the mark).
* ``tab_ablation.tex`` -- demand and mapping ablations on both hosts, with gains over each host's equal-step uniform
  curve at matched preservation / matched alignment (linear interpolation between the two neighbouring curve points;
  if several segments bracket the value, the one most favourable to the baseline; outside the curve range -> "out",
  no extrapolation; paper_strategy.md section 3).

A cell without a value -- missing table, empty column, or a whole ablation block whose uniform curve is incomplete --
prints the paper's gray placeholder ``\ph`` (defined in the paper's preamble.tex); CF is a placeholder column until it is
scored; WE's steps are a placeholder until R71's cost is audited. No NFE column: the paper matches steps only. WE rows read R71 (normalized mapping
m = round(49 q)). The paper \input's these files verbatim (sec/4_experiments.tex), so they are the only copy of the tables:

    python -I evaluation/paper_tables.py --out_dir 6ae3275401185b12182f9f41/tables
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

N_PAIRS = 415
CSV_DIR = Path("evaluation/csv")

Row = Dict[str, Optional[float]]


# suite columns the tables use (five_acc is derived from its two parts)
USED_KEYS = ("structure_distance", "psnr_unedit_part", "lpips_unedit_part", "mse_unedit_part", "ssim_unedit_part",
             "clip_similarity_target_image", "editreward", "psc", "niqe_target_image", "motion_fidelity_score",
             "five_acc_yes_no", "five_acc_multi_choice")


def _num(v: str) -> Optional[float]:
    if v in ("", "nan", "NaN"):
        return None
    f = float(v)
    return None if f != f else f


def load(stem: str) -> Optional[Row]:
    """Plain per-clip means of ``suite_<stem>.csv`` (415 pairs; an empty / NaN cell is left out of its column's mean,
    FiVE's rule, as the R57 merge does), or None if the table is absent.

    The means are computed from the per-clip table because ``suite_<stem>_overall.csv`` stores them with 6 significant
    digits, which flips the last displayed digit when the exact mean sits next to a rounding boundary (3 cells on
    2026-11-09). Every mean is cross-checked against that overall row; a disagreement beyond its precision is fatal.
    """
    path = CSV_DIR / f"suite_{stem}.csv"
    if not path.exists():
        return None
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != N_PAIRS or len({(r["edit_type"], r["video_name"]) for r in rows}) != N_PAIRS:
        raise SystemExit(f"[paper_tables] {path}: expected {N_PAIRS} distinct pairs, got {len(rows)} rows")
    out: Row = {}
    for k in USED_KEYS:
        vals = [x for x in (_num(r.get(k, "")) for r in rows) if x is not None]
        out[k] = sum(vals) / len(vals) if vals else None
    ov_path = CSV_DIR / f"suite_{stem}_overall.csv"
    with ov_path.open() as f:
        ov = [r for r in csv.DictReader(f) if r["scope"] == "overall"]
    if len(ov) != 1 or int(float(ov[0]["n"])) != N_PAIRS:
        raise SystemExit(f"[paper_tables] {ov_path}: expected one overall row with n = {N_PAIRS}")
    for k in USED_KEYS:
        o = _num(ov[0].get(k, ""))
        if (o is None) != (out[k] is None) or (o is not None and abs(o - out[k]) > 1e-5 * max(1.0, abs(o))):
            raise SystemExit(f"[paper_tables] {stem}.{k}: per-clip mean {out[k]} vs overall row {o}")
    yn, mc = out["five_acc_yes_no"], out["five_acc_multi_choice"]
    out["five_acc"] = (yn + mc) / 2.0 if yn is not None and mc is not None else None  # project convention (R67)
    return out


# Motion fidelity is incomplete (0010_giant-slalom edit1-4 NaN in the motion part, user 2026-11-12): every MF cell and the
# dMF@Acc delta print PH until the motion part is complete; set True then.
MF_READY = False
PH = r"\ph"   # gray placeholder cell, \newcommand{\ph}[1][xx.xx]{\textcolor{gray}{#1}} in the paper's preamble.tex


@dataclass(frozen=True)
class Col:
    key: str             # suite column
    scale: float         # displayed value = raw * scale
    digits: int
    higher_better: bool
    placeholder: bool = False   # not shown yet (CF: user, 2026-11-12): always PH, never read


def fmt(v: Optional[float], col: Col) -> str:
    return PH if v is None or col.placeholder else f"{v * col.scale:.{col.digits}f}"


# ----------------------------------------------------------------------------------------------------------------
# Main table
# ----------------------------------------------------------------------------------------------------------------
MAIN_COLS: Tuple[Col, ...] = (
    Col("structure_distance", 1e3, 2, False),
    Col("psnr_unedit_part", 1.0, 2, True),
    Col("lpips_unedit_part", 1e3, 2, False),
    Col("mse_unedit_part", 1e4, 2, False),
    Col("ssim_unedit_part", 1e2, 2, True),
    Col("cf", 1.0, 2, True, placeholder=True),
    Col("clip_similarity_target_image", 1.0, 2, True),
    Col("editreward", 1.0, 2, True),
    Col("psc", 1e2, 2, True),
    Col("niqe_target_image", 1.0, 2, False),
    Col("motion_fidelity_score", 1e2, 2, True, placeholder=not MF_READY),
    Col("five_acc", 1e2, 2, True),
)

MAIN_HEADER = r"""        \multirow{2}{*}{\textbf{Method}} & \multirow{2}{*}{\textbf{Backbone}} & \multirow{2}{*}{\textbf{Steps}} & \textbf{Struc.} & \multicolumn{5}{c|}{\textbf{Background Preservation}} & \multicolumn{3}{c|}{\textbf{Text Alignment}} & \textbf{IQA} & \textbf{Temp.} & \textbf{FiVE} \\
        & & & Dist.$_{\times 10^3}\!\downarrow$ & PSNR$\uparrow$ & LPIPS$_{\times 10^3}\!\downarrow$ & MSE$_{\times 10^4}\!\downarrow$ & SSIM$_{\times 10^2}\!\uparrow$ & CF$\uparrow$ & CLIP-T$\uparrow$ & EditRwd.$\uparrow$ & PSC$_{\times 10^2}\!\uparrow$ & NIQE$\downarrow$ & MF$_{\times 10^2}\!\uparrow$ & Acc$\uparrow$ \\"""


@dataclass(frozen=True)
class MainRow:
    label: str
    stem: str
    backbone: str
    steps: str           # solver steps per clip, inversion and our draft included (per chunk for StreamEdit)
    nfe: str             # backbone forward passes per clip (provenance below); NOT printed: the paper matches steps only (user, 2026-11-12)
    ours: bool = False


# Cost columns, audited in the code on 2026-11-09 (one pass = one backbone evaluation of one input stream; a forward on a
# batch of k streams counts k):
#   Wan-Edit (wan-edit/wan/text2video.py edit()): 4 passes per active step (cond/uncond x src/tgt, n_avg 1). R71:
#     baseline 90-step grid from step 27 (s 0.92, R71 plan's matched63 N/skip) = 63 steps, 252. Ours: 14-step draft
#     (r48_wan_draft.py asserts 4*(17-3) calls) + 49 steps (50-grid from s_min 1) = 63 steps, 252. (R58: 59 / 236.)
#   FlowDirector (wan/text2video.py edit()): 50-grid, n_max 40 -> 40 steps, each n_avg 4 x 4 passes (one temporal
#     window: 11 latent frames, window 11) = 640.
#   UniEdit-Flow (UniEdit-Flow_diffusers schedulers, alpha 0.8 of 25): 20 inversion steps x 1 + 20 edit steps x 2
#     (src + trg, guidance_scale 1.0 -> no CFG) = 40 steps, 60.
#   AnyV2V (i2vgen-xl pipeline): 500 DDIM inversion steps x 1 (cfg 1.0) + 50 PnP steps x 3 (inverted src, negative,
#     edit prompt) = 550 steps, 650.
#   ContextFlow (wan/image2video.py, fm_new = 2 calls per step): 50 inversion steps x 2 x 1 + 50 edit steps x 2 x 3
#     (origin, cond, uncond) = 100 steps, 400 (generate_reconstruction's reconstruction loop is dead code).
#   StreamEdit (pipeline/edit_causal_inference.py): per 3-latent-frame chunk 1 clean source pass + N steps x 2 (src,
#     trg) + 1 clean target rerun = 2N + 2; plus 2 context passes per rollout window (SF VP: every window, the first
#     holds the anchor, windows = 1 + ceil((L-18)/18); SF text: every window but the first, windows = 1 + ceil((L-21)/18);
#     LL VP: one call, 2 for the x3 anchor block). L = latent frames; the NFE shown is the mean over the 415 rendered
#     clips (21-117 frames): baseline N = 20 -> SF 251.4, text 248.6, LL 250.6; ours = 5-step draft + 15-step render,
#     each a full rollout -> SF 266.1, text 260.5, LL 264.4. DINO / LPIPS (divergence) are not backbone passes.


@dataclass(frozen=True)
class MainTable:
    filename: str
    label: str
    caption: str
    rows: Tuple[MainRow, ...]   # baselines first, "Ours" rows last (one rule above them), as in StreamEdit's table


WE_STEPS, WE_NFE = r"\ph[xx]", r"\ph[xxx]"   # R71 cost (63 / 252 per its plan) not audited yet: placeholders

COST_CAPTION = (
    r"Steps: solver steps per clip, inversion and the \method draft included (per chunk for SE). "
)
CAPTION_TAIL = (
    r"Background metrics use FiVE's reference frames, so LPIPS is not comparable to the values in~\cite{streamedit2026}. "
    r"Best and second best in \textbf{bold} and \underline{underlined}."
)

MAIN_TABLES: Tuple[MainTable, ...] = (
    MainTable(
        "tab_main_t2v.tex", "tab:main_t2v",
        r"\textbf{Text-guided editing} on FiVE-Bench-C. SE runs in its text mode on Self-Forcing (SF); WE and SE use their "
        r"official strength with the step count of their \method variant. Each method is scored on its own output "
        r"(41 frames for the Wan-based editors, up to 117 for SE). " + COST_CAPTION + CAPTION_TAIL,
        (
            MainRow(r"FlowDirector~\cite{flowdirector}", "r64_r64_flowdirector", "Wan-1.3B", "40", "640"),
            MainRow(r"UniEdit-Flow~\cite{unieditflow}", "r63_r63_uniedit_flow", "Wan-1.3B", "40", "60"),
            MainRow(r"Wan-Edit~\cite{five2025}", "r71_wan_matched63_s0p92_correct", "Wan-1.3B", WE_STEPS, WE_NFE),
            MainRow(r"StreamEdit~\cite{streamedit2026}", "r60_r60_n20_rho2_correct", "SF-1.3B", "20", "249"),
            MainRow(r"WE\,+\,\method", "r71_wan_max_s1_correct", "Wan-1.3B", WE_STEPS, WE_NFE, ours=True),
            MainRow(r"SE\,+\,\method", "r60_r60_max_novp_correct", "SF-1.3B", "20", "261", ours=True),
        ),
    ),
    MainTable(
        "tab_main_i2v.tex", "tab:main_i2v",
        r"\textbf{Image-guided editing} on FiVE-Bench-C. All methods receive the same edited first frame; SE runs on "
        r"Self-Forcing (SF) or LongLive (LL) at its official strength with the step count of \method. Each method is scored on "
        r"its own output (41 frames for ContextFlow, 32 for AnyV2V, up to 117 for SE). " + COST_CAPTION + CAPTION_TAIL,
        (
            MainRow(r"AnyV2V~\cite{anyv2v}", "r65_r65_anyv2v", "I2VGen-XL", "550", "650"),
            MainRow(r"ContextFlow~\cite{contextflow}", "r66_r66_contextflow", "Wan-I2V-14B", "100", "400"),
            MainRow(r"StreamEdit~\cite{streamedit2026} (SF)", "r59_r59_n20_rho2_vp", "SF-1.3B", "20", "251"),
            MainRow(r"StreamEdit~\cite{streamedit2026} (LL)", "r61_r61_ll_rho2_n20", "LL-1.3B", "20", "251"),
            MainRow(r"SE\,+\,\method (SF)", "r59_r59_max_full", "SF-1.3B", "20", "266", ours=True),
            MainRow(r"SE\,+\,\method (LL)", "r61_r61_max_ll", "LL-1.3B", "20", "264", ours=True),
        ),
    ),
)


def rank_marks(values: Sequence[Optional[str]], higher_better: bool) -> List[str]:
    """Bold the best and underline the second-best distinct displayed value (ties share the mark); a column with
    fewer than two filled cells gets no mark."""
    blank = (None, "--")

    def filled(v: Optional[str]) -> bool:
        return v not in blank and not str(v).startswith(PH)
    if sum(filled(v) for v in values) < 2:
        return [PH if v is None else v for v in values]
    distinct = sorted({float(v) for v in values if filled(v)}, reverse=higher_better)
    best = distinct[0] if distinct else None
    second = distinct[1] if len(distinct) > 1 else None
    out = []
    for v in values:
        if not filled(v):
            out.append(PH if v is None else v)
        elif float(v) == best:
            out.append(rf"\textbf{{{v}}}")
        elif float(v) == second:
            out.append(rf"\underline{{{v}}}")
        else:
            out.append(v)
    return out


def main_table(table: MainTable) -> str:
    ncols = 3 + len(MAIN_COLS)
    rows = table.rows
    data = [load(r.stem) for r in rows]
    cells = [[fmt(d.get(c.key) if d else None, c) for c in MAIN_COLS] for d in data]
    for j, col in enumerate(MAIN_COLS):  # rank each column over the whole table
        marked = rank_marks([cells[i][j] for i in range(len(rows))], col.higher_better)
        for i in range(len(rows)):
            cells[i][j] = marked[i]
    lines = [
        r"% Generated by evaluation/paper_tables.py from the R57 suite _overall tables -- do not edit numbers by hand.",
        r"\begin{table*}[t]",
        r"    \centering",
        rf"    \caption{{{table.caption}}}",
        r"    \resizebox{0.99\textwidth}{!}{",
        r"    \begin{tabular}{l|c|c|c|ccccc|ccc|c|c|c}",
        r"        \toprule",
        MAIN_HEADER,
        r"        \midrule",
    ]
    for i, (r, c) in enumerate(zip(rows, cells)):
        if r.ours and (i == 0 or not rows[i - 1].ours):  # one rule before the "Ours" rows
            lines.append(r"        \midrule")
        lines.append(f"        {r.label} & {r.backbone} & {r.steps} & " + " & ".join(c) + r" \\")
    lines += [r"        \bottomrule", r"    \end{tabular}}", rf"    \label{{{table.label}}}", r"\end{table*}", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------
# Ablation table
# ----------------------------------------------------------------------------------------------------------------
ABL_COLS: Tuple[Col, ...] = (
    Col("psnr_unedit_part", 1.0, 2, True),
    Col("lpips_unedit_part", 1e3, 1, False),
    Col("ssim_unedit_part", 1e2, 2, True),
    Col("five_acc", 1e2, 2, True),
    Col("clip_similarity_target_image", 1.0, 2, True),
    Col("editreward", 1.0, 3, True),
    Col("psc", 1e2, 2, True),
    Col("motion_fidelity_score", 1e2, 2, True, placeholder=not MF_READY),
)

# Equal-step uniform curves, in knob order (StreamEdit rho ascending, Wan-Edit strength ascending).
CURVE_KNOBS = {
    "se": ("2", "3", "4", "6", "8", "10", "20", "50"),
    "wan": ("0.5", "0.85", "0.92", "0.95", "0.978", "0.996"),   # R71 matched-63 curve
}
CURVES = {
    "se": tuple(f"r59_r59_n20_rho{r}_vp" for r in CURVE_KNOBS["se"]),
    "wan": tuple(f"r71_wan_matched63_s{s.replace('.', 'p')}_correct" for s in CURVE_KNOBS["wan"]),
}
AUTO = ""   # AblRow.stem of a reference row picked from the curve: closest point to the full method in LPIPS


@dataclass(frozen=True)
class AblRow:
    label: str
    demand: str          # q (or mapping) column
    stem: str
    reference: bool = False   # point of the uniform curve itself: no matched delta (stem AUTO: "<knob>" in label)
    ours: bool = False
    short: str = ""           # one-column label of the main-paper table (empty: row only in the full table)


# Main paper (user, 2026-11-12): ESM components on WE, control assignment on SE. StreamEdit's ESM components (R62:
# r62_r62_{nodiv,dino,lpips,nosupp}_full) go to the supplementary.
ABL_BLOCKS: Tuple[Tuple[str, str, Tuple[AblRow, ...]], ...] = (
    ("wan", r"\textit{ESM components: WE\,+\,\method, " + WE_STEPS + r" steps, control $s$}", (
        AblRow(r"Uniform $s{=}<knob>$", r"--", AUTO, reference=True, short=r"Uniform $s{=}<knob>$"),
        # R71 (batch r71.txt, tag r71): A1-A4 and A0 at s_min 1.
        AblRow(r"w/o divergence", r"$\mathbf S$", "r71_wan_nodiv_s1_correct", short=r"$\mathbf S$"),
        AblRow(r"DINO only", r"$\mathbf S\odot\mathbf D^{\mathrm{sem}}$", "r71_wan_dino_s1_correct",
               short=r"$\mathbf S\odot\mathbf D^{\mathrm{sem}}$"),
        AblRow(r"LPIPS only", r"$\mathbf S\odot\mathbf D^{\mathrm{per}}$", "r71_wan_lpips_s1_correct",
               short=r"$\mathbf S\odot\mathbf D^{\mathrm{per}}$"),
        AblRow(r"w/o support", r"$\mathbf D$", "r71_wan_nosupp_s1_correct", short=r"$\mathbf D$"),
        AblRow(r"\textbf{Full}", r"$\mathbf S\odot\mathbf D$", "r71_wan_max_s1_correct", ours=True,
               short=r"$\mathbf S\odot\mathbf D$ (\textbf{full})"),
    )),
    ("se", r"\textit{Control assignment: SE\,+\,\method, 20 steps, control $\rho$}", (
        AblRow(r"Uniform $\rho{=}6$", r"--", "r59_r59_n20_rho6_vp", reference=True, short=r"Uniform $\rho{=}6$"),
        # R68 B1 (not scored yet): adjust the stem to R68's batch name once it exists.
        AblRow(r"Linear", r"$\rho_i{=}\rho^{\min}{+}(\rho^{\max}{-}\rho^{\min})\,d_i$", "r68_r68_linrho_full",
               short=r"Linear"),
        AblRow(r"\textbf{Full}", r"exposure-matched, Eq.~\ref{eq:stream_mapping}", "r59_r59_max_full", ours=True,
               short=r"Exposure-matched (\textbf{full})"),
    )),
)


def matched(curve: Sequence[Row], xkey: str, ykey: str, x: float) -> Optional[float]:
    """Baseline y at x on the curve: linear interpolation between neighbouring points; the value most favourable to
    the baseline (max, all y used here are higher-is-better) if several segments bracket x; None outside the range.
    The caller guarantees every curve point has both values (no interpolation across a missing point)."""
    pts = [(p[xkey], p[ykey]) for p in curve]
    vals = [y0 + (x - x0) * (y1 - y0) / (x1 - x0)
            for (x0, y0), (x1, y1) in zip(pts, pts[1:])
            if x1 != x0 and min(x0, x1) <= x <= max(x0, x1)]
    return max(vals) if vals else None


def deltas(d: Row, curve: Sequence[Row]) -> List[str]:
    """[dAcc @ matched PSNR, dAcc @ matched LPIPS, dCLIP-T @ matched PSNR, dPSNR @ matched Acc, dMF @ matched Acc];
    'out' outside the curve range, '--' if the row or any curve point lacks a value. CLIP-T = target prompt vs whole
    edited frame (the paper reports CLIP-T, not CLIP-D: user, 2026-11-09); MF = FiVE motion fidelity, read at matched
    alignment."""
    specs = (("psnr_unedit_part", "five_acc", 1e2), ("lpips_unedit_part", "five_acc", 1e2),
             ("psnr_unedit_part", "clip_similarity_target_image", 1.0), ("five_acc", "psnr_unedit_part", 1.0),
             ("five_acc", "motion_fidelity_score", 1e2))
    out = []
    for xkey, ykey, scale in specs:
        if ykey == "motion_fidelity_score" and not MF_READY:
            out.append(PH)
            continue
        if d.get(xkey) is None or d.get(ykey) is None or any(p.get(k) is None for p in curve for k in (xkey, ykey)):
            out.append("--")
            continue
        base = matched(curve, xkey, ykey, d[xkey])
        out.append("out" if base is None else f"{(d[ykey] - base) * scale:+.2f}")
    return out


ABL_CAPTION = (
    r"\textbf{Ablation} on FiVE-Bench-C. Top: components of the edit strength map (ESM) on WE\,+\,\method; $\mathbf S$: "
    r"support, $\mathbf D^{\mathrm{sem}}$, $\mathbf D^{\mathrm{per}}$: normalized DINOv3 and LPIPS divergences "
    r"(Eq.~\ref{eq:div}). Bottom: control assignment on SE\,+\,\method (image-guided, Self-Forcing). Each row changes one "
    r"component and keeps the rest fixed. The last five columns compare each row with the uniform-control sweep at the "
    r"same step count, at matched background preservation ($\Delta$Acc; $\Delta$CLIP-T) or matched FiVE-Acc "
    r"($\Delta$PSNR in dB; $\Delta$MF), interpolated linearly between neighbouring sweep points (out: outside the sweep, "
    r"not extrapolated). Uniform: the sweep point closest to the full method in LPIPS."
)


def block_rows(host: str, rows: Sequence[AblRow]) -> List[Tuple[AblRow, str, Optional[Row], List[str]]]:
    """(row, resolved label, values or None, 5 matched deltas) per row of one ablation block. Deltas need the whole
    uniform curve, so an incomplete curve leaves the whole block empty (placeholders)."""
    curve = [load(s) for s in CURVES[host]]
    complete = all(c is not None for c in curve)
    if not complete:
        missing = [s for s, c in zip(CURVES[host], curve) if c is None]
        print(f"[paper_tables] host {host}: uniform curve incomplete ({len(missing)} tables missing), block left as placeholders")
    full = next(load(r.stem) for r in rows if r.ours)
    out = []
    for r in rows:
        d, label = (load(r.stem) if complete and r.stem != AUTO else None), r.label
        if complete and r.stem == AUTO and full is not None and full.get("lpips_unedit_part") is not None:
            i = min(range(len(curve)), key=lambda k: abs(curve[k]["lpips_unedit_part"] - full["lpips_unedit_part"]))
            d, label = curve[i], r.label.replace("<knob>", CURVE_KNOBS[host][i])
            r = AblRow(r.label, r.demand, r.stem, r.reference, r.ours, r.short.replace("<knob>", CURVE_KNOBS[host][i]))
        dl = (["--"] * 5 if r.reference else deltas(d, curve)) if d is not None else [PH] * 5
        out.append((r, label, d, dl))
    return out


def knob_ph(label: str) -> str:
    return label.replace("$s{=}<knob>$", "$s{=}$" + PH + "[x.xx]").replace("<knob>", PH + "[x.xx]")


def ablation_table() -> str:
    """Full ablation (every metric, both blocks, full-width): supplementary material."""
    ncols = 2 + len(ABL_COLS) + 5
    lines = [
        r"% Generated by evaluation/paper_tables.py from the R57 suite _overall tables -- do not edit numbers by hand.",
        r"\begin{table*}[t]",
        r"    \centering",
        rf"    \caption{{{ABL_CAPTION}}}",
        r"    \resizebox{0.99\textwidth}{!}{",
        r"    \begin{tabular}{ll|ccc|cccc|c|ccccc}",
        r"        \toprule",
        r"        \multirow{2}{*}{\textbf{Variant}} & \multirow{2}{*}{\textbf{Configuration}} & \multicolumn{3}{c|}{\textbf{Background Preservation}} & \multicolumn{4}{c|}{\textbf{Text Alignment}} & \textbf{Temp.} & \multicolumn{5}{c}{\textbf{vs.\ uniform sweep, same steps}} \\",
        r"        & & PSNR$\uparrow$ & LPIPS$_{\times 10^3}\!\downarrow$ & SSIM$_{\times 10^2}\!\uparrow$ & FiVE Acc$\uparrow$ & CLIP-T$\uparrow$ & EditRwd.$\uparrow$ & PSC$_{\times 10^2}\!\uparrow$ & MF$_{\times 10^2}\!\uparrow$ & $\Delta$Acc@PSNR & $\Delta$Acc@LPIPS & $\Delta$CLIP-T@PSNR & $\Delta$PSNR@Acc & $\Delta$MF@Acc \\",
    ]
    for host, title, rows in ABL_BLOCKS:
        lines += [r"        \midrule", rf"        \multicolumn{{{ncols}}}{{l}}{{{title}}} \\", r"        \midrule"]
        for r, label, d, dl in block_rows(host, rows):
            if r.ours:
                lines.append(r"        \cmidrule{1-" + str(ncols) + "}")
            vals = [fmt(d.get(c.key) if d else None, c) for c in ABL_COLS]
            lines.append(f"        {knob_ph(label)} & {r.demand} & " + " & ".join(vals + dl) + r" \\")
    lines += [r"        \bottomrule", r"    \end{tabular}}", r"    \label{tab:supp_ablation}", r"\end{table*}", ""]
    return "\n".join(lines)


# Main-paper ablations (user, 2026-11-12): one column each, a FIXED subset chosen before the WE / Linear results exist
# (primary metrics of paper_strategy.md section 3 + MF + one matched delta per side of the claim), never re-picked
# from the results; the full table above goes to the supplementary.
SHORT_COLS: Tuple[Col, ...] = (
    Col("psnr_unedit_part", 1.0, 2, True),
    Col("ssim_unedit_part", 1e2, 2, True),
    Col("five_acc", 1e2, 2, True),
    Col("clip_similarity_target_image", 1.0, 2, True),
    Col("motion_fidelity_score", 1e2, 2, True, placeholder=not MF_READY),
)
SHORT_DELTAS = (0, 3)   # indices into deltas(): dAcc @ matched PSNR, dPSNR @ matched Acc
SHORT_HEADER = (
    r"        & PSNR$\uparrow$ & SSIM$_{\times 10^2}\!\uparrow$ & Acc$\uparrow$ & CLIP-T$\uparrow$ & "
    r"MF$_{\times 10^2}\!\uparrow$ & $\Delta$Acc & $\Delta$PSNR \\"
)
DELTA_CAPTION = (
    r"$\Delta$: gain over the uniform-control sweep at the same step count, in FiVE-Acc at matched PSNR and in PSNR (dB) "
    r"at matched FiVE-Acc, interpolated linearly between neighbouring sweep points (out: outside the sweep, not "
    r"extrapolated). Uniform: the sweep point closest to the full method in LPIPS. All metrics are in the supplementary "
    r"material."
)
SHORT_TABLES = (
    ("tab_abl_esm.tex", "tab:abl_esm", "wan", r"\textbf{ESM}",
     r"\textbf{ESM components} (WE\,+\,\method, " + WE_STEPS + r" steps, FiVE-Bench-C). $\mathbf S$: support; "
     r"$\mathbf D^{\mathrm{sem}}$, $\mathbf D^{\mathrm{per}}$: normalized DINOv3 and LPIPS divergences (Eq.~\ref{eq:div}). "
     + DELTA_CAPTION),
    ("tab_abl_assign.tex", "tab:abl_assign", "se", r"\textbf{Assignment}",
     r"\textbf{Control assignment} (SE\,+\,\method, image-guided, Self-Forcing, 20 steps, FiVE-Bench-C). Linear: "
     r"$\rho_i{=}\rho^{\min}{+}(\rho^{\max}{-}\rho^{\min})\,d_i$; exposure-matched: Eq.~\ref{eq:stream_mapping}. "
     r"$\Delta$ and Uniform as in Tab.~\ref{tab:abl_esm}."),
)


def short_table(filename: str, label: str, host: str, head: str, caption: str) -> str:
    rows = dict((h, r) for h, _, r in ABL_BLOCKS)[host]
    lines = [
        r"% Generated by evaluation/paper_tables.py from the R57 suite _overall tables -- do not edit numbers by hand.",
        r"\begin{table}[t]",
        r"    \centering",
        rf"    \caption{{{caption}}}",
        r"    \resizebox{\columnwidth}{!}{",
        r"    \begin{tabular}{l|cc|cc|c|cc}",
        r"        \toprule",
        rf"        \multirow{{2}}{{*}}{{{head}}} & \multicolumn{{2}}{{c|}}{{\textbf{{Preservation}}}} & "
        r"\multicolumn{2}{c|}{\textbf{Alignment}} & \textbf{Temp.} & \multicolumn{2}{c}{\textbf{vs.\ uniform}} \\",
        SHORT_HEADER,
        r"        \midrule",
    ]
    for r, _, d, dl in block_rows(host, [r for r in rows if r.short]):
        if r.ours:
            lines.append(r"        \midrule")
        vals = [fmt(d.get(c.key) if d else None, c) for c in SHORT_COLS]
        lines.append(f"        {knob_ph(r.short)} & " + " & ".join(vals + [dl[i] for i in SHORT_DELTAS]) + r" \\")
    lines += [r"        \bottomrule", r"    \end{tabular}}", rf"    \label{{{label}}}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path, default=Path("."), help="where the .tex files are written")
    args = ap.parse_args()
    outputs = ([(t.filename, main_table(t)) for t in MAIN_TABLES] + [(t[0], short_table(*t)) for t in SHORT_TABLES]
               + [("supp_tab_ablation.tex", ablation_table())])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in outputs:
        path = args.out_dir / name
        path.write_text(text)
        print(f"[paper_tables] wrote {path}")


if __name__ == "__main__":
    main()
