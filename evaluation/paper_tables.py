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
prints the paper's gray placeholder ``\ph`` (defined in the paper's preamble.tex); WE's 63 steps = 14 draft + 49 final (audited 2026-11-13). No NFE column: the paper matches steps only. WE rows read R71 (normalized mapping
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
             "cf", "five_acc_yes_no", "five_acc_multi_choice")


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
MF_READY = True   # 2026-11-13: giant-slalom filled; the remaining NaN cells are single clips (<= 2 per arm), left out of the mean
PH = r"\ph"   # gray placeholder cell, \newcommand{\ph}[1][xx.xx]{\textcolor{gray}{#1}} in the paper's preamble.tex


@dataclass(frozen=True)
class Col:
    key: str             # suite column
    scale: float         # displayed value = raw * scale
    digits: int
    higher_better: bool
    placeholder: bool = False   # not shown yet: always PH, never read (was CF until 2026-11-13)


def fmt(v: Optional[float], col: Col) -> str:
    return PH if v is None or col.placeholder else f"{v * col.scale:.{col.digits}f}"


# ----------------------------------------------------------------------------------------------------------------
# Main table
# ----------------------------------------------------------------------------------------------------------------
# Full FiVE-style table (12 metrics): supplementary (supp_tab_main_{t2v,i2v}.tex).
FULL_COLS: Tuple[Col, ...] = (
    Col("structure_distance", 1e3, 2, False),
    Col("psnr_unedit_part", 1.0, 2, True),
    Col("lpips_unedit_part", 1e3, 2, False),
    Col("mse_unedit_part", 1e4, 2, False),
    Col("ssim_unedit_part", 1e2, 2, True),
    Col("cf", 1.0, 2, True),   # IVEBench CF, Qwen2.5-VL-72B int8 judge (R57 cf part, complete 2026-11-12)
    Col("clip_similarity_target_image", 1.0, 2, True),
    Col("editreward", 1.0, 2, True),
    Col("psc", 1e2, 2, True),
    Col("niqe_target_image", 1.0, 2, False),
    Col("motion_fidelity_score", 1e2, 2, True, placeholder=not MF_READY),
    Col("five_acc", 1e2, 2, True),
)

FULL_HEADER = r"""        \multirow{2}{*}{\textbf{Backbone}} & \multirow{2}{*}{\textbf{Method}} & \multirow{2}{*}{\textbf{Steps}} & \textbf{Struc.} & \multicolumn{5}{c|}{\textbf{Background Preservation}} & \multicolumn{3}{c|}{\textbf{Text Alignment}} & \textbf{IQA} & \textbf{Temp.} & \textbf{FiVE} \\
        & & & Dist.$_{\times 10^3}\!\downarrow$ & PSNR$\uparrow$ & LPIPS$_{\times 10^3}\!\downarrow$ & MSE$_{\times 10^4}\!\downarrow$ & SSIM$_{\times 10^2}\!\uparrow$ & CF$\uparrow$ & CLIP-T$\uparrow$ & EditRwd.$\uparrow$ & PSC$_{\times 10^2}\!\uparrow$ & NIQE$\downarrow$ & MF$_{\times 10^2}\!\uparrow$ & Acc$\uparrow$ \\"""
FULL_SPEC = r"l|l|c|c|ccccc|ccc|c|c|c"

# Main paper (user, 2026-11-13): preservation SSIM, LPIPS (FiVE) + CF (IVEBench); semantic alignment FiVE-Acc,
# CLIP-T (FiVE) + PSC (IVEBench); temporal MF. No PSNR anywhere in the main paper (SSIM is the preservation axis of
# Fig. tradeoff); PSNR, MSE, structure distance, NIQE and EditReward are in the full supplementary tables only.
MAIN_COLS: Tuple[Col, ...] = (
    Col("ssim_unedit_part", 1e2, 2, True),
    Col("lpips_unedit_part", 1e3, 2, False),
    Col("cf", 1.0, 2, True),
    Col("five_acc", 1e2, 2, True),
    Col("clip_similarity_target_image", 1.0, 2, True),
    Col("psc", 1e2, 2, True),
    Col("motion_fidelity_score", 1e2, 2, True, placeholder=not MF_READY),
)
MAIN_HEADER = r"""        \multirow{2}{*}{\textbf{Backbone}} & \multirow{2}{*}{\textbf{Method}} & \multirow{2}{*}{\textbf{Steps}} & \multicolumn{3}{c|}{\textbf{Background Preservation}} & \multicolumn{3}{c|}{\textbf{Semantic Alignment}} & \textbf{Temp.} \\
        & & & SSIM$_{\times 10^2}\!\uparrow$ & LPIPS$_{\times 10^3}\!\downarrow$ & CF$\uparrow$ & FiVE-Acc$\uparrow$ & CLIP-T$\uparrow$ & PSC$_{\times 10^2}\!\uparrow$ & MF$_{\times 10^2}\!\uparrow$ \\"""
MAIN_SPEC = r"l|l|c|ccc|ccc|c"


@dataclass(frozen=True)
class MainRow:
    label: str
    stem: str
    backbone: str
    steps: str           # solver steps per clip, inversion and our draft included (per chunk for StreamEdit)
    nfe: str             # backbone forward passes per clip (provenance below); NOT printed: the paper matches steps only (user, 2026-11-12)
    ours: bool = False
    base: str = ""       # ours rows: stem of the base editor (grey Delta line after the row if deltas_rows)


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


WE_STEPS, WE_NFE = "63", r"\ph[xxx]"   # R71 cost audited 2026-11-13: 14 draft + 49 final steps (r49_wan_render A_ACTIVE_63; matched-63 baseline: N/skip 90/27); NFE not printed

COST_CAPTION = (
    r"Steps: solver steps per clip, inversion and the \method draft included (per chunk for SE). "
    r"Rows are grouped by backbone; \method rows are shaded. "
)
MAIN_CAPTION_EXTRA = (r" PSNR, MSE, structure distance, NIQE and EditReward are reported in the supplementary material.")
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
            MainRow(r"WE\,+\,\method", "r71_wan_max_s1_correct", "Wan-1.3B", WE_STEPS, WE_NFE, ours=True,
                    base="r71_wan_matched63_s0p92_correct"),
            MainRow(r"StreamEdit~\cite{streamedit2026}", "r60_r60_n20_rho2_correct", "SF-1.3B", "20", "249"),
            MainRow(r"SE\,+\,\method", "r60_r60_max_novp_correct", "SF-1.3B", "20", "261", ours=True,
                    base="r60_r60_n20_rho2_correct"),
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
            MainRow(r"StreamEdit~\cite{streamedit2026}", "r59_r59_n20_rho2_vp", "SF-1.3B", "20", "251"),
            MainRow(r"SE\,+\,\method", "r59_r59_max_full", "SF-1.3B", "20", "266", ours=True,
                    base="r59_r59_n20_rho2_vp"),
            MainRow(r"StreamEdit~\cite{streamedit2026}", "r61_r61_ll_rho2_n20", "LL-1.3B", "20", "251"),
            MainRow(r"SE\,+\,\method", "r61_r61_max_ll", "LL-1.3B", "20", "264", ours=True,
                    base="r61_r61_ll_rho2_n20"),
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


def delta_cells(d: Optional[Row], b: Optional[Row], cols: Sequence[Col]) -> List[str]:
    """Grey signed difference ours - base per column, in displayed units (rounded values subtracted, so the line
    agrees with the two rows above it); no colour judgement (sign meaning depends on the column's arrow)."""
    out = []
    for c in cols:
        if d is None or b is None or d.get(c.key) is None or b.get(c.key) is None or c.placeholder:
            out.append(PH)
            continue
        diff = round(d[c.key] * c.scale, c.digits) - round(b[c.key] * c.scale, c.digits)
        out.append(rf"\textcolor{{gray}}{{{diff:+.{c.digits}f}}}")
    return out


def main_table(table: MainTable, cols: Sequence[Col] = MAIN_COLS, header: str = MAIN_HEADER, spec: str = MAIN_SPEC,
               label_suffix: str = "", caption_extra: str = "", deltas_rows: bool = False) -> str:
    MAIN_COLS = cols  # noqa: N806 (local alias: the body below predates the parameter)
    rows = table.rows
    data = [load(r.stem) for r in rows]
    cells = [[fmt(d.get(c.key) if d else None, c) for c in MAIN_COLS] for d in data]
    for j, col in enumerate(MAIN_COLS):  # rank each column over the method rows (Delta lines are not ranked)
        marked = rank_marks([cells[i][j] for i in range(len(rows))], col.higher_better)
        for i in range(len(rows)):
            cells[i][j] = marked[i]
    lines = [
        r"% Generated by evaluation/paper_tables.py from the R57 suite _overall tables -- do not edit numbers by hand.",
        r"\begin{table*}[t]",
        r"    \centering",
        rf"    \caption{{{table.caption}{caption_extra}}}",
        r"    \resizebox{0.99\textwidth}{!}{",
        rf"    \begin{{tabular}}{{{spec}}}",
        r"        \toprule",
        header,
        r"        \midrule",
    ]
    by_stem = dict(zip((r.stem for r in rows), data))
    # Rows grouped by backbone (user 2026-11-13): consecutive rows with the same backbone form a group, one merged
    # backbone cell per group drawn in its LAST row (negative \multirow, so the shaded SNaC row cannot paint over it),
    # a rule between groups; the backbone cell itself is never shaded.
    starts = [i for i in range(len(rows)) if i == 0 or rows[i].backbone != rows[i - 1].backbone]
    for i, (r, c) in enumerate(zip(rows, cells)):
        if i in starts and i > 0:
            lines.append(r"        \midrule")
        g0 = max(k for k in starts if k <= i)
        g1 = min([k for k in starts if k > i] + [len(rows)])          # group = rows g0 .. g1-1
        bb = "" if i != g1 - 1 else (r.backbone if g1 - g0 == 1 else rf"\multirow{{-{g1 - g0}}}{{*}}{{{r.backbone}}}")
        if r.ours:   # SNaC rows shaded (oursbg: paper preamble.tex)
            lines.append(rf"        \rowcolor{{oursbg}}\cellcolor{{white}}{bb} & {r.label} & {r.steps} & " + " & ".join(c) + r" \\")
        else:
            lines.append(f"        {bb} & {r.label} & {r.steps} & " + " & ".join(c) + r" \\")
        if r.ours and r.base and deltas_rows:
            lines.append(r"        \quad\textcolor{gray}{$\Delta$} & & & "
                         + " & ".join(delta_cells(by_stem[r.stem], load(r.base), MAIN_COLS)) + r" \\")
    lines += [r"        \bottomrule", r"    \end{tabular}}", rf"    \label{{{table.label}{label_suffix}}}", r"\end{table*}", ""]
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
    ("wan", r"\textit{ESM components: WE\,+\,\method, " + WE_STEPS + r" steps, control $m$}", (
        AblRow(r"Uniform $t_s{=}<knob>$", r"--", AUTO, reference=True, short=r"Uniform $t_s{=}<knob>$"),
        # R71 (batch r71.txt, tag r71): A1-A4 and A0 at s_min 1.
        AblRow(r"w/o divergence", r"$\mathbf S$", "r71_wan_nodiv_s1_correct", short=r"$\mathbf S$ only"),
        AblRow(r"DINO only", r"$\mathbf S\odot\mathbf D^{\mathrm{sem}}$", "r71_wan_dino_s1_correct",
               short=r"$\mathbf S\odot\mathbf D^{\mathrm{sem}}$"),
        AblRow(r"LPIPS only", r"$\mathbf S\odot\mathbf D^{\mathrm{per}}$", "r71_wan_lpips_s1_correct",
               short=r"$\mathbf S\odot\mathbf D^{\mathrm{per}}$"),
        AblRow(r"w/o support", r"$\mathbf D$", "r71_wan_nosupp_s1_correct", short=r"$\mathbf D$ only"),
        AblRow(r"\textbf{Full}", r"$\mathbf S\odot\mathbf D$", "r71_wan_max_s1_correct", ours=True,
               short=r"$\mathbf S\odot\mathbf D$ (\textbf{full})"),
    )),
    ("se", r"\textit{Control assignment: SE\,+\,\method, 20 steps, control $\rho$}", (
        AblRow(r"Uniform $\rho{=}6$", r"--", "r59_r59_n20_rho6_vp", reference=True, short=r"Uniform $\rho{=}6$"),
        # R68 B1, scored 2026-11-13 (batch r68.txt: r68_max_linrho_full).
        AblRow(r"Linear", r"$\rho_i{=}\rho^{\min}{+}(\rho^{\max}{-}\rho^{\min})\,\mathrm{ESM}_i$", "r68_r68_max_linrho_full",
               short=r"Linear"),
        AblRow(r"\textbf{Full}", r"analytic, Eq.~\ref{eq:stream_mapping}", "r59_r59_max_full", ours=True,
               short=r"Analytic (\textbf{full})"),
    )),
)


LOWER_BETTER = {"lpips_unedit_part", "mse_unedit_part", "structure_distance", "niqe_target_image"}


def matched(curve: Sequence[Row], xkey: str, ykey: str, x: float) -> Optional[float]:
    """Baseline y at x on the curve: linear interpolation between neighbouring points; the value most favourable to
    the baseline (max, all y used here are higher-is-better) if several segments bracket x; None outside the range.
    The caller guarantees every curve point has both values (no interpolation across a missing point)."""
    pts = [(p[xkey], p[ykey]) for p in curve]
    vals = [y0 + (x - x0) * (y1 - y0) / (x1 - x0)
            for (x0, y0), (x1, y1) in zip(pts, pts[1:])
            if x1 != x0 and min(x0, x1) <= x <= max(x0, x1)]
    if not vals:
        return None
    return min(vals) if ykey in LOWER_BETTER else max(vals)


def deltas(d: Row, curve: Sequence[Row]) -> List[str]:
    """[dAcc @ SSIM, dAcc @ LPIPS, dCLIP-T @ SSIM, dSSIM @ Acc, dMF @ Acc, dAcc @ PSNR, dPSNR @ Acc] (matched);
    'out' outside the curve range, '--' if the row or any curve point lacks a value. CLIP-T = target prompt vs whole
    edited frame (the paper reports CLIP-T, not CLIP-D: user, 2026-11-09); MF = FiVE motion fidelity, read at matched
    alignment."""
    specs = (("ssim_unedit_part", "five_acc", 1e2), ("lpips_unedit_part", "five_acc", 1e2),
             ("ssim_unedit_part", "clip_similarity_target_image", 1.0), ("five_acc", "ssim_unedit_part", 1e2),
             ("five_acc", "motion_fidelity_score", 1e2), ("psnr_unedit_part", "five_acc", 1e2),
             ("five_acc", "psnr_unedit_part", 1.0))   # last two: PSNR-matched, supplementary table only
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
    r"($\Delta$SSIM, $\Delta$MF, $\Delta$PSNR in dB), interpolated linearly between neighbouring sweep points (out: outside the sweep, "
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
        dl = (["--"] * 7 if r.reference else deltas(d, curve)) if d is not None else [PH] * 7
        out.append((r, label, d, dl))
    return out


def knob_ph(label: str) -> str:
    return label.replace("$t_s{=}<knob>$", "$t_s{=}$" + PH + "[x.xx]").replace("<knob>", PH + "[x.xx]")


def ablation_table() -> str:
    """Full ablation (every metric, both blocks, full-width): supplementary material."""
    ncols = 2 + len(ABL_COLS) + 7
    lines = [
        r"% Generated by evaluation/paper_tables.py from the R57 suite _overall tables -- do not edit numbers by hand.",
        r"\begin{table*}[t]",
        r"    \centering",
        rf"    \caption{{{ABL_CAPTION}}}",
        r"    \resizebox{0.99\textwidth}{!}{",
        r"    \begin{tabular}{ll|ccc|cccc|c|ccccccc}",
        r"        \toprule",
        r"        \multirow{2}{*}{\textbf{Variant}} & \multirow{2}{*}{\textbf{Configuration}} & \multicolumn{3}{c|}{\textbf{Background Preservation}} & \multicolumn{4}{c|}{\textbf{Text Alignment}} & \textbf{Temp.} & \multicolumn{7}{c}{\textbf{vs.\ uniform sweep, same steps}} \\",
        r"        & & PSNR$\uparrow$ & LPIPS$_{\times 10^3}\!\downarrow$ & SSIM$_{\times 10^2}\!\uparrow$ & FiVE Acc$\uparrow$ & CLIP-T$\uparrow$ & EditRwd.$\uparrow$ & PSC$_{\times 10^2}\!\uparrow$ & MF$_{\times 10^2}\!\uparrow$ & $\Delta$Acc@SSIM & $\Delta$Acc@LPIPS & $\Delta$CLIP-T@SSIM & $\Delta$SSIM@Acc & $\Delta$MF@Acc & $\Delta$Acc@PSNR & $\Delta$PSNR@Acc \\",
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
    Col("ssim_unedit_part", 1e2, 2, True),
    Col("lpips_unedit_part", 1e3, 1, False),
    Col("five_acc", 1e2, 2, True),
    Col("clip_similarity_target_image", 1.0, 2, True),
    Col("motion_fidelity_score", 1e2, 2, True, placeholder=not MF_READY),
)
SHORT_DELTAS = (0, 3)   # indices into deltas(): dAcc @ matched SSIM, dSSIM @ matched Acc (PSNR dropped, user 2026-11-13)
SHORT_HEADER = (
    r"        & SSIM$_{\times 10^2}\!\uparrow$ & LPIPS$_{\times 10^3}\!\downarrow$ & Acc$\uparrow$ & CLIP-T$\uparrow$ & "
    r"MF$_{\times 10^2}\!\uparrow$ & $\Delta$Acc & $\Delta$SSIM \\"
)
DELTA_CAPTION = (
    r"$\Delta$: gain over the uniform-control sweep at the same step count, in FiVE-Acc at matched SSIM and in SSIM "
    r"at matched FiVE-Acc, interpolated linearly between neighbouring sweep points (out: outside the sweep, not "
    r"extrapolated). All metrics are in the supplementary material."
)
SHORT_TABLES = (
    ("tab_abl_esm.tex", "tab:abl_esm", "wan", r"\textbf{ESM}",
     r"\textbf{ESM components} (WE\,+\,\method, " + WE_STEPS + r" steps, FiVE-Bench-C). $\mathbf S$: support; "
     r"$\mathbf D^{\mathrm{sem}}$, $\mathbf D^{\mathrm{per}}$: normalized DINOv3 and LPIPS divergences (Eq.~\ref{eq:div}). "
     + DELTA_CAPTION),
    ("tab_abl_assign.tex", "tab:abl_assign", "se", r"\textbf{Assignment}",
     r"\textbf{Control assignment} (SE\,+\,\method, image-guided, Self-Forcing, 20 steps, FiVE-Bench-C). Linear: "
     r"$\rho_i{=}\rho^{\min}{+}(\rho^{\max}{-}\rho^{\min})\,\mathrm{ESM}_i$; analytic: Eq.~\ref{eq:stream_mapping}. "
     r"$\Delta$ as in Tab.~\ref{tab:abl_esm}."),
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
    # The uniform reference row is not printed in the main paper (user, 2026-11-12); the deltas still use its curve.
    for r, _, d, dl in block_rows(host, [r for r in rows if r.short]):
        if r.reference:
            continue
        if r.ours:
            lines.append(r"        \midrule")
        vals = [fmt(d.get(c.key) if d else None, c) for c in SHORT_COLS]
        lines.append(f"        {knob_ph(r.short)} & " + " & ".join(vals + [dl[i] for i in SHORT_DELTAS]) + r" \\")
    lines += [r"        \bottomrule", r"    \end{tabular}}", rf"    \label{{{label}}}", r"\end{table}", ""]
    return "\n".join(lines)


# Gains over each editor's equal-step uniform sweep (Sec. 4.2, Fig. tradeoff), for every editor/mode with a sweep.
# Columns fixed by paper_strategy.md section 3 (alignment at matched PSNR and LPIPS, preservation at matched Acc,
# motion always) + SSIM, the axis of Fig. tradeoff; rows = every configuration we have a sweep for (none left out).
RHOS_SWEEP = ("2", "3", "4", "6", "8", "10", "20", "50")
MATCHED_ROWS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    (r"WE, text (Wan)", "r71_wan_max_s1_correct", CURVES["wan"]),
    (r"SE, text (SF)", "r60_r60_max_novp_correct", tuple(f"r60_r60_n20_rho{r}_correct" for r in RHOS_SWEEP)),
    (r"SE, image (SF)", "r59_r59_max_full", tuple(f"r59_r59_n20_rho{r}_vp" for r in RHOS_SWEEP)),
    (r"SE, image (LL)", "r61_r61_max_ll", tuple(f"r61_r61_ll_rho{r}_n20" for r in RHOS_SWEEP)),
)
MATCHED_SPECS = (("ssim_unedit_part", "five_acc", 1e2), ("lpips_unedit_part", "five_acc", 1e2),
                 ("five_acc", "ssim_unedit_part", 1e2), ("five_acc", "lpips_unedit_part", 1e3),
                 ("five_acc", "motion_fidelity_score", 1e2))
MATCHED_CAPTION = (
    r"\textbf{Spatial \vs global control.} Gain of \method over the uniform sweep of the same editor's global "
    r"control at the same number of solver steps (Fig.~\ref{fig:tradeoff}): FiVE-Acc points at matched background "
    r"SSIM or LPIPS, and SSIM ($\times 10^2$), LPIPS ($\times 10^3$, lower is better) or MF ($\times 10^2$) at "
    r"matched FiVE-Acc, interpolated linearly between "
    r"neighbouring sweep points (out: outside the sweep, not extrapolated). FiVE-Bench-C, 415 pairs."
)


def matched_table() -> str:
    lines = [
        r"% Generated by evaluation/paper_tables.py from the R57 suite tables -- do not edit numbers by hand.",
        r"\begin{table}[t]",
        r"    \centering",
        rf"    \caption{{{MATCHED_CAPTION}}}",
        r"    \resizebox{\columnwidth}{!}{",
        r"    \begin{tabular}{l|cc|ccc}",
        r"        \toprule",
        r"        \multirow{2}{*}{\textbf{Editor}} & \multicolumn{2}{c|}{$\Delta$FiVE-Acc at matched} & "
        r"\multicolumn{3}{c}{at matched FiVE-Acc} \\",
        r"        & SSIM & LPIPS & $\Delta$SSIM$\uparrow$ & $\Delta$LPIPS$\downarrow$ & $\Delta$MF$\uparrow$ \\",
        r"        \midrule",
    ]
    for label, ours, stems in MATCHED_ROWS:
        curve = [load(s) for s in stems]
        d = load(ours)
        if d is None or any(c is None for c in curve):
            cells = [PH] * len(MATCHED_SPECS)
        else:
            cells = []
            for xkey, ykey, scale in MATCHED_SPECS:
                if ykey == "motion_fidelity_score" and not MF_READY:
                    cells.append(PH)
                    continue
                base = matched(curve, xkey, ykey, d[xkey])
                v = (d[ykey] - base) * scale if base is not None else None
                cells.append("out" if v is None else ("0.00" if abs(v) < 0.005 else f"{v:+.2f}"))
        lines.append(f"        {label} & " + " & ".join(cells) + r" \\")
    lines += [r"        \bottomrule", r"    \end{tabular}}", r"    \label{tab:matched}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path, default=Path("."), help="where the .tex files are written")
    args = ap.parse_args()
    outputs = ([(t.filename, main_table(t, caption_extra=MAIN_CAPTION_EXTRA)) for t in MAIN_TABLES]
               + [("supp_" + t.filename, main_table(t, FULL_COLS, FULL_HEADER, FULL_SPEC, label_suffix="_full",
                                                    deltas_rows=False)) for t in MAIN_TABLES]
               # tab_matched.tex dropped (user 2026-11-13): the dAcc @ matched SSIM is an arrow in Fig. tradeoff instead
               + [(t[0], short_table(*t)) for t in SHORT_TABLES]
               + [("supp_tab_ablation.tex", ablation_table())])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in outputs:
        path = args.out_dir / name
        path.write_text(text)
        print(f"[paper_tables] wrote {path}")


if __name__ == "__main__":
    main()
