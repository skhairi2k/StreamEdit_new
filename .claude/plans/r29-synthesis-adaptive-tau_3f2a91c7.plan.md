---
name: R29 — Synthesis-Driven Adaptive Tau
overview: "Measure SYNTHESIS DEMAND -- how much of an edit's content has no counterpart in the source -- and test whether it orders edit classes the way depth displacement provably cannot. R27 is blocked because a depth delta measures how far depth MOVED, which for an addition is bounded by the added object's own extent: additions measured BELOW swaps (0.906 vs 1.346) while the required tau ordering puts them above, and budget-linear is monotone so no remapping can invert an order. R29 measures a different quantity. For dense DINO ViT-B/8 patch features of the source first frame and the Qwen anchor, N_fwd(p) = 1 - max_q cos(F_edit(p), F_src(q)) is content that had to be INVENTED and N_bwd(q) = 1 - max_p cos(F_src(q), F_edit(p)) is content that had to be ERASED, each maximised over the WHOLE other image so a patch scores high only when nothing anywhere resembles it -- which is blind to displacement by construction. The task is VALIDATION ONLY: it emits no tau map and renders nothing. It computes N on the same 22 pairs as R25 and R27, selects an aggregation from a small family by criteria fixed in advance, and scores the resulting signal against the expected semantic ordering (colour < material < addition < form swap < removal) alongside R27's depth and R25's IoU. The pre-registered prediction is that N will NOT put add/remove above swap, because a swap fires both directions while an addition fires one; the asymmetry |N_fwd - N_bwd| is therefore computed and scored in the same pass, so the fallback is named before the result is seen rather than after."
task_id: R29
todos:
  - id: write-novelty
    content: "evaluation/r29_novelty.py -- per case, the bidirectional correspondence residual. No video generation, no tau. Load DINO ViT-B/8 from the torch hub cache (`torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')`, verified to resolve offline from ~/.cache/torch/hub; 85.8 M params, 768-dim, patch 8). Extract dense patch tokens for the source first frame and the anchor on the SHARED ANCHOR GRID 480x832 (R27's convention -- the pipeline builds the anchor via transforms.Resize((480,832)), an anisotropic squeeze, so resampling to the source's 864x480 would re-introduce a 3.7% warp), giving 60x104 = 6240 tokens per image. Use the KEY facet of the last attention block rather than the final output tokens: keys are the standard choice for semantic correspondence and are less dominated by the CLS objective. L2-normalise, then compute the full 6240x6240 cosine matrix (trivial on GPU) and take row/column maxima: N_fwd(p) = 1 - max_q cos(F_edit(p), F_src(q)), N_bwd(q) = 1 - max_p cos(F_src(q), F_edit(p)). Maximise over the WHOLE other image -- the locked choice -- so a patch is novel only if nothing anywhere resembles it, which makes background inside the edit region score ~0 because it matches background elsewhere. Read the edit region from R25's frame-0 masks (evaluation/figures/r25_iou_masks/{case_id}.npz, union M_src | M_edit) downsampled to the token grid with area-mean then thresholded at 0.5. Emit ALL candidate aggregations in one pass so the selection step needs no re-run: for each of N_fwd, N_bwd, max(N_fwd,N_bwd), N_fwd+N_bwd and |N_fwd-N_bwd|, the WHOLE-FRAME top-n mean with n = ceil(0.005 * 6240) = 32 (the PRIMARY -- a constant-size pool, so the percentile is identical across cases and no mask enters the magnitude), plus the region mean, region median, top-20%-of-region mean, the region mean after a 1-TOKEN EROSION of the region (all four region-relative ones emitted to EXHIBIT the dilution, not to be adopted: R = M_src | M_edit holds the unchanged source object too, so the novel fraction spans 4.3% for the hat to 63% for the flamingo) (boundary-token control: a region of k tokens carries ~3.5*sqrt(k) boundary tokens whose patches straddle edited and unedited content, so the boundary FRACTION ~3.5/sqrt(k) is ~20% at 312 tokens against ~8% at 1872 -- a systematic size dependence the region-mean derivation does not cover), and R27's winner (region median x coverage, coverage = min(1, |region tokens| / n_ref), emitted for comparability only). Columns: case_id, video_name, edit_type, n_tokens_region, region_frac, plus one column per (quantity, aggregation) pair. **Also emit the sorted top-200 token values of each quantity** (as a separate npz keyed by case_id, not in the CSV), so `n` can be swept downstream without recomputing features -- `n` enters only the aggregation, which happens in r29_compare.py. Optional --viz_dir writes a per-case 2x3 panel (source / anchor / N_fwd / N_bwd / |N_fwd-N_bwd| / region overlay). HF_HUB_OFFLINE=1 and TORCH_HOME set; nothing downloads."
    status: completed
  - id: write-compare
    content: "evaluation/r29_compare.py -- the selection and the three-way comparison, in that order and with the criteria applied in the order written here so the choice cannot be fitted to the outcome. (0) SWEEP n for the whole-frame top-n slice over {10, 20, 32, 50}, rejecting any n > 49 (the smallest novel object, 0091_A_hawk's hawk at 3122 px), and read the top-200 npz rather than recomputing. Report how much the ranking moves across admissible n: if the Spearman against the expected ordering is stable, n is not load-bearing and that fact should be stated; if it swings, n is a second min_effect-style hidden parameter and must be reported with the result. (1) SELECT the aggregation: for each of the four aggregations of the primary quantity, compute Spearman rho against the expected semantic ordering (colour 1 < material 2 < addition 3 < form swap 4 < removal 5, assigned from edit SEMANTICS not FiVE type codes -- 0057_dog -> robotic dog and 0074_rhino -> robotic dinosaur are material, not swaps), the Pearson correlation with edited region area computed **WITHIN the 12 form swaps** (not across all 22 -- class and area are confounded there, and the across-class figure flatters the signal: R27's depth reads +0.188 across all 22 but +0.230 within swaps), and the colour/non-colour separation. Rank by Spearman; DISQUALIFY any aggregation whose within-swap |corr(area)| exceeds R27's +0.230, since a signal more area-driven than the one it replaces cannot be sold as surface-independent; break remaining ties by |corr(area)|. (2) SCORE every quantity (N_fwd, N_bwd, max, sum, asymmetry) under the selected aggregation, so the pre-registered prediction that add/remove will NOT exceed swap is tested explicitly. (3) COMPARE against the two existing signals on the same 22 cases: R27's depth D_norm (evaluation/csv/r27_depth.csv) and R25's IoU (evaluation/csv/r25_iou.csv), one Spearman per signal plus per-class means. Emits evaluation/csv/r29_signals.csv (per-case, every signal side by side) and prints the ranking table. Must NOT write a tau map -- R29 is validation only."
    status: pending
  - id: write-figures
    content: "evaluation/r29_figures.py -- two pages. (a) r29_class_order.pdf: per-class strip/box plot of each candidate signal (R25 IoU, R27 depth, R29 N variants) against the expected ordering, one row per signal, so the question the task exists to answer -- does any signal put add/remove above swap -- is answerable at a glance. (b) r29_scatter.pdf: R29 novelty against R27 depth, one marker per edit class, with the 22 case labels. If the two signals rank cases the same way, R29 measured an expensive restatement of R27 and the task ends there; the interesting outcome is the off-diagonal cases, especially the three additions."
    status: pending
steps:
  - id: novelty
    type: local
    command: |
      source ~/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate streamgve
      TORCH_HOME=~/.cache/torch HF_HUB_OFFLINE=1 python evaluation/r29_novelty.py \
        --cases evaluation/cases.json \
        --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \
        --anchor_root /projects/dataggen/outputs/five_bench/anchors \
        --mask_dir evaluation/figures/r25_iou_masks \
        --facet key --layer -1 \
        --viz_dir evaluation/figures/r29_novelty_panels \
        -o evaluation/csv/r29_novelty.csv
    output_paths:
      - evaluation/csv/r29_novelty.csv
      - evaluation/figures/r29_novelty_panels/
    status: completed
    completed_at: 2026-08-31
    job_id: "971472"

  - id: check-novelty
    type: manual
    wait_for: novelty
    check_hint: "Four gates, read the panels not just the CSV. (1) SANITY OF THE CORRESPONDENCE: on a case with NO edit content change, N should be near 0 almost everywhere. 0058_boat (white -> pink fishing boat) is the cleanest probe -- if its region N_fwd is not near the frame minimum, the features are colour-dominated rather than semantic and the whole measure is compromised. Note DINO features ARE somewhat colour-sensitive, so expect small but non-zero. (2) THE ADDITION PROBE, the reason this task exists: 0011_lucia_e5 (add a dog) and 0007_guitar-violin (add a hat) must show N_fwd HIGH and localised on the added object in the panel, and N_bwd LOW -- that asymmetry is the mechanism the whole design rests on. If N_fwd on the dog is not clearly elevated, the correspondence is matching the dog to something spurious in the source and the measure fails on its motivating case. (3) THE REMOVAL PROBE, and it is what justifies keeping both directions: 0042_gym-ball must show the mirror image, N_bwd high on the vacated ball region and N_fwd LOW -- low because the anchor shows background there, which matches background elsewhere in the source. If N_bwd does not fire on this case, a one-directional measure would have sufficed and the design is more complex than it needs to be; if it does, that single case is the entire evidence for the second direction, so record it explicitly (n = 1). Note R25's M_edit is EMPTY by construction for type 6, so the union region is M_src = where the ball WAS -- which is correct for N_bwd but means N_fwd has no region of its own. (4) LOCALISATION: the high-N tokens should fall inside the edit region. Report the fraction; there is no threshold to tune here, so unlike R27 this is pure diagnosis. Expect the two known-bad R27 cases to reappear: 0016_horsejump-high (R25's mask under-covers the real change) and 0017_kid-football (mask is the cap alone, 4.9% of frame)."
    status: pending

  - id: compare
    type: local
    wait_for: check-novelty
    command: |
      source ~/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate streamgve
      python evaluation/r29_compare.py \
        --novelty_csv evaluation/csv/r29_novelty.csv \
        --depth_csv evaluation/csv/r27_depth.csv \
        --iou_csv evaluation/csv/r25_iou.csv \
        --primary max --n_sweep 10,20,32,50 \
        -o evaluation/csv/r29_signals.csv
    output_paths:
      - evaluation/csv/r29_signals.csv
    status: pending

  - id: figures
    type: local
    wait_for: compare
    command: |
      source ~/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate streamgve
      python evaluation/r29_figures.py \
        --signals_csv evaluation/csv/r29_signals.csv \
        --out_dir evaluation/figures
    output_paths:
      - evaluation/figures/r29_class_order.pdf
      - evaluation/figures/r29_scatter.pdf
    status: pending

  - id: verdict
    type: manual
    wait_for: figures
    check_hint: "The task exists to answer ONE question, so answer it first and plainly. (a) DOES ANY QUANTITY PUT ADD/REMOVE ABOVE SWAP? The target ordering is colour/material < swap < add/remove; R27's depth measured add BELOW swap and that is why it is blocked. State the per-class means for N_fwd, N_bwd, max, sum and asymmetry, and say which -- if any -- achieves it. The PRE-REGISTERED PREDICTION was that the magnitude would NOT, because a swap fires both correspondence directions while an addition fires one; record whether the prediction held, because a prediction that survives is worth far more than one written afterwards. (b) If the ASYMMETRY |N_fwd - N_bwd| is the quantity that separates them, say so explicitly and note the cost: routing tau on an asymmetry rather than a magnitude is a much harder claim to defend, and it needs a reason why one-sidedness should imply detachment. (c) Spearman for all three signals side by side (R25 IoU, R27 depth, R29 novelty) against the expected semantic ordering, with R27's best of +0.577 (p 0.005) as the number to beat. (d) Does R29 rank the cases DIFFERENTLY from R27 (r29_scatter.pdf)? If the two agree case-for-case, R29 measured an expensive restatement of R27 and neither unblocks the task -- that is a real and reportable outcome. (e) THE DECISION: does this signal unblock R27, and on which quantity? If yes, the follow-up is a tau map and an arm. If no, state what the next signal would have to capture and whether the target ordering itself should be revisited -- the possibility that additions do NOT need more detachment than swaps, and that the 0011_lucia_e5 failure had another cause, must stay live. (f) SURFACE-INDEPENDENCE, R27's founding claim against IoU: report `corr(signal, area)` WITHIN the 12 swaps for every candidate, against R27 depth's +0.230 and R25 IoU's value on the same subset. Note explicitly that the across-all-22 figure (+0.188 for depth) UNDERSTATES the bias because class and area are confounded, and do not quote it as the headline. (g) Caveats: n = 3 additions and n = 1 removal, so every class statement here is weak evidence and must be labelled as such."
    sets_status: analyzed
    status: pending
isProject: true
---

# R29: Synthesis-Driven Adaptive Tau

## Context

R27 measures how far depth *moved* between the source first frame and the Qwen anchor. That quantity is bounded, for an addition, by the added object's own extent — a hat displaces almost nothing — so additions measured **below** swaps (0.906 vs 1.346) while the required `tau` ordering puts them above. Budget-linear interpolation is monotone, so no choice of `tau_min`, `tau_max`, curvature, `D_lo` or `D_hi` can invert an order. R27 is blocked at `taumap` for that reason.

R29 measures a different quantity: **synthesis demand**, how much of an edit's content has no counterpart in the source at all. For dense DINO ViT-B/8 patch features,

$$N_{\rightarrow}(p) = 1 - \max_{q} \cos\bigl(F_{\text{edit}}(p), F_{\text{src}}(q)\bigr), \qquad
N_{\leftarrow}(q) = 1 - \max_{p} \cos\bigl(F_{\text{src}}(q), F_{\text{edit}}(p)\bigr)$$

with each maximum taken over the **whole** other image. A patch scores high only when nothing anywhere in the other image resembles it — which is blind to displacement by construction, and is exactly the property the depth signal lacks.

**Done when:** the verdict states, for the 22 shared cases, whether any quantity derived from `N` puts add/remove above swap; whether the pre-registered prediction that it would not held; and how R29's Spearman against the expected semantic ordering compares with R27's depth (+0.577, p = 0.005) and R25's IoU.

**This task is validation only.** It emits no `tau` map and renders nothing.

## Execution steps

| # | Step id | Type | What it does | sets_status |
|---|---------|------|--------------|-------------|
| — | *(prep)* | todos | `r29_novelty.py`, `r29_compare.py`, `r29_figures.py` | — |
| 1 | `novelty` | local (GPU) | `N_fwd`/`N_bwd` + all candidate aggregations → `r29_novelty.csv` + panels | — |
| 2 | `check-novelty` | manual | Correspondence sanity, addition probe, removal probe, localisation | — |
| 3 | `compare` | local | Select aggregation, score every quantity, three-way signal comparison | — |
| 4 | `figures` | local | Class-ordering plot, R29-vs-R27 scatter | — |
| 5 | `verdict` | manual | Does any quantity order add/remove above swap? Decision on R27 | `analyzed` |

Invoke with `/run-step R29 <step-id>`, e.g. `/run-step R29 novelty`. Bare `/run-step R29` takes the first `pending` step whose `wait_for` is satisfied.

## Decisions

| Question | Locked choice | Why |
|---|---|---|
| Quantity | Two per-token MAPS on the 60×104 grid: `N_fwd(p) = 1 − max_q S[p,q]` (one value per **anchor** token, row-wise) and `N_bwd(q) = 1 − max_p S[p,q]` (one value per **source** token, column-wise), where `S = F_edit F_srcᵀ`; plus the derived `max`, `sum` and `\|N_fwd − N_bwd\|` | Displacement-blind by construction, which is what R27 lacked. ⚠ `N_fwd` indexes the ANCHOR grid and `N_bwd` the SOURCE grid — same 60×104 shape, and token `p` is the same spatial LOCATION in both, but not the same content. Combining them is therefore a **positional** pairing, reading as "at location `x`, either something new appeared or something disappeared" — not a correspondence pairing |
| Why BOTH directions | `N_fwd` alone cannot see a removal | Remove the gym ball and the anchor shows background, which matches background elsewhere in the source, so `N_fwd ≈ 0`; only `N_bwd` fires, on the vacated region. With removals targeted at `tau` 30–50, a one-directional measure would floor the one removal case. Mechanistically both also argue for detachment: the blend pulls generation toward the source latent, so content that must be **erased** is content the anchor actively fights, exactly as much as content that must be **created** |
| Correspondence test | **Max similarity only** — cycle consistency considered and rejected | `N_fwd`/`N_bwd` are the forward/backward occlusion test of the optical-flow literature: `N_bwd` high = occlusion (content disappeared), `N_fwd` high = disocclusion (content appeared). That literature's stronger test is **mutual nearest neighbour** (`p → q* → p*`, novel if `p* ≠ p`), which handles displacement correctly since it tests identity not position. **Rejected because it fails in textureless/repetitive regions**: many grass patches are near-identical, so the cycle lands on a different-but-equally-good patch and reports false novelty. This benchmark's backgrounds are grass (`lucia`), sky and water (`planes-water`), open sky (`hawk`) — the same cases that broke R27's σ estimate. Max similarity is immune: a grass patch has an excellent match, so `N_fwd ≈ 0` |
| Search domain | **Whole other image** | Strictest reading of "no counterpart"; background inside the edit region correctly scores ≈ 0 because it matches background elsewhere. No margin parameter to justify |
| Features | DINO ViT-B/8, **key facet, last block** | Cached at `~/.cache/torch/hub` (85.8 M params, 768-dim, patch 8) so it runs offline; keys are the standard choice for semantic correspondence and are less dominated by the CLS objective |
| Grid | Anchor grid 480×832 → 60×104 = 6240 tokens | R27's convention. The pipeline builds the anchor via `transforms.Resize((480,832))`, an anisotropic squeeze, so resampling to 864×480 would re-introduce a 3.7% warp |
| Normalisation | **None** | Cosine distance is already bounded in [0,2] and unitless — R27's `÷ IQR` existed only because a depth residual has arbitrary per-image units |
| Erosion | **Not ported by default** | R27 needed it for sub-pixel boundary halo at depth discontinuities; DINO tokens are 8× downsampled, so a 1-px shift is ⅛ of a token. To be confirmed at `check-novelty`, not assumed |
| Boundary-token control | Moot for the primary; retained on the region-relative columns | The `O(k^{-1/2})` boundary-fraction argument applies to aggregation over a region of `k` tokens. The whole-frame top-`n` slice does not aggregate over a region at all, so the boundary term does not arise — it is kept only on the region-relative columns that exist to exhibit dilution |
| Surface-independence | **Fixed-count top slice over the WHOLE FRAME** — `S = mean of the largest n of all 6240 tokens`. `n` is **swept in `compare`, not fixed here** | ⚠ Region-relative aggregation is surface-DEPENDENT and was nearly adopted in error: `R = M_src ∪ M_edit` holds the unchanged source object too, so the novel fraction of `R` spans 15× within the addition class alone (hat 4.3%, dog 46%, flamingo 63%) — worst for exactly the class R29 serves. A whole-frame pool is a CONSTANT 6240 tokens, so the percentile is identical across cases by construction and any fully-novel object above `n` tokens fills the slice to the same level |
| Choice of `n` | Swept over {10, 20, 32, 50} in `compare`; **hard admissibility bound `n ≤ 49`** | ⚠ The 0.005 fraction was inherited from R27, where it was swept on a DEPTH residual against a different binding constraint, and has no independent justification here. The constraint for `N` is that `n` must not exceed the smallest novel object, or the slice fills with non-novel tokens and dilutes. Measured per direction: `N_bwd` is bound by `0091_A_hawk` (the hawk, 3122 px = **49 tokens**), `N_fwd` by `0007_guitar-violin` (the added hat, ~4448 px = **70 tokens**). R27's carried-over `n = 32` leaves only a **1.5× margin** on the binding case. Against that, small `n` is noisy — averaging 12 tokens is a poor estimate — so the trade is real and is settled by the same pre-registered criteria as the aggregation, not by assertion |
| Surface-independence TEST | `corr(signal, area)` computed **WITHIN the 12 form swaps**, not across all 22 | Measured: across all 22 the depth signal reads +0.188, but within the swaps alone — one homogeneous class, 4.8× area spread — it reads **+0.230**. Class and area are confounded across the full set (swaps are larger on average), so the across-class figure FLATTERS the signal. +0.230 is the number R29 must beat |
| Aggregation | **Selected at `compare`**, not chosen now | Primary is the whole-frame top-`n` slice (see the Surface-independence row). The region-relative family (region mean, region median, top-20% of region, region mean after a 1-token erosion) is still emitted **as evidence of the dilution effect, not as candidates to adopt** — the hat-vs-flamingo spread should be visible in those columns and is worth showing. R27's median × coverage is emitted for comparability only. All in the same pass. Criteria fixed **in advance**: Spearman vs expected ordering, then `\|corr(area)\|` as tie-break |
| Expected ordering | colour 1 < material 2 < addition 3 < form swap 4 < removal 5 | Assigned from edit **semantics**, not FiVE type codes: `0057_dog` → robotic dog and `0074_rhino` → robotic dinosaur are material, not swaps |
| Region | R25's frame-0 union mask, area-downsampled to the token grid, threshold 0.5 | Already on disk; no new segmentation. Type 6's `M_edit` is empty by construction, so the region is where the ball **was** — correct for `N_bwd` |
| Pre-registered prediction | `N` magnitude will **NOT** put add/remove above swap | A swap fires both directions, an addition one. Written before measuring so the asymmetry fallback is not a post-hoc rescue |
| Scope | Validation only — **no tau map, no render** | R27's block was caused by committing to a signal before checking its ordering; this task exists not to repeat that |
| Cases | The same 22 pairs of `evaluation/cases.json` | Direct comparability with R27 depth and R25 IoU |
| Out of scope | `tau` mapping, arms, evaluation, any render | Follow-up, and only if the verdict says the ordering works |

## Step commands

### novelty

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate streamgve
TORCH_HOME=~/.cache/torch HF_HUB_OFFLINE=1 python evaluation/r29_novelty.py \
  --cases evaluation/cases.json \
  --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \
  --anchor_root /projects/dataggen/outputs/five_bench/anchors \
  --mask_dir evaluation/figures/r25_iou_masks \
  --facet key --layer -1 \
  --viz_dir evaluation/figures/r29_novelty_panels \
  -o evaluation/csv/r29_novelty.csv
```

### compare

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate streamgve
python evaluation/r29_compare.py \
  --novelty_csv evaluation/csv/r29_novelty.csv \
  --depth_csv evaluation/csv/r27_depth.csv \
  --iou_csv evaluation/csv/r25_iou.csv \
  --primary max --n_sweep 10,20,32,50 \
  -o evaluation/csv/r29_signals.csv
```

### figures

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate streamgve
python evaluation/r29_figures.py \
  --signals_csv evaluation/csv/r29_signals.csv \
  --out_dir evaluation/figures
```

## Pipeline

```mermaid
flowchart TD
  SRC[FiVE images/&lt;video&gt;/00001.jpg] --> NOV[r29_novelty.py]
  ANC[anchors/edit&lt;T&gt;/&lt;video&gt;.png] --> NOV
  MSK[r25_iou_masks/*.npz<br/>edit region] --> NOV
  DINO[DINO ViT-B/8<br/>key facet, last block] --> NOV
  NOV --> NCSV[csv/r29_novelty.csv<br/>N_fwd N_bwd max sum asym<br/>x 4 aggregations]
  NOV --> NPAN[figures/r29_novelty_panels/]
  NCSV --> CMP[r29_compare.py<br/>select aggregation, score]
  R27[csv/r27_depth.csv] --> CMP
  R25[csv/r25_iou.csv] --> CMP
  CMP --> SIG[csv/r29_signals.csv]
  SIG --> FIG[r29_figures.py]
  FIG --> F1[figures/r29_class_order.pdf]
  FIG --> F2[figures/r29_scatter.pdf]
```

## Code to touch

| File | Status | Change |
|---|---|---|
| `evaluation/r29_novelty.py` | **written, smoke-tested 4/4** | DINO ViT-B/8 key tokens on the anchor grid → 6240×6240 cosine → row/col maxima → 5 quantities × 6 aggregations + top-200 npz for the `n` sweep. Flags: `--facet`, `--layer`, `--top_n`, `--viz_dir`, `--dump_top` |
| `evaluation/r29_compare.py` | new | Select aggregation by pre-registered criteria, score every quantity, three-way Spearman vs R27 depth and R25 IoU |
| `evaluation/r29_figures.py` | new | `r29_class_order.pdf`, `r29_scatter.pdf` |
| `evaluation/r27_depth_delta.py` | **unchanged** | Reused only as a data source (`r27_depth.csv`) and for `load_frame0_masks` / `resize_nearest` |
| `evaluation/run_fivebench.py` | **unchanged** | R29 renders nothing |

Feature extraction and correspondence, for reference:

```python
# keys of the last attention block, the standard facet for semantic correspondence
feats = model.get_intermediate_layers(img, n=1)[0][:, 1:]   # drop CLS -> (1, 6240, 768)
F = torch.nn.functional.normalize(feats[0], dim=-1)
S = F_edit @ F_src.T                       # (6240, 6240) cosine, both L2-normalised
N_fwd = 1.0 - S.max(dim=1).values          # per anchor token: nothing in src resembles it
N_bwd = 1.0 - S.max(dim=0).values          # per source token: nothing in edit resembles it
```
