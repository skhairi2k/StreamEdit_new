---
name: R25 — IoU-Driven Adaptive Tau
overview: "Make StreamGVE's Eq. 4 release exponent a per-video quantity driven by how much the edit changes the region the edited content occupies. Ground BOTH the source phrase and the target phrase on BOTH the source first frame and the Qwen anchor with GroundingDINO + SAM2, union the two masks on each side, and take the IoU of the two unions; map the measured IoU range linearly onto tau in [1, 12] (high IoU = small shape change = tau_min = hold source longest; low IoU = large shape change = tau_max = release fast), then render one adaptive arm over the 22 cases of evaluation/cases.json in vp mode against the stored full-bench Eq.4 vp reference. One phrase is grounded per side (revised 2026-08-26 after measurement showed GroundingDINO never abstains); removals are handled by construction — type 6 sets M_edit empty so IoU is exactly 0 — while additions (type 5) are measured identically to a swap, M_edit = m(trg_word, anchor) alone (reverted 2026-09-01 from an earlier version that unioned M_edit with M_src so the addition would read as regional growth; see the Decisions table). tau reaches the model as a per-case `blend_power` override, which IS the Eq. 4 exponent at causal_model.py:334 — the `--rho_min/--rho_max/--edit_field` flags of the earlier R25 V0 per-token path no longer exist (removed from the tree 2026-08-26), so `blend_power` is the only exponent knob and this task needs no pipeline change."
task_id: R25
todos:
  - id: write-iou
    content: "evaluation/r25_iou.py — per case, text-grounded object masks and their IoU, no video generation. SINGLE PHRASE PER SIDE (revised 2026-08-26): M_src = m(src_word, I0) for every type; M_edit = m(trg_word, anchor) for types 1-5, = EMPTY for type 6. IoU = |M_src & M_edit| / |M_src | M_edit|. GroundingDINO (`IDEA-Research/grounding-dino-base`) + SAM2 (`facebook/sam2-hiera-large`), both cached, `--box_thr 0.35`. The earlier symmetric two-phrase union was REMOVED because its founding premise — a phrase naming something absent contributes nothing to that side — is false for this detector: it never abstains. Measured at box_thr 0.25 on this exact case set, all 4 groundings fired on 22/22 cases and the absent phrase scored up to 0.94 ('A dragon' on the unedited cow), above most genuine detections, so no threshold separates them. Only the removal type is handled BY CONSTRUCTION rather than by detection: type 6 sets M_edit empty so IoU is exactly 0 without grounding anything on the anchor. (Revision 2026-09-01: type 5 (addition) was originally ALSO handled by construction — M_edit unioned the anchor-side target mask with M_src, e.g. 0069_car-turn: M_src = SUV, M_edit = SUV | flamingo, so the addition read as regional growth rather than a disjoint object. That union is now REMOVED at the user's request: type 5 is measured exactly like a swap, M_edit = m(trg_word, anchor) alone, no union. See the 'Reverted: type-5 union with M_src' Decisions row for the consequence — IoU_hi in the 2026-08-27 run was set by a type-5 case (0007_guitar-violin, 0.957), so this changes the measured range and therefore the tau of every case, not only the addition ones.) Emits evaluation/csv/r25_iou.csv with case_id, video_name, edit_type, iou, the TWO groundings' box/score/fired flags, both areas, and `iou_gt_src` = IoU(M_src, FiVE-Bench GT `bmasks/{video}/00001.jpg` > 127). Failure rules: src_word not firing on the source is always fatal (no M_src); trg_word not firing on the anchor is fatal for types 1-5 (for a swap or addition it would read IoU = 0 and route to tau_max on a detector miss); type 6 grounds nothing on the anchor so it cannot fail that way. Optional --viz_dir writes one 2x2 audit panel per case. HF_HUB_OFFLINE=1: both checkpoints are in ~/.cache/huggingface/hub, nothing downloads."
    status: completed
  - id: write-taumap
    content: "evaluation/r25_tau_map.py — read r25_iou.csv, take IoU_lo = min and IoU_hi = max over the 22 cases, and emit tau = tau_max - (tau_max - tau_min) * (iou - IoU_lo) / (IoU_hi - IoU_lo) with tau_min=2.0, tau_max=10.0. Anchoring on the MEASURED range rather than a nominal [0,1] is the locked choice: a person->lion swap still occupies roughly the same image region, so IoU never approaches 0, and a nominal domain would leave tau_max unreached. Writes evaluation/csv/r25_tau_map.csv (video_name, edit_type, iou, tau) and prints the realized tau spread. Refuses to write if IoU_hi - IoU_lo < 0.05 (a degenerate range means the signal carries nothing and the arm is not worth rendering)."
    status: completed
  - id: plumb-taumap
    content: "evaluation/run_fivebench.py — add `--tau_map` (path to r25_tau_map.csv, default None). Load it into a dict keyed (video_name, edit_type); at the rollout_inference call (:303) pass `blend_power=tau_map[(video_name, args.edit_type)]` when the map is given, else `args.blend_power` unchanged. Hard-error at startup if the map is given but does not cover every pair that survives the --cases_json filter — a silent fallback to 2.0 would render a half-adaptive arm that looks like a result. Print the per-pair tau in the existing [ok] line so the log is self-documenting. `--tau_map` absent => byte-identical to today. There is no other exponent flag to collide with: --edit_field/--rho_min/--rho_max were removed with the R25 V0 per-token path on 2026-08-26."
    status: completed
  - id: write-smoke
    content: "slurm_scripts/five_bench/r25_smoke.sh — three gates. (1) Override reachability: render 0001_bus/edit1 with `--tau_map` forced to tau=2.0 for every case and assert it is sha256-identical to the same clip rendered with plain `--blend_power 2` — this is what proves the map actually reaches blend_power rather than being ignored. (2) Non-degeneracy: the same clip at tau=8 must DIFFER from tau=2 (a map that is read but discarded would pass gate 1 and fail here). (3) Coverage error: an intentionally truncated map must SystemExit rather than render. Cheap — 3 renders of one clip. L40S, --mem=64G, --time=02:00:00."
    status: completed
  - id: write-infer
    content: "slurm_scripts/five_bench/r25_infer.sh — `#SBATCH --array=0-5`, one edit type per task (types 1..6), each loading the model once and running its slice of evaluation/cases.json. `--method r25_adaptive_vp --vp_mode vp --first_frame_edit_dir /projects/dataggen/outputs/five_bench/anchors --cases_json evaluation/cases.json --tau_map evaluation/csv/r25_tau_map.csv`, OUT_ROOT /projects/dataggen/outputs/five_bench/r25_adaptive_tau. L40S, --mem=64G, --time=04:00:00 (22 clips total at R22's measured ~5.3 min/clip is ~2 h for the whole arm, and the largest task is edit2 with 13). Type 3 and 4 tasks carry 2 clips each."
    status: completed
  - id: write-eval
    content: "slurm_scripts/five_bench/r25_eval.sh — single task (not an array; 22 clips at R20's measured ~45 s/clip is well under an hour). **L40S,A100** (retargeted 2026-08-27: H100 unusable, both nodes draining and fully allocated; A100 added for its 11 nodes vs L40S's 5), --exclude=node52, --mem=64G, --time=06:00:00, an explicit `--metrics` list of 9 metrics which OMITS five_acc + motion_fidelity_score{,_edit_part} (config.yaml's `metrics:` key is vestigial -- evaluate.py:200 reads args.metrics, so only the flag works), `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the R21 fix for R20's CoTracker OOMs and niqe failures). evaluate.py over all six edit{T}_FiVE.json with `--cases_json evaluation/cases.json`, csv stem r25_adaptive_vp. No code change to evaluate.py / metrics_calculator.py."
    status: completed
  - id: write-summarize
    content: "evaluation/r25_summarize.py — join the six edit{T}_FiVE_r25_adaptive_vp_frame_stride8.csv against the stored edit{T}_FiVE_r21_ref_vp_frame_stride8.csv on `file_id` per edit type. The join is sound because evaluate.py assigns file_id from `enumerate` over the FULL annotation file and `--cases_json` only `continue`s (evaluate.py:268-272), so ids are stable between a filtered and a full run. Re-average the R21 reference over ONLY the 22 R25 cases — never against its 419-pair mean. Emits evaluation/csv/r25_adaptive.csv: one row per case with iou, tau, both arms' 16 metrics and per-metric deltas, plus an overall-mean row and a per-edit-type mean block."
    status: completed
  - id: write-figure
    content: "evaluation/r25_figures.py — two pages. (a) r25_tau_iou.pdf: the realized mapping, tau against measured IoU, one marker shape per edit type, with the Eq.4 rho=2 line drawn across — shows at a glance whether colour/material (types 3/4) really did land at the tau_min end and object swaps at tau_max. (b) r25_delta.pdf: per-case delta (adaptive minus Eq.4) for clip_similarity_target_image and lpips_unedit_part plotted against |tau - 2|. This is the diagnostic that decides the task: if adaptive helps, the gain must concentrate on the cases whose tau moved FURTHEST from the baseline exponent. A flat cloud means any win is sampling noise, not the IoU signal. (c) ADDED 2026-08-27 at user request: evaluation/figures/r25_grids/{case_id}.pdf, one qualitative page per video, F x 4 -- rows are every 10th RENDERED frame, columns are source / Eq.4 baseline render / IoU mask overlay / adaptive render, with the case's iou and tau in the title. Rows align the source column by NORMALIZED TIME because the VAE changes frame count (80 source vs 69 rendered on 0001_bus), so indexing both by the same integer would drift the source ahead of the renders. The mask column draws M_src red / M_edit blue / intersection white on frame 0 and is blank in later rows -- both masks are frame-0 quantities and painting them over a later frame would imply a per-frame segmentation that was never computed. Needs r25_iou.py --dump_masks (added the same day, inert by default)."
    status: completed
  - id: verdict
    content: "Read r25_adaptive.csv with both figures. Record in daily.md: (a) the measured IoU range and whether it separated types 3/4 from types 1/2 as predicted; (b) whether the adaptive arm beats Eq.4 rho=2 on the 22-case means, and on which metrics; (c) whether the per-case gain correlates with |tau - 2| or is flat. (d) whether 0042_gym-ball read IoU = 0 as designed; if not, whether the anchor PNG shows a phantom detection or an anchor edit that never removed the ball — n=1, so this is the only evidence on whether type 6 is measurable under a phrase-grounded IoU, and it decides whether R26's mask-union route needs to own removals instead. Note explicitly what this design CANNOT answer — with Eq.4 as the only comparator, a win does not distinguish 'adaptivity helps' from 'some constant tau != 2 helps'; the constant-tau endpoint arms and the shuffled-IoU control were scoped out and are the natural follow-up if the arm wins."
    status: completed
  - id: write-control-maps
    content: "evaluation/r25_tau_map.py — add `--assign {iou,reversed,constant}` (default `iou` = today's behaviour, byte-identical). `reversed` keeps the SAME 22 tau values but pairs them with the INVERTED IoU ranking (highest tau to the highest IoU), so colour edits get tau_max and object swaps get tau_min. `constant` writes the realized mean tau (6.57) for every pair. Both emit the same (video_name, edit_type, iou, tau) schema so run_fivebench.py --tau_map needs no change. Reversed is chosen over a random permutation deliberately: one random draw of 22 is noisy and could land near or far from the true ordering by luck, while the inversion is deterministic and maximises the contrast."
    status: completed
  - id: param-arm-scripts
    content: "slurm_scripts/five_bench/r25_infer.sh and r25_eval.sh — take the arm from the environment instead of hardcoding it: `ARM=${R25_ARM:-adaptive}` selecting METHOD (`r25_${ARM}_vp`) and TAU_MAP (`evaluation/csv/r25_tau_map_${ARM}.csv`), defaulting to today's adaptive names so an unset variable reproduces the completed phase-1 run. Everything else — seed 0, anchors, cases_json, sampler defaults, --exclude=node52, the explicit --metrics list — must stay identical across arms, because the whole attribution argument rests on the three arms differing ONLY in the tau assignment."
    status: completed
  - id: write-controls-summarize
    content: "evaluation/r25_controls.py — three-way table: adaptive vs reversed vs constant, all re-averaged over the same 22 cases, plus the stored Eq. 4 reference for context. Report per-metric means and the two contrasts that matter: adaptive − reversed (does the PAIRING matter) and adaptive − constant (does VARYING tau matter at all). Also the per-case paired deltas and a sign test on each contrast, since n=22 and the phase-1 mean hid a 15/7 split. Emits evaluation/csv/r25_controls.csv. WRITTEN 2026-09-01 (was never actually built during phase 2 despite the plan calling for it — caught when `resummarize` was about to run it and the file didn't exist). Joins all three arms + the Eq.4 reference on (video_name, edit_type) directly (not file_id — each arm already covers the identical 22 pairs, hard-verified before joining), reusing r25_summarize.py's edit_type_of/read_metric_csv/load_video_names conventions duplicated per this repo's self-contained-script pattern. Sign test is scipy.stats.binomtest (two-sided, exact, distribution-free — deliberately not a paired t-test, since phase 1's own mean hid a lopsided split), computed per contrast x metric and written to a r25_controls.meta.json sidecar (a scalar p-value doesn't fit the per-case CSV row shape). `--arms` order is load-bearing: contrasts are always arms[0]-arms[1] and arms[0]-arms[2]."
    status: completed
    completed_at: 2026-09-01
  # === PHASE 3: type-5 addition mask fix (added 2026-09-01, user request) ===
  - id: fix-type5-mask
    content: "evaluation/r25_iou.py — drop the type-5 (addition) special case. M_edit is now m(trg_word, anchor) alone for types 1-5, exactly the swap formula; only type 6 (removal) still builds M_edit by construction (empty). Removed: the ADDITION_EDIT_TYPE union branch in measure_case, its dump_masks branch, and the save_panel branch that drew the inherited M_src beneath the added object. Updated: module docstring, the M_edit formula the main() banner prints, and the TRG-NOT-FOUND failure comment (no longer claims a missed addition would read IoU = 1). Motivation is the user's, not a measured failure of the union design — recorded as the 'Reverted: type-5 union with M_src' row in Decisions, together with its consequence: the completed run's IoU_hi (0.957) was set by a type-5 case (0007_guitar-violin), so re-measuring without the union is expected to move IoU_hi and therefore shift the linear tau map for every case, not only the addition ones."
    status: completed
    completed_at: 2026-09-01
  - id: type5-rerun-scope
    content: "Re-derive evaluation/csv/r25_iou.csv, then GATE on whether IoU_hi/IoU_lo moved (check-iou gate 4, added above). If they moved — expected, since IoU_hi was a type-5 value — re-derive ALL THREE tau maps (evaluation/r25_tau_map.py --assign iou/reversed/constant) and re-render ALL THREE arms over the full 22 cases (adaptive, reversed, constant), not just the 3 addition-only clips (0007_guitar-violin, 0069_car-turn, 0011_lucia_e5). This also supersedes the in-flight phase-2 renders at jobs 962628 (reversed) and 962629 (constant): both were built from the pre-fix tau multiset and must not be carried into control-eval / control-summarize once superseded. Only if the range-shift gate finds the endpoints unchanged (e.g. because 0017_kid-football or another case already exceeded the old IoU_hi) would an addition-only re-render of the 3 type-5 clips be sufficient — record which branch applies before submitting any sbatch job. RESOLVED 2026-09-01 (check-reiou): IoU_hi moved 0.957 -> 0.908, full re-render branch confirmed."
    status: completed
    completed_at: 2026-09-01
  - id: fix-tau-mapping-family
    content: "evaluation/r25_tau_map.py — add `--mapping {budget_linear, linear}`, default `budget_linear` (revised 2026-09-01, user decision after being shown the quantified divergence). Reuses r27_tau_map.py's `t_next_schedule`/`budget`/`budget_linear_tau` machinery verbatim (duplicated, not imported, matching this repo's self-contained-script convention) so R25 corrects the same tau-saturation distortion R27 already fixed: tau is an exponent whose effect saturates, so the physically meaningful quantity is the anchoring budget A(tau)=sum t_i**tau over the --step 15 schedule, not tau itself. `iou_to_tau_linear` kept for `--mapping linear` (reproduces the original R25 mapping exactly). `--tau_max` default changed 10.0 -> 20.0, valid only together with budget_linear — quantified before deciding: at [2,20] under the OLD linear-in-tau mapping, mean tau over the 22 post-type5-fix cases would hit 13.39 (dragging the swap-heavy middle of the distribution toward the R21 `zero`-schedule preservation-collapse regime, LPIPS 143 vs `ref_vp` 53.6); under budget_linear the same range gives mean tau 7.57, with the widening concentrated on the already-extreme low-IoU tail (removal, the 2 most disjoint additions) rather than dragging the middle. `--assign reversed/constant` unaffected — both operate on the `tau` column after mapping, verified byte-for-byte: `reversed` keeps an IDENTICAL tau multiset under the new mapping too. Also added a `.meta.json` sidecar (mapping, tau_min/max, step, flow_shift, A(tau_min)/A(tau_max), iou range) mirroring r27_tau_map.py's provenance record, since the CSV's 4 columns alone no longer determine how `tau` was derived. Smoke-tested locally (no GPU needed): `--mapping budget_linear`, `--mapping linear` (legacy), `--assign reversed` (multiset-identical check passed), `--assign constant` (flat-value check passed) all exit 0."
    status: completed
    completed_at: 2026-09-01
steps:
  - id: iou
    type: srun
    command: |
      srun --partition=L40S --gres=gpu:1 --mem=32G --time=01:00:00 --pty \
        env HF_HUB_OFFLINE=1 python evaluation/r25_iou.py \
          --cases evaluation/cases.json \
          --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \
          --anchor_root /projects/dataggen/outputs/five_bench/anchors \
          --box_thr 0.35 \
          --viz_dir evaluation/figures/r25_iou_panels \
          -o evaluation/csv/r25_iou.csv
    output_paths:
      - evaluation/csv/r25_iou.csv
      - evaluation/figures/r25_iou_panels/
    status: completed
    completed_at: 2026-08-26
    job_id: "960251"

  - id: check-iou
    type: manual
    wait_for: iou
    check_hint: "Three gates, all hard (revised 2026-08-26 for the single-phrase design). (1) SEGMENTER TRUST: median iou_gt_src must be >= 0.7 against FiVE-Bench GT bmasks. EXCLUDE any case whose src_word does not name the object FiVE annotated — for those the column measures a disagreement about WHICH object, not mask quality, and a near-zero value is correct rather than a segmenter failure. As of 2026-08-26 that is 0042_gym-ball (src_word names the removed ball, GT marks the man) and 0017_kid-football (src_word names the cap, GT marks the boy). Check the exclusion list against cases.json whenever a src_word changes, and read the remaining rows' minimum, not just the median. (2) GROUNDING SANITY: there are now only TWO groundings per case (src_word on the source, trg_word on the anchor; type 6 has one). Absence is no longer the mechanism, so the question is no longer 'did the right things fail to fire' but 'did the two boxes land on the right objects'. READ THE PANELS IN evaluation/figures/r25_iou_panels/ — each is a 2x2 (source/anchor over M_src and M_edit) and it is the only artefact that shows WHICH object was grounded; a box coordinate cannot. Check especially: any case whose trg box on the anchor sits on the background rather than the edited object. (Revised 2026-09-01: the type-5 panel no longer draws an inherited M_src beneath the added object — M_edit is now just the trg grounding, same panel layout as a swap.) (3) SIGNAL: types 3+4 must sit at visibly higher iou than type 2 — and note the margin, since under the two-phrase design it was only 0.004 and that is not a separation worth routing on. 0042_gym-ball now reads exactly 0 BY CONSTRUCTION and is no longer evidence of anything — do not read it as a passing signal. **(4) RANGE-SHIFT CHECK (added 2026-09-01, mandatory before reusing any downstream tau map):** compare the new IoU_hi/IoU_lo against the values the current r25_tau_map.csv / r25_tau_map_reversed.csv / r25_tau_map_constant.csv were derived from (IoU_hi = 0.957 from 0007_guitar-violin [type 5], IoU_lo = 0.0 from 0042_gym-ball [type 6]). Since type 5 no longer unions M_edit with M_src, its IoU is expected to DROP once measured — if it no longer sets IoU_hi (the next-highest case was 0017_kid-football at 0.908), the normalization range changes, and because the tau map is linear on that range EVERY case's tau shifts, not only the three type-5 cases. Do not proceed to `taumap` believing 'only the addition videos need re-rendering' until this comparison is made. Read the two within-video pairs first (0028_kite-walk type2 vs type4, 0011_lucia type2 vs type5): they hold framing and object scale fixed, so they isolate the edit, and both were correctly ordered under the previous design."
    status: completed
    completed_at: 2026-08-26

  - id: taumap
    type: local
    wait_for: check-iou
    command: |
      python evaluation/r25_tau_map.py \
        --iou_csv evaluation/csv/r25_iou.csv \
        --tau_min 2.0 --tau_max 10.0 \
        -o evaluation/csv/r25_tau_map.csv
    output_paths:
      - evaluation/csv/r25_tau_map.csv
    status: completed
    completed_at: 2026-08-26

  - id: smoke
    type: sbatch
    wait_for: taumap
    command: sbatch slurm_scripts/five_bench/r25_smoke.sh
    status: completed
    completed_at: 2026-08-26
    job_id: "960406"

  - id: wait-smoke
    type: manual
    wait_for: smoke
    check_hint: "grep -c '^IDENTICAL' logs/r25_smoke_{job_id}.out  # expect 1 (tau=2 map == --blend_power 2); grep -c '^DIFFERS' logs/r25_smoke_{job_id}.out  # expect 1 (tau=8 != tau=2); grep -c '^COVERAGE-OK' logs/r25_smoke_{job_id}.out  # expect 1 (truncated map SystemExits). Gate 1 alone is not enough -- a map that is parsed and then discarded also passes it, which is what gate 2 exists to catch."
    status: completed
    completed_at: 2026-08-27

  - id: infer
    type: sbatch
    wait_for: wait-smoke
    command: sbatch --array=0-5 slurm_scripts/five_bench/r25_infer.sh
    sets_status: running
    status: completed
    completed_at: 2026-08-27
    job_id: "960704"

  - id: wait-infer
    type: manual
    wait_for: infer
    check_hint: "sacct -j {job_id} --format=JobID,State,Elapsed; OUT=/projects/dataggen/outputs/five_bench/r25_adaptive_tau/r25_adaptive_vp; for t in 1 2 3 4 5 6; do echo -n \"edit$t \"; ls -d $OUT/edit$t/*/ 2>/dev/null | grep -vc '_resize/$'; done  # expect 1 13 2 2 3 1 = 22; grep -h 'failures:' logs/r25_infer_*.out  # all 0. Also confirm the per-pair tau printed in each [ok] line matches r25_tau_map.csv -- that is the cheapest proof the arm is actually adaptive and not 22 renders at 2.0."
    sets_status: finished
    status: completed
    completed_at: 2026-08-27

  - id: evaluate
    type: sbatch
    wait_for: wait-infer
    command: sbatch slurm_scripts/five_bench/r25_eval.sh
    status: completed
    completed_at: 2026-08-27
    job_id: "960721"

  - id: wait-eval
    type: manual
    wait_for: evaluate
    check_hint: "ls evaluation/csv/edit?_FiVE_r25_adaptive_vp_frame_stride8_avg.csv | wc -l  # expect 6; grep -c 'out of memory' logs/r25_eval_*.metrics.log  # MUST be 0; grep -c 'Error: niqe' logs/r25_eval_*.metrics.log  # MUST be 0. Row counts per type: 1/13/2/2/3/1. 0010_giant-slalom is NOT in evaluation/cases.json, so the known empty-mask motion_fidelity degeneracy (R2/R14/R21) should not appear at all here -- any 'Error: motion_fidelity_score_edit_part' line IS a real failure in this task, unlike in R21/R24."
    status: completed
    completed_at: 2026-08-27
    job_id: "961490"

  - id: summarize
    type: local
    wait_for: wait-eval
    command: |
      python evaluation/r25_summarize.py \
        --arm_csv_glob 'evaluation/csv/edit?_FiVE_r25_adaptive_vp_frame_stride8.csv' \
        --ref_csv_glob 'evaluation/csv/edit?_FiVE_r21_ref_vp_frame_stride8.csv' \
        --tau_map evaluation/csv/r25_tau_map.csv \
        --iou_csv evaluation/csv/r25_iou.csv \
        -o evaluation/csv/r25_adaptive.csv
    output_paths:
      - evaluation/csv/r25_adaptive.csv
    status: completed
    completed_at: 2026-08-27

  - id: figures
    type: local
    wait_for: summarize
    command: |
      python evaluation/r25_figures.py \
        --summary_csv evaluation/csv/r25_adaptive.csv \
        --clip_col clip_similarity_target_image \
        --lpips_col lpips_unedit_part \
        --tau_baseline 2.0 \
        --out_dir evaluation/figures
    output_paths:
      - evaluation/figures/r25_tau_iou.pdf
      - evaluation/figures/r25_delta.pdf
      - evaluation/figures/r25_grids/
    status: completed
    completed_at: 2026-08-27

  - id: verdict
    type: manual
    wait_for: figures
    check_hint: "Read r25_adaptive.csv against both figures and record in daily.md: (a) measured IoU range and whether types 3/4 separated from 1/2; (b) does the adaptive arm beat Eq.4 rho=2 on the 22-case means, on which metrics, and by how much; (c) does per-case gain track |tau - 2| (r25_delta.pdf) or is it a flat cloud -- a flat cloud means the IoU routing contributed nothing even if the mean moved. (d) did 0042_gym-ball read IoU = 0 as designed; if not, open the anchor PNG to tell a phantom box from an anchor edit that never removed the ball -- n=1, so this single case is all the evidence there is on whether type 6 is measurable this way. State explicitly that Eq.4 is the ONLY comparator, so a win cannot be attributed to adaptivity over a better constant tau until the endpoint and shuffled-IoU arms are run."
    sets_status: analyzed
    status: completed
    completed_at: 2026-08-27
  # === PHASE 2: attribution controls (added 2026-08-27 after the phase-1 verdict) ===
  - id: control-maps
    type: local
    wait_for: verdict
    command: |
      python evaluation/r25_tau_map.py --assign reversed \
        --iou_csv evaluation/csv/r25_iou.csv --tau_min 2.0 --tau_max 10.0 \
        -o evaluation/csv/r25_tau_map_reversed.csv
      python evaluation/r25_tau_map.py --assign constant \
        --iou_csv evaluation/csv/r25_iou.csv --tau_min 2.0 --tau_max 10.0 \
        -o evaluation/csv/r25_tau_map_constant.csv
    output_paths:
      - evaluation/csv/r25_tau_map_reversed.csv
      - evaluation/csv/r25_tau_map_constant.csv
    status: completed
    completed_at: 2026-08-27

  - id: check-control-maps
    type: manual
    wait_for: control-maps
    check_hint: "Both maps must hold the SAME 22 (video_name, edit_type) pairs as r25_tau_map.csv -- run_fivebench.py hard-errors on missing coverage, so a short map fails at startup rather than silently. reversed: the multiset of tau values must be IDENTICAL to the adaptive map (sort both tau columns and diff -- that identity is the whole point, since it is what makes the metric's bias against shape change cancel between the two arms), and the pairing must be inverted BY RANK, so the extremes swap ends: 0042_gym-ball (lowest IoU 0.000, adaptive tau 10.00) must now carry **2.00** and 0007_guitar-violin (highest IoU 0.957, adaptive tau 2.00) must carry **10.00**; 0075_A_bicycle goes 9.46 -> **2.42** and 0017_kid-football 2.42 -> **9.46**. Expect all 22 rows to change. Rank reversal, NOT an IoU mirror: mirroring the IoU axis through the linear map would give a different multiset unless the IoUs were symmetric, and the identical multiset is what makes the metric bias cancel. constant: every row must read tau = 6.57 to 2 dp."
    status: completed
    completed_at: 2026-08-27

  - id: control-infer
    type: sbatch
    wait_for: check-control-maps
    command: |
      R25_ARM=reversed sbatch --export=ALL,R25_ARM --exclude=node52 --array=0-5 slurm_scripts/five_bench/r25_infer.sh
      R25_ARM=constant sbatch --export=ALL,R25_ARM --exclude=node52 --array=0-5 slurm_scripts/five_bench/r25_infer.sh
    sets_status: running
    status: completed
    completed_at: 2026-08-28
    job_id: "962628 (reversed), 962629 (constant)"

  # *** SUPERSEDED 2026-09-01 by Phase 3 (reinfer/wait-reinfer -> reeval -> resummarize ***
  # *** -> refigures -> type5-verdict). Jobs 962628 (reversed) / 962629 (constant) were  ***
  # *** built from the PRE-type5-fix tau multiset and are void (see check-reiou's         ***
  # *** verdict). The 5 steps below (wait-control-infer .. control-verdict) must NOT be   ***
  # *** run -- Phase 3's `resummarize` step already runs r25_controls.py on the correct,  ***
  # *** re-rendered data with the identical command `control-summarize` has below. A bare ***
  # *** `/run-step R25` (no step id) would otherwise pick `wait-control-infer` next as     ***
  # *** the first `pending` step whose `wait_for` (control-infer) is `completed` -- it is, ***
  # *** but checking on a void job is wrong. Target `reeval` explicitly instead.           ***
  - id: wait-control-infer
    type: manual
    wait_for: control-infer
    check_hint: "Per arm: 1/13/2/2/3/1 frame dirs = 22, `failures: 0` in all six task logs, and the per-pair tau echoed in each [ok] line matching that arm's map. For `constant` every [ok] line must read tau=6.57 -- if any row differs the wrong map was picked up. Check the GPU diagnostics header too (node, nvidia-smi -L, torch cuda_ok): node52 advertises gpu:8 while exposing no device to batch jobs, which killed two phase-1 submissions."
    sets_status: finished
    status: pending

  - id: control-eval
    type: sbatch
    wait_for: wait-control-infer
    command: |
      R25_ARM=reversed sbatch --export=ALL,R25_ARM slurm_scripts/five_bench/r25_eval.sh
      R25_ARM=constant sbatch --export=ALL,R25_ARM slurm_scripts/five_bench/r25_eval.sh
    status: pending

  - id: wait-control-eval
    type: manual
    wait_for: control-eval
    check_hint: "Per arm: 6 per-edit-type _avg.csv, 0 `Error:` lines, 0 OOM, row counts 1/13/2/2/3/1, no column misalignment. Both arms MUST be scored with the same explicit --metrics list as phase 1 (9 metrics) -- a different metric set would make the three-way table incomparable. Remember the metrics: key in config.yaml is vestigial; only the --metrics flag is read (evaluate.py:200)."
    status: pending

  - id: control-summarize
    type: local
    wait_for: wait-control-eval
    command: |
      python evaluation/r25_controls.py \
        --arms adaptive reversed constant \
        --ref_csv_glob 'evaluation/csv/edit?_FiVE_r21_ref_vp_frame_stride8.csv' \
        --iou_csv evaluation/csv/r25_iou.csv \
        -o evaluation/csv/r25_controls.csv
    output_paths:
      - evaluation/csv/r25_controls.csv
    status: pending

  - id: control-verdict
    type: manual
    wait_for: control-summarize
    check_hint: "The two contrasts decide it, and they answer different questions. (1) adaptive - reversed: does the PAIRING carry signal? Same 22 tau values in both, so the *_unedit_part mask bias against shape change cancels and this contrast is trustworthy even with the confounded metric. adaptive ~= reversed means the IoU routing contributes nothing and R25 should not be promoted. (2) adaptive - constant: does VARYING tau help at all? constant ~= adaptive means the whole IoU machinery is unnecessary regardless of (1) -- this is the control a reviewer asks for first. Record BOTH contrasts with per-case sign tests, not just means: phase 1's +1.064 mean CLIP gain hid a 15-positive/7-negative split (sign test p = 0.134). State explicitly which of the two hypotheses each result rules out."
    sets_status: analyzed
    status: pending
  # *** end superseded block ***
  # === PHASE 3: type-5 addition mask fix (added 2026-09-01) -- run BEFORE resuming the
  # phase-2 steps above (wait-control-infer onward): jobs 962628/962629 were rendered from
  # the pre-fix tau multiset and are expected to be superseded by `reinfer` below. ===
  - id: reiou
    type: srun
    command: |
      srun --partition=L40S --gres=gpu:1 --mem=32G --time=01:00:00 --pty \
        env HF_HUB_OFFLINE=1 python evaluation/r25_iou.py \
          --cases evaluation/cases.json \
          --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \
          --anchor_root /projects/dataggen/outputs/five_bench/anchors \
          --box_thr 0.35 \
          --viz_dir evaluation/figures/r25_iou_panels \
          --dump_masks evaluation/figures/r25_iou_masks \
          -o evaluation/csv/r25_iou.csv
    output_paths:
      - evaluation/csv/r25_iou.csv
      - evaluation/figures/r25_iou_panels/
      - evaluation/figures/r25_iou_masks/
    status: completed
    completed_at: 2026-09-01
    job_id: "973779"
    # AMENDED 2026-09-01: the first pass (this command originally, minus --dump_masks) did
    # not refresh evaluation/figures/r25_iou_masks/*.npz, which r25_figures.py --mask_dir
    # reads for the r25_grids mask-overlay column -- those stayed dated Aug 28 (the
    # pre-type5-fix union masks) even after the CSV was correct. User caught this
    # ("r25_grids are not re-generated") before `refigures` ran, so no wrong figure was
    # ever produced -- but it would have been silently wrong otherwise, since only the 3
    # type-5 npz files actually differ and a stale dump does not fail loudly. Re-ran with
    # --dump_masks added: CSV md5 unchanged (f16283d5cada8c632423faa405b8a42a, confirming
    # determinism), all 22 npz refreshed (mtime 2026-09-01), spot-checked
    # 0007_guitar-violin.npz decodes to area_edit=16814 / iou=0.11843, matching the CSV
    # exactly (not the old union-based area_edit=104416). r25_grids/*.pdf themselves are
    # STILL the Aug 28 pre-fix pages -- this only fixes the npz INPUT to `refigures`, which
    # has not run yet (waits on reeval -> resummarize first).

  - id: check-reiou
    type: manual
    wait_for: reiou
    check_hint: "Same three gates as `check-iou`, PLUS gate 4 (range shift) from that step's hint, now run for real: diff the new IoU_hi/IoU_lo against the old (0.957 / 0.000). Expect the three type-5 rows (0007_guitar-violin, 0069_car-turn, 0011_lucia_e5) to drop, and 0007_guitar-violin in particular to fall well below 0.957 since a hat mask alone shares far less area with the musician mask than SUV|flamingo did with the SUV. Record the new IoU_hi/IoU_lo and whether they moved. If they moved (expected): every case's tau changes and the FULL 22-case arm needs re-rendering for `adaptive`, `reversed`, AND `constant` -- proceed to `retaumaps` / `reinfer` below, and treat jobs 962628/962629 as void. If they did NOT move (i.e. some other case already exceeded 0.957 -- unlikely but check 0017_kid-football at 0.908 first): only the 3 type-5 clips need re-rendering under `adaptive`, and phase 2's in-flight reversed/constant renders remain valid since the tau multiset is unchanged -- if so, skip `retaumaps`/`reinfer` below and resume `wait-control-infer` instead. **RESULT (2026-09-01): all 4 gates PASS, range MOVED.** Gate 1 median `iou_gt_src` 0.975 (min 0.726), excluding `0042_gym-ball`/`0017_kid-football` as before. Gate 2: visually verified all 3 type-5 panels (`M_edit` is now the hat/flamingo/dog alone, correctly grounded, small-to-near-zero overlap with `M_src`) plus the new `IoU_hi` case `0017_kid-football` (red cap → yellow cap, same head position, genuine 0.908 — not a detector artefact). Gate 3: types 3+4 range [0.522, 0.908] still cleanly above types 1+2 range [0.065, 0.518], zero overlap, unaffected by the fix. Gate 4: `IoU_hi` **0.957 → 0.908**, now set by `0017_kid-football` (type 3) instead of a type-5 case; `IoU_lo` unchanged at 0.0. New qualitative finding: the 3 type-5 rows (0.023, 0.026, 0.118) now sit BELOW the entire type-1+2 swap range [0.065, 0.518] — additions read as a LARGER shape change than any swap in this set, not an intermediate one. **Verdict: branch (a) applies — full 22-case re-render required for all three arms; jobs 962628 (reversed) / 962629 (constant) are void and must not be carried into `wait-control-infer`/`control-eval`. Proceed to `retaumaps` -> `reinfer`, not an addition-only re-render.**"
    status: completed
    completed_at: 2026-09-01

  - id: retaumaps
    type: local
    wait_for: check-reiou
    command: |
      python evaluation/r25_tau_map.py \
        --iou_csv evaluation/csv/r25_iou.csv \
        --tau_min 2.0 --tau_max 20.0 --mapping budget_linear \
        --step 15 --flow_shift 1.0 \
        -o evaluation/csv/r25_tau_map.csv
      python evaluation/r25_tau_map.py --assign reversed \
        --iou_csv evaluation/csv/r25_iou.csv --tau_min 2.0 --tau_max 20.0 \
        --mapping budget_linear --step 15 --flow_shift 1.0 \
        -o evaluation/csv/r25_tau_map_reversed.csv
      python evaluation/r25_tau_map.py --assign constant \
        --iou_csv evaluation/csv/r25_iou.csv --tau_min 2.0 --tau_max 20.0 \
        --mapping budget_linear --step 15 --flow_shift 1.0 \
        -o evaluation/csv/r25_tau_map_constant.csv
    output_paths:
      - evaluation/csv/r25_tau_map.csv
      - evaluation/csv/r25_tau_map_reversed.csv
      - evaluation/csv/r25_tau_map_constant.csv
    status: completed
    completed_at: 2026-09-01

  - id: check-retaumaps
    type: manual
    wait_for: retaumaps
    check_hint: "Same checks as `check-control-maps` above, against the NEW r25_iou.csv, PLUS the mapping-family change (2026-09-01, user decision): `adaptive` now uses `--mapping budget_linear --tau_max 20.0` (was `linear`/10.0), so `tau` is no longer a simple linear function of IoU -- verify the `.meta.json` sidecar records `mapping: budget_linear`, `tau_max: 20.0`, and `A(2)/A(20)` before trusting the CSV. reversed keeps the identical tau multiset as adaptive with the pairing inverted by rank (re-verify which case is now the IoU-min / IoU-max, since 0007_guitar-violin may no longer hold the max, and confirm the reversed run also reports `mapping: budget_linear` in its own sidecar); constant is every row at the new realized mean (smoke-tested at ~7.57 with the new mapping/range, NOT 6.57 -- recompute and do not treat 6.57 as a sanity anchor any more). All three maps must cover the same 22 (video_name, edit_type) pairs. **RESULT (2026-09-01): ALL PASS.** Coverage identical across all three (22 pairs). Tau multiset identical adaptive vs reversed. Rank-inversion verified exactly at both extremes: `0042_gym-ball` (lowest IoU 0.000) adaptive tau **20.00** -> reversed tau **2.00**; `0017_kid-football` (highest IoU 0.908) adaptive tau **2.00** -> reversed tau **20.00**. Constant: single unique value **7.5745** across all 22 rows. All three `.meta.json` sidecars report `mapping: budget_linear, tau_min: 2.0, tau_max: 20.0, A_tau_min: 4.5062, A_tau_max: 0.3200, step: 15, flow_shift: 1.0` identically."
    status: completed
    completed_at: 2026-09-01

  - id: reinfer
    type: sbatch
    wait_for: check-retaumaps
    command: |
      sbatch --array=0-5 slurm_scripts/five_bench/r25_infer.sh
      R25_ARM=reversed sbatch --export=ALL,R25_ARM --exclude=node52 --array=0-5 slurm_scripts/five_bench/r25_infer.sh
      R25_ARM=constant sbatch --export=ALL,R25_ARM --exclude=node52 --array=0-5 slurm_scripts/five_bench/r25_infer.sh
    sets_status: running
    status: completed
    completed_at: 2026-09-01
    job_id: "973779 (interactive — all 3 sbatch --array submissions above hit QOSMaxSubmitJobPerUserLimit; ran the 18 run_fivebench.py invocations directly, sequentially, inside the held allocation instead; see daily.md)"

  - id: wait-reinfer
    type: manual
    wait_for: reinfer
    check_hint: "Per arm (adaptive/reversed/constant): 1/13/2/2/3/1 frame dirs = 22, `failures: 0` in all six task logs per arm, and the per-pair tau echoed in each [ok] line matching that arm's regenerated map -- for the 3 type-5 cases specifically, confirm the tau printed differs from the phase-1/phase-2 value recorded in r25_adaptive.csv / r25_controls.csv. **RESULT (2026-09-01): PASS.** All 18 tasks (3 arms x 6 edit types) `OK`, `total_failures=0` in `logs/r25_reinfer_master.log`. Frame-dir counts 1/13/2/2/3/1=22 per arm, verified BOTH from the master log's per-task check and independently on disk (`ls -d .../edit{T}/*/`). Cross-checked all 66 (video, arm) `[ok] ... tau=...` log lines against the corresponding tau-map CSV: **0/66 mismatches**. Type-5 tau, adaptive vs phase-1 (verified from the stored `r25_adaptive.csv`, union+linear design): `0007_guitar-violin` **10.4281** (was **2.0000** phase-1 — it was the old `IoU_hi` anchor, pinned exactly to Eq.4), `0069_car-turn` **16.5635** (was **6.9090**), `0011_lucia_e5` **16.9526** (was **5.4750**) — all now route markedly faster than under the old union+linear design, consistent with additions reading as a larger shape change post-fix. No `FAIL`/`Traceback`/`Error` string in any of the 18 per-task logs. Total wall time ~1h40m serialized on one GPU."
    sets_status: finished
    status: completed
    completed_at: 2026-09-01

  - id: reeval
    type: sbatch
    wait_for: wait-reinfer
    command: |
      sbatch slurm_scripts/five_bench/r25_eval.sh
      R25_ARM=reversed sbatch --export=ALL,R25_ARM slurm_scripts/five_bench/r25_eval.sh
      R25_ARM=constant sbatch --export=ALL,R25_ARM slurm_scripts/five_bench/r25_eval.sh
    status: completed
    completed_at: 2026-09-01
    job_id: "973779 (interactive — the 3 sbatch submissions 976245/976246/976247 queued PENDING/QOSMaxGRESPerUser behind the R26 array backlog; cancelled per user instruction and ran evaluate.py directly x3, sequentially, inside the held allocation instead, using the five-bench conda env's python by absolute path; see daily.md)"

  - id: wait-reeval
    type: manual
    wait_for: reeval
    check_hint: "Per arm: 6 per-edit-type _avg.csv, 0 OOM, 0 niqe errors, row counts 1/13/2/2/3/1, same explicit 9-metric --metrics list as phase 1/2 so the three arms stay comparable. **RESULT (2026-09-01): PASS.** All 3 arms `exit=0 error_lines=0 oom_lines=0 avg_csv=6` in `logs/r25_reeval_master.log` (adaptive 855s, reversed 843s, constant 853s, ~42min total). Independently verified on disk: 18 avg CSVs total (6x3), and every `edit{T}_FiVE_r25_{arm}_vp_frame_stride8.csv` has the exact expected row count 1/13/2/2/3/1 for all three arms."
    status: completed
    completed_at: 2026-09-01

  - id: resummarize
    type: local
    wait_for: wait-reeval
    command: |
      python evaluation/r25_summarize.py \
        --arm_csv_glob 'evaluation/csv/edit?_FiVE_r25_adaptive_vp_frame_stride8.csv' \
        --ref_csv_glob 'evaluation/csv/edit?_FiVE_r21_ref_vp_frame_stride8.csv' \
        --tau_map evaluation/csv/r25_tau_map.csv \
        --iou_csv evaluation/csv/r25_iou.csv \
        -o evaluation/csv/r25_adaptive.csv
      python evaluation/r25_controls.py \
        --arms adaptive reversed constant \
        --ref_csv_glob 'evaluation/csv/edit?_FiVE_r21_ref_vp_frame_stride8.csv' \
        --iou_csv evaluation/csv/r25_iou.csv \
        -o evaluation/csv/r25_controls.csv
    output_paths:
      - evaluation/csv/r25_adaptive.csv
      - evaluation/csv/r25_controls.csv
    status: completed
    completed_at: 2026-09-01
    # r25_controls.py did not exist (phase 2 never actually built it -- see write-
    # controls-summarize). Written before this step could run; see that todo. Both
    # scripts ran clean, exit 0: r25_adaptive.csv 22 cases + 7 mean rows, r25_controls.csv
    # same shape + r25_controls.meta.json (per contrast x metric sign tests). Headline
    # result: adaptive beats Eq.4 on clip_similarity_target_image (+1.38) and lpips_
    # unedit_part is WORSE (+0.013, higher=worse) -- same qualitative shape as phase 1.
    # Neither adaptive-reversed nor adaptive-constant CLIP contrast reaches significance
    # at n=22 (13pos/9neg both, sign_p=0.523) -- see type5-verdict for the full read.

  - id: refigures
    type: local
    wait_for: resummarize
    command: |
      python evaluation/r25_figures.py \
        --summary_csv evaluation/csv/r25_adaptive.csv \
        --clip_col clip_similarity_target_image \
        --lpips_col lpips_unedit_part \
        --tau_baseline 2.0 \
        --out_dir evaluation/figures
    output_paths:
      - evaluation/figures/r25_tau_iou.pdf
      - evaluation/figures/r25_delta.pdf
      - evaluation/figures/r25_grids/
    status: completed
    completed_at: 2026-09-01
    # Exit 0, no stderr. All 22 r25_grids/*.pdf independently verified fresh (mtime
    # 2026-09-01, not Aug 28) -- the staleness the user caught is now resolved end to
    # end, since its input (r25_iou_masks/*.npz) was fixed by the reiou amendment first.
    # delta-vs-|tau-2| correlations printed: clip_similarity_target_image r=+0.076
    # p=0.736 (flat -- CLIP gain does NOT track how far tau moved from Eq.4), lpips_
    # unedit_part r=+0.778 p<0.001 (strong -- background-preservation loss DOES track
    # |tau-2|, consistent with the plan's documented *_unedit_part metric bias: the GT
    # mask is source-only, so a shape-changing edit scores its own new pixels as
    # "unedited", and that mismeasurement scales with (1-IoU), which tau is a function
    # of -- see the "Why the phase-2 controls are trustworthy" decision row).

  - id: type5-verdict
    type: manual
    wait_for: refigures
    check_hint: "Re-read the phase-1 and phase-2 verdict questions against the new numbers, PLUS: did the addition cases (0007_guitar-violin, 0069_car-turn, 0011_lucia_e5) move toward tau_max now that they read like swaps instead of regional growth? Does that change their per-case delta vs Eq.4 in a way that shifts the overall mean materially, given types 1/5/6 are single-case anecdotes but 0007_guitar-violin previously anchored tau_min for the WHOLE arm (every other case's tau was defined relative to it)?

**VERDICT (2026-09-01), full record in daily.md Progress/outcomes:**

(a) IoU separation unaffected by the fix: types 3+4 [0.522,0.908] vs types 1+2 [0.065,0.518], zero overlap, same as before.

(b) Adaptive still beats Eq.4 on edit alignment and loses on preservation -- SAME shape as phase 1, now with sign tests computed for the first time (r25_summarize.py reports means only): CLIP delta 17pos/4neg/1tie, sign_p=**0.0072** (significant); LPIPS delta 21pos/1neg, sign_p=**0.000011** (significant, worse). Phase 1's analogous split was 15pos/7neg, p=0.134 (not significant) -- the type-5 fix made this comparison MORE decisive, not less.

(c) Per-case gain vs |tau-2|: CLIP still flat (r=+0.076, p=0.736 -- was +0.101, p=0.656) -- the disqualifier for 'IoU routing specifically' still holds. LPIPS still tracks strongly (r=+0.778, p<0.001 -- was +0.614, p=0.002), still attributable to the *_unedit_part mask bias (scales with 1-IoU, which tau is a function of) rather than necessarily real damage -- see (f).

(d) 0042_gym-ball still IoU=0 by construction, tau=20 (tau_max), n=1, still an anecdote.

(e) TYPE-5 SPECIFIC: yes, all 3 moved sharply toward tau_max (2.00->10.43, 6.91->16.56, 5.48->16.95). This STRENGTHENED the overall mean, not weakened it: the 3 cases now score an ABOVE-average CLIP gain (mean +2.30 vs overall +1.38), with 0069_car-turn the 3rd-largest single-case gain of all 22 (+4.26). Structurally, the fix also improved the map's face validity: 0007_guitar-violin no longer anchors tau_min for the whole arm -- that anchor is now 0017_kid-football, a genuine colour edit, a far more defensible 'mildest edit' reference point than an addition (a hat) was. This is a calibration-soundness improvement independent of whether the aggregate numbers moved.

(f) THE HEADLINE CHANGE FROM PHASE 1: the phase-2 contrasts are, for the first time, ACTUALLY MEASURED -- jobs 962628/962629 were void and r25_controls.py did not exist before this session (see write-controls-summarize). Neither reaches significance on CLIP: adaptive-reversed 13pos/9neg sign_p=0.523; adaptive-constant 13pos/9neg sign_p=0.523. **This directly answers the phase-1 'Eq.4 is the only comparator' caveat, and the answer is negative: nothing here demonstrates the IoU-specific routing beats a single well-chosen constant tau (~7.57).** niqe_target_image is the closest either contrast gets to significance (adaptive-reversed 6pos/16neg, p=0.052, favoring reversed).

(g) CROSS-REFERENCE TO R28 (not previously connected in this plan): R28 independently measured that *_unedit_part metrics (which lpips_unedit_part/psnr_unedit_part/etc. all are) mis-score the edited region as background, inflating the apparent fidelity cost of aggressive edits by ~5-7x on the R26 spatial-tau arms. The -3.27 dB PSNR / +0.013 LPIPS costs reported here almost certainly overstate R25's TRUE preservation cost for the same reason -- R25 has not been re-scored under R28's union-corrected metric family, and doing so is the natural next step before any preservation number here is treated as final.

**BOTTOM LINE: do not promote the IoU-driven routing as validated.** The arm still shows the same editability/fidelity trade phase 1 found, now on firmer statistical footing in both directions -- but the phase-2 controls, run and measured for the first time in this session, show no evidence that the SPECIFIC per-case IoU-based tau assignment does anything a single constant tau near 7.5 would not. n=22 may simply be underpowered to detect a real but modest pairing effect (13/9 splits are wide at this n), so 'no evidence of an effect' is not the same as 'evidence of no effect' -- but as measured, this arm does not clear the bar to promote past R25 into a paper claim."
    sets_status: analyzed
    status: completed
    completed_at: 2026-09-01

isProject: true
---

# R25: IoU-Driven Adaptive Tau

## Context

StreamGVE's blender rate (Eq. 4) is implemented at [causal_model.py:334](../Self-Forcing_StreamEdit/wan/modules/causal_model.py#L334) as `blender_rate = 1 - t_{i+1} ** blend_power`, i.e. a source-anchoring weight

$$W^{src}(t_i) = t_i^{\tau}, \qquad r = 1 - W^{src}$$

with a single global exponent `tau = blend_power = 2` for every video and every token. Because `t in (0, 1)`, `t**tau` *decreases* in tau: a large tau releases the source anchor fast, a small tau holds it. The curve family is [f_t_power_tau_comparison.png](../f_t_power_tau_comparison.png).

R25 makes that exponent a property of *the edit*. The driver is the shape change the edit demands, measured once on clean pixels before any denoising, as the IoU between the region the edited content occupies in the source first frame and in the Qwen anchor:

$$\tau = f\bigl(\mathrm{IoU}(I_0, I_{0,\mathrm{edit}})\bigr), \qquad \mathrm{IoU} \to 1 \Rightarrow \tau_{\min}, \quad \mathrm{IoU} \to 0 \Rightarrow \tau_{\max}$$

A colour edit leaves the silhouette intact (IoU near 1) and should stay anchored to the source; a person-to-lion swap needs the silhouette to move (low IoU) and should release early. **One phrase is grounded per side** (revised 2026-08-26): `M_src = m(src_word, I0)` on the source frame and `M_edit = m(trg_word, anchor)` on the anchor. The original design unioned both phrases on both images, on the assumption that a phrase naming something absent would contribute nothing to that side — but GroundingDINO never abstains, and measuring it on this case set showed all four groundings firing on 22/22 cases with the "absent" phrase scoring up to 0.94, so the union added a false positive everywhere and broke worst on the add/remove cases it existed for. Only type 6 (removal) is handled by construction: `M_edit` is set empty so a removal reads as IoU = 0, the maximal shape change. **Type 5 (addition), revised 2026-09-01:** an earlier version also unioned `M_edit` with `M_src` so an addition would read as regional growth (`M_src` = SUV, `M_edit` = SUV ∪ flamingo on 0069_car-turn) rather than as a disjoint new object; that union has been reverted, and `M_edit = m(trg_word, anchor)` alone for type 5, exactly as for a swap. Consequence: since IoU_hi in the completed run was set by a type-5 case (`0007_guitar-violin`, 0.957) and is expected to drop once measured without the union, the measured-range normalization shifts and every case's tau — not only the three addition cases — needs to be re-derived and every arm re-rendered.

`f` is linear over the *measured* IoU range with `tau_min = 2`, `tau_max = 10` (revised 2026-08-26). `tau_min = 2` pins the mildest edits at the Eq. 4 default rather than bracketing it, so the arm only ever releases the source *at least* as fast as the paper baseline, never slower.

This is the per-video scalar counterpart of a per-token field. An earlier per-token implementation ("R25 V0" — `build_rho_field`, `--edit_field/--rho_min/--rho_max`) was removed from the tree on 2026-08-26, so nothing of it remains to build on or conflict with; the three self-attention blend sites are back to the pristine scalar `blender_rate` ([causal_model.py:371-399](../Self-Forcing_StreamEdit/wan/modules/causal_model.py#L371-L399)). This task therefore needs no pipeline change at all — it only varies the existing `blend_power` per pair. R26 is the spatial counterpart.

**Done when:** `evaluation/csv/r25_adaptive.csv` holds the scored FiVE metrics per case (9 in the phase-1 run; see the metric-environment row) for the adaptive arm and the Eq. 4 vp reference re-averaged over the same 22 cases, with per-metric deltas; `evaluation/figures/r25_tau_iou.pdf` shows the realized IoU-to-tau mapping by edit type; `evaluation/figures/r25_delta.pdf` shows whether the per-case gain tracks `|tau - 2|`.

## Execution steps

| # | Step id | Type | What |
|---|---------|------|------|
| — | *(prep)* | — | `write-iou`, `write-taumap`, `plumb-taumap`, `write-smoke`, `write-infer`, `write-eval`, `write-summarize`, `write-figure` — see **Code to touch** |
| 1 | `iou` | srun | Both phrases × both frames → SAM2 masks → per-side union → `r25_iou.csv` |
| 2 | `check-iou` | manual | GT-mask agreement ≥ 0.7, firing pattern matches each edit, type 3/4 separate from type 2, add/remove strictly below 1.0 — hard gate before any render |
| 3 | `taumap` | local | Linear map on the measured IoU range → `r25_tau_map.csv` |
| 4 | `smoke` | sbatch | Override reaches `blend_power`; non-degenerate; coverage error raises |
| 5 | `wait-smoke` | manual | 1 × `IDENTICAL`, 1 × `DIFFERS`, 1 × `COVERAGE-OK` |
| 6 | `infer` | sbatch | Array 0-5, one edit type per task, 22 cases, vp mode → `running` |
| 7 | `wait-infer` | manual | 1/13/2/2/3/1 frame dirs, 0 failures, logged tau matches the map → `finished` |
| 8 | `evaluate` | sbatch | `evaluate.py` on L40S,A100 with the metric-crash env fix and a 9-metric `--metrics` list |
| 9 | `wait-eval` | manual | 6 `_avg.csv`, 0 OOM, 0 niqe errors, 0 motion-fidelity errors |
| 10 | `summarize` | local | Join on `file_id` vs stored R21 vp reference → `r25_adaptive.csv` |
| 11 | `figures` | local | Mapping page + per-case delta-vs-`\|tau-2\|` page |
| 12 | `verdict` | manual | Attribution reading → `analyzed` |

```
/run-step R25 iou
/run-step R25 check-iou
/run-step R25 taumap
/run-step R25 smoke
/run-step R25 wait-smoke
/run-step R25 infer
/run-step R25 wait-infer
/run-step R25 evaluate
/run-step R25 wait-eval
/run-step R25 summarize
/run-step R25 figures
/run-step R25 verdict
```

## Phase 2 — attribution controls

Phase 1 answered *does this arm beat Eq. 4* (yes on edit alignment, 15/22 cases positive,
sign test p = 0.134). It could not answer *does the IoU routing produce that gain*, because
the arm changed the tau values and their assignment at the same time.

Two arms, one 22-clip render each, settle it. Everything else — seed 0, anchors,
`cases_json`, sampler defaults, the 9-metric `--metrics` list — is held identical, so the
arms differ **only** in the tau assignment.

| arm | tau assignment | question it answers | if it matches adaptive |
|---|---|---|---|
| `adaptive` | IoU → tau (phase 1, done) | — | — |
| `reversed` | same 22 values, inverted pairing | does the **pairing** carry signal? | the IoU routing contributes nothing |
| `constant` | tau = 6.57 for every pair | does **varying** tau help at all? | the IoU machinery is unnecessary |

The two contrasts are independent and both are needed: `constant` could match adaptive even
if the pairing carries real signal (a flat optimum), and `reversed` could differ from
adaptive even if a single constant would do just as well.

```
/run-step R25 control-maps
/run-step R25 check-control-maps
/run-step R25 control-infer
/run-step R25 wait-control-infer
/run-step R25 control-eval
/run-step R25 wait-control-eval
/run-step R25 control-summarize
/run-step R25 control-verdict
```

## Phase 3 — type-5 addition mask fix

At the user's request (2026-09-01), type 5 (addition) is no longer handled by construction.
`M_edit` was `m(trg_word, anchor) | M_src` — the edited region read as the source object
plus what was added, e.g. `0069_car-turn`: SUV ∪ flamingo. It is now `m(trg_word, anchor)`
alone, exactly the swap formula, so an added object that is spatially disjoint from the
source (the common case) now reads a low IoU close to a swap's rather than a "grew"
reading.

**This is not scoped to the 3 addition clips.** `IoU_hi` in the completed run was `0.957`,
set by `0007_guitar-violin` — a type-5 case. The tau map is linear on
`[IoU_lo, IoU_hi]`, so if the re-measured type-5 IoU no longer sets that endpoint (expected:
a hat mask alone should share far less area with the musician mask than the old union did),
**every case's tau shifts**, not only the addition cases — and both phase-2 control renders
(`reversed`, `constant`) inherit the same tau multiset by construction, so they are
implicated too. `check-reiou` makes this an explicit gate rather than an assumption before
anything is re-rendered.

| step | what |
|---|---|
| `reiou` | Re-run `r25_iou.py` with the type-5 union removed |
| `check-reiou` | Gate: did `IoU_hi`/`IoU_lo` move? Decides whether the re-render is 3 clips or 66 (3 arms × 22) |
| `retaumaps` | Re-derive all three tau maps (`iou`/`reversed`/`constant`) from the new IoU |
| `reinfer` | Re-render `adaptive`, `reversed`, `constant` — supersedes jobs 962628/962629 |
| `reeval` → `resummarize` → `refigures` | Same scripts as phases 1-2, re-pointed at the new renders |
| `type5-verdict` | Does the fix change which cases anchor the map, and does that move the overall verdict |

```
/run-step R25 reiou
/run-step R25 check-reiou
/run-step R25 retaumaps
/run-step R25 check-retaumaps
/run-step R25 reinfer
/run-step R25 wait-reinfer
/run-step R25 reeval
/run-step R25 wait-reeval
/run-step R25 resummarize
/run-step R25 refigures
/run-step R25 type5-verdict
```

## Decisions

| Question | Choice |
|---|---|
| What `tau` is | The Eq. 4 exponent `blend_power`, `W_src = t_{i+1} ** tau`. **Not** R24's cosine release-point tau, which is a different parametrization of the same bridge; R25 does not depend on R24 landing. |
| **How tau reaches the model (premise corrected)** | Per-case override of **`blend_power`**, via a new `--tau_map` CSV. The chosen option was originally phrased as "`--rho_min = --rho_max`", but those flags belonged to the R25 V0 per-token path, which was **removed from the tree on 2026-08-26** — so they no longer exist at all. `blend_power` *is* the global per-video exponent, read directly as `1 - t**blend_power` at [causal_model.py:334](../Self-Forcing_StreamEdit/wan/modules/causal_model.py#L334). Same intent, and now the only knob. |
| Mapping `f` | Linear on the **measured** IoU range: `tau = tau_max - (tau_max - tau_min) * (iou - IoU_lo)/(IoU_hi - IoU_lo)`, `IoU_lo/hi` = min/max over the 22 cases. |
| `tau_min`, `tau_max` | **`2.0` and `10.0`** (revised 2026-08-26 from `1.0`/`12.0`). `tau_min = 2` places the highest measured IoU exactly at the Eq. 4 default `blend_power = 2`, so a near-silhouette-preserving edit behaves like the paper baseline instead of holding the source *longer* than it. **Consequence, accepted deliberately:** the arm is now one-directional — every case gets `tau >= 2`, so it can only ever release faster than Eq. 4, and the converse hypothesis (that a colour or material edit benefits from holding the source longer than the default) is no longer testable within this arm. Under the previous `[1, 12]` map two cases sat below 2 (`0007_guitar-violin` 1.00, `0017_kid-football` 1.57) and did test it. `tau_max = 10` pulls the fast end back from 12 toward the range where preservation has been measured. |
| ⚠️ `tau_max` is still set blind | Nothing between Eq. 4 (`rho = 2`) and full release has ever been scored. The only measured point in the fast direction is R21's `zero` schedule, at **LPIPS 143.0 against `ref_vp`'s 53.6** — ~3x worse background preservation. Integrated source mass over the 15 steps: `tau=1` 7.00, `tau=2` **4.51**, `tau=8` 1.21, `tau=12` 0.72, `zero` 0.00. `tau_max = 10` therefore sits between the 1.21 and 0.72 anchors — still much nearer `zero` than Eq. 4 in integrated terms, so the low-IoU cases routed there may still hit preservation collapse. Lowering 12 → 10 narrows that exposure without removing it; the value remains extrapolation from one endpoint, not a measurement. Cheap mitigation, already in the pipeline: `taumap` prints the realized per-case tau and `check-iou` reads it **before** anything renders. |
| Consequence: the arm is also a "release faster" arm | The map is linear in tau, so a mid-range IoU lands near `tau ≈ 6.5` against Eq. 4's `2` — most cases are routed to substantially faster release than the reference. If the arm wins, *"releasing faster helps on a swap-heavy set"* competes with *"adaptive routing helps"* as an explanation, and with Eq. 4 as the sole comparator they are not separable. This compounds the attribution ceiling recorded below rather than being a separate issue; `r25_delta.pdf` (gain vs `|tau - 2|`) is the only partial discriminator in scope. |
| Why the measured range, not `[0, 1]` | Object swaps still occupy roughly the same image region, so their IoU does not approach 0 (expected floor ~0.4-0.7), while colour/material edits sit near 1.0. On a nominal `[0, 1]` domain the realized tau spread would be a fraction of `[1, 12]` and the arm would be far less adaptive than intended. **Note the removal case pins `IoU_lo = 0` by construction** (`0042_gym-ball` reads exactly 0 after the `src_word` fix), so at the low end the measured range coincides with the nominal one; the anchoring still matters at the top, where `IoU_hi ≈ 0.95` rather than 1.0. Cost: `f` is calibrated on this case set and is not transferable as-is to a different one. |
| IoU definition | **One phrase per side** (revised 2026-08-26). `M_src = m(src_word, I0)`; `M_edit = m(trg_word, anchor)` for types 1-5, and **empty** for type 6. GroundingDINO box → SAM2 mask, `--box_thr 0.35`. |
| Rejected: symmetric two-phrase union | The original design, **removed after measuring it**. It assumed a phrase naming something absent would drop out of that side's union. GroundingDINO does not abstain — at `box_thr 0.25` all 4 groundings fired on **22/22** cases, and the phrase that should have been absent scored up to **0.94** (`'A dragon'` on the unedited cow), higher than most genuine detections, so no global threshold separates them (should-fire cells run down to 0.28). The union therefore injected a false positive on every case and failed hardest on the add/remove cases it was introduced to serve. Rephrasing does not rescue it either: narrowing `0007_guitar-violin`'s `trg_word` to `'wearing a hat'` moved the false positive from the violinist to the seated guitarist rather than removing it. |
| Remove handled by construction | Type 6 sets `M_edit` empty, so IoU is **exactly 0** → `tau_max`, the maximal shape change, which is the correct reading of a removal. ⚠ Type 6 is now an **assumption, not a measurement**: the row reads 0 whether or not the anchor actually removed the object, which forfeits the free anchor-quality check the two-phrase design had. |
| **Reverted: type-5 union with `M_src`** (2026-09-01) | Type 5 (addition) was originally **also** handled by construction, on the same reasoning as removal: `M_edit = m(trg_word, anchor) \| M_src`, so the edited region read as the source object PLUS what was added (`0069_car-turn`: `M_src` = SUV, `M_edit` = SUV ∪ flamingo), and the IoU measured how much the region **grew** rather than how much it moved. **Reverted at user request**: type 5 is now measured identically to a swap — `M_edit = m(trg_word, anchor)` alone, no union — so an addition whose new object is spatially disjoint from the source (a flamingo strapped to an SUV roof, a hat on a musician's head) reads a low IoU close to a swap's, rather than the earlier "regional growth" reading. |
| ⚠️ Consequence: the measured-range normalization shifts, not just the 3 addition cases' tau | `tau_min`/`tau_max` are anchored on the **measured** `IoU_lo`/`IoU_hi` over all 22 cases (see the "Why the measured range" row below), and in the completed 2026-08-27 run `IoU_hi = 0.957` was set by a **type-5 case**, `0007_guitar-violin` (next-highest was `0017_kid-football`, type 3, at 0.908). Once `M_edit` no longer unions with `M_src`, that case's IoU is expected to **drop** — the hat mask alone overlaps the musician mask far less than SUV∪flamingo overlaps SUV — which very likely moves `IoU_hi` down. Because the tau map is **linear** on `[IoU_lo, IoU_hi]`, changing either endpoint changes **every** case's tau, not only the three type-5 cases (`0007_guitar-violin`, `0069_car-turn`, `0011_lucia_e5`). "Re-run the videos involving addition" is therefore necessary but is **not sufficient** for a rigorous re-render — `check-iou`'s new gate (4) makes the range-shift check explicit before `taumap` is re-run, and the phase-3 plan below re-renders the full 22-case arm accordingly. This also invalidates the **in-flight phase-2 controls** (`reversed`/`constant`, jobs 962628/962629): both were derived from the old tau multiset via `--assign reversed/constant`, so once the multiset changes those renders are stale and must not be evaluated as-is. |
| Rejected: CLIPSeg grounding | `CIDAS/clipseg-rd64-refined` is also cached and was the first pick when GroundingDINO appeared to be absent. Rejected once `IDEA-Research/grounding-dino-*` was found in the HF cache: CLIPSeg emits a 352×352 heatmap needing a threshold and a connected-component heuristic to become a box, and that threshold silently sets object extent — the exact quantity IoU measures. GroundingDINO returns a scored box directly, so the mask is not a function of a hand-tuned cutoff. |
| Detection failure handling | A phrase that does not fire is now a **hard failure**, not a silent empty contribution — the opposite of the two-phrase design, where non-firing WAS the mechanism. `src_word` missing on the source is always fatal (no `M_src`). `trg_word` missing on the anchor is fatal for types 1-5: for a swap it would read IoU = 0 and route to `tau_max` on a detector miss rather than on the edit, and for an addition it would read IoU = 1 and hide the addition entirely. Type 6 grounds nothing on the anchor and so cannot fail this way. |
| Single best-scoring box | **Now the design**, one box per side. The earlier objection — that it cannot describe types 5/6, which involve two objects — is answered by construction rather than by grounding a second phrase: see the add/remove row above. |
| `--box_thr` | **0.35** (0.25 → 0.5 → 0.35 on 2026-08-26). 0.25 let the detector return a box for every phrase on every image. **0.5 was tried and rejected**: it hard-failed `0058_boat`, whose anchor shows an unmistakably pink boat, because `'A pink fishing boat'` scores 0.45 while `'a white fishing boat'` scores 0.92 on the same hull — detector confidence tracks the PHRASING, not the image, so a high bar discards correct edits. 0.35 recovers `0058_boat` (0.45) and `0028_kite-walk` (`'Pikachu'` 0.41). Still a prior, not a derivation: with one phrase per side there is no absence to detect, so the threshold's only job is to reject weak boxes, and a phrase that stops firing fails loudly rather than silently emptying a mask. |
| Removal edits: phrase convention | **For type 6, `src_word` in `cases.json` names the object being REMOVED, not the untouched subject.** `0042_gym-ball` was `'a man'` / `'without a heavy gym ball'` and is now `'with a heavy gym ball'` / `'without a heavy gym ball'`. Both phrases then ground the ball: `M_src` = the ball's region, `M_edit` = empty once it is gone, so **IoU = 0 → tau_max**. That is the correct reading of a removal — the source content in that region must be entirely replaced by background, which is the maximal shape change and the case for releasing the anchor fastest. Under the old phrasing both sides grounded the unchanged man and the removal read as IoU ≈ 0.95, a no-op routed to tau_min. **Safe by construction:** the render's trigger words come from the benchmark's own `edit{T}_FiVE.json` (`e['source_object']` / `e['target_object']`, [run_fivebench.py:307-308](../evaluation/run_fivebench.py#L307-L308)) and `cases.json` is used only as a `video_name` filter ([:195](../evaluation/run_fivebench.py#L195)), so this edits the measurement and nothing about the generated video. |
| `cases.json` phrases are grounding phrases | Following from the above: `src_word`/`trg_word` in `cases.json` are the **IoU grounding phrases**, deliberately decoupled from the trigger words the model conditions on. They started as copies of the benchmark's `source_object`/`target_object` and remain equal for 21 of 22 cases, but they are not required to be, and `0042_gym-ball` no longer is. Anyone reading a trigger word off `cases.json` is reading the wrong file — the render reads `edit{T}_FiVE.json`. |
| Empty union | An empty union on the **edit** side means IoU = 0 and is legitimate **for type 6 only**, where it is the signature of a completed removal. For every other type it stays a hard failure: a swap whose target phrase misses on the anchor while the source phrase also misses would otherwise report IoU = 0 and route to tau_max on a detector miss rather than on the edit. An empty union on the **source** side is always a hard failure — there is nothing to compare against. The two cases are logged distinctly so a genuine removal is never confused with a grounding failure. |
| `0042_gym-ball` | Reads **exactly 0 by construction** now, since type 6 sets `M_edit` empty. The 2026-08-26 run under the two-phrase union read 0.153 instead, because with the ball genuinely gone from the anchor the detector handed both phrases *the man*; the panel confirmed the anchor edit itself succeeded. |
| Mask validation | `iou_gt_src` = IoU(source mask, FiVE-Bench GT `bmasks/{video}/00001.jpg`) per case, gated at median >= 0.7. The GT mask is used **only** as a gate, never as one side of the reported IoU — mixing a GT mask with a SAM2 mask would bias the ratio by producer. ⚠ It is only meaningful where `src_word` names the object FiVE annotated. Wherever the grounding phrase deliberately targets something else — `0042_gym-ball` (the removed ball) and `0017_kid-football` (the cap, after the 2026-08-26 phrase fix) — the column reads near zero by construction and must be excluded from the gate, not treated as a failure. |
| Frame rendition | Source mask read from `images/{video}/00001.jpg`, the same rendition `bmasks/` is aligned to. The mp4-vs-jpg gap (2026-08-20 log) is perceptual and does not move object extent, so it does not affect a shape IoU. |
| Resolution | Anchor mask resized to the source frame size with NEAREST before intersecting — the anchor PNG is at the model's working resolution and the jpg is not. |
| Degenerate-range guard | `r25_tau_map.py` refuses to write if `IoU_hi - IoU_lo < 0.05`. A collapsed range means the signal carries nothing, and rendering would burn GPU time on 22 near-identical taus. |
| Case set | **`evaluation/cases.json` as it stands — no new case file.** 22 cases, types 1/2/3/4/5/6 at **1/13/2/2/3/1**. The colour/material rows the task called for are already in it: `0017_kid-football` (red→yellow cap) and `0058_boat` (white→pink boat) at type 3, `0089_A_dog` (→plush) and `0028_kite-walk_e4` (→wooden person) at type 4. Verified: all four anchors exist under `five_bench/anchors/edit{3,4}/`, and all four videos are present in the corresponding `edit{T}_FiVE.json`. |
| Type balance | **Deliberately unbalanced toward type 2 (13/22).** The set is built to attack StreamEdit's weakness on object swaps, which is where a shape-driven tau should matter most and where a constant Eq. 4 exponent is most likely to be wrong. Consequence to respect when reading results: the overall mean is effectively a type-2 mean, and types 1 and 6 hold a single case each — their per-type rows in `r25_adaptive.csv` are anecdotes and must not be tabled as per-type measurements. |
| Overlapping videos | `0028_kite-walk` appears at type 2 (→Pikachu) and type 4 (→wooden person), and `0011_lucia` at type 2 (→lion) and type 5 (→dog added). Those two pairs are the only within-video contrasts available, and they are the cleanest evidence the IoU axis tracks the edit rather than the clip — same framing, same object scale, different edit magnitude. Every other case contributes a between-video comparison. |
| FiVE-Bench type labels | **type 3 = colour, type 4 = material** (verified against `edit{3,4}_FiVE.json`); the request had the two labels swapped, the set requested is unchanged. |
| Anchoring regime | **vp only** (paper §4.5 / R7 path), `--first_frame_edit_dir .../five_bench/anchors`. `novp` is meaningless here — with no anchor there is no `I_{0,edit}` in the render. `pvp` is a separate follow-up. |
| Eq. 4 reference | **Reuse, do not re-render:** the stored per-video `evaluation/csv/edit{T}_FiVE_r21_ref_vp_frame_stride8.csv` (R7 render, all 6 types, 100/100/100/100/9/10 rows), re-averaged over only the 22 R25 cases. Verified valid: the 2026-08-11 log records that R7's frames are mtime 07-22 23:06, i.e. the **post-seeding-fix generation**, and that R7's flag set is config-matched (`flow_shift 1.0`, `vp_mode vp`, `blend_sched None`) — so `r21_ref_vp` "is sound and needed no re-render". |
| Join key | `file_id`, per edit type. Sound because `evaluate.py` assigns `file_id` from `enumerate` over the **full** annotation file and `--cases_json` only `continue`s ([evaluate.py:268-272](../evaluation/fivebench/evaluate.py#L268-L272)) — ids do not renumber under a subset. |
| Subset comparability | Safe: `run_fivebench.py` re-seeds before **every** pair since 2026-07-22, so a `--cases_json` subset is bit-comparable to a full run. This is exactly the property R20's smoke gate 2 was added to protect. |
| Comparators — PHASE 1 | Eq. 4 `rho = 2` **only**. Recorded up front as an attribution ceiling, and the phase-1 verdict hit it exactly: the arm changes TWO things at once relative to the baseline — the SET of tau values (all moved off 2, mean 6.57) and the ASSIGNMENT of which video gets which. Every phase-1 number measures both together, so no re-analysis of that data can separate them. |
| Comparators — PHASE 2 (added 2026-08-27) | Two more arms, one 22-clip render each (~2 h L40S), each isolating one of those two changes. **`reversed`** keeps the same 22 tau values and inverts the pairing → tests whether the PAIRING carries signal. **`constant`** gives every pair tau = 6.57, the realized mean → tests whether VARYING tau helps at all. The second is the control a reviewer asks for first: if constant matches adaptive, the IoU machinery is unnecessary whatever the pairing does. |
| Why `reversed` and not a random shuffle | A single random permutation of 22 items is one draw and can land near or far from the true ordering by luck; the inversion is deterministic and maximises the contrast (colour edits get tau_max, object swaps get tau_min). If the pairing carries signal, adaptive must beat reversed clearly. |
| ⚠️ Why the controls are trustworthy even though the preservation metric is not | `*_unedit_part` masks come from FiVE's GT `bmasks/`, which annotate the SOURCE object, and evaluate.py passes that same mask as both `src_mask` and `tgt_mask` ([evaluate.py:302-316](../evaluation/fivebench/evaluate.py#L302-L316)). A shape-changing edit therefore puts the new object on pixels scored as "unedited", and the size of that mismeasurement scales with (1 − IoU) — which tau is a linear function of. Measured: `delta lpips` vs `|tau−2|` and vs `(1 − IoU)` give the **identical** r = +0.614, p = 0.002, because they are the same variable. So phase 1's −3.04 dB PSNR is confounded and cannot be read as pure background damage (nor as pure artefact — the blend acts on all tokens, so some real drift is expected). **The phase-2 contrasts are immune**: all three arms share the same tau distribution, hence the same average shape-change profile, so the bias applies equally and cancels in the difference. Run the controls BEFORE fixing the metric. |
| Fixing the preservation metric (separate, later) | Needs a per-frame mask of the EDITED object, which nothing currently produces: FiVE's `bmasks/` are per-frame but source-only, and `r25_iou.py --dump_masks` covers **frame 0 only**. The fix is to ground `trg_word` on sampled frames of BOTH arms' renders and score preservation outside `GT_src(f) ∪ M_trg^A(f) ∪ M_trg^B(f)` — the union across arms, so the excluded region is arm-independent. ⚠️ A per-arm exclusion region would be gameable in exactly the direction under test. Implementation path is `r21_bgonly`'s: build an alternate src root whose `bmasks/` holds the corrected masks and point `--src_image_folder` at it; no evaluate.py change. Required before quoting absolute preservation numbers in a paper, NOT required for the phase-2 contrasts. |
| Metric environment | **L40S,A100** with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `--exclude=node52` (retargeted 2026-08-27). H100 was the plan, but the partition is unusable — both nodes draining and fully allocated behind another user's 13-task array, and the eval sat PENDING for hours. Fitting L40S required dropping the two models that caused R20's exhaustion in the first place: **`five_acc`** (Qwen2.5-VL-7B) and **`motion_fidelity_score{,_edit_part}`** (CoTracker), via an explicit `--metrics` flag (a reduced `config_l40s.yaml` was tried first and did NOT work: `config.yaml`'s `metrics:` key is vestigial, `evaluate.py:200` reads `args.metrics`, so job 961342 loaded both models anyway and OOMed on a 40GB A100). 10 metrics remain, including both niqe (which was collateral damage in R20, not a cause). `evaluate.py` / `metrics_calculator.py` / `config.yaml` all untouched. **A100 is accepted alongside L40S**: R21 rejected it because "a 40GB A100 would be worse than the L40S that already failed", but that judged the FULL 13-metric set with Qwen2.5-VL + CoTracker resident — with those dropped the ceiling is far lower and 40GB is ample. GPU VRAM is still not exposed via `scontrol`, so this is inference rather than measurement; an A100 OOM would be caught by the job's `out of memory` grep. |
| ⚠️ Consequence: no temporal metric | `motion_fidelity` is the ONLY temporal-consistency measure in the set, and it is precisely the one most likely to catch this arm's main risk: 20 of 22 cases route to `tau > 2`, i.e. faster source release than Eq. 4, which is exactly what would degrade temporal stability. With it dropped, a preservation blowout can present as a WIN on the surviving metrics. The verdict metrics (`clip_similarity_target_image`, `lpips_unedit_part`) do survive, so the primary read stands — but any result from this config is **provisional on the temporal axis** and must be re-scored on H100 with the full config before publication. |
| Expected metric errors | **None.** `0010_giant-slalom`, the empty-mask `motion_fidelity_score_edit_part` degeneracy of R2/R14/R21, is not in `evaluation/cases.json`, so unlike R21/R24 any such error here is a real failure. |
| Sampler config | `--step 15`, `--flow_shift 1.0`, `--fg_boost_factor 4`, `--seed 0`, `rollout_chunk_size=21`, `rollout_overlap_block_num=1`, `sink_size=0` — the `run_fivebench.py` defaults shared by R1/R7/R20/R21. Only `blend_power` varies, per case. |
| Naming | Method dir `r25_adaptive_vp` under `/projects/dataggen/outputs/five_bench/r25_adaptive_tau`, never mixed with `r21_blend_full` or `r24_blend_tau`. |
| Compute budget | ~2 h inference (22 clips at R22's measured ~5.3 min/clip) on L40S + ~1 h eval on H100. Two orders of magnitude cheaper than R24 — the sweep is replaced by a single prior-fixed arm. |
| Default behaviour | `--tau_map` absent ⇒ `run_fivebench.py` bit-identical to today. A map that fails to cover a filtered pair is a startup `SystemExit`, never a silent fall back to 2.0. |
| Out of scope | Per-token tau (the R25 V0 implementation was deleted 2026-08-26), spatial tau (R26), fitting `f` against a per-video oracle over a constant-tau grid, `pvp`/`novp` regimes, other `f` families, full-bench scale. |

## Step commands

### iou

```bash
srun --partition=L40S --gres=gpu:1 --mem=32G --time=01:00:00 --pty \
  env HF_HUB_OFFLINE=1 python evaluation/r25_iou.py \
    --cases evaluation/cases.json \
    --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \
    --anchor_root /projects/dataggen/outputs/five_bench/anchors \
    --box_thr 0.35 \
    --viz_dir evaluation/figures/r25_iou_panels \
    -o evaluation/csv/r25_iou.csv
```

### check-iou

```bash
column -s, -t evaluation/csv/r25_iou.csv | less -S
python - <<'PY'
import csv, statistics as st
rows = list(csv.DictReader(open("evaluation/csv/r25_iou.csv")))
gt = [float(r["iou_gt_src"]) for r in rows]
print("n", len(rows), "median iou_gt_src", round(st.median(gt), 3), "min", round(min(gt), 3))
for grp, types in (("swap 1+2", {"1", "2"}), ("col/mat 3+4", {"3", "4"}), ("add/rm 5+6", {"5", "6"})):
    v = [float(r["iou"]) for r in rows if r["edit_type"] in types]
    if v:
        print(f"{grp:12s} n={len(v):2d} iou min {min(v):.3f} median {st.median(v):.3f} max {max(v):.3f}")
PY
```

### taumap

```bash
python evaluation/r25_tau_map.py \
  --iou_csv evaluation/csv/r25_iou.csv \
  --tau_min 2.0 --tau_max 10.0 \
  -o evaluation/csv/r25_tau_map.csv
column -s, -t evaluation/csv/r25_tau_map.csv
```

### smoke

```bash
sbatch slurm_scripts/five_bench/r25_smoke.sh
```

### wait-smoke

```bash
tail -40 logs/r25_smoke_{job_id}.out
grep -c '^IDENTICAL'   logs/r25_smoke_{job_id}.out   # expect 1
grep -c '^DIFFERS'     logs/r25_smoke_{job_id}.out   # expect 1
grep -c '^COVERAGE-OK' logs/r25_smoke_{job_id}.out   # expect 1
```

### infer

```bash
sbatch --array=0-5 slurm_scripts/five_bench/r25_infer.sh
```

### wait-infer

```bash
sacct -j {job_id} --format=JobID,State,Elapsed
OUT=/projects/dataggen/outputs/five_bench/r25_adaptive_tau/r25_adaptive_vp
for t in 1 2 3 4 5 6; do
  echo "edit$t $(ls -d "$OUT/edit$t"/*/ 2>/dev/null | grep -vc '_resize/$')"   # 1 13 2 2 3 1
done
grep -h 'failures:' logs/r25_infer_*.out
grep -h '^\[ok\]' logs/r25_infer_*.out | sort            # per-pair tau vs r25_tau_map.csv
```

### evaluate

```bash
sbatch slurm_scripts/five_bench/r25_eval.sh
```

### wait-eval

```bash
ls evaluation/csv/edit?_FiVE_r25_adaptive_vp_frame_stride8_avg.csv | wc -l   # expect 6
grep -c 'out of memory' logs/r25_eval_*.metrics.log                          # MUST be 0
grep -c 'Error: niqe'   logs/r25_eval_*.metrics.log                          # MUST be 0
# motion_fidelity is NOT in the --metrics list -- expect its columns absent, not zero
python - <<'PY'
import csv, glob
for f in sorted(glob.glob("evaluation/csv/edit?_FiVE_r25_adaptive_vp_frame_stride8.csv")):
    rows = list(csv.reader(open(f))); w = len(rows[0])
    print(f.split('/')[-1], "cols", w, "rows", len(rows) - 1,
          "misaligned", [i for i, r in enumerate(rows) if len(r) != w])
PY
```

### summarize

```bash
python evaluation/r25_summarize.py \
  --arm_csv_glob 'evaluation/csv/edit?_FiVE_r25_adaptive_vp_frame_stride8.csv' \
  --ref_csv_glob 'evaluation/csv/edit?_FiVE_r21_ref_vp_frame_stride8.csv' \
  --tau_map evaluation/csv/r25_tau_map.csv \
  --iou_csv evaluation/csv/r25_iou.csv \
  -o evaluation/csv/r25_adaptive.csv
```

### figures

```bash
python evaluation/r25_figures.py \
  --summary_csv evaluation/csv/r25_adaptive.csv \
  --clip_col clip_similarity_target_image \
  --lpips_col lpips_unedit_part \
  --tau_baseline 2.0 \
  --out_dir evaluation/figures
```

### verdict

```bash
column -s, -t evaluation/csv/r25_adaptive.csv | less -S
xdg-open evaluation/figures/r25_tau_iou.pdf
xdg-open evaluation/figures/r25_delta.pdf
```

## Pipeline

```mermaid
flowchart TD
    CR["evaluation/cases.json<br/>22 cases, 13 of them object swaps"]
    SRC["FiVE-Bench images/{video}/00001.jpg"] --> IOU
    ANC["anchors/edit{T}/{video}.png"] --> IOU
    GT["FiVE-Bench bmasks/{video}/00001.jpg"] --> IOU
    CR --> IOU["evaluation/r25_iou.py<br/>GroundingDINO x2 phrases x2 frames<br/>-> SAM2 -> per-side union -> IoU"]
    IOU --> IC["csv/r25_iou.csv<br/>iou + iou_gt_src gate"]
    IC --> TM["evaluation/r25_tau_map.py<br/>linear on measured range, tau in [1,8]"]
    TM --> TMC["csv/r25_tau_map.csv"]
    TMC --> RUN["evaluation/run_fivebench.py<br/>--tau_map --vp_mode vp --cases_json"]
    CR --> RUN
    ANC --> RUN
    RUN --> ARM["r25_adaptive_tau/r25_adaptive_vp/<br/>edit{T}/{video}/"]
    ARM --> EV["fivebench/evaluate.py<br/>H100 + expandable_segments<br/>--cases_json evaluation/cases.json"]
    EV --> ARMCSV["csv/edit{T}_FiVE_r25_adaptive_vp_frame_stride8.csv"]
    REF["csv/edit{T}_FiVE_r21_ref_vp_frame_stride8.csv<br/>stored Eq.4 vp reference"] --> SUM
    ARMCSV --> SUM["evaluation/r25_summarize.py<br/>join on file_id, re-average on the 22"]
    IC --> SUM
    TMC --> SUM
    SUM --> SC["csv/r25_adaptive.csv"]
    SC --> FIG["evaluation/r25_figures.py"]
    FIG --> F1["figures/r25_tau_iou.pdf"]
    FIG --> F2["figures/r25_delta.pdf"]
```

## Code to touch

| File | Change |
|---|---|
| `evaluation/r25_iou.py` | **New**, revised 2026-09-01 — one grounding per side (`src_word` on the source, `trg_word` on the anchor) → SAM2 masks → IoU. `M_edit = m(trg_word, anchor)` for types 1-5 (type 5 no longer unions with `M_src` — reverted 2026-09-01); type 6 sets `M_edit` empty and grounds nothing on the anchor. `--box_thr 0.35`. Writes `csv/r25_iou.csv` with `iou`, both boxes/scores/fired-flags, both areas and `iou_gt_src`. Optional `--viz_dir` writes 2×2 audit panels. |
| `evaluation/r25_tau_map.py` | **New** — linear map on the measured IoU range into `[tau_min, tau_max]`; writes `csv/r25_tau_map.csv`; refuses a degenerate range. |
| `evaluation/run_fivebench.py` | Add `--tau_map` (default `None`); load to `{(video_name, edit_type): tau}`; hard-error if it misses any filtered pair; at the `rollout_inference` call ([:303](../evaluation/run_fivebench.py#L303), beside `blend_power=args.blend_power` at [:317](../evaluation/run_fivebench.py#L317)) pass the per-pair tau as `blend_power`; echo it in the `[ok]` line. `--tau_map` is the only new flag; no other exponent knob exists since the R25 V0 removal. |
| `slurm_scripts/five_bench/r25_smoke.sh` | **New** — override-reaches-`blend_power` parity, non-degeneracy, coverage-error gates. L40S, `--mem=64G`, `--time=02:00:00`. |
| `slurm_scripts/five_bench/r25_infer.sh` | **New** — `--array=0-5`, one edit type per task, `--method r25_adaptive_vp --vp_mode vp --tau_map ... --cases_json ...`. L40S, `--mem=64G`, `--time=04:00:00`. |
| `slurm_scripts/five_bench/r25_eval.sh` | **New** — single task, L40S (was H100), `--mem=64G`, `--time=06:00:00`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, 16 metrics, `--cases_json evaluation/cases.json`, stem `r25_adaptive_vp`. |
| `evaluation/r25_summarize.py` | **New** — join arm vs stored R21 vp reference on `file_id` per type, re-average the reference over the 22 cases only, attach `iou`/`tau`, emit `csv/r25_adaptive.csv`. |
| `evaluation/r25_tau_map.py` *(phase 2)* | `--assign {iou,reversed,constant}`, default `iou` and byte-identical to phase 1. `reversed` permutes the SAME tau multiset by inverted IoU **rank** — not an IoU mirror, which would change the multiset unless the IoUs were symmetric, and the identical multiset is what makes the metric bias cancel. Ties break on `(video_name, edit_type)` for determinism. `constant` writes the realized mean (6.5697). |
| `slurm_scripts/five_bench/r25_infer.sh`, `r25_eval.sh` *(phase 2)* | `ARM=${R25_ARM:-adaptive}` → `METHOD=r25_${ARM}_vp`; TAU_MAP is `r25_tau_map_${ARM}.csv` except for `adaptive`, which keeps the phase-1 filename `r25_tau_map.csv`. Unknown arm exits 1. Unset reproduces phase 1. `#SBATCH --exclude=node52` moved INTO r25_infer.sh (phase 1 passed it on the command line, so the exclusion lived only in shell history). |
| `evaluation/r25_controls.py` *(phase 2, not yet built)* | **New** — three-way table (adaptive / reversed / constant) over the same 22 cases, reporting the two contrasts: adaptive − reversed (does the PAIRING matter) and adaptive − constant (does VARYING tau matter), each with per-case paired deltas and a sign test. |
| `evaluation/r25_figures.py` | **New** — `r25_tau_iou.pdf` (realized mapping by edit type), `r25_delta.pdf` (per-case Δ vs `\|tau − 2\|`), and `r25_grids/{case_id}.pdf` (F×4 qualitative page per video: source / Eq.4 / IoU masks / adaptive, every 10th rendered frame, source aligned by normalized time). |
| `evaluation/r25_iou.py` | `--dump_masks DIR` — persist `M_src`/`M_edit` per case as bit-packed npz so the grids can draw the overlap. Inert by default; the CSV is unaffected. |

Mapping, with the measured-range anchoring that is the locked choice:

```python
def iou_to_tau(iou: float, iou_lo: float, iou_hi: float,
               tau_min: float = 2.0, tau_max: float = 10.0) -> float:
    """High IoU (small shape change) -> tau_min: W_src = t**tau decays slowly, source held.
    Low IoU (large shape change) -> tau_max: W_src decays fast, source released early."""
    if iou_hi - iou_lo < 0.05:
        raise ValueError(f"degenerate IoU range [{iou_lo:.3f}, {iou_hi:.3f}]: the signal "
                         f"carries no per-video information, do not render")
    u = (min(max(iou, iou_lo), iou_hi) - iou_lo) / (iou_hi - iou_lo)   # 0 at lo, 1 at hi
    return tau_max - (tau_max - tau_min) * u
```

Two-stage grounded mask, both checkpoints already in the HF cache:

```python
def phrase_mask(img, phrase, box_thr=0.25):
    """SAM2 mask for the best GroundingDINO box for `phrase`, or None if it does not fire."""
    inputs = gdino_proc(images=img, text=f"{phrase}.", return_tensors="pt").to(device)
    det = gdino_proc.post_process_grounded_object_detection(
        gdino(**inputs), inputs.input_ids,
        threshold=box_thr, text_threshold=box_thr, target_sizes=[img.size[::-1]],
    )[0]
    if len(det["scores"]) == 0:
        return None, 0.0                       # absent from THIS image -- meaningful, not an error
    i = det["scores"].argmax()
    predictor.set_image(np.array(img))         # SAM2 sets object extent, no tuned threshold
    mask, _, _ = predictor.predict(box=det["boxes"][i].cpu().numpy(), multimask_output=False)
    return mask[0].astype(bool), det["scores"][i].item()


def side_union(img, src_word, trg_word, hw):
    """Union of both phrases on ONE image, resampled to the source frame grid."""
    parts, scores = [], {}
    for phrase in (src_word, trg_word):
        m, sc = phrase_mask(img, phrase)
        scores[phrase] = sc
        if m is not None:
            parts.append(resize_nearest(m, hw))
    if not parts:
        raise RuntimeError(f"neither {src_word!r} nor {trg_word!r} fired -- empty union has no IoU")
    return np.logical_or.reduce(parts), scores


# The whole measurement. A phrase absent from one side simply drops out of that union,
# which is how an addition (target only on the anchor) and a removal (target only on the
# source) register at all -- a single best-scoring box can describe only one of the two
# objects those edits involve.
hw = src_img.size[::-1]
m_src,  s_src  = side_union(src_img,    case["src_word"], case["trg_word"], hw)
m_edit, s_edit = side_union(anchor_img, case["src_word"], case["trg_word"], hw)
iou = (m_src & m_edit).sum() / max((m_src | m_edit).sum(), 1)
```

`run_fivebench.py` override — the one line that makes the arm adaptive:

```python
# R25: per-video release exponent. `blend_power` IS Eq. 4's rho at causal_model.py:334,
# so a per-case value here is exactly a per-case W_src = t ** tau. Missing coverage is a
# hard error: falling back to 2.0 would render a half-adaptive arm that still looks valid.
blend_power=(tau_map[(video_name, args.edit_type)] if tau_map else args.blend_power),
```
