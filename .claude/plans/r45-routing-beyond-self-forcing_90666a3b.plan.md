---
name: "R45 plan — routing beyond Self-Forcing (LongLive, gated method)"
overview: "Port the R35/R38 spatial divergence routing (stage 1 draft -> DINO/LPIPS divergence -> budget-linear rho field -> gated render) and the R36/R39 VP rho sweeps to the LongLive backbone (LongLive-native config), build all 8 method combos (draft N x metric x regime) ahead, and let a user method-gate pick which combos (and which matched-compute sweep N) actually run. Readout: routing gain over the same-backbone sweep curve, SF vs LL side by side."
task_id: R45
todos:
  - id: weights
    content: "USER downloads the LongLive weights (not built by /build-step). /build-step only verifies, before writing any GPU script: LongLive_StreamEdit/longlive_models/models/{longlive_base.pt,lora.pt} (configs/longlive_inference.yaml:28-29) and wan_models/Wan2.1-T2V-1.3B resolve; stop and report if missing."
    status: completed
    note: "VERIFIED 2026-10-06: user downloaded. longlive_models/models/longlive_base.pt (5.68 GB, zip OK, 829 entries) + lora.pt (2.80 GB, 1204 entries); wan_models/Wan2.1-T2V-1.3B with diffusion_pytorch_model.safetensors (825 tensors), Wan2.1_VAE.pth, models_t5_umt5-xxl-enc-bf16.pth. Both dirs git-ignored (LongLive_StreamEdit/.gitignore:3-4). Not yet loaded into the model -- smoke G1 is the real load test."
  - id: ll-utils
    content: "LongLive_StreamEdit/pipeline/utils.py -- copy `_schedule_blend_rate` (incl. tgt<tau>, t_cur), `build_rho_field_from_mask`, `blender_rate_from_rho` verbatim from Self-Forcing_StreamEdit/pipeline/utils.py. blender_rate_from_rho is called with t_next on both backbones (paper Eq. 4)."
    status: completed
    note: "BUILT 2026-10-06. The 3 functions copied verbatim (AST-identical to SF, docstrings aside) + header comment on LL's current_timestep = t_next key. VERIFIED on CPU: 0 schedule mismatches vs SF over 10 schedules x 15 steps; blender_rate_from_rho 0 diffs (rate + comp) over 5010 exponents x the 15-step t_next grid; tgt0.9 echoes N=5 [0,1,1,1,1], N=15 [0,0,1x13]. Only removed line: the old typing import. 2026-10-06 (user-requested): _find_phrase_token_indices_single made case-insensitive, AST-identical to SF's. Measured with the real umT5 tokenizer on all 838 FiVE (prompt, trigger) pairs: SF vs new LL 0 mismatches; stock LL found NO trigger tokens on 269/838 (32%), i.e. grounding + boosting silently off there."
  - id: ll-bridge
    content: "LongLive_StreamEdit/wan/modules/causal_model.py:360 -- blender_rate override as in SF causal_model.py:352-400: shared_dict['blender_rate'] (scalar, from blend_sched) or a per-token rate installed in the KV cache (rho field); absent => stock formula `1 - current_timestep ** blend_power`, bit-for-bit. NOTE: LL's shared_dict['current_timestep'] holds t_NEXT (edit_causal_inference.py:317-318), while SF's holds t_cur -- same key, different meaning; never copy SF bridge lines that read it."
    status: completed
    note: "BUILT 2026-10-06. Ported SF's blend_off / blender_rate / 4-field spatial rate (R10/R20/R26) only -- not R10 vp_kv, R32 seg_bias, R31 src_kv_full (out of scope). Prev-token rate gathered with LL's attn_seq_slice_list (sink + local window), same as trg_fg_mask. Stock line kept on shared_dict['current_timestep'] (= t_next on LL). VERIFIED on CPU, bf16: stock vs new scalar vs uniform rho=2 spatial field through a sink+window slice list -> 0/16 mismatches over t_next in {0.933, 0.533, 0.067, 0}. GPU parity is smoke G1/G2."
  - id: ll-pipeline
    content: "LongLive_StreamEdit/pipeline/edit_causal_inference.py -- inference() gains blend_sched, rho_frames (per-latent-frame exponent field [n_latent, 1560]) and step_dump (R22 per-step x0 hook), ported from SF edit_causal_inference.py (:106-:175 guards, :577-:610 _chunk_rho, :900-:910 scalar rate, _inject_blend_rates_to_kv_cache :1382). Rho rows indexed by ABSOLUTE latent frame per 3-frame block (LL has no rollout windows). All new kwargs default None => stock path."
    status: completed
    note: "BUILT 2026-10-06. inference(blend_sched, step_dump, rho_frames) -- kwarg named rho_frames (SF rollout_inference's name, not rho_rows) so evaluation/backbone.py passes one signature to both backbones. Ported: SF R31 rho_frames validation (LL has no rollout to host it), _chunk_rho with VP-prefix borrow of frame 0 (num_trim = 3 for tripled VP), blend_rho_tok buffer rolled with LL's own temp_mask arithmetic (incl. is_recompute / roped_offset), _inject_blend_rates_to_kv_cache, per-step scalar rate with t_cur from the loop variable, R22 step_out + identity assert. VERIFIED on CPU: compiles; simulated LL cache (3-frame VP sink + 10 blocks, 12-frame window) -> sink pinned, oldest block evicted from block 4 on, rho/mask misalignments 0, mask unchanged by the rho path. GPU parity is smoke G1-G3. STOCK-LL NOTE: under eviction the bridge reads the pre-roll mask cache with post-roll slices (causal_model.py:244-266 vs pipeline update after the clean forward), i.e. prev-key fg mask lags one block from block 4 on; rho rides the same slices, so it inherits the same lag. Left as stock (affects sweep and routing equally; fixing it breaks G1)."
  - id: ll-text-encoder
    content: "LongLive_StreamEdit/wan/modules/t5.py + utils/wan_wrapper.py -- WanTextEncoder loads via SF's umt5_xxl_encoder_from_checkpoint (fp16, cuda, mmap) instead of the stock fp32 CPU->GPU build that saturated GPU memory (user-requested 2026-10-06)."
    status: completed
    note: "BUILT 2026-10-06. t5.py replaced by SF's (byte-identical; the only diff was the missing function + __all__ entry). wan_wrapper.py: import + WanTextEncoder.__init__ switched to umt5_xxl_encoder_from_checkpoint(..., dtype=torch.float16, device='cuda'). VERIFIED: compiles; checkpoint vs meta T5Encoder 0 missing / 0 unexpected / 0 shape mismatches; no other LL code imports umt5_xxl. Numerics: both drivers cast the pipeline to bf16, but bf16->fp16->bf16 flushes 267,815 / 5.68 B tiny weights (4.7e-5), so LL text embeddings now equal SF's, not stock LL's."
  - id: ll-ckpt-load
    content: "LongLive_StreamEdit/inference_edit_streamedit.py load_pipe -- SF's memory-lean generator load (SF inference_edit_streamedit.py:74-79 vs the original hh.py): torch.load(map_location=device, weights_only=True), load_state_dict(assign=True), del the dict (user-requested 2026-10-06)."
    status: completed
    note: "BUILT 2026-10-06. longlive_base.pt = {'generator': 825 fp32 tensors}, use_ema false -> the strict non-EMA branch runs; it now uses assign=True. EMA branch unchanged except it reads the device-loaded dict. VERIFIED: compiles; weights_only load works for longlive_base.pt and lora.pt; CPU proxy shows assign=True keeps nn.Parameter and gives identical values before and after the bf16 cast -> numerically a no-op. NOT ported: SF's low_memory threshold change (free VRAM < 50 instead of < 40), not requested."
  - id: backbone-loader
    content: "evaluation/backbone.py (new, shared) -- load_backbone(args) for sf|ll (LL: LongLive_StreamEdit/inference_edit_streamedit.py:40-147 incl. LoRA, flow_shift override, arange step grid), encode_vp(backbone, ...) (LL: tripled first frame), edit_call(backbone, pipeline, ...) (SF rollout_inference vs LL inference, same return contract incl. step_dump)."
    status: completed
    note: "BUILT 2026-10-06. evaluation/backbone.py: activate(backbone) (sys.path + chdir, returns the build's own find_closest_num_frame / read_json / find_phrase_token_indices / set_seed), load_backbone (sf = the build's load_pipe unchanged; ll = LL load_pipe + LL driver's post-load steps: bf16 cast, DynamicSwap if low_memory, generator/vae to device), encode_vp (src frame then anchor, one clear_cache; ll repeats x3 -> triple_first_frame), edit_call (sf -> rollout_inference; ll -> inference, refusing any set SF-only option: vp_latent, union_*, tau_bg/fg, seg_bias_table, vp_head_gate, src_kv_full). VERIFIED on CPU: compiles; activate + imports work for both builds in separate processes; every kwarg the drivers pass exists on SF rollout_inference and LL inference. Not wired into any driver yet (driver-flags)."
  - id: driver-flags
    content: "evaluation/run_fivebench.py, evaluation/r31_stage1.py, evaluation/r31_stage3.py -- add `--backbone {sf,ll}` (default sf, byte-identical) routed through evaluation/backbone.py; ll sets root LongLive_StreamEdit + configs/longlive_inference.yaml. --fg_boost_factor stays explicit in every script (2 for LL)."
    status: completed
    note: "BUILT 2026-10-06. All 3 drivers: `import backbone as bb`; bb.add_backbone_args replaces --sf_root/--config_path (None => per-backbone default via bb.resolve_defaults); --vp_mode pvp refused on ll; sys.path/chdir/imports -> bb.activate (helpers from the active build); load_pipe -> bb.load_backbone; vp encode -> bb.encode_vp; rollout_inference -> bb.edit_call(+triple_first_frame, False on sf = its default). r31_stage1 npz gains `backbone`. Manifest CSVs left unchanged (downstream guards parse them positionally). VERIFIED on CPU: all compile; sf defaults == pre-R45 (root + config); encode_vp == old inline code on a stateful mock VAE (same latents, same call log); ll x3 tripled frames + flags; edit_call routing + SF-only refusal. GPU byte-parity for sf is smoke G0."
  - id: select
    content: "evaluation/r45_select.py (new) -- single source of truth for the 8-combo table + the method choice. `--set --draft_n {5,15} --combos <names>` writes evaluation/r45_method.json (refuses combos whose N != draft_n unless --allow_mixed); subcommands stage1|stage2|stage3|sweepM|eval print the sbatch --array list (or the matched N) from it; `check <array_id> <stage>` exits 1 if a task's combo is not selected (every R45 GPU script calls it first)."
    status: completed
    note: "BUILT 2026-10-07. evaluation/r45_select.py: --set (draft_n in {5,15}, combos validated, mixed-N refused without --allow_mixed, matched_n = 15 + draft_n); stage1 / stage2 / stage3 / sweepM / eval [--table] / check <id> <stage> / framecheck (per-clip PNG counts of every table method vs R36 r36_rho2_vp). Table order: sweep n15 x8, sweep n{matched} x8, r45_{combo} (frame root <method>/step14, as r31_stage3 writes). R45_METHOD_JSON env override for tests only. VERIFIED on a temp json: missing json refused; N=5 x4 -> stage1 2,3 / stage2-3 4,5,6,7 / sweepM 20 / eval 0-19; N=15 x2 -> 0,1 / 0,3 / 30 / 0-17; check accepts selected and exits 1 on unselected; mixed N refused. Real evaluation/r45_method.json NOT created (method-gate writes it)."
  - id: smoke
    content: "slurm_scripts/five_bench/r45_smoke.sh + evaluation/r45_smoke_parity.py -- 2-clip GPU smoke with gates G0-G6 (see Code to touch). Blocks step `smoke`."
    status: completed
    note: "BUILT 2026-10-07. r45_smoke.sh: G0 (run_fivebench --backbone sf, 0001_bus, sha256 vs R36 r36_rho2_vp) and G6 (run_fivebench --backbone ll end to end on 0001_bus; frames LL == SF == find_closest_num_frame = 69), then the stock reference via `git archive HEAD LongLive_StreamEdit` (spec said git worktree; archive = same pristine code, LL dir only) + copied t5.py / wan_wrapper.py / inference_edit_streamedit.py + symlinked weights, guarded by a no-`rho_frames` check, then r45_smoke_parity.py run stock / run ported / gates (G1-G5). G0/G6 use 0001_bus (not 0034_cows): G6 needs R36's reference frames and G1 needs clips where stock trigger matching agrees. VERIFIED on CPU: both files compile/parse; HEAD archive has no rho_frames and the case-sensitive tokenizer; inline find_closest formula == the builds' function on 13..499 (80 -> 69). 1 GPU, L40S, 4 h."
  - id: sweep-script
    content: "slurm_scripts/five_bench/r45_sweep.sh -- r36_infer.sh with --backbone ll, fg_boost 2, positional N (15 or matched), array 0-7 over RHOS=(2 3 4 6 8 10 20 50), METHOD=ll_rho{R}_n{N}_vp, OUT_ROOT=r45_sweep/n{N}."
    status: completed
    note: "BUILT 2026-10-07. Based on r39_infer.sh (edit-type resume skip, conda.sh activation, node exclusions), not r36_infer.sh. N = $1 in {15,20,30}; any N != 15 must equal `r45_select.py sweepM` (matched_n), else refuse. run_fivebench.py --backbone ll, fg_boost 2, flow_shift 1.0, seed 0. --time=24:00:00 for every N (an #SBATCH line cannot depend on $1; spec said 'scaled by N') -- re-check against smoke G4's s/frame before launching N=30. VERIFIED: bash -n; refusal paths tested (bad N, no array id, N=20 without method json, N=20 vs matched_n 30), none reached rendering; no output dir or method json created."
  - id: stage-scripts
    content: "slurm_scripts/five_bench/r45_stage1.sh (array over 4 runs: n15_unblended n15_tgt09 n5_unblended n5_tgt09), r45_stage2.sh / r45_stage3.sh (array over the 8 combos), r45_absnorm.sh (CPU; r31_rho_map.py unchanged) -- copies of the r38_* scripts with --backbone ll, fg_boost 2, r45_ roots, and an `r45_select.py check` guard at the top."
    status: completed
    note: "BUILT 2026-10-07. r45_stage1.sh (RUNS n15_unblended/n15_tgt09/n5_unblended/n5_tgt09, SCHEDS zero/tgt0.9, NSTEPS 15/5, dump/measure N-1), r45_stage2.sh (8-combo case table -> arm/run/N, five-bench env), r45_absnorm.sh (no args = the method-gate selection via r45_select.py stage2; div_max by metric 0.65/1.0; r31_rho_map.py unchanged at --step 15), r45_stage3.sh (step 15 pinned, dump 14, per-file manifest guard). All renders --backbone ll --fg_boost_factor 2; --rollout_chunk_size dropped (SF-only). Each GPU task runs `r45_select.py check <id> <stage>` right after conda activation. VERIFIED: bash -n x4; RUNS/SCHEDS/NSTEPS and all three COMBOS tables == r45_select.py index for index; stage-2 case table consistent with combo names; absnorm with a temp selection picks exactly the selected combos then stops at preflight on empty roots; unselected array ids exit at `check` in stage1/2/3 before any model load. No outputs or method json created."
  - id: eval-scripts
    content: "slurm_scripts/five_bench/r45_eval.sh, r45_fiveacc.sh, r45_clipd.sh -- R38's three, indexing the method table printed by `r45_select.py eval --table` (sweep n15 x8, sweep nM x8, selected routed combos)."
    status: completed
    note: "BUILT 2026-10-07. [2026-10-22: r45_fiveacc.sh REWRITTEN as ONE sequential, resumable job looping over all table rows (user: fiveacc must be sequential; Qwen2.5-VL-7B, ~30 min/method, 1 queue entry instead of one array task per method -- the 17-row table made eval 17 + fiveacc 17 + clipd 1 = 35 > the 30-job cap, now 19). `--array=i` still reruns one row; positional args select rows; a method whose CSVs are already complete is skipped.] r45_eval.sh / r45_fiveacc.sh (originally): array id -> row of `r45_select.py eval --table` (method, frame root), guarded by `check <id> eval`; stems r45_{method} (sweeps) / {method} (routed, already r45_) and r45_fiveacc_{method minus r45_}; same 9 metrics / five_acc-only + R38's guards. r45_clipd.sh: one job, all table methods or a positional subset (so the N=15 sweep can be scored before sweepM exists), labels as r45_eval stems, -> evaluation/csv/r45_clip_directional.csv, --time 24 h (up to 24 methods). VERIFIED: bash -n x3; dry runs on a temp selection resolve tasks 0 / 9 / 16 / 17 to the right method, stem and frame root, then stop at the missing-render preflight before evaluate.py; out-of-table id 18 refused; clipd subset preflight + unknown-method refusal; no CSVs or method json created."
  - id: divstats
    content: "evaluation/r45_div_stats.py -- reuse r38_div_stats.py functions on r45_div/{combo} next to r35_div/r38_div (SF): mean, p50/p90/p99, saturation fraction at div_max 0.65/1.0."
    status: completed
    note: "BUILT 2026-10-07. evaluation/r45_div_stats.py: imports pooled_d / DIV_MAX / FIVE from r38_div_stats.py; for each SELECTED LL combo, pairs the SF combo at the same N, metric, regime (N=15: r35_div/{metric}_{unblended,first2}; N=5: r38_div/{metric}_n5_{unblended,tgt09}); rows with backbone + N; prints LL/SF d_mean and sat_frac ratios and FLAGs LL sat_frac > 2x SF (plan Decisions). VERIFIED: compiles; all four SF mappings exist on disk; end-to-end on real data with the LL side symlinked to SF's own maps -> ratios 1.000, 0 flags, SF sat_frac 0.00516 / 0.00604 == r38_div_stats.csv. No real CSV or method json written."
  - id: score
    content: "evaluation/r45_score.py -- 3 two-panel figures (SF | LL) + evaluation/csv/r45_arms.csv (routing gain at matched bg LPIPS vs each same-backbone curve)."
    status: completed
    note: "BUILT 2026-10-07. evaluation/r45_score.py: LL curves built with r36_summarize.py on the R45 stems (r45_rho_sweep_n{15,M}.csv; --no_summarize to skip), SF curves R36 + R39 n{M}; arms via r35_score loaders (SF counterpart: R35 at N=15 with tau=0.9 -> first2, R38 at N=5); gains with r39_pareto.interp_at (never extrapolated, flags ok/outside_range/nonmonotone) vs N=15 and matched curves; r45_arms.csv long format (backbone, combo, x_metric, value, lpips, ref/gain/flag x2); 3 two-panel PDFs (x = CLIP-T / CLIP-D / FiVE-Acc yes-no, y = lpips_unedit_part). VERIFIED with LL aliased to SF data (N=15 selection, R36 standing in for the matched curve since r39_rho_sweep_n*.csv do not exist yet): 24 rows, SF == LL on every row; SF arm values == r35_arms.csv (24 values, max |diff| 0.0); clipd figure inspected. Hard-fails at run time if R39's matched-N sweep CSV is missing."
steps:
  - id: smoke
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r45_smoke.sh
    status: completed
    completed_at: 2026-10-07
    job_id: "1027277"
  - id: wait-smoke
    type: manual
    wait_for: smoke
    check_hint: "grep GATE logs/r45_smoke_{job_id}.out -- G0-G6 all PASS; echoed vectors N=15 [0,0,1x13], N=5 [0,1,1,1,1]; peak VRAM + s/frame logged for the longest clip; NFE counts logged (G5)"
    status: completed
    completed_at: 2026-10-07
    note: "verified 2026-10-07 from logs/r45_smoke_1027277.out (node39, 'failures: 0'): G0 sf via --backbone == R36 r36_rho2_vp sha256 on 0001_bus (69 PNGs); G6 LL 69 == SF 69 == find_closest 69; G1 stock == ported LL, max |d| = 0, on 0021_landing + 0025_baseball (stock reference = git archive HEAD + R45 encoder/ckpt files); G2 uniform rho=2 field == scalar; G3 tgt0.9 == first2 at N=15, echoes N=15 [0,0,1x13] / N=5 [0,1,1,1,1]; G4 longest clip 0010_giant-slalom (127 src -> 117 frames) at N=30: 205 s = 1.76 s/frame, peak VRAM 29.9 GB on L40S; G5 0001_bus (6 blocks): draft5 b2 30 + render15 b2 90 == single20 b2 120, draft15 90 + 90 == single30 180, extra batch-1 = 14 = the draft's own clean-context forwards (2 VP prefix + 2/block) -> same accounting as SF/R39."
  - id: sweep15
    type: sbatch
    command: sbatch --array=0-7 slurm_scripts/five_bench/r45_sweep.sh 15
    wait_for: wait-smoke
    sets_status: running
    output_paths:
      - ~/Data/dataggen/outputs/five_bench/r45_sweep/n15
    status: completed
    note: "job 1027752 submitted 2026-10-07, then CANCELLED 2026-10-07 at user request (QOSMaxSubmitJobPerUserLimit blocked the stage2->absnorm->stage3 chain; user chose to free slots and focus on R44 first). 0 clips rendered on any task. Resubmit when returning to R45."
    completed_at: 2026-10-12
    job_id: "1035812"

  - id: method-gate
    type: manual
    wait_for: wait-smoke
    check_hint: "USER DECISION (needs R38 analyzed). Pick draft N (5|15) and the (metric, regime) combos, then: python evaluation/r45_select.py --set --draft_n N --combos ...  ; cat evaluation/r45_method.json"
    status: completed
    completed_at: 2026-10-07
    note: "USER DECISION 2026-10-07: inherited, not reselected on LongLive (R46's plan already assumed this) -- draft_n=5, combos=['ll_dino_n5_tgt09'] (DINO . tau=0.9 . N=5), matched_n=20. Basis: R38's decision (daily.md 2026-10-07 'decision: final routing config') on Self-Forcing -- DINO.tau0.9 vs LPIPS.tau0.9 statistically indistinguishable overall, DINO significantly better on non-rigid substitutions, N=5 chosen on cost (~1.54x a plain render vs 2x at N=15). evaluation/r45_method.json written via r45_select.py --set --draft_n 5 --combos ll_dino_n5_tgt09."
  - id: sweepM
    type: sbatch
    command: sbatch --array=0-7 slurm_scripts/five_bench/r45_sweep.sh $(python evaluation/r45_select.py sweepM)
    wait_for: method-gate
    output_paths:
      - ~/Data/dataggen/outputs/five_bench/r45_sweep/n20
      - ~/Data/dataggen/outputs/five_bench/r45_sweep/n30
    status: completed
    completed_at: 2026-10-12
    job_id: "1035813"

  - id: stage1
    type: sbatch
    command: sbatch --array=$(python evaluation/r45_select.py stage1) slurm_scripts/five_bench/r45_stage1.sh
    wait_for: method-gate
    output_paths:
      - ~/Data/dataggen/outputs/five_bench/r45_stage1
      - ~/Data/dataggen/outputs/five_bench/r45_latents
    status: completed
    note: "job 1027775 (array task 3, run n5_tgt09) submitted 2026-10-07, then CANCELLED 2026-10-07 at user request, same as sweep15 -- QOSMaxSubmitJobPerUserLimit blocked the downstream chain, user chose to free slots and focus on R44 first. Still PENDING at cancellation (0 of 419 clips rendered). Resubmit when returning to R45."
    completed_at: 2026-10-12
    job_id: "1035814"
    note: "run n5_tgt09 (array index 3), ll_dino_n5_tgt09."

  - id: wait-stage1
    type: manual
    wait_for: stage1
    check_hint: "419 npz per selected run, 100/100/100/100/9/10; echoed blender_rate vectors as in wait-smoke; npz backbone=ll"
    status: pending
  - id: stage2
    type: sbatch
    command: sbatch --array=$(python evaluation/r45_select.py stage2) slurm_scripts/five_bench/r45_stage2.sh
    wait_for: wait-stage1
    output_paths:
      - ~/Data/dataggen/outputs/five_bench/r45_div
    status: completed
    completed_at: 2026-10-12
    job_id: "1035815"
    note: "combo ll_dino_n5_tgt09 (array index 5), --dependency=afterok:1035814."

  - id: wait-stage2
    type: manual
    wait_for: stage2
    check_hint: "419 npz per selected combo, d shape matches its stage-1 z_trg, 0 NaN, 0 [ERROR]"
    status: pending
  - id: divstats
    type: local
    wait_for: wait-stage2
    command: python evaluation/r45_div_stats.py
    output_paths:
      - evaluation/csv/r45_div_stats.csv
    status: completed
    completed_at: 2026-10-26
    note: "run by the user on gpu-gw (/store mounted), exit 0, 0 saturation flags. dino_n5_tgt09 at N=5, LL vs SF: d_mean 0.0902 vs 0.0794 (x1.136), p50 0.0391 vs 0.0338, p90 0.2472 vs 0.2037, p99 0.6248 vs 0.6090, sat_frac(d >= 0.65) 0.00644 vs 0.00516 (x1.25, below the 2x flag). The fixed div_max 0.65 fits LL's map; LL diverges ~14% more on average."

  - id: absnorm
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r45_absnorm.sh
    wait_for: wait-stage2
    output_paths:
      - ~/Data/dataggen/outputs/five_bench/r45_div_abs
      - ~/Data/dataggen/outputs/five_bench/r45_rho
    status: completed
    completed_at: 2026-10-12
    job_id: "1035816"
    note: "--dependency=afterok:1035815."

  - id: stage3
    type: sbatch
    command: sbatch --array=$(python evaluation/r45_select.py stage3) slurm_scripts/five_bench/r45_stage3.sh
    wait_for: absnorm
    output_paths:
      - ~/Data/dataggen/outputs/five_bench/r45_arms
    status: completed
    completed_at: 2026-10-12
    job_id: "1035817"
    note: "combo ll_dino_n5_tgt09 (array index 5), --dependency=afterok:1035816. Whole chain sweep15+sweepM+stage1->stage2->absnorm->stage3 submitted together 2026-10-12, queue 1/30 -> 21/30."

  - id: wait-renders
    type: manual
    wait_for: [sweep15, sweepM, stage3]
    check_hint: "16 sweep methods + selected routed combos: 419 frame dirs each, 0 non-ok manifest rows (count per file); per-clip frame count == R36 r36_rho2_vp on all 419 (python evaluation/r45_select.py framecheck)"
    sets_status: finished
    status: completed
    completed_at: 2026-10-22
    note: "verified 2026-10-22 from disk (R45 outputs restored from /store to /home): 16 sweep methods + the routed arm, 419 clip dirs each, 6704 sweep clips = 16x419 all manifest rows ok, per-clip PNG count == manifest n_frames (0 mismatches), all 456048 sweep PNGs have a valid signature + IEND trailer, r45_sweep byte-compared to /store with rsync -rtni (only a directory mtime differs). NOT run: framecheck vs R36's r36_rho2_vp frames (R36 still on /store, unreadable from this node)."

  - id: eval
    type: sbatch
    command: sbatch --array=$(python evaluation/r45_select.py eval) slurm_scripts/five_bench/r45_eval.sh
    wait_for: wait-renders
    status: completed
    job_id: "1057624/1064402"
    note: "Two submissions covering all 17 table rows: 1057624 = rows 0-7 (N=15 sweep) + 16 (routed arm ll_dino_n5_tgt09), all OK by 2026-10-25; 1064402 = rows 8-15 (N=20 sweep), submitted by the user 2026-10-25 (array 8-15, PENDING at submit). Step is completed once both are submitted; wait-eval verifies the results."
    completed_at: 2026-10-25

  - id: fiveacc
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r45_fiveacc.sh
    wait_for: wait-renders
    status: completed
    completed_at: 2026-10-22
    job_id: "1057622"
    note: "ONE sequential job over all 17 table rows (~30 min each)."

  - id: clipd
    type: sbatch
    command: sbatch slurm_scripts/five_bench/r45_clipd.sh
    wait_for: wait-renders
    status: completed
    completed_at: 2026-10-22
    job_id: "1057623"
    note: "single job over all 17 table methods."

  - id: wait-eval
    type: manual
    wait_for: [eval, fiveacc, clipd]
    check_hint: "17 methods: 6 metric CSVs + 6 five_acc CSVs each, 419 rows, 0 NaN; r45_clip_directional.csv 17*419 rows; 0 Error:/OOM; the single fiveacc job exits 0 with no FAILED rows (grep FAILED logs/r45_fiveacc_*.out)"
    status: completed
    completed_at: 2026-10-26
    note: "verified 2026-10-26: 17 methods, all 6 per-edit-type 9-metric CSVs + 6 five_acc CSVs complete (0 problems), r45_clip_directional.csv 17 x 419, 0 Error:/OOM lines in any eval/fiveacc metric log; eval jobs 1057624 (rows 0-7,16) + 1064402 (rows 8-15) all OK."

  - id: figures
    type: local
    wait_for: wait-eval
    command: python evaluation/r45_score.py --all
    output_paths:
      - evaluation/csv/r45_arms.csv
      - evaluation/figures/r45_clip_vs_lpips.pdf
      - evaluation/figures/r45_clipd_vs_lpips.pdf
      - evaluation/figures/r45_fiveacc_vs_lpips.pdf
    sets_status: analyzed
    status: completed
    completed_at: 2026-10-26
    note: "python evaluation/r45_score.py --all, exit 0: r45_rho_sweep_n15/n20.csv rebuilt via r36_summarize, r45_arms.csv (6 rows), 3 two-panel figures; all 6 gains flagged ok (inside both curves). divstats (step above) NOT run: r45_div/r38_div/r35_div are on /store, unreadable from nodemm06 -- run on gpu-gw."

isProject: true
---

# R45: Routing Generalizes Beyond Self-Forcing

## Context

**The routing config is settled, inherited from R38's Self-Forcing decision (2026-10-07), not reselected on LongLive**: DINO · τ = 0.9 · draft N = 5 (`ll_dino_n5_tgt09`), matched N = 20. All 8 combos were still built ahead so the choice did not block the port; `evaluation/r45_method.json` records it.

Is the spatial divergence routing (R35 at draft N=15, R38 at N=5) a property of the Self-Forcing backbone, or does it transfer? R45 runs the same pipeline on **LongLive** (text + Visual Prompt). Each routed arm is read against a **same-backbone VP rho sweep**: the R36 analog at N=15 plus the R39 analog at the matched step count (15 + draft N). A user **method-gate** selects what runs. The N=15 sweep does not depend on that choice and starts right after the smoke test. The LongLive build in this repo is the stock StreamEdit code and has none of the routing machinery, so the port is the main build cost.

**Done when:** the selected LongLive combos and both LongLive sweeps are rendered and scored on all 419 pairs. The three two-panel Pareto figures place the SF and LL routing gains side by side. `r45_arms.csv` reports each arm's gain at matched bg LPIPS over its same-backbone curves. Answer: does routing sit above the same-backbone sweep curve on LL as it does on SF?

## Execution steps

| # | Step id | Type | What it does | Sets status |
|---|---------|------|--------------|-------------|
| — | *(prep)* | — | User downloads the weights; `/build-step` the other 12 `todos` (LL port, loader, selector, scripts) | `implemented` |
| 1 | `smoke` | sbatch | 2 clips, gates G0-G6 (SF regression, LL stock parity, rho plumbing, tgt0.9, VRAM, NFE, frame parity) | |
| 2 | `wait-smoke` | manual | All gates PASS | |
| 3 | `sweep15` | sbatch | LL VP rho sweep at N=15, array 0-7 | `running` |
| 4 | `method-gate` | manual | **User** picks draft N + combos → `r45_method.json` (after R38 is analyzed) | |
| 5 | `sweepM` | sbatch | LL VP rho sweep at N = 15 + draft N (20 or 30), array 0-7 | |
| 6 | `stage1` | sbatch | LL drafts for the selected (N, regime) runs | |
| 7 | `wait-stage1` | manual | 419 npz per run | |
| 8 | `stage2` | sbatch | Divergence npz per selected combo | |
| 9 | `wait-stage2` | manual | 419 npz per combo | |
| 10 | `divstats` | local | LL vs SF raw-divergence distribution + saturation | |
| 11 | `absnorm` | sbatch | CPU: absolute norm + rho fields (unchanged mapping) | |
| 12 | `stage3` | sbatch | Gated 15-step LL render per selected combo | |
| 13 | `wait-renders` | manual | Sweeps + routed arms complete | `finished` |
| 14 | `eval` | sbatch | 9-metric harness over the method table | |
| 15 | `fiveacc` | sbatch | `five_acc` over the method table: **one sequential, resumable job** (17 rows, ~30 min each, ~12 h), not an array | |
| 16 | `clipd` | sbatch | CLIP-directional, single job | |
| 17 | `wait-eval` | manual | CSV coverage, 0 errors | |
| 18 | `figures` | local | SF \| LL Pareto figures + `r45_arms.csv` | `analyzed` |

```
/run-step R45 smoke         /run-step R45 stage1        /run-step R45 stage3
/run-step R45 wait-smoke    /run-step R45 wait-stage1   /run-step R45 wait-renders
/run-step R45 sweep15       /run-step R45 stage2        /run-step R45 eval|fiveacc|clipd
/run-step R45 method-gate   /run-step R45 wait-stage2   /run-step R45 wait-eval
/run-step R45 sweepM        /run-step R45 divstats      /run-step R45 figures
                            /run-step R45 absnorm
```

## Decisions

| Question | Choice |
|---|---|
| Backbone config | **LongLive-native**: `configs/longlive_inference.yaml` (base + LoRA r256, `local_attn_size 12`, `sink_size 3`), VP as a **tripled first frame** (LL default), `fg_boost 2` (paper: ω=2 outside Ours(SF)), `flow_shift 1.0` pinned (overrides the config's 5.0 as SF runs do), seed 0, R7 anchors (`~/Data/dataggen/outputs/five_bench/anchors`) |
| Step count | **15** for the N=15 sweep and every stage-3 render. Paper: "15 steps for standard comparisons, 5 for long-video" applies to both backbones. The StreamGVE driver replaces LongLive's native 4-step grid with `np.arange(1000, 0, -1000/step)` |
| LL blend rate | Stock LL already follows **paper Eq. 4** (`r = 1 − t_{i−1}^ρ`, t_{i−1} = next timestep, since t_N = 1 → t_0 = 0): `causal_model.py:360` reads `shared_dict['current_timestep']`, which LL fills with `timestep_next` (`edit_causal_inference.py:317-318`). Same rate as SF, which keeps t_next under `current_timestep_next`. No change |
| Rho field | `budget_linear`, `tau_min 2`, `tau_max 50`, `--step 15`, `r31_rho_map.py` **unchanged**: its t_next grid is the grid LL's bridge uses. Per-token rate = `1 − t_next^ρ(f,p)` |
| Regimes | `zero` (unblended) and `tgt0.9` on the warped input t, as in R38. N=15 anchors t = 1.0, 0.933. N=5 anchors t = 1.0 |
| Normalisation | SF's absolute values: `div_min 0`, `div_max` **0.65 DINO / 1.0 LPIPS**, not recalibrated. They are scales on the perceptual metrics (settled 2026-09-24, `r31_div_abs.py:16-20`), not on the backbone, so both backbones use the **same d → m function**. Recalibrating on LL would tune the method to LL's results. `r45_div_stats.csv` logs the shift and saturation against SF. If LL's saturated fraction is > 2× SF's for a combo, flag it at `divstats` before stage 3 |
| Combo table (array idx) | 0 `ll_dino_n15_unblended` · 1 `ll_dino_n15_tgt09` · 2 `ll_lpips_n15_unblended` · 3 `ll_lpips_n15_tgt09` · 4 `ll_dino_n5_unblended` · 5 `ll_dino_n5_tgt09` · 6 `ll_lpips_n5_unblended` · 7 `ll_lpips_n5_tgt09`. Stage-1 runs: 0 `n15_unblended` · 1 `n15_tgt09` · 2 `n5_unblended` · 3 `n5_tgt09` |
| Open method | **Build all 8, gate the launch.** `evaluation/r45_method.json` (written at `method-gate` by `r45_select.py --set`) is the single source of truth. Every array is derived from it, and every GPU task refuses an unselected combo. Mixed-N selections need `--allow_mixed`. **Resolved 2026-10-07:** `ll_dino_n5_tgt09` only, draft N=5, matched N=20 -- inherited from R38's Self-Forcing decision, not reselected on LongLive |
| Sweeps | **Full** R36 grid ρ ∈ {2,3,4,6,8,10,20,50}. `sweep15` at N=15 runs right after smoke. `sweepM` at N = 15 + draft N (**20 if N=5, 30 if N=15**, the R39 single-pass logic) is gated on the method choice. Method names `ll_rho{R}_n{N}_vp` |
| SF reference | Same combos on SF: R35 arms (N=15) / R38 arms (N=5), R36 curve, R39 curve at the matched N. Read from their CSVs, nothing rerun |
| Fairness controls | Held identical across backbones: the 419-pair manifest, R7 anchors, seed 0, `flow_shift 1.0`, `--step 15` render, the ρ grid, `tau_min/tau_max`, `div_max`, the eval harness (stride 8, same FiVE-Acc VLM and CLIP-D code), plain per-clip mean, same selected combos and draft N (one `r45_method.json` drives both sides). Native per backbone (paper settings): `fg_boost` 4/2, VP single/tripled frame, checkpoint + LoRA. Native settings shift absolute scores, so **only within-backbone gains are compared across backbones, never absolute scores**. Checked by gates: per-clip frame counts equal SF's (G6 + `wait-renders`); LL NFE accounting for draft + render vs single pass at 15+N (G5), the counterpart of R39's SF check |
| Trigger matching | LL's `_find_phrase_token_indices_single` made **case-insensitive**, identical to SF's (user-requested 2026-10-06). Stock LL found no trigger tokens on 269/838 FiVE (prompt, trigger) pairs (32%), which leaves grounding and boosting off for that branch. This differs from the stock LL behaviour behind the paper's Ours(LL) numbers |
| Text encoder | LL's `WanTextEncoder` uses SF's `umt5_xxl_encoder_from_checkpoint` (fp16 on GPU, mmap; ~11 GB instead of ~23 GB fp32, which saturated memory), user-requested 2026-10-06. Both drivers then cast to bf16. The fp16 round trip flushes 4.7e-5 of the weights, so the text embeddings are **identical to SF's** and slightly different from stock LL's |
| Readout | Two-panel figures (SF \| LL), x = CLIP-target / `clip_d_prompt` / FiVE-Acc, y = `lpips_unedit_part`. **Routing gain** = x_arm − x_curve(y_arm), linear interpolation in y on the same-backbone curve. Reported against N=15 and against matched-N (**primary**: compute-fair). If y_arm is outside the curve's y-range, the gain is NaN with an `outside_range` flag, and the figure shows it. Plain per-clip mean over 419, never `_avg.csv` |
| Scope | All **419** pairs. Stages 1, 3 and the sweeps pass no `--cases_json`. Stage 2 uses `evaluation/r7_anchor_manifest.json` |
| Storage | `~/Data/dataggen/outputs/five_bench/r45_*`. ~15 GB per 419-clip render, ~17 GB per draft run. Typical (one N, 4 combos): sweeps 240 + drafts 34 + arms 60 ≈ **335 GB**. All 8 combos: ≈ 430 GB. R35 hit `Errno 122` once |
| Eval layout + queue cap | Selection = `ll_dino_n5_tgt09` → the eval table has **17 rows** (8 sweep N=15 + 8 sweep N=20 + 1 routed arm). `eval` = array 0-16 (17 entries), `clipd` = 1, **`fiveacc` = 1 sequential job** (user: fiveacc must be sequential) → **19 queue entries**, under the 30-job cap. An arrayed fiveacc would have been 17 + 17 + 1 = 35. All three can be submitted together |
| Output location | The R45 renders (and the R36/R39 reference frames) were moved to `/store/PERSO/skhairi/Outputs/five_bench/`, with symlinks under `~/Data/dataggen/outputs/five_bench/`. `/store` is mounted on only some nodes (gpu-gw, nodecpu12, node39; not nodemm06, node50), so a job on another node stops at its render-dir preflight. **2026-10-22: user restores `r45_sweep` and `r45_arms` to `/home`** (rsync to `$d.tmp`, file-count check, swap the symlink; the `/store` copy is kept). `r45_select.py framecheck` also reads `r36_rho_sweep/r36_rho2_vp` and `wait-renders` should be run from gpu-gw, or with that folder restored too |
| Compute | From R39 (~0.7 L40S GPU-h per step per 419 clips on SF): sweep15 ≈ 84 h, sweepM ≈ 112 h (N=20) / 168 h (N=30), drafts N × 0.7 h per run, stage 3 ≈ 10.5 h per combo. LL speed is measured in smoke (G4) and these numbers updated |
| Node / partition | `--partition=L40S`, `--exclude=node01,node51,node52,node57`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Out of scope | Text-only (no-VP) LL arms (that is R44's axis); recalibrated `div_max`; tau grids; binary mapping (R41); FiVE-Bench-Correct anchors (R43) |

## Step commands

### smoke

```bash
sbatch slurm_scripts/five_bench/r45_smoke.sh          # 1 GPU, ~2 h
```

### wait-smoke

```bash
grep -h "GATE\|blender_rate per step\|peak_vram\|s/frame" logs/r45_smoke_*.out
```

### sweep15

```bash
sbatch --array=0-7 slurm_scripts/five_bench/r45_sweep.sh 15
```

### method-gate

```bash
# after R38 is analyzed; example: draft N=5, all four (metric, regime) combos
python evaluation/r45_select.py --set --draft_n 5 \
  --combos ll_dino_n5_unblended,ll_dino_n5_tgt09,ll_lpips_n5_unblended,ll_lpips_n5_tgt09
cat evaluation/r45_method.json
for s in stage1 stage2 stage3 sweepM eval; do echo "$s: $(python evaluation/r45_select.py $s)"; done
```

### sweepM

```bash
sbatch --array=0-7 slurm_scripts/five_bench/r45_sweep.sh $(python evaluation/r45_select.py sweepM)   # 20 or 30
```

### stage1

```bash
sbatch --array=$(python evaluation/r45_select.py stage1) slurm_scripts/five_bench/r45_stage1.sh
```

### wait-stage1

```bash
D=~/Data/dataggen/outputs/five_bench
for r in $(ls $D/r45_latents); do
  echo -n "$r: $(ls $D/r45_latents/$r/edit*/*.npz 2>/dev/null | wc -l) (expect 419) "
  for T in 1 2 3 4 5 6; do echo -n " e$T=$(ls $D/r45_latents/$r/edit$T/*.npz 2>/dev/null | wc -l)"; done; echo
done
grep -h "blender_rate per step" logs/r45_stage1_*.out | sort | uniq -c
```

### stage2

```bash
sbatch --array=$(python evaluation/r45_select.py stage2) slurm_scripts/five_bench/r45_stage2.sh
```

### wait-stage2

```bash
D=~/Data/dataggen/outputs/five_bench
for c in $(ls $D/r45_div); do echo "$c: $(ls $D/r45_div/$c/edit*/*.npz | wc -l) (expect 419)"; done
grep -c "\[ERROR\]" logs/r45_stage2_*.out
```

### divstats

```bash
python evaluation/r45_div_stats.py       # -> evaluation/csv/r45_div_stats.csv
```

### absnorm

```bash
sbatch slurm_scripts/five_bench/r45_absnorm.sh        # selected combos from r45_method.json, CPU node
```

### stage3

```bash
sbatch --array=$(python evaluation/r45_select.py stage3) slurm_scripts/five_bench/r45_stage3.sh
```

### wait-renders

```bash
D=~/Data/dataggen/outputs/five_bench
for d in $D/r45_sweep/n*/ll_* $D/r45_arms/r45_*; do
  n=$(ls -d $d/edit*/*/ $d/step14/edit*/*/ 2>/dev/null | grep -vc '_resize/$')
  bad=$(for f in $d/_manifest_edit*.csv $d/edit*/_manifest.csv; do [ -f "$f" ] && awk -F, 'NR>1 && $3!="ok"' "$f"; done | wc -l)
  echo "$(basename $d): $n dirs (expect 419), non-ok rows: $bad"
done
```

### eval

```bash
python evaluation/r45_select.py eval --table          # method table the array indexes
sbatch --array=$(python evaluation/r45_select.py eval) slurm_scripts/five_bench/r45_eval.sh
```

### fiveacc

```bash
sbatch slurm_scripts/five_bench/r45_fiveacc.sh          # ONE sequential job over every table row (resumable)
```

### clipd

```bash
sbatch slurm_scripts/five_bench/r45_clipd.sh          # single job, all methods in the table
```

### wait-eval

```bash
ls evaluation/csv/edit?_FiVE_r45_*_frame_stride8.csv | wc -l   # expect 2 x 6 x n_methods
grep -c "Error:\|out of memory" logs/r45_eval_*.metrics.log
wc -l evaluation/csv/r45_clip_directional.csv                  # expect n_methods*419 + 1
```

### figures

```bash
python evaluation/r45_score.py --all     # 3 PDFs + evaluation/csv/r45_arms.csv
```

## Pipeline

```mermaid
flowchart TD
  W["LongLive weights<br/>base + LoRA"] --> PORT["LL port<br/>blend_sched · rho_frames · step_dump"]
  PORT --> SM["r45_smoke.sh<br/>G0-G6"]
  SM --> SW15["r45_sweep.sh 15<br/>8 rho, N=15"]
  SM --> GATE{"method-gate<br/>user: draft N + combos<br/>(after R38)"}
  R38["R38 results"] --> GATE
  GATE --> JSON["r45_method.json"]
  JSON --> SWM["r45_sweep.sh 15+N<br/>8 rho, N=20|30"]
  JSON --> S1["r45_stage1.sh<br/>LL drafts"]
  S1 --> S2["r45_stage2.sh<br/>DINO / LPIPS divergence"]
  S2 --> DS["r45_div_stats.py"]
  SFDIV["R35/R38 div (SF)"] --> DS
  S2 --> ABS["r45_absnorm.sh<br/>div_max 0.65/1.0, budget_linear"]
  ABS --> S3["r45_stage3.sh<br/>15-step gated LL render"]
  SW15 --> EV["r45_eval · r45_fiveacc · r45_clipd"]
  SWM --> EV
  S3 --> EV
  EV --> SC["r45_score.py"]
  SFREF["SF: R35/R38 arms, R36, R39 CSVs"] --> SC
  SC --> FIG["r45_{clip,clipd,fiveacc}_vs_lpips.pdf<br/>r45_arms.csv"]
```

## Code to touch

**Weights** — downloaded by the user, no code. `/build-step` checks that `LongLive_StreamEdit/longlive_models/models/{longlive_base.pt,lora.pt}` and `LongLive_StreamEdit/wan_models/Wan2.1-T2V-1.3B` exist before anything that loads the model.

**`LongLive_StreamEdit/pipeline/utils.py`** — copy `_schedule_blend_rate` (SF `utils.py:24-78`), `build_rho_field_from_mask`, `blender_rate_from_rho` verbatim. `blender_rate_from_rho(rho, t)` is called with `t_next`, as in SF.

**`LongLive_StreamEdit/wan/modules/causal_model.py:360`** — replace the single line with SF's precedence (SF `causal_model.py:352-400`):
```python
if shared_dict.get('blend_off'):            blender_rate = 1.0
elif shared_dict.get('blender_rate') is not None:  blender_rate = shared_dict['blender_rate']   # blend_sched
elif <per-token rate installed in kv_cache>:      use it (rho field; raise if its complement is missing)
else:  blender_rate = 1 - shared_dict['current_timestep'] ** shared_dict['blend_power']          # stock LL
```
Keep the stock line as is: on LL the key `current_timestep` already holds t_next. `tgt<tau>` must take its `t_cur` from the loop variable (`current_timestep / 1000`), NOT from `shared_dict['current_timestep']`.

**`LongLive_StreamEdit/pipeline/edit_causal_inference.py`** — `inference(..., blend_sched=None, rho_frames=None, step_dump=None)` (named as SF's `rollout_inference` kwarg):
- `blend_sched`: per denoising index, `shared_dict['blender_rate'] = 1 - _schedule_blend_rate(blend_sched, i, n, t_cur=float(t)/1000)` (SF `:900-:910`), where `t` is the warped input timestep.
- `rho_frames` `[n_latent, 1560]`: for the block covering latent frames `[f0, f0+3)`, take `rho_frames[f0:f0+3]` (absolute index, VP frames excluded) and install `rate = 1 − t_next^ρ` and its complement on the current/prev KV entries via a port of `_inject_blend_rates_to_kv_cache` (SF `:1382`). Mutually exclusive with `blend_sched`, as in SF `:155-:175`.
- `step_dump`: append each step's x0 prediction for the block (SF R22 hook, `:292-:373`).

**`evaluation/backbone.py`** *(new, shared)* — `load_backbone(args) -> (pipeline, low_memory, device)`: `sf` = the current `run_fivebench.py` loader moved here unchanged; `ll` = `LongLive_StreamEdit/inference_edit_streamedit.py:40-147` (config, `timestep_shift = flow_shift`, arange grid, generator ckpt, LoRA via `configure_lora_for_model` + `peft.set_peft_model_state_dict`), bf16. `encode_vp(backbone, pipeline, src0, anchor)`: LL repeats each latent ×3 and sets `triple_first_frame=True`. `edit_call(backbone, pipeline, **kw)`: SF → `rollout_inference`, LL → `inference` (whole clip; LL's sink + local window bound the cache), same `(video, latents)` return incl. `step_dump`.

**`evaluation/run_fivebench.py`, `evaluation/r31_stage1.py`, `evaluation/r31_stage3.py`** — `--backbone {sf,ll}` (default `sf`); `--sf_root` / `--config_path` / `--checkpoint_path` defaults resolved from the backbone. All loading/VP/inference calls go through `evaluation/backbone.py`. Write `backbone` into the stage-1 npz and the manifest.

**`evaluation/r45_select.py`** *(new)* — `COMBOS` (8, order as in Decisions), `RUNS` (4). `--set --draft_n N --combos a,b,...` validates and writes `evaluation/r45_method.json` `{draft_n, combos, matched_n: 15+draft_n, set_at}`. Subcommands print: `stage1` → run indices of the selected combos; `stage2|stage3` → combo indices; `sweepM` → `matched_n`; `eval` → `0-{k-1}` over the method table (`--table` prints it: `ll_rho{R}_n15_vp` ×8, `ll_rho{R}_n{M}_vp` ×8, `r45_{combo}` per selection, each with its frame root); `check <id> <stage>` → exit 1 if not selected; `framecheck` → per-clip frame counts of every rendered method vs R36 `r36_rho2_vp`, exit 1 on any mismatch. Missing json ⇒ `SystemExit("run method-gate first")`.

**`slurm_scripts/five_bench/r45_smoke.sh`** *(new)* + **`evaluation/r45_smoke_parity.py`** *(new)* — 1 GPU, L40S, 3 h, clips `0001_bus`/edit1 and `0034_cows`/edit2, out root `~/Data/dataggen/outputs/five_bench/r45_smoke/`. Each gate prints `GATE<k> PASS|FAIL`:
- **G0** SF regression: `run_fivebench.py --backbone sf`, ρ=2, VP, on `0001_bus` == R36 `r36_rho2_vp` frames (sha256). Licenses the loader refactor.
- **G1** LL stock parity: stock LL `pipeline.inference(return_latents=True)` from a pristine `git worktree add $TMPDIR/ll_ref HEAD` vs the ported pipeline with every new kwarg `None`, two subprocesses, same seed/anchor, `torch.equal` on latents. The reference worktree gets the same text-encoder change (copy `wan/modules/t5.py`, `utils/wan_wrapper.py` and `inference_edit_streamedit.py` into it before the run; the checkpoint-load change is numerically a no-op but avoids the memory peak), so G1 isolates the routing port. Runs on **`0021_landing`/edit1 + `0025_baseball`/edit2** (90 frames), where stock and case-insensitive matching return the same non-empty trigger tokens on both prompts. `0001_bus` and `0034_cows` can't be used: stock finds no source-trigger tokens there.
- **G2** rho plumbing: uniform field ρ≡2 == scalar `blend_power 2` on LL (latents equal).
- **G3** `tgt0.9` == `first2` at N=15 on LL. Echoed vectors: N=15 `[0,0,1×13]`, N=5 `[0,1,1,1,1]`.
- **G4** longest FiVE clip at N=30 completes on L40S; log `peak_vram` (GB) and `s/frame`.
- **G5** NFE accounting on LL: count generator forwards (hook on `generator.forward`) for a draft at N=5 and N=15, the N=15 render, and single passes at N=20 and N=30, all on `0001_bus`. Log batch-2 vs batch-1 calls. PASS if draft(N) + render(15) − single(15+N) equals only the draft's clean-context forwards, as R39 found for SF.
- **G6** frame parity: per clip, the number of LL output frames == R36 `r36_rho2_vp` (SF) frames, and both equal `find_closest_num_frame(len(src))`.

**`slurm_scripts/five_bench/r45_sweep.sh`** *(new)* — copy of `r36_infer.sh`. `N=$1` (required, ∈ {15,20,30}), `METHOD=ll_rho${R}_n${N}_vp`, `OUT_ROOT=$FIVE_ROOT/r45_sweep/n$N`, `run_fivebench.py --backbone ll --step $N --fg_boost_factor 2`. Same skip/guard logic. `--time` scaled by N (20 h at 15).

**`slurm_scripts/five_bench/r45_stage1.sh`, `r45_stage2.sh`, `r45_absnorm.sh`, `r45_stage3.sh`** *(new)* — copies of the `r38_*` scripts. First line after env setup: `python evaluation/r45_select.py check $SLURM_ARRAY_TASK_ID <stage> || exit 1`. Stage 1: `RUNS`/`SCHEDS`/`NSTEPS` tables over the 4 runs, `--backbone ll --fg_boost_factor 2 --step $N --dump_steps $((N-1)) --measure_index $((N-1))`. Stage 2: combo → (arm, run, N) case table, `TARGET_ROOT=r45_stage1/$RUN/step$(printf %02d $((N-1)))`. Absnorm: reads the selected combos from the json, `div_max` by prefix, `r31_rho_map.py --mapping budget_linear --tau_min 2 --tau_max 50 --step 15` (as R38). Stage 3: `r31_stage3.py --backbone ll --fg_boost_factor 2 --step 15 --dump_steps 14`, per-file manifest guard.

**`slurm_scripts/five_bench/r45_eval.sh`, `r45_fiveacc.sh`, `r45_clipd.sh`** *(new)* — R38's three. `r45_eval.sh`: array id indexes `r45_select.py eval --table`. **`r45_fiveacc.sh`: ONE sequential job looping over every table row** (Qwen2.5-VL-7B, ~30 min per method; skips a method whose five_acc CSVs are already complete and valid, so a timed-out job is resubmitted as-is; continues past a failed row and exits non-zero at the end; `--array=i` / positional row ids rerun single rows). Stems `r45_{method}` / `r45_fiveacc_{method}`. `r45_clipd.sh` passes one `--methods` entry per table row → `evaluation/csv/r45_clip_directional.csv`.

**`evaluation/r45_div_stats.py`** *(new)* — import the stats helpers from `r38_div_stats.py`. Rows for the selected `r45_div/{combo}` plus the matching SF combos (`r35_div` N=15, `r38_div` N=5), with `backbone` and `N` columns.

**`evaluation/r45_score.py`** *(new)* — import `clip_index`, `load_harness`, `load_clipd`, `_check_coverage` from `r35_score.py`. Per x-metric, one figure with two panels sharing axes. **SF**: R36 curve (N=15), R39 curve at `matched_n`, SF arms for the selected (metric, regime) at `draft_n` (R35 if 15, R38 if 5). **LL**: `sweep15` curve, `sweepM` curve, R45 arms. Hue = metric, marker = regime, curves gray (solid N=15, dashed matched). `evaluation/csv/r45_arms.csv`: backbone, combo, N, n_clips, three x metrics, `lpips_unedit_part`, `gain_vs_n15`, `gain_vs_matched` (interpolated per x-metric). Hard-fail on missing CSVs for any selected combo. Adds an `outside_range` column.
