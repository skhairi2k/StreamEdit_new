---
name: R66 — ContextFlow Baseline on FiVE-Correct
overview: "ContextFlow (AAAI 2026, training-free, Wan2.1-I2V-14B-480P, released code + a frame-count generalisation patch) as the second external I2V baseline on the 415 FiVE-Bench-Correct pairs, replacing the dropped VACE task. RENDERED ON JEAN ZAY with the authors' own setup (2 H100 per job: torchrun --nproc_per_node=2, --dit_fsdp --ulysses_size 2 --t5_cpu --offload_model True; env contextflow built from the official instructions only), SCORED ON THE LAB L40S with the R57 suite like every main-table row; renders brought back once to /work for scoring, then moved to /store. First frame = the Correct Qwen anchor, editing prompt = Correct target_prompt, FiVE type -> ContextFlow mode (1/2/3/4 swap, 5 insert, 6 delete); every pair rendered on the first min(41, n) source frames truncated to 4k+1 (the protocol of Wan-Edit, UniEdit-Flow, FlowDirector); RF-Solver inversion once per video; a 4-clip smoke test at 41 frames (plus 81 frames as a diagnostic of the patch) sets the shard layout. The lab L40S phase (2026-10-27 .. 11-02) is kept below as history: 46 GB is too small for the authors' 2-GPU setup (smoke job 1080831)."
task_id: R66
todos:
  # ---- lab L40S phase (history; completed or superseded) -------------------------------------------------------------
  - id: cf-env-script
    content: "SUPERSEDED (single-GPU env, removed 2026-11-01). evaluation/r66_make_contextflow_env.sh -- clone yyChen233/ContextFlow to ~/Code/ContextFlow at commit a9b903edf4; conda env contextflow: python 3.10, torch 2.5.1 + torchvision 0.20.1 (cu121), flash-attn 2.7.4.post1 wheel, the repo's requirements.txt (numpy<2) without gradio and dashscope; Wan2.1-I2V-14B-480P to $CF_WEIGHTS (/work/PERSO/skhairi/weights/Wan2.1-I2V-14B-480P); applies evaluation/r66_contextflow_frames.patch; freeze evaluation/r66_contextflow_env_freeze.txt"
    needed_by: [setup-env]
    status: completed
  - id: cf-official-env-script
    content: "evaluation/r66_make_contextflow_official_env.sh -- conda env contextflow built only from the official instructions: ContextFlow README (PyTorch >= 2.4.0, pip install -r requirements.txt, 'follow the official Wan2.1 setup') + Wan2.1 README for multi-GPU (pip install \"xfuser>=0.4.1\"); every open-ended requirement pinned at its stated minimum via a constraints file: torch 2.4.0 / torchvision 0.19.0 (cu121), diffusers 0.31.0, transformers 4.49.0, accelerate 1.1.1, numpy 1.23.5, opencv-python 4.9.0.80, gradio 5.0.0, xfuser 0.4.1; tokenizers 0.21.0 (stated minimum 0.20.3 conflicts with transformers 4.49); flash_attn 2.7.4.post1 prebuilt wheel for torch 2.4 first, then requirements.txt verbatim (gradio and dashscope included); python 3.10; no torchaudio; verifies every pin, import check incl. xfuser; freeze -> evaluation/r66_contextflow_official_env_freeze.txt (lab build 2026-11-01: 131 packages)"
    needed_by: [setup-env]
    status: completed
  - id: cf-frames-patch
    content: "evaluation/r66_contextflow_frames.patch -- unified diff against wan/image2video.py at a9b903edf4: in generate_with_pnp and generate_reconstruction the hard-coded clip length (latent T 21 / 13 in the shape asserts, mask 81 / 49 frames, first-frame zero padding 80 / 48 frames) becomes F = frame_num for every mode, deletion included (latent T = (F-1)//4 + 1, mask F, padding F-1); nothing else changes (RF-Solver, PnP layers, injection step, CFG, delete branch logic)"
    needed_by: [setup-env]
    status: completed
  - id: cases-script
    content: "evaluation/r66_build_cases.py -- writes evaluation/r66_cases_correct.json: the 415 pairs of r43_anchor_manifest_correct.json with case_id, edit_type, video_name, target_prompt (evaluation/five_bench_correct), anchor_image, mode (types 1-4 swap, 5 insert, 6 delete), pnp_layers (swap 27 29 31 32, insert 6 7 8 9, delete 21 26 27 29), is_delete, n81 = 4*((min(81, n_src)-1)//4)+1 and n41 = 4*((min(41, n_src)-1)//4)+1; and evaluation/r66_cases_smoke.json: the first Correct pair of types 2, 3, 5 and 6 whose source has >= 81 frames; exit 1 if the count != 415 or a source/anchor is missing"
    needed_by: [build-cases]
    status: completed
  - id: driver-script
    content: "evaluation/r66_contextflow_run.py -- builds wan.WanI2V ONCE; --cases --frames 41 (render) | 81 (smoke diagnostic only) --out_root --shard i --n_shards K; video j -> shard j mod K; inputs at 832x480; per video generate_reconstruction (fm_new, 51 steps, empty prompt, source first frame, first n{F} frames) -> {out_root}/latents/n{F}/{video}.pt, reused on resume; per pair generate_with_pnp (case's pnp_layers, injection_step 0.5, guide 3.0, prompt = target_prompt, prompt_origin '', image = anchor, image_origin = source first frame, z_T at the repo's load t, seed 0, is_delete per case) -> {out_root}/n{F}/edit{T}/{video}/%05d.png; resume; manifest_n{F}_shard{i}.csv"
    needed_by: [jz-smoke, jz-render]
    status: completed
  - id: cf-bf16-linears
    content: "SUPERSEDED 2026-11-01 by driver-usp (the cast is removed). Single-GPU L40S path: cast every nn.Linear in model.blocks to bf16 behind a block-0 bit-identity check (smoke job 1074397 had OOMed with the fp32 DiT)"
    needed_by: [jz-smoke]
    status: completed
  - id: driver-usp
    content: "evaluation/r66_contextflow_run.py -- the authors' setup: torchrun --nproc_per_node=2; dist init (nccl) + xfuser init_distributed_environment / initialize_model_parallel(sequence_parallel_degree=2, ring_degree=1, ulysses_degree=2) as in generate.py; WanI2V(dit_fsdp=True, use_usp=True, t5_cpu=True); rank 0 decides skips / inversions (broadcast) and alone writes latents, frames, manifest; barrier after the latent save; per-case status and peak memory all-reduced (MAX); exit code also written to exitcode_n{F}_shard{i}.txt (torchrun does not pass it through)"
    needed_by: [jz-smoke, jz-render]
    status: completed
  - id: smoke-sbatch
    content: "SUPERSEDED (lab L40S). slurm_scripts/five_bench/r66_smoke.sh -- 1 GPU, then 2 L40S (--exclude=node52,node51, 16 CPUs, 192G, NCCL_P2P_DISABLE=1, OMP_NUM_THREADS=8), torchrun, 41 then 81 frames; jobs 1074397, 1076785, 1080609, 1080618, 1080831"
    needed_by: [smoke]
    status: completed
  - id: smoke-sbatch-2gpu
    content: "SUPERSEDED (lab L40S). 2-GPU version of slurm_scripts/five_bench/r66_smoke.sh (see smoke-sbatch)"
    needed_by: [smoke]
    status: completed
  - id: smoke-check-script
    content: "evaluation/r66_smoke_check.py -- contact sheet evaluation/figures/r66_smoke_sheet.jpg (per smoke pair: source | anchor | ContextFlow n81 | ContextFlow n41, frames 0 / middle / last) and evaluation/csv/r66_smoke_timing.csv (pair, mode, frames, sec_inversion, sec_edit, peak_mem_gb) plus a projection for the 415 pairs at 41 frames (100 inversions + 415 edits, measured means, x1.15 overhead); 81-frame timings for reference only"
    needed_by: [jz-smoke-check]
    status: completed
  # ---- Jean Zay phase ------------------------------------------------------------------------------------------------
  - id: cases-data-relative
    content: "evaluation/r66_build_cases.py -- write anchor_image and src_first_frame relative to the data root ~/Data (e.g. 'dataggen/outputs/five_bench/anchors_r43_correct/edit1/0001_bus.png', 'FiVE-Fine-Grained-Video-Editing-Benchmark/images/0001_bus/00001.jpg') instead of absolute lab paths, so the same case files work on the lab and on Jean Zay (~/Data -> $WORK/Data there); everything else unchanged"
    needed_by: [build-cases]
    status: completed
  - id: driver-jz
    content: "evaluation/r66_contextflow_run.py -- (1) --data_dir (default ~/Data) joined with the case's Data-relative anchor_image / src_first_frame; (2) an OOM prints the exception message (what was tried, total / free / reserved) on each rank; (3) after an OOM or any other error the run STOPS (manifest row + exit-code file written, remaining pairs left for a resume), because FSDP cannot recover from an exception inside its forward (smoke job 1080831: 'expected to be in states [IDLE] but current state is FORWARD_BACKWARD' on every later pair); --cf_dir / --ckpt_dir defaults also readable from CF_DIR / CF_WEIGHTS as now"
    needed_by: [jz-smoke, jz-render]
    status: completed
  - id: jz-env-script
    content: "evaluation/r66_make_contextflow_official_env.sh -- make it run unchanged on the Jean Zay frontal: ENV_PREFIX overridable (Jean Zay: $WORK/envs/contextflow), CONDA_BIN overridable (Jean Zay: the conda of `module load anaconda-py3`), and when CF_DIR has no clone yet: git clone + checkout a9b903edf4 + git apply evaluation/r66_contextflow_frames.patch (as the old script did); FREEZE_OUT overridable (Jean Zay: evaluation/r66_contextflow_official_env_freeze_jz.txt); same pins, same verification; on Jean Zay first prints `idr_quota_user` (read by the user, as in R50: its output is not machine-stable, so no automatic refusal; the env is ~51k files of the 500k $WORK quota), sends the conda package cache to $WORK/.conda/pkgs and turns pip's cache off (nothing in the small $HOME), and prints the env's file count at the end; lab behaviour unchanged"
    needed_by: [jz-env]
    status: completed
  - id: jz-push-script
    content: "jz/r66_push.sh -- ON THE LAB FRONTAL (gpu-gw), in tmux: sources jz/r50_jz.env; rsync -a --partial (no --delete, no staging copy) of (a) /work/PERSO/skhairi/weights/Wan2.1-I2V-14B-480P -> $JZ_WORK/weights/Wan2.1-I2V-14B-480P (77 GB, 33 files) and (b) ~/Data/dataggen/outputs/five_bench/anchors_r43_correct -> $JZ_WORK/Data/dataggen/outputs/five_bench/anchors_r43_correct (222 MB, 416 PNG), skipping a part that is already identical on Jean Zay; verify per part: same file count and bytes (remote find/du vs local), sha256 of the weights' index json and config.json; --check sends nothing; ends with '[r66_push] failures: N'. Never deletes anything on either side"
    needed_by: [jz-push]
    status: completed
  - id: smoke-jz-sbatch
    content: "slurm_scripts/five_bench/r66_smoke_jz.sh -- Jean Zay header written by hand (not via r50_port.sh, which targets the five-bench wrapper): -A omt@h100, -C h100, --qos=qos_gpu_h100-t3, --nodes=1, --gres=gpu:2, --cpus-per-task=48 (24 per GPU; memory follows the CPU count), --hint=nomultithread, 8 h; module purge, python = $WORK/envs/contextflow/bin/python; HF_HUB_OFFLINE=1, OMP_NUM_THREADS=24, TORCH_NCCL_ASYNC_ERROR_HANDLING=1, NO NCCL_P2P_DISABLE (NVLink); CF_DIR=$WORK/code/ContextFlow, CF_WEIGHTS=$WORK/weights/Wan2.1-I2V-14B-480P, --data_dir $WORK/Data; out = ~/Data/dataggen/outputs/five_bench/r66_contextflow/smoke (a symlink to $SCRATCH/five_bench_out/r66_contextflow, created by the script if missing, like r50_scratch_links.sh); torchrun --standalone --nproc_per_node=2 at --frames 41 then 81 (diagnostic); nvidia-smi per-GPU peak; driver code read from the exit-code file; 'failures: N'"
    needed_by: [jz-smoke]
    status: completed
  - id: smoke-check-jz
    content: "evaluation/r66_smoke_check.py -- runs on the Jean Zay frontal with $WORK/envs/contextflow/bin/python (cv2, numpy, PIL are in the env): --smoke_root (default ~/Data/dataggen/outputs/five_bench/r66_contextflow/smoke) and --data_dir (default ~/Data) for the Data-relative case paths; projection in H100-hours for 2 GPUs per job (GPU-h = job wall-clock x 2) and the wall-clock with K concurrent jobs; per-job time must stay < 18 h of the 20 h qos limit; peak flagged against 76 GB (H100 80 GB)"
    needed_by: [jz-smoke-check]
    status: completed
  - id: render-jz-sbatch
    content: "slurm_scripts/five_bench/r66_render_jz.sh -- same Jean Zay header and environment as r66_smoke_jz.sh, 20 h, --array=0-(K-1) with K set at the gate; task i runs torchrun --standalone --nproc_per_node=2 r66_contextflow_run.py --cases evaluation/r66_cases_correct.json --frames 41 --out_root ~/Data/dataggen/outputs/five_bench/r66_contextflow/correct --shard i --n_shards K --data_dir $WORK/Data; post-run guard: every pair of shard i has its n41 frames; 'failures: N'"
    needed_by: [jz-render]
    status: pending
  - id: fetch-script
    content: "jz/r66_fetch_renders.sh -- ON THE LAB FRONTAL, in tmux, modelled on jz/r62_fetch_renders.sh: rsync -a --partial (no --delete) of $SCRATCH/five_bench_out/r66_contextflow/correct/n41/ (frames only; latents and manifests stay on Jean Zay, the manifests' CSVs are fetched to evaluation/csv/r66_render_manifest_shard{i}.csv) -> /work/PERSO/skhairi/five_bench/r66_contextflow/correct/n41/ (node-visible, not /store); pre-check: 415 clip dirs on Jean Zay; post-check: 415 clip dirs locally and each holds exactly its n41 PNGs (evaluation/r66_cases_correct.json); >= 15 GB free; --check sends nothing; '[r66_fetch] failures: N'. Never deletes"
    needed_by: [fetch-renders]
    status: pending
  - id: window-links-script
    content: "evaluation/r66_window_links.py -- for r59_n15_rho2_vp ({D}/r59_rho_sweep/r59_n15_rho2_vp) and r59_max_full ({D}/r59_se_arms/r59_max_full/step14), builds {D}/r66_window/{method}/edit{T}/{video}/ with symlinks to the first n frames, n = frame count of the fetched ContextFlow render of that pair (/work/PERSO/skhairi/five_bench/r66_contextflow/correct/n41); refuses if a render has < n frames or a target is not a symlink into the method root; idempotent; exit 1 unless 415 dirs per method"
    needed_by: [window-links]
    status: pending
  - id: eval-driver
    content: "slurm_scripts/five_bench/r66_eval.sh -- lab launcher (bash): eval_suite.sh --method r66_contextflow --root /work/PERSO/skhairi/five_bench/r66_contextflow/correct/n41 --layout edit_video --tag r66 --src_window render --allow_nan motion_fidelity_score_edit_part; plus (always) r59_n15_rho2_vp and r59_max_full on {D}/r66_window/{method}, tag r66w; preflight: 415 dirs per root, eval_suite.sh accepts --src_window (R65 suite edits built), queue + 3 x n_methods <= 30; appends job ids to logs/r66_eval_jobs.txt, refuses if a method is already logged"
    needed_by: [eval]
    status: pending
  - id: score-script
    content: "evaluation/r66_score.py -- reads evaluation/csv/suite_r66_r66_contextflow.csv and suite_r66w_r59_n15_rho2_vp.csv / suite_r66w_r59_max_full.csv; plain per-clip means overall and per edit type, types 3/4 flagged 'swap mode' -> evaluation/csv/r66_table.csv; paired per-clip deltas vs StreamEdit native and se_max on the r66w window rows (the same first n frames per pair) with 95% bootstrap CI (2000 resamples, seed 0) -> evaluation/csv/r66_deltas.csv"
    needed_by: [score]
    status: pending
  - id: archive-script
    content: "jz/r66_archive.sh -- ON THE LAB FRONTAL, after score: (1) move (mv, not copy) /work/PERSO/skhairi/five_bench/r66_contextflow/correct -> /store/PERSO/skhairi/Outputs/five_bench/r66_contextflow/correct (refuse unless [ -d /store/PERSO/skhairi/Outputs/five_bench ]; never mkdir -p the store root; verify 415 clip dirs and the PNG count after the move) and link ~/Data/dataggen/outputs/five_bench/r66_contextflow -> /store/PERSO/skhairi/Outputs/five_bench/r66_contextflow (the R35-R45 convention); (2) prints, and with --cleanup-jz runs over ssh, the removal of the Jean Zay copies: $SCRATCH/five_bench_out/r66_contextflow (renders + latents + smoke) and the link in $WORK/Data; weights / env on Jean Zay only with --cleanup-jz-model; every deletion only after the post-move check passed and only with its explicit flag; '[r66_archive] failures: N'"
    needed_by: [archive]
    status: pending
steps:
  # ---- lab L40S phase (history) --------------------------------------------------------------------------------------
  - id: probe-paths
    type: manual
    check_hint: "User: R66_ROOT (/work/PERSO/skhairi/five_bench/r66_contextflow) and CF_WEIGHTS (/work/PERSO/skhairi/weights) writable from the frontal and readable from the compute nodes: `mkdir -p /work/PERSO/skhairi/five_bench/r66_contextflow /work/PERSO/skhairi/weights && df -h /work/PERSO/skhairi`; `srun -p L40S --exclude=node52 --gres=gpu:1 -t 00:05:00 bash -c 'hostname; ls -d /work/PERSO/skhairi/weights /work/PERSO/skhairi/five_bench/r66_contextflow'`"
    status: completed
    completed_at: 2026-10-27
  - id: setup-env
    type: local
    wait_for: probe-paths
    command: bash evaluation/r66_make_contextflow_official_env.sh
    status: completed
    completed_at: 2026-11-01
  - id: smoke
    type: sbatch
    wait_for: setup-env
    command: sbatch slurm_scripts/five_bench/r66_smoke.sh
    sets_status: running
    status: completed
    completed_at: 2026-11-02
    job_id: "1080831"
  # ---- Jean Zay phase ------------------------------------------------------------------------------------------------
  - id: build-cases
    type: local
    command: ~/anaconda3/envs/five-bench/bin/python -I evaluation/r66_build_cases.py
    output_paths: [evaluation/r66_cases_correct.json, evaluation/r66_cases_smoke.json]
    status: completed
    completed_at: 2026-11-02
  - id: jz-sync
    type: manual
    wait_for: build-cases
    check_hint: "User: commit + push the R66 files (evaluation/r66_*, jz/r66_*, slurm_scripts/five_bench/r66_*_jz.sh, the case JSONs). On Jean Zay: `cd $WORK/StreamEdit_new && git pull && ls evaluation/r66_cases_correct.json evaluation/r66_contextflow_frames.patch slurm_scripts/five_bench/r66_smoke_jz.sh`; `ls $WORK/Data/FiVE-Fine-Grained-Video-Editing-Benchmark/images | wc -l` (the FiVE sources staged by R50); paste the output"
    status: pending
  - id: jz-push
    type: manual
    wait_for: jz-sync
    check_hint: "User, on the lab frontal in tmux: `bash jz/r66_push.sh 2>&1 | tee logs/r66_push.log`; paste the tail: per part '<n> files, <bytes>: identical on Jean Zay' and '[r66_push] failures: 0'"
    status: pending
  - id: jz-env
    type: manual
    wait_for: jz-push
    check_hint: "User, on the Jean Zay frontal (PyPI and GitHub reachable there): `cd $WORK/StreamEdit_new && module purge && module load anaconda-py3 && CONDA_BIN=$(which conda) ENV_PREFIX=$WORK/envs/contextflow CF_DIR=$WORK/code/ContextFlow CF_WEIGHTS=$WORK/weights/Wan2.1-I2V-14B-480P FREEZE_OUT=evaluation/r66_contextflow_official_env_freeze_jz.txt bash evaluation/r66_make_contextflow_official_env.sh 2>&1 | tee logs/r66_setup_env_jz.log`; paste the [pins] lines and 'import check OK'; then bring the freeze file to the lab (rsync) and `diff` it with evaluation/r66_contextflow_official_env_freeze.txt (expect identical)"
    status: pending
  - id: jz-smoke
    type: manual
    wait_for: jz-env
    check_hint: "User, on Jean Zay: `cd $WORK/StreamEdit_new && sbatch slurm_scripts/five_bench/r66_smoke_jz.sh`; when it ends paste `grep -E '\\[r66_run\\] (model|edit|done)|\\[r66_smoke\\]' logs/r66_smoke_jz_<jobid>.out`. Record the job id"
    sets_status: running
    status: pending
  - id: jz-smoke-check
    type: manual
    wait_for: jz-smoke
    check_hint: "User, on the Jean Zay frontal: `cd $WORK/StreamEdit_new && $WORK/envs/contextflow/bin/python -I evaluation/r66_smoke_check.py`; then on the lab frontal: `rsync -a zay:<JZ_REPO>/evaluation/figures/r66_smoke_sheet.jpg evaluation/figures/ && rsync -a zay:<JZ_REPO>/evaluation/csv/r66_smoke_timing.csv evaluation/csv/` (two small files; JZ_REPO from jz/r50_jz.env)"
    output_paths: [evaluation/figures/r66_smoke_sheet.jpg, evaluation/csv/r66_smoke_timing.csv]
    status: pending
  - id: gate
    type: manual
    wait_for: jz-smoke-check
    check_hint: "With the user, from r66_smoke_sheet.jpg and r66_smoke_timing.csv: (1) the anchor is kept at frame 0, the edit persists, background and motion follow the source, in all 4 modes incl. the type-3 colour edit in swap mode; (1b) the faint vertical stripes seen at frames 20 / 40 of the one L40S render (job 1080831, edit2/0002_girl-dog) are absent, or present in the authors' own outputs too; (2) peak < 76 GB per H100; (3) 41 vs 81 diagnostic: the 41-frame renders are not visibly worse than the 81-frame ones in any mode, deletion at 41 included (the authors used 49); keep the sheet for the supplementary; (4) K so that ceil(100/K) videos x measured (inversion + 4.15 edits per video) x 1.3 < 18 h; record K in r66_render_jz.sh. Visual failure in a mode -> stop and decide (e.g. n/a for that type)"
    status: pending
  - id: lab-cleanup
    type: manual
    wait_for: gate
    check_hint: "User, on the lab, only after the gate passed on Jean Zay (the lab copies are then never used again; the freeze file and env script keep the env reproducible): `rm -rf /work/PERSO/skhairi/weights/Wan2.1-I2V-14B-480P` (77 GB) and `conda env remove -p ~/anaconda3/envs/contextflow` (4.6 GB); `rm -rf /work/PERSO/skhairi/five_bench/r66_contextflow/smoke` (L40S smoke leftovers and archives); paste `df -h /work/PERSO/skhairi`"
    status: pending
  - id: jz-render
    type: manual
    wait_for: gate
    check_hint: "User, on Jean Zay: `cd $WORK/StreamEdit_new && sbatch slurm_scripts/five_bench/r66_render_jz.sh`; record the array job id"
    sets_status: running
    status: pending
  - id: jz-wait-render
    type: manual
    wait_for: jz-render
    check_hint: "User, on Jean Zay: `sacct -j <jobid> --format=JobID,State,Elapsed`; `grep -h failures logs/r66_render_jz_*.out | sort | uniq -c` -> every line 'failures: 0'; `find -L ~/Data/dataggen/outputs/five_bench/r66_contextflow/correct/n41 -mindepth 2 -maxdepth 2 -type d | wc -l` = 415"
    sets_status: finished
    status: pending
  - id: fetch-renders
    type: manual
    wait_for: jz-wait-render
    check_hint: "User, on the lab frontal in tmux: `bash jz/r66_fetch_renders.sh 2>&1 | tee logs/r66_fetch.log`; paste the tail: '415 clip dirs, every clip has its n41 PNGs' and '[r66_fetch] failures: 0'"
    status: pending
  - id: wait-deps
    type: manual
    check_hint: "R65's suite edits built (eval_suite.sh accepts --src_window: todos suite-*-srcwindow, psc-srcwindow, cf-srcwindow completed) and R57 lab smoke passed; R59 renders complete (415 dirs in r59_n15_rho2_vp and r59_max_full/step14)"
    status: pending
  - id: window-links
    type: local
    wait_for: [fetch-renders, wait-deps]
    command: ~/anaconda3/envs/five-bench/bin/python -I evaluation/r66_window_links.py
    status: pending
  - id: eval
    type: local
    wait_for: [fetch-renders, wait-deps, window-links]
    command: bash slurm_scripts/five_bench/r66_eval.sh
    status: pending
  - id: wait-eval
    type: manual
    wait_for: eval
    check_hint: "Job ids in logs/r66_eval_jobs.txt; evaluation/csv/suite_r66_r66_contextflow.csv with 415 rows, and the two suite_r66w_*.csv with 415 rows each; merge by hand if afterok was blocked by the edit-part NaN line"
    status: pending
  - id: score
    type: local
    wait_for: wait-eval
    command: ~/anaconda3/envs/five-bench/bin/python -I evaluation/r66_score.py
    output_paths: [evaluation/csv/r66_table.csv, evaluation/csv/r66_deltas.csv]
    status: pending
  - id: verdict
    type: manual
    wait_for: score
    check_hint: "Read r66_table.csv / r66_deltas.csv with the user: ContextFlow vs AnyV2V vs StreamEdit native / se_max per edit type, swap-mode types 3/4 separately; record in daily.md Progress / outcomes"
    sets_status: analyzed
    status: pending
  - id: archive
    type: manual
    wait_for: verdict
    check_hint: "User, on the lab frontal: `bash jz/r66_archive.sh` (move to /store + ~/Data link, post-move check), then, once satisfied, `bash jz/r66_archive.sh --cleanup-jz` (Jean Zay renders, latents, smoke); `--cleanup-jz-model` only when no rerun is planned; paste '[r66_archive] failures: 0'"
    status: pending
isProject: true
---

# R66: ContextFlow Baseline on FiVE-Correct

## Context

The main table needs a second, recent external I2V baseline next to AnyV2V (R65). VACE was dropped (no native first-frame propagation mode) and per-video-optimization methods are excluded for cost. ContextFlow (AAAI 2026) is training-free, takes an edited first frame and the source video, and runs on Wan2.1-I2V-14B-480P, so the Correct Qwen anchors plug in directly. Inversion depends only on the source video, so it is done once per video.

**Platform history.** Planned on the lab L40S (46 GB). The single-GPU path does not fit the fp32 DiT and its edit call crashes (the authors only ran 2 GPUs); the authors' 2-GPU setup needed `NCCL_P2P_DISABLE=1` on the L40S and then ran at 45.0–45.4 GB of 46 GB: one pair rendered, the next three OOMed (smoke job 1080831, 2026-11-02). 4 L40S per job are not available to the user, so rendering moves to **Jean Zay H100 (80 GB) with the authors' 2-GPU command unchanged**; scoring stays on the lab L40S like every main-table row.

Done when: the Jean Zay smoke gate is passed, the 415 Correct pairs are rendered on Jean Zay, fetched once to the lab, scored with the R57 suite, the ContextFlow row plus its paired deltas against StreamEdit native and se_max on the same frame windows are written, and the renders are moved to `/store`.

## Execution steps

| Step | Type | Where | Waits for | Sets status |
|---|---|---|---|---|
| `probe-paths`, `setup-env`, `smoke` | — | lab (history, completed) | — | — |
| `build-cases` | local | lab | — | — |
| `jz-sync` | manual | lab → Jean Zay (git) | `build-cases` | — |
| `jz-push` | manual | lab frontal → Jean Zay | `jz-sync` | — |
| `jz-env` | manual | Jean Zay frontal | `jz-push` | — |
| `jz-smoke` | manual (sbatch on JZ, 2 H100) | Jean Zay | `jz-env` | `running` |
| `jz-smoke-check` | manual | Jean Zay frontal → lab (2 small files) | `jz-smoke` | — |
| `gate` | manual | — | `jz-smoke-check` | — |
| `lab-cleanup` | manual | lab | `gate` | — |
| `jz-render` | manual (sbatch array on JZ) | Jean Zay | `gate` | `running` |
| `jz-wait-render` | manual | Jean Zay | `jz-render` | `finished` |
| `fetch-renders` | manual | lab frontal ← Jean Zay | `jz-wait-render` | — |
| `wait-deps` | manual | lab | R65 suite edits, R57 smoke, R59 renders | — |
| `window-links` | local | lab | `fetch-renders`, `wait-deps` | — |
| `eval` | local (submits eval_suite jobs, L40S) | lab | `fetch-renders`, `wait-deps`, `window-links` | — |
| `wait-eval` | manual | lab | `eval` | — |
| `score` | local | lab | `wait-eval` | — |
| `verdict` | manual | — | `score` | `analyzed` |
| `archive` | manual | lab frontal (+ ssh JZ) | `verdict` | — |

*(prep)* `/build-step build-cases R66` · `/build-step jz-push R66` · `/build-step jz-env R66` · `/build-step jz-smoke R66` · `/build-step jz-smoke-check R66` · `/build-step jz-render R66` · `/build-step fetch-renders R66` · `/build-step window-links R66` · `/build-step eval R66` · `/build-step score R66` · `/build-step archive R66`

```
/run-step R66 build-cases
/run-step R66 jz-sync
/run-step R66 jz-push
/run-step R66 jz-env
/run-step R66 jz-smoke
/run-step R66 jz-smoke-check
/run-step R66 gate
/run-step R66 lab-cleanup
/run-step R66 jz-render
/run-step R66 jz-wait-render
/run-step R66 fetch-renders
/run-step R66 wait-deps
/run-step R66 window-links
/run-step R66 eval
/run-step R66 wait-eval
/run-step R66 score
/run-step R66 verdict
/run-step R66 archive
```

Every Jean Zay step is run by the user (only the lab frontal can `ssh zay`; Claude cannot reach Jean Zay) and its output pasted back.

## Decisions

| Decision | Choice |
|---|---|
| Method / backbone | ContextFlow, commit `a9b903edf4` + `r66_contextflow_frames.patch` (frame count only; the only change to the authors' code), run with the authors' own command line (2 GPUs, FSDP + Ulysses), Wan2.1-I2V-14B-480P (no 1.3B I2V exists; a port would change the authors' layer selection) |
| Render platform | **Jean Zay H100** (decided 2026-11-02): `-A omt@h100`, `qos_gpu_h100-t3` (20 h per job), 1 node, `--gres=gpu:2`, 48 CPUs (24 per GPU), **the authors' setup as in `inversion.sh` and every demo script**: `torchrun --nproc_per_node=2`, `--dit_fsdp --ulysses_size 2 --t5_cpu --offload_model True`. 2 GPUs, not 1 (the single-GPU path crashes as released and the fp32 DiT + activations exceed 80 GB) and not 4 (not needed on 80 GB). NVLink: no `NCCL_P2P_DISABLE` (L40S-only workaround) |
| Env | **`contextflow` built from the official instructions only** (decided 2026-11-01): ContextFlow `requirements.txt` verbatim + Wan2.1's `xfuser>=0.4.1`, every open-ended requirement at its stated minimum (torch 2.4.0+cu121, diffusers 0.31.0, transformers 4.49.0, tokenizers 0.21.0, accelerate 1.1.1, numpy 1.23.5, opencv 4.9.0.80, gradio 5.0.0, xfuser 0.4.1), flash-attn 2.7.4.post1 wheel, python 3.10. Same script on both hosts; Jean Zay copy = conda env at `$WORK/envs/contextflow` (~51k files of the 500k `$WORK` quota), its freeze must equal the lab freeze. The lab env is removed at `lab-cleanup`. Never installed into `five-bench` / `streamgve` |
| Eval platform | Lab **L40S**, R57 `eval_suite.sh`, same as every main-table row (renders are fetched; never scored on Jean Zay) |
| Disk / copies | One transfer per item, no staging copies: weights and anchors lab → Jean Zay once (`rsync`, then the lab weights are deleted at `lab-cleanup`); the ContextFlow clone is made on Jean Zay from GitHub + the patch from the repo; code through git; FiVE sources already on Jean Zay (R50). Renders and latents written to Jean Zay `$SCRATCH`; only the 415 render dirs (~10 GB) come back, once, to `/work` (node-visible, needed by the eval jobs); after `score` they are **moved** to `/store` with a `~/Data` link and the Jean Zay copies are deleted (`archive`) |
| DiT weight precision | As released: FSDP mixed precision (bf16 compute, fp32 shards), nothing added. History: single-GPU path OOM with the fp32 DiT (job 1074397) and `latent_output_dir` crash (job 1076785); bf16-Linear cast dropped for the official setup |
| Inputs | First frame = `anchors_r43_correct/edit{T}/{video}.png`; editing prompt = Correct `target_prompt`; reconstruction path = source first frame + empty prompt (repo default); all inputs resized to the anchor size 832×480 so both paths share one latent shape. Case files hold paths relative to `~/Data` (works on both hosts) |
| Mode mapping | Types 1, 2, 3, 4 → swap (layers 27 29 31 32); 5 → insert (6 7 8 9); 6 → delete (21 26 27 29, `--is_delete`); types 3/4 in swap mode are footnoted |
| Sampling | Repo settings: RF-Solver `fm_new`, 51 steps (inversion and edit; 2 model calls per step), shift 3.0 (inert: `fm_new` uses 5.0), CFG 3.0, `injection_step` 0.5, top-4 layers per mode; seed 0 (the repo default −1 is random) |
| Frames | n = 4⌊(min(41, n_src) − 1)/4⌋ + 1 for every pair and every mode (41 for most videos, 25–37 for the 7 short ones), the protocol of Wan-Edit / UniEdit-Flow / FlowDirector; inversion uses the same n. StreamEdit renders up to 81 frames, so every StreamEdit comparison uses the window rows below |
| Frame-count patch | The released code hard-codes 81 frames (49 for deletion: latent T 21 / 13, mask, zero padding) in `generate_with_pnp` and `generate_reconstruction`; 237 / 415 pairs have < 81 source frames. `r66_contextflow_frames.patch` replaces the constants by F = `frame_num`. Paper: "the released code fixes the clip length to 81 frames (49 for deletion); we generalise it to an arbitrary frame count without changing any other setting" |
| Inversion | Once per video at n41, z_T kept on Jean Zay `$SCRATCH` (`r66_contextflow/correct/latents/n41/`, ~22 MB each) and reused on resume; never fetched |
| Smoke | 4 Correct pairs (first pair with ≥ 81 source frames of types 2, 3, 5, 6), each at 41 frames (render setting) then 81 frames (diagnostic of the patch at the authors' length, for the supplementary), on 2 H100. Reference from the L40S (job 1080831): inversion 402 s, edit 1345 s per pair at 41 frames |
| Failure handling | An OOM / error inside FSDP's forward leaves the model unusable, so the driver stops after the first failure (manifest row + exit-code file); a resubmit resumes |
| Sharding | By video, video j → shard j mod K, `--array=0-(K-1)`, < 18 h per task (20 h qos); K from the gate |
| Paths | Jean Zay: repo `$WORK/StreamEdit_new`, env `$WORK/envs/contextflow`, clone `$WORK/code/ContextFlow`, weights `$WORK/weights/Wan2.1-I2V-14B-480P`, data `$WORK/Data` (= `~/Data`), outputs `~/Data/dataggen/outputs/five_bench/r66_contextflow` → `$SCRATCH/five_bench_out/r66_contextflow`. Lab: fetched renders `/work/PERSO/skhairi/five_bench/r66_contextflow/correct/n41` until `archive`, then `/store/PERSO/skhairi/Outputs/five_bench/r66_contextflow` with the `~/Data` link; window views `{D}/r66_window/{method}`, `D = ~/Data/dataggen/outputs/five_bench` |
| Scoring | R57 suite, tag `r66`, `--src_window render` (needs R65's suite edits), `--allow_nan motion_fidelity_score_edit_part`; renders scored at their own 832×480 size like the Wan and StreamEdit rows; plain per-clip means; paired bootstrap 2000 resamples, seed 0 |
| Window control rows | Always: StreamEdit native + se_max on the same first n frames per pair as the ContextFlow render, tag `r66w`; all paired deltas use these rows. Open item beyond R66: the main table mixes 41-frame (Wan baselines), 32-frame (AnyV2V) and 69–81-frame (StreamEdit) rows; a common-window table is not decided yet |
| Main-table presentation | Separate external I2V group with AnyV2V; Backbone and NFE columns (inversion 100 model calls; edit 100 calls of a batch of 3); no wall-clock column |
| Out of scope | VACE, optimization-based methods, other ContextFlow image editors (AnyDoor / InsertAnything / MagicQuill), hyperparameter tuning, official-FiVE runs |

## Step commands

### build-cases
```bash
~/anaconda3/envs/five-bench/bin/python -I evaluation/r66_build_cases.py
```

### jz-sync
```bash
# lab: git add/commit/push the R66 files; then on Jean Zay:
cd $WORK/StreamEdit_new && git pull
ls $WORK/Data/FiVE-Fine-Grained-Video-Editing-Benchmark/images | wc -l
```

### jz-push
```bash
bash jz/r66_push.sh 2>&1 | tee logs/r66_push.log            # lab frontal, tmux
```

### jz-env
```bash
# Jean Zay frontal
cd $WORK/StreamEdit_new && module purge && module load anaconda-py3
CONDA_BIN=$(which conda) ENV_PREFIX=$WORK/envs/contextflow CF_DIR=$WORK/code/ContextFlow \
CF_WEIGHTS=$WORK/weights/Wan2.1-I2V-14B-480P FREEZE_OUT=evaluation/r66_contextflow_official_env_freeze_jz.txt \
  bash evaluation/r66_make_contextflow_official_env.sh 2>&1 | tee logs/r66_setup_env_jz.log
```

### jz-smoke
```bash
cd $WORK/StreamEdit_new && sbatch slurm_scripts/five_bench/r66_smoke_jz.sh        # Jean Zay
```

### jz-smoke-check
```bash
$WORK/envs/contextflow/bin/python -I evaluation/r66_smoke_check.py                   # Jean Zay frontal
# lab frontal: rsync the sheet + timing CSV back (two small files)
```

### gate
```bash
cat evaluation/csv/r66_smoke_timing.csv
# view evaluation/figures/r66_smoke_sheet.jpg (41 vs 81 per mode); set K in slurm_scripts/five_bench/r66_render_jz.sh
```

### lab-cleanup
```bash
rm -rf /work/PERSO/skhairi/weights/Wan2.1-I2V-14B-480P /work/PERSO/skhairi/five_bench/r66_contextflow/smoke
conda env remove -p ~/anaconda3/envs/contextflow
df -h /work/PERSO/skhairi
```

### jz-render
```bash
cd $WORK/StreamEdit_new && sbatch slurm_scripts/five_bench/r66_render_jz.sh       # Jean Zay
```

### jz-wait-render
```bash
sacct -j <jobid> --format=JobID,State,Elapsed
grep -h failures logs/r66_render_jz_*.out | sort | uniq -c
```

### fetch-renders
```bash
bash jz/r66_fetch_renders.sh 2>&1 | tee logs/r66_fetch.log  # lab frontal, tmux
```

### wait-deps
```bash
grep -c src_window slurm_scripts/five_bench/eval_suite_common.sh evaluation/eval_suite/psc_score.py evaluation/eval_suite/cf_score.py
ls evaluation/csv/suite_r57_smoke_lab_*.csv
```

### window-links
```bash
~/anaconda3/envs/five-bench/bin/python -I evaluation/r66_window_links.py
```

### eval
```bash
bash slurm_scripts/five_bench/r66_eval.sh            # ContextFlow + the two r66w window rows
```

### wait-eval
```bash
cat logs/r66_eval_jobs.txt
wc -l evaluation/csv/suite_r66_r66_contextflow.csv
```

### score
```bash
~/anaconda3/envs/five-bench/bin/python -I evaluation/r66_score.py
```

### verdict
```bash
cat evaluation/csv/r66_table.csv evaluation/csv/r66_deltas.csv
```

### archive
```bash
bash jz/r66_archive.sh                 # lab frontal: mv to /store + ~/Data link + check
bash jz/r66_archive.sh --cleanup-jz    # then: Jean Zay renders, latents, smoke
```

## Pipeline

```mermaid
flowchart LR
  A[r43_anchor_manifest_correct.json<br/>five_bench_correct JSONs] --> B[r66_build_cases.py<br/>Data-relative paths]
  B --> G1[git push / pull]
  W[lab: Wan2.1-I2V weights<br/>anchors_r43_correct] --> P[r66_push.sh<br/>rsync once]
  G1 --> E[JZ: r66_make_contextflow_official_env.sh<br/>conda env + clone + patch]
  P --> S[JZ: r66_smoke_jz.sh<br/>2 H100, torchrun]
  E --> S
  S --> C[r66_smoke_check.py<br/>sheet + timing]
  C --> GT{gate: modes, 41 vs 81, K}
  GT --> LC[lab-cleanup<br/>weights + env]
  GT --> R[JZ: r66_render_jz.sh<br/>array K, 2 H100 each]
  R --> F[r66_fetch_renders.sh<br/>JZ SCRATCH -> lab /work]
  F --> WL[r66_window_links.py]
  R59[R59 renders] --> WL
  F --> EV[r66_eval.sh<br/>R57 suite, lab L40S]
  WL --> EV
  EV --> SC[r66_score.py<br/>table + deltas]
  SC --> AR[r66_archive.sh<br/>mv to /store, JZ cleanup]
```

## Code to touch

| File | Change | Todo |
|---|---|---|
| `evaluation/r66_make_contextflow_env.sh` | Superseded single-GPU env script (kept for its record) | `cf-env-script` |
| `evaluation/r66_make_contextflow_official_env.sh` | Official-recipe env (lab, 2026-11-01) | `cf-official-env-script` |
| `evaluation/r66_make_contextflow_official_env.sh` | Runs unchanged on Jean Zay: `ENV_PREFIX`, `CONDA_BIN`, `FREEZE_OUT` overridable; clones + patches ContextFlow when `CF_DIR` is empty; `$WORK` file-quota check | `jz-env-script` |
| `evaluation/r66_contextflow_frames.patch` | Hard-coded 81 / 49 frames → F = `frame_num` | `cf-frames-patch` |
| `evaluation/r66_build_cases.py` | 415 cases + 4 smoke cases | `cases-script` |
| `evaluation/r66_build_cases.py` | Anchor / source paths relative to `~/Data` | `cases-data-relative` |
| `evaluation/r66_contextflow_run.py` | Single-GPU driver → 2-GPU torchrun driver (authors' setup) | `driver-script`, `cf-bf16-linears` (superseded), `driver-usp` |
| `evaluation/r66_contextflow_run.py` | `--data_dir`; OOM message printed; stop after the first OOM / error | `driver-jz` |
| `jz/r66_push.sh` *(new)* | Weights + anchors lab → Jean Zay, rsync once, verify | `jz-push-script` |
| `slurm_scripts/five_bench/r66_smoke.sh` | Lab L40S smoke (history) | `smoke-sbatch`, `smoke-sbatch-2gpu` |
| `slurm_scripts/five_bench/r66_smoke_jz.sh` *(new)* | Jean Zay smoke: 2 H100, authors' command, 41 then 81 frames | `smoke-jz-sbatch` |
| `evaluation/r66_smoke_check.py` | Contact sheet + timing CSV + projection | `smoke-check-script` |
| `evaluation/r66_smoke_check.py` | Runs on Jean Zay; `--smoke_root` / `--data_dir`; H100-hours, 18 h per job, 76 GB flag | `smoke-check-jz` |
| `slurm_scripts/five_bench/r66_render_jz.sh` *(new)* | Jean Zay array over K video shards at 41 frames, per-shard guard | `render-jz-sbatch` |
| `jz/r66_fetch_renders.sh` *(new)* | JZ `$SCRATCH` renders → lab `/work`, once, verified | `fetch-script` |
| `evaluation/r66_window_links.py` *(new)* | First-n-frame symlink views of R59 native + se_max | `window-links-script` |
| `slurm_scripts/five_bench/r66_eval.sh` *(new)* | `eval_suite.sh` for ContextFlow (tag `r66`) and the two window rows (tag `r66w`); guards; job-id log | `eval-driver` |
| `evaluation/r66_score.py` *(new)* | Per-clip means overall / per type (3/4 flagged), paired deltas with bootstrap CI | `score-script` |
| `jz/r66_archive.sh` *(new)* | mv renders to `/store` + `~/Data` link; Jean Zay cleanup behind explicit flags | `archive-script` |
