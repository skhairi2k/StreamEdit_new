---
name: "R50 plan — Jean Zay H100 Setup Kit"
overview: "Move the Wan-Edit half of R49 to Jean Zay (H100) in one verified pass (StreamEdit stays on the lab cluster): code by shallow git clone; Python from the IDRIS H100 module pytorch-gpu/py3/2.5.1 plus a pinned pip overlay of every package that differs from our five-bench env; the exact weights and data from this cluster by rsync with SHA-256 checks; a port script that rewrites the hard-coded paths and Slurm headers from one config file; an offline preflight job on a compute node; and an 18-clip reproduction of Wan-Edit max_s5 and the official baseline native_s0.92 whose paired deltas are checked against our L40S confidence intervals before any full-stage run."
task_id: R50
todos:
  - id: env-template-file
    content: "jz/r50_jz.env.example -- the only file with site-specific values, copied to jz/r50_jz.env and filled in: JZ_HOST (user@host for rsync), JZ_WORK (absolute $WORK on Jean Zay), JZ_REPO (clone path, default $JZ_WORK/StreamEdit_new), JZ_ACCOUNT_GPU (default empty = no -A line; <project>@h100 only for multi-project logins or CPU+GPU projects), JZ_QOS_GPU (default qos_gpu_h100-t3), JZ_CPUS_PER_GPU (default 24), JZ_MODULE (pytorch-gpu/py3/2.5.1, matches our five-bench env), LOCAL_STAGE (symlink stage on this cluster, default ~/Data/r50_stage)"
    needed_by: [fill-config]
    status: completed
  - id: gitignore-edit
    content: ".gitignore -- add jz/r50_jz.env (filled config never committed)"
    needed_by: [fill-config]
    status: completed
  - id: readme-file
    content: "jz/r50_README.md -- the ordered procedure (this cluster: fill config, make bundle, push; Jean Zay: clone + setup + verify, pip overlay, port, preflight, reproduction, compare), the host split (Wan-Edit on Jean Zay, StreamEdit on the lab cluster), what each step checks, the inode budget, and how to recover from a failed step"
    needed_by: [fill-config]
    status: completed
  - id: bundle-script
    content: "jz/r50_make_bundle.sh -- stages in $LOCAL_STAGE as SYMLINKS to our files (no copy, works across filesystems): weights/wan_models/Wan2.1-T2V-1.3B, cache/huggingface/hub/{models--facebook--dinov3-vitb16-pretrain-lvd1689m, models--openai--clip-vit-large-patch14, models--Qwen--Qwen2.5-VL-7B-Instruct}, cache/torch/hub (DINO ViT-B/8 repo + weights, AlexNet/VGG for LPIPS, RAFT, pyiqa NIQE), data/FiVE-Fine-Grained-Video-Editing-Benchmark/{videos,images,bmasks,edit_prompt}, ref/evaluation_csv (a frozen copy of the 23 CSVs r50_compare.py reads: wan_max_s5 and wan_native_s0p92 harness + FiVE-Acc per type, CLIP-D main / base, r49_deltas.csv), code/iqa_cli, freeze/five-bench.{txt,editable.json,versions.txt}; writes MANIFEST.sha256 over every file reachable in the stage and CODE.sha256 over the tracked + new (not ignored) files of slurm_scripts/five_bench, evaluation, Self-Forcing_StreamEdit, wan-edit, jz, warns about uncommitted code; refuses unfilled config values, missing sources and broken links; prints GiB and file count to send"
    needed_by: [make-bundle]
    status: completed
  - id: push-script
    content: "jz/r50_push.sh -- rsync -a --copy-unsafe-links --partial --info=progress2 $LOCAL_STAGE/ $JZ_HOST:$JZ_WORK/r50_stage/ (stage links are sent as real files, the HF cache's internal relative links stay links; resumable) plus jz/r50_jz.env, then prints the remote total size; reads jz/r50_jz.env"
    needed_by: [push]
    status: completed
  - id: setup-script
    content: "jz/r50_setup_dirs.sh -- on Jean Zay: moves stage contents into $JZ_WORK/{weights,cache,Data,code} (stage data/ -> $JZ_WORK/Data) and ref/evaluation_csv -> $JZ_WORK/r50_ref/evaluation_csv made read-only (the compare reference, never written by jobs); symlinks ~/Data -> $JZ_WORK/Data, ~/.cache/huggingface -> $JZ_WORK/cache/huggingface, ~/.cache/torch -> $JZ_WORK/cache/torch, repo Self-Forcing_StreamEdit/wan_models -> $JZ_WORK/weights/wan_models; repo evaluation/csv is a real directory seeded with a copy of the reference (jobs overwrite the copy, not the reference); idempotent (safe to rerun)"
    needed_by: [jz-clone-setup]
    status: completed
  - id: verify-script
    content: "jz/r50_verify.sh -- on Jean Zay: sha256sum -c MANIFEST.sha256 (weights, data, caches) and CODE.sha256 (the cloned repo files); prints '[r50_verify] failures: N'"
    needed_by: [jz-clone-setup]
    status: completed
  - id: overlay-script
    content: "jz/r50_pip_overlay.sh -- on the Jean Zay frontal: module purge; module load arch/h100 $JZ_MODULE; pip freeze of the module; every package of freeze/five-bench.txt that is missing or at another version, except the module-provided torch/torchvision/torchaudio/triton/nvidia-*, is installed with PYTHONUSERBASE=$JZ_WORK/pyenv/five-bench pip install --user --no-deps --no-cache-dir <pkg>==<ours>; flash-attn (Wan-Edit asserts FLASH_ATTN_2_AVAILABLE; no PyPI wheel, never built from source on the frontal): kept if the module ships a working one, else the prebuilt release wheel matching the module's torch/CUDA/Python/C++ ABI, at our version 2.7.4.post1 if available; editable installs from freeze/five-bench.editable.json: pyiqa with --user --no-deps -e $JZ_WORK/code/iqa_cli, cotracker skipped (no motion-fidelity metric in R49) and listed; writes the wrapper $JZ_WORK/pyenv/five-bench/python (module purge + load arch/h100 $JZ_MODULE, export PYTHONUSERBASE, exec python \"$@\"); then through the wrapper: pip check, a freeze compared with ours (same exclusions), transformers/diffusers/torch versions and the file each is imported from (must be the overlay for every pinned package); writes $JZ_WORK/pyenv/five-bench/overlay.txt and prints '[r50_overlay] installed N, mismatches M, files F' and '[r50_overlay] failures: K'"
    needed_by: [jz-overlay]
    status: completed
  - id: div-hosts-edit
    content: "slurm_scripts/five_bench/r49_div.sh -- optional first argument HOSTS ('se', 'wan' or 'se wan', default 'se wan' = current behaviour on the lab cluster): the loop runs only the {host}_dino / {host}_lpips names of those hosts; nothing else changes"
    needed_by: [jz-repro]
    status: completed
  - id: field-hosts-edit
    content: "slurm_scripts/five_bench/r49_field.sh -- optional first argument HOSTS (default 'se wan' = current behaviour): passes --hosts to r49_field.py and r48_field.py, and the post-run guard checks only the output dirs of those hosts (wan: wan_max_s9, wan_max_s5 + start maps, wan_dino_SD reference); built by R49 todo field-full (HOSTS + MODE) -- mark completed when that is built"
    needed_by: [jz-repro]
    status: completed
  - id: units-group-edit
    content: "evaluation/r49_units.py -- new group wan_pair: wan max_s5 (r49_wan_arms/max_s5_subset) and wan native_s0.92 (r48_wan_arms/native_s0.92_subset, the official Wan-Edit baseline); metric stems r49_wan_max_s5 / r49_wan_native_s0p92 as in our groups"
    needed_by: [jz-repro]
    status: completed
  - id: eval-group-edit
    content: "slurm_scripts/five_bench/r49_eval_all.sh -- accept GROUP wan_pair (usage line and case statement); CLIP-D output r49_clip_directional_wan_pair.csv"
    needed_by: [jz-repro]
    status: completed
  - id: official-render-sbatch
    content: "slurm_scripts/five_bench/r50_wan_official.sh -- one GPU job: the official Wan-Edit config native_s0.92 on the 18 clips of evaluation/r49_cases.json with r48_wan_render.py (same call as r49_newclip_baselines.sh) into $D/r48_wan_arms/native_s0.92_subset; resumable; post-run guard: one complete render per clip (r48_units.py --missing); prints '[r50_wan_official] failures: N'"
    needed_by: [jz-repro]
    status: completed
  - id: port-script
    content: "jz/r50_port.sh -- reads jz/r50_jz.env and rewrites the Wan-side job scripts (slurm_scripts/five_bench/r49_wan_draft.sh, r49_div.sh, r49_field.sh, r49_wan_render.sh, r49_eval_all.sh, r50_wan_official.sh, r50_preflight.sh, the R48 scripts the R49 Wan full stage runs: r48_wan_draft.sh, r48_div.sh, r48_wan_render.sh, and the R49 Wan chain driver r49_wan_full_chain.sh (REPO only, no Slurm header); idempotent, rerun after pulling changed scripts): REPO= -> this clone (absolute path; also the #SBATCH --output/--error=logs/ paths, so sbatch works from any folder); the five-bench interpreter (~/anaconda3/envs/five-bench/bin/python, absolute or ~) -> $JZ_WORK/pyenv/five-bench/python; Slurm headers --partition=L40S|CPU / --exclude=node52 / --mem -> -C h100, --gres=gpu:1, --qos=$JZ_QOS_GPU, --cpus-per-task=$JZ_CPUS_PER_GPU, -A $JZ_ACCOUNT_GPU only if set; --time capped at 20:00:00; adds export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 where missing; evaluation/fivebench/config.yaml IQA_PyTorch_model_path -> $JZ_WORK/code/iqa_cli; then greps the same files for /home/ids/skhairi, anaconda3, L40S, node52, --partition=CPU, streamgve and prints '[r50_port] leftovers: N' (must be 0)"
    needed_by: [jz-port]
    status: completed
  - id: preflight-py
    content: "evaluation/r50_preflight.py -- offline (HF_HUB_OFFLINE=1): prints python, torch, CUDA, transformers, diffusers versions and where each is imported from; loads each model the Wan-side R49 jobs use once and runs one tiny forward pass: DINOv3 ViT-B/16, LPIPS (alex), CLIP ViT-L/14, Qwen2.5-VL-7B processor + model, DINO ViT-B/8 via torch.hub (cached repo), pyiqa NIQE; reports OK/FAIL per model and '[r50_preflight] model failures: N'"
    needed_by: [jz-preflight]
    status: completed
  - id: preflight-sbatch
    content: "slurm_scripts/five_bench/r50_preflight.sh -- one H100 job: nvidia-smi, Slurm CPU/memory granted, torch/CUDA through the wrapper, r50_preflight.py, then a one-clip Wan-Edit draft (r48_wan_draft.py) on evaluation/r50_preflight_case.json into $D/r50_preflight; prints '[r50_preflight] failures: N'"
    needed_by: [jz-preflight]
    status: completed
  - id: preflight-case-file
    content: "evaluation/r50_preflight_case.json -- one case copied from evaluation/r49_cases.json (0076_A_rabbit, edit 2) for the preflight draft"
    needed_by: [jz-preflight]
    status: completed
  - id: repro-chain-script
    content: "jz/r50_repro_chain.sh -- on Jean Zay, submits the Wan-Edit side of R49 on the 18 clips with afterok: r49_wan_draft -> r49_div.sh wan -> r49_field.sh wan -> r49_wan_render.sh arms (--array=1: max_s5); r50_wan_official.sh (no dependency); both renders -> r49_eval_all.sh wan_pair; writes logs/r50_repro_jobs.txt"
    needed_by: [jz-repro]
    status: completed
  - id: compare-script
    content: "evaluation/r50_compare.py -- reads the Jean Zay CSVs of wan_max_s5 and wan_native_s0p92 (harness, fiveacc, CLIP-D wan_pair) and the same units from the read-only reference $JZ_WORK/r50_ref/evaluation_csv (r49 main/base CSVs); per metric (FiVE-Acc, CLIP-D, bg LPIPS, bg PSNR): the paired delta max_s5 - native_s0.92 over the 18 clips with its 95% CI on each platform (same CI method as r49_score.py), and the per-arm platform shift JZ - L40S with its CI and the 3 clips that moved most; PASS when every metric's Jean Zay delta lies inside the L40S 95% CI of that delta; writes evaluation/csv/r50_compare.csv and prints PASS/FAIL"
    needed_by: [jz-compare]
    status: completed
steps:
  - id: fill-config
    type: manual
    check_hint: "cp jz/r50_jz.env.example jz/r50_jz.env and fill JZ_HOST, JZ_WORK (JZ_ACCOUNT_GPU stays empty for a single GPU-only project); no value contains '<'; git check-ignore jz/r50_jz.env must print the file"
    status: completed
    completed_at: 2026-10-20
  - id: make-bundle
    type: local
    wait_for: fill-config
    command: bash jz/r50_make_bundle.sh
    status: completed
    completed_at: 2026-10-20
  - id: push
    type: manual
    wait_for: make-bundle
    check_hint: "ON THE LAB CLUSTER'S FRONTAL (compute nodes such as nodemm06 cannot reach Jean Zay), inside tmux: bash jz/r50_push.sh 2>&1 | tee logs/r50_push.log; logs/r50_push.log ends with '[r50_push] OK' and '[r50_push] failures: 0'"
    sets_status: running
    status: completed
    completed_at: 2026-10-20
  - id: jz-clone-setup
    type: manual
    wait_for: push
    check_hint: "ON JEAN ZAY frontal: git clone --depth 1 git@github.com:skhairi2k/StreamEdit_new.git $JZ_WORK/StreamEdit_new; cp the pushed r50_jz.env to jz/; bash jz/r50_setup_dirs.sh; bash jz/r50_verify.sh -> '[r50_verify] failures: 0'"
    status: completed
    completed_at: 2026-10-21
  - id: jz-overlay
    type: manual
    wait_for: jz-clone-setup
    check_hint: "ON JEAN ZAY frontal: bash jz/r50_pip_overlay.sh -> '[r50_overlay] failures: 0', mismatches 0; paste the summary lines and idr_quota_user"
    status: completed
    completed_at: 2026-10-21
  - id: jz-port
    type: manual
    wait_for: jz-overlay
    check_hint: "ON JEAN ZAY: bash jz/r50_port.sh -> '[r50_port] leftovers: 0'; git diff --stat shows only the job scripts and config.yaml"
    status: completed
    completed_at: 2026-10-21
  - id: jz-preflight
    type: manual
    wait_for: jz-port
    check_hint: "ON JEAN ZAY: sbatch slurm_scripts/five_bench/r50_preflight.sh; its log ends with '[r50_preflight] failures: 0' (every model OK offline, the one-clip Wan draft written)"
    status: completed
    completed_at: 2026-10-21
    job_id: "1399496"
  - id: jz-repro
    type: manual
    wait_for: jz-preflight
    check_hint: "ON JEAN ZAY: bash jz/r50_repro_chain.sh; when the queue is empty, every log in logs/r50_repro_jobs.txt ends with 'failures: 0' / 'unit failures: 0'"
    sets_status: finished
    status: pending
  - id: jz-compare
    type: manual
    wait_for: jz-repro
    check_hint: "ON JEAN ZAY: $JZ_WORK/pyenv/five-bench/python evaluation/r50_compare.py -> PASS, evaluation/csv/r50_compare.csv written; paste the table here"
    sets_status: analyzed
    status: pending
isProject: true
---

# R50: Jean Zay H100 Setup Kit

## Context

Run the Wan-Edit half of R49 on Jean Zay (H100); StreamEdit stays on the lab cluster (L40S). The port must work in one pass and be checked at every step. Code arrives by a shallow clone of the repo. Python is the IDRIS H100 module matching our `five-bench` env (every Wan-side job runs in that env), plus a small pinned pip overlay. Weights, data and our CSVs are copied byte for byte from this cluster by rsync and verified with SHA-256. One script rewrites the hard-coded paths and Slurm headers from one config file. An offline job confirms every model loads on a compute node. H100 will not reproduce L40S bit for bit, so the check is on what the paper claims: the 18-clip paired delta of `max_s5` against the official Wan-Edit baseline must fall inside our L40S confidence interval.

Done when: on Jean Zay, the checksums verify, the overlay leaves no version mismatches, the port reports no leftovers, the preflight passes offline, and the reproduced Wan-Edit paired deltas (`max_s5 − native_s0.92`) lie inside the L40S 95% CIs for FiVE-Acc, CLIP-D, bg LPIPS and bg PSNR. The R49 full stage itself stays in R49.

## Execution steps

| # | Step id | Type | Waits for | Sets status |
|---|---|---|---|---|
| 0 | *(prep)* todos `env-template-file`, `gitignore-edit`, `readme-file`, `bundle-script`, `push-script`, `setup-script`, `verify-script`, `overlay-script`, `div-hosts-edit`, `field-hosts-edit`, `units-group-edit`, `eval-group-edit`, `official-render-sbatch`, `port-script`, `preflight-py`, `preflight-sbatch`, `preflight-case-file`, `repro-chain-script`, `compare-script` | /build-step | — | implemented |
| 1 | `fill-config` | manual (this cluster) | — | — |
| 2 | `make-bundle` | local | fill-config | — |
| 3 | `push` | manual (lab frontal) | make-bundle | running |
| 4 | `jz-clone-setup` | manual (Jean Zay) | push | — |
| 5 | `jz-overlay` | manual (Jean Zay) | jz-clone-setup | — |
| 6 | `jz-port` | manual (Jean Zay) | jz-overlay | — |
| 7 | `jz-preflight` | manual (Jean Zay) | jz-port | — |
| 8 | `jz-repro` | manual (Jean Zay) | jz-preflight | finished |
| 9 | `jz-compare` | manual (Jean Zay) | jz-repro | analyzed |

Invoke: `/run-step R50 make-bundle`, `/run-step R50 push`, then the Jean Zay steps in order (`/run-step R50 jz-clone-setup` … `jz-compare`) after running their commands there.

## Decisions

| Item | Choice |
|---|---|
| Host split | Wan-Edit (draft, divergence, field, renders, baselines, matched-NFE, eval) on Jean Zay H100; StreamEdit on the lab cluster L40S (user, 2026-10-20). Each host's arms, baselines and curves come from one platform, so every within-host comparison is same-platform |
| Target | Jean Zay H100 partition (`gpu_p6`) for every job, headers as in the IDRIS `exec_batch` H100 example: `-C h100`, `--gres=gpu:1`, `--cpus-per-task=24`, `--hint=nomultithread`, `module purge` + `module load arch/h100`; `-A omt@h100` (required: first `sbatch` on 2026-10-21 failed with "Multiple accounts available (omt@a100, omt@h100, omt@v100)"; the earlier "no -A" decision was wrong); the short CPU-only `r49_field.sh` also runs there with 1 GPU |
| Site values | only in `jz/r50_jz.env` (from `jz/r50_jz.env.example`, git-ignored): host, `$WORK` path, account `omt@h100`, QOS `qos_gpu_h100-t3` (confirmed by the user), CPUs per GPU 24 (IDRIS example), module name |
| Code | `git clone --depth 1` of `skhairi2k/StreamEdit_new` on the frontal into `$WORK/StreamEdit_new`; byte-identity checked with `CODE.sha256` |
| Python | IDRIS module `arch/h100` + `pytorch-gpu/py3/2.5.1` (torch 2.5.1, as our `five-bench`); every other package that differs from our `pip freeze` pinned to our version with `pip install --user --no-deps` into `$WORK/pyenv/five-bench` (the module's own transformers 4.55.4 is shadowed by our pin); torch build from IDRIS, not byte-identical to ours; `streamgve` not needed (no StreamEdit job on Jean Zay). Measured on the module (2026-10-21): python 3.12.7 (lab 3.11.14), torch 2.5.1 + CUDA 12.4, C++ ABI True, flash-attn 2.7.4.post1 already shipped (= our pin), numpy 1.26.x (its torch is compiled against it). Therefore numpy and OpenCV (4.12+ needs numpy>=2) stay the module's, and peft 0.18.1 is added (diffusers 0.36 refuses the module's peft 0.14; 0.17.x does not import with transformers 5.3). Python 3.12 vs 3.11 is one more platform difference that `r50_compare` has to tolerate |
| Why not conda-pack | `$WORK` inode quota: 500k, 253k used; unpacked `five-bench` alone is 121k files; modules live on shared storage and cost no user inodes |
| Interpreter | wrapper `$WORK/pyenv/five-bench/python` (loads `arch/h100` + the module, sets `PYTHONUSERBASE`, `exec python "$@"`), substituted for our env's python path in the job scripts |
| Editable installs | none: `pyiqa` is reached by putting `$WORK/code/iqa_cli` on `PYTHONPATH` in the interpreter wrapper (same effect as the lab's editable install, no build step); `cotracker` skipped (no motion-fidelity metric in R49) |
| Weights | rsync of our own copies (byte-identical, no Hugging Face token needed): Wan2.1-T2V-1.3B (17 GB; Wan-Edit loads it from `Self-Forcing_StreamEdit/wan_models`), HF cache entries DINOv3 ViT-B/16 (0.3 GB), CLIP ViT-L/14 (1.6 GB), Qwen2.5-VL-7B-Instruct (16 GB), the torch hub cache (1.1 GB: DINO ViT-B/8, AlexNet/VGG for LPIPS, RAFT, pyiqa NIQE); ~36 GB |
| Data | FiVE-Bench `videos`, `images`, `bmasks`, `edit_prompt` (~0.4 GB), a FROZEN COPY of the 23 CSVs `r50_compare.py` reads (the live `evaluation/csv` is never linked: running evals create and delete files there, which broke the first re-push) as the read-only compare reference `$WORK/r50_ref/evaluation_csv` (the repo's `evaluation/csv` on Jean Zay is a separate copy that jobs overwrite), `iqa_cli` (19 MB) |
| Inode budget on `$WORK` | ~246k free: bundle ~16.4k (FiVE data 16k files, 23 CSVs, caches), overlay ~5–15k (measured by the overlay step), outputs ~500 files per clip-edit and arm |
| Layout on Jean Zay | `$WORK/{StreamEdit_new, pyenv, weights, cache, Data, code, r50_ref}`; `~/Data` and `~/.cache/{huggingface,torch}` are symlinks into `$WORK`, so the Python code runs unchanged; outputs under `$WORK/Data/dataggen/outputs/five_bench` |
| Transfer | direct rsync from this cluster (`jz/r50_push.sh`, resumable, `--partial`, `--copy-unsafe-links`) of `$LOCAL_STAGE`, a folder of symlinks to our files (no copy here) |
| Script port | `jz/r50_port.sh` edits only the Wan-side job scripts and `config.yaml`: repo path (this clone, absolute; log paths too), interpreter → wrapper, Slurm headers (`-C h100`, `--gres=gpu:1`, QOS, `--cpus-per-task`, no `--exclude`, no `--mem`, `--time` ≤ 20 h), offline flags; must report 0 leftovers; rerun it on scripts added later (R49 full-stage Wan scripts) |
| Host-only R49 jobs | `r49_div.sh` and `r49_field.sh` get an optional `HOSTS` argument (default `se wan` = unchanged behaviour on the lab cluster); Jean Zay calls them with `wan` |
| Offline check | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; preflight loads every model once on a compute node and runs a one-clip Wan-Edit draft |
| Reproduction | Wan-Edit side of R49 on the 18 clips: draft, divergence, field, `max_s5` (the full-stage arm) and the official baseline `native_s0.92`, eval of both (group `wan_pair`); `max_s9`, the other baselines and matched-59 are not rerun |
| Pass criterion | H100 ≠ L40S bit for bit (kernels, bf16 accumulation order, torch build; integer start steps and VLM yes/no can flip near boundaries); PASS when, for FiVE-Acc, CLIP-D, bg LPIPS and bg PSNR, the Jean Zay paired delta `max_s5 − native_s0.92` lies inside the L40S 95% CI of the same delta (L40S bg LPIPS: −0.018); per-arm platform shifts reported with CIs |
| Out of scope | StreamEdit on Jean Zay, the R49 full stage (stays in R49), refactoring the method code, a separate repo |

## Step commands

### fill-config
```bash
cp jz/r50_jz.env.example jz/r50_jz.env
${EDITOR:-vi} jz/r50_jz.env
git check-ignore jz/r50_jz.env
```

### make-bundle
```bash
bash jz/r50_make_bundle.sh
```

### push
```bash
# on the lab cluster's frontal (ssh zay works only from there), inside tmux
cd ~/Code/StreamEdit_bigchantier
bash jz/r50_push.sh --dry-run | tail -n 5
bash jz/r50_push.sh 2>&1 | tee logs/r50_push.log
```

### jz-clone-setup
```bash
# on the Jean Zay frontal
git clone --depth 1 git@github.com:skhairi2k/StreamEdit_new.git $WORK/StreamEdit_new
cd $WORK/StreamEdit_new
cp $WORK/r50_stage/r50_jz.env jz/r50_jz.env
bash jz/r50_setup_dirs.sh
bash jz/r50_verify.sh
```

### jz-overlay
```bash
cd $WORK/StreamEdit_new
bash jz/r50_pip_overlay.sh
idr_quota_user
```

### jz-port
```bash
cd $WORK/StreamEdit_new
bash jz/r50_port.sh
git diff --stat
```

### jz-preflight
```bash
cd $WORK/StreamEdit_new
sbatch slurm_scripts/five_bench/r50_preflight.sh
tail -n 20 logs/r50_preflight_<job>.out
```

### jz-repro
```bash
cd $WORK/StreamEdit_new
bash jz/r50_repro_chain.sh
squeue -u $USER
grep -E 'failures' logs/r49_*.out logs/r50_*.out
```

### jz-compare
```bash
cd $WORK/StreamEdit_new
$WORK/pyenv/five-bench/python evaluation/r50_compare.py
```

## Pipeline

```mermaid
flowchart LR
  CFG[jz/r50_jz.env] --> MB[r50_make_bundle.sh: Wan weights, caches, FiVE data, our CSVs, five-bench freeze + SHA-256]
  MB --> PU[r50_push.sh: rsync to Jean Zay $WORK]
  GIT[GitHub: StreamEdit_new] --> CL[clone --depth 1 on the frontal]
  PU --> SU[r50_setup_dirs.sh + r50_verify.sh]
  CL --> SU
  MOD[IDRIS arch/h100 + pytorch-gpu/py3/2.5.1] --> OV[r50_pip_overlay.sh: pinned pip --user + wrapper]
  SU --> OV
  OV --> PO[r50_port.sh: paths, Slurm headers, offline flags]
  PO --> PF[r50_preflight.sh: models offline + one-clip Wan draft]
  PF --> WD[r49_wan_draft] --> DV[r49_div.sh wan] --> FI[r49_field.sh wan] --> MX[r49_wan_render.sh arms: max_s5]
  PF --> OF[r50_wan_official.sh: native_s0.92]
  MX --> EV[r49_eval_all.sh wan_pair]
  OF --> EV
  EV --> CO[r50_compare.py: paired deltas vs L40S CIs]
  CO --> R49[R49 Wan full stage on Jean Zay]
```

## Code to touch

| File | Change |
|---|---|
| `jz/r50_jz.env.example` | new: commented template of every site value with its default |
| `.gitignore` | add `jz/r50_jz.env` |
| `jz/r50_README.md` | new: host split, ordered procedure, the check after each step, inode budget, recovery notes |
| `jz/r50_make_bundle.sh` | new: symlink stage of the Wan weights, the three HF cache entries, the torch hub cache, FiVE data subset, `ref/evaluation_csv` (a frozen COPY of the 23 CSVs `r50_compare.py` reads, hashed once: the live folder is written by running evals), `iqa_cli`; `five-bench` package lists into `freeze/`; `MANIFEST.sha256` over the stage (parallel `sha256sum`), `CODE.sha256` over tracked + new files of `slurm_scripts/five_bench`, `evaluation`, `Self-Forcing_StreamEdit`, `wan-edit`, `jz` with an uncommitted-code warning |
| `jz/r50_push.sh` | new: `rsync -a --copy-unsafe-links --partial --info=progress2` of `$LOCAL_STAGE` to `$JZ_HOST:$JZ_WORK/r50_stage/` plus the filled env file; `--meta` sends only CODE.sha256, MANIFEST.sha256, freeze/ and r50_jz.env (seconds): every refresh after `r50_setup_dirs.sh` must use it, because setup moves the big folders out of the stage and a full push re-sends all 38 GB |
| `jz/r50_setup_dirs.sh` | new (built, tested on a fake tree): `mv` the stage folders into `$JZ_WORK/{weights,cache,Data,code}` and the read-only reference `$JZ_WORK/r50_ref/evaluation_csv` (MANIFEST / CODE checklists, `freeze/` and `r50_jz.env` stay in the stage); links `~/Data`, `~/.cache/{huggingface,torch}`, `Self-Forcing_StreamEdit/wan_models`; seeds a writable `evaluation/csv` only when empty; creates `$WORK/Data/dataggen/outputs/five_bench` and `logs/`; idempotent, reports CONFLICT and never overwrites; `R50_ENV` overrides the config for tests |
| `jz/r50_verify.sh` | new (built, tested): maps the stage-relative manifest paths to their final locations and runs `sha256sum -c` with 8 parallel readers; then `CODE.sha256` against the clone, naming every differing / missing file; then the layout (links, FiVE videos, Wan weights, `evaluation/csv`); `--code-only` hashes just the repo files (after a `git pull`); prints `[r50_verify] failures: N` |
| `jz/r50_pip_overlay.sh` | built (tested end to end on a fake module: plan, real `pip install`, real flash-attn wheel download, wrapper, verify; idempotent rerun; failing pin; lab guard): loads `arch/h100` + `$JZ_MODULE`, refuses python < 3.10; plans from the stage freeze (every pin that is missing or differs, except torch family / nvidia-* / pip / setuptools / wheel; `name @ file://` lines tolerated); `pip install --user --no-deps --ignore-installed --prefer-binary` into `PYTHONUSERBASE=$WORK/pyenv/five-bench` (bulk, then one by one to name failures); flash-attn from the prebuilt wheel named from the module's torch / CUDA / python / C++ ABI (`$WORK/pyenv/wheels/` first, else GitHub release, never compiled); wrapper `$WORK/pyenv/five-bench/python` (module purge + load, `PYTHONPATH` = overlay first + `$WORK/code/iqa_cli`, so subprocesses such as the NIQE call see it); verification through the wrapper (versions, no shadowing, key imports from the overlay, `pip check` for our packages) and `overlay.txt`; `--dry-run` plans only AND reports the flash-attn wheel it needs (exact file name; `HTTP 200` reachable, `404` none published for this torch / CUDA / python / ABI, `000` GitHub unreachable; the last two count as failures), so the module's python / torch / ABI and the flash-attn situation are known before anything is installed (replaces the manual three-command check); ends with `installed N, mismatches M, files F` and `failures: K`; 2026-10-21 first real run on Jean Zay (module: python 3.12.7, torch 2.5.1 cu124, C++ ABI True, numpy 1.26.x) exposed two things the lab env cannot: numpy 2.x breaks the module's torch<->numpy bridge, and diffusers 0.36 refuses the module's peft 0.14 -> numpy and OpenCV are now LEFT TO THE MODULE (OpenCV 4.10.0.84 only if the module has none), peft 0.18.1 is added when the module's is < 0.17 (0.17.x does not import with transformers 5.3), anything of that kind already in the overlay is removed first, verify gained a real torch<->numpy round trip (warnings recorded) and imports of numpy / scipy / pandas / cv2 / PIL / torchvision, and `pip check` counts only complaints made BY packages we installed (full list in `overlay_pipcheck.txt`) |
| `slurm_scripts/five_bench/r49_div.sh` | built: optional first argument `HOSTS` (`"se wan"` default = unchanged; `se` or `wan`), bad values refused; the loop skips the other host's names (Jean Zay calls `r49_div.sh wan`) |
| `slurm_scripts/five_bench/r49_field.sh` | optional `HOSTS` argument (default `se wan`); `--hosts` to `r49_field.py` and `r48_field.py`; guard checks only those hosts' outputs |
| `evaluation/r49_units.py` | built: subset group `wan_pair` = `wan_max_s5` (`r49_wan_arms/max_s5_subset`) + `wan_native_s0p92` (`r48_wan_arms/native_s0.92_subset`); unit stems `r49_wan_max_s5` / `r49_wan_native_s0p92` are exactly the lab CSVs' that `r50_compare.py` reads; other groups unchanged |
| `slurm_scripts/five_bench/r49_eval_all.sh` | built: `wan_pair` accepted in subset mode (usage line and group case; CLIP-D → `r49_clip_directional_wan_pair.csv`); refused in full mode; the other groups' arguments are unchanged |
| `slurm_scripts/five_bench/r50_wan_official.sh` | built: `native_s0.92` (the official Wan-Edit) on the 18 clips with `r48_wan_render.py --mode subset --cases r49_cases.json` into `r48_wan_arms/native_s0.92_subset` (same call as `r49_newclip_baselines.sh`), resumable, `r48_units.py --missing` guard, prints the render wall time; starts with `REPO=` so the port picks it up |
| `jz/r50_port.sh` | new (built, tested on a scratch copy of the real scripts): per-file rules for `r48_wan_draft/div/wan_render`, `r49_wan_draft/div/field/wan_render/eval_all`, `r50_wan_official`, `r50_preflight`, `r49_wan_full_chain` and `jz/r50_repro_chain.sh` (missing ones skipped): `REPO=` → this clone, interpreter → `$JZ_WORK/pyenv/five-bench/python`, `#SBATCH` partition / exclude / mem / cpus / gres / hint / qos / account lines → one H100 block (`-C h100`, `--gres=gpu:1`, `--cpus-per-task`, `--hint=nomultithread`, `--qos`, optional `-A`), `--time` capped at 20 h, `TRANSFORMERS_OFFLINE=1` added, `config.yaml` paths → `$JZ_WORK/code`; then `bash -n` and a leftover grep (`leftovers: N`, must be 0); idempotent; `--check` changes nothing; refuses to run on the lab cluster (repo under `/home/ids`, `/projects`, `/store`, `/scratch` or missing `$JZ_WORK`); drivers added later must start with a `REPO=` line |
| `evaluation/r50_preflight.py` | new: offline report of versions + the file each is imported from, then seven checks that use the real jobs' own loaders: `cuda`, `flash_attn` (tiny `flash_attn_func`), `dinov3` (`r31_divergence.load_dinov3`), `lpips_alex` (`r30_divergence.load_lpips`), `clip_d` (`r33_clip_directional.Clip`), `harness` (the FiVE `MetricsCalculator` exactly as `evaluate.py` builds it: CLIPScore, PSNR, LPIPS-squeeze, SSIM, DINO ViT-B/8 via `torch.hub`, Qwen2.5-VL with flash attention; asserts `five_acc_calculator` is not None because the constructor swallows a failed Qwen load and `five_acc` would silently be NaN; short generation), `niqe` (the harness' own `inference_iqa.py` subprocess); OK/FAIL table, `model failures: N`; `--device cpu`, `--only`, `--skip` |
| `slurm_scripts/five_bench/r50_preflight.sh` | new (lab-style header, rewritten by `jz/r50_port.sh`): node info (GPU, granted CPUs / memory), torch / transformers through `$PY_FB`, `r50_preflight.py`, then one real Wan-Edit draft (`r48_wan_draft.py` on `r50_preflight_case.json`) into `$D/r50_preflight/<jobid>/` with its wall time printed (the per-clip cost for extrapolating the full stage); `failures: N`; `sbatch --qos=qos_gpu_h100-dev` starts sooner |
| `evaluation/r50_preflight_case.json` | new (built): the `0076_A_rabbit` (edit 2) entry copied verbatim from `r49_cases.json` |
| `jz/r50_repro_chain.sh` | built (dry-run tested with a fake `sbatch`): guards (JZ_WORK exists, scripts ported, wrapper present, data and Wan weights linked), then `afterok` chain `r49_wan_draft` → `r49_div.sh wan` → `r49_field.sh wan` → `r49_wan_render.sh arms --array=1`; `r50_wan_official.sh` with no dependency; `r49_eval_all.sh wan_pair` afterok both renders; ids and the git commit to `logs/r50_repro_jobs.txt`; refuses a second chain |
| `evaluation/r50_compare.py` | built (tested: null case, realistic noise, a +0.05 bg LPIPS regression, missing inputs): reuses `r48_score.unit_values` / `paired_ci` (the lab's own loaders and bootstrap, 2000 resamples, seed 0); for FiVE-Acc, CLIP-D, bg LPIPS, bg PSNR: the paired delta `wan_max_s5 - wan_native_s0p92` on Jean Zay vs the lab's 95% CI (PASS = inside for all four), plus the platform shift of each delta and of each arm with CIs, the 3 clips that moved most, a NOTE for metrics whose lab effect is not distinguishable from 0 (FiVE-Acc and CLIP-D here: a weak check), and a cross-check of the recomputed lab means against `r49_deltas.csv`; `evaluation/csv/r50_compare.csv` (`kind` = delta / arm_shift / worst_clip); exit 0 PASS, 1 FAIL, 2 inputs missing |
