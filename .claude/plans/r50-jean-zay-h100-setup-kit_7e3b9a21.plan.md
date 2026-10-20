---
name: "R50 plan — Jean Zay H100 Setup Kit"
overview: "Move R49 to Jean Zay (H100) in one verified pass: code by shallow git clone; Python from the IDRIS H100 PyTorch modules (2.8.0 for streamgve, 2.5.1 for five-bench) plus a pinned pip overlay of every package that differs from our envs; the exact weights and data from this cluster by rsync with SHA-256 checks; a port script that rewrites the hard-coded paths and Slurm headers from one config file; an offline preflight job on a compute node; and a reproduction of the R49 main arms on the 18 clips compared with our CSVs before any full-stage run."
task_id: R50
todos:
  - id: env-template-file
    content: "jz/r50_jz.env.example -- the only file with site-specific values, copied to jz/r50_jz.env and filled in: JZ_HOST (user@host for rsync), JZ_WORK (absolute $WORK on Jean Zay), JZ_REPO (clone path, default $JZ_WORK/StreamEdit_new), JZ_ACCOUNT_GPU (<project>@h100), JZ_ACCOUNT_CPU (<project>@cpu), JZ_QOS_GPU (default qos_gpu_h100-t3), JZ_CPU_PARTITION (default cpu_p1), JZ_CPUS_PER_GPU (default 24), JZ_MODULE_streamgve (pytorch-gpu/py3/2.8.0), JZ_MODULE_five_bench (pytorch-gpu/py3/2.5.1), LOCAL_STAGE (staging dir on this cluster, default ~/Data/r50_stage)"
    needed_by: [fill-config]
    status: completed
  - id: gitignore-edit
    content: ".gitignore -- add jz/r50_jz.env (filled config never committed)"
    needed_by: [fill-config]
    status: completed
  - id: readme-file
    content: "jz/r50_README.md -- the ordered procedure (this cluster: fill config, make bundle, push; Jean Zay: clone + setup + verify, pip overlay, port, preflight, reproduction, compare), what each step checks, the inode budget, and how to recover from a failed step"
    needed_by: [fill-config]
    status: completed
  - id: bundle-script
    content: "jz/r50_make_bundle.sh -- stages in $LOCAL_STAGE (hard links / rsync --link-dest, no second copy): weights/{wan_models/Wan2.1-T2V-1.3B, checkpoints/self_forcing_dmd.pt}, cache/huggingface/hub/{models--facebook--dinov3-vitb16-pretrain-lvd1689m, models--openai--clip-vit-large-patch14, models--Qwen--Qwen2.5-VL-7B-Instruct}, cache/torch/hub (DINO ViT-B/8 repo + weights, AlexNet/VGG for LPIPS, pyiqa), data/FiVE-Fine-Grained-Video-Editing-Benchmark/{videos,images,bmasks,edit_prompt}, data/dataggen/outputs/five_bench/anchors, data/evaluation_csv (evaluation/csv, ignored by git), code/iqa_cli, freeze/{streamgve,five-bench}.txt (pip freeze --exclude-editable of each env) and freeze/{streamgve,five-bench}.editable (pip list --editable); writes $LOCAL_STAGE/MANIFEST.sha256 over every staged file and $LOCAL_STAGE/CODE.sha256 over every repo file the R49 chain uses (job scripts, Python, case files); prints size and file count"
    needed_by: [make-bundle]
    status: pending
  - id: push-script
    content: "jz/r50_push.sh -- rsync -a --partial --info=progress2 $LOCAL_STAGE/ $JZ_HOST:$JZ_WORK/r50_stage/ (resumable) plus jz/r50_jz.env, then prints the remote total size; reads jz/r50_jz.env"
    needed_by: [push]
    status: pending
  - id: setup-script
    content: "jz/r50_setup_dirs.sh -- on Jean Zay: moves stage contents into $JZ_WORK/{weights,cache,Data,code}; symlinks ~/Data -> $JZ_WORK/Data, ~/.cache/huggingface -> $JZ_WORK/cache/huggingface, ~/.cache/torch -> $JZ_WORK/cache/torch, repo Self-Forcing_StreamEdit/wan_models and /checkpoints -> $JZ_WORK/weights/..., evaluation/csv -> the shipped CSVs; idempotent (safe to rerun)"
    needed_by: [jz-clone-setup]
    status: pending
  - id: verify-script
    content: "jz/r50_verify.sh -- on Jean Zay: sha256sum -c MANIFEST.sha256 (weights, data, caches) and CODE.sha256 (the cloned repo files R49 runs); prints '[r50_verify] failures: N'"
    needed_by: [jz-clone-setup]
    status: pending
  - id: overlay-script
    content: "jz/r50_pip_overlay.sh -- on the Jean Zay frontal, per env (streamgve -> $JZ_MODULE_streamgve, five-bench -> $JZ_MODULE_five_bench): module purge; module load arch/h100 <module>; pip freeze of the module; every package of our freeze that is missing or at another version, except the module-provided torch/torchvision/torchaudio/triton/nvidia-*, is installed with PYTHONUSERBASE=$JZ_WORK/pyenv/<env> pip install --user --no-deps --no-cache-dir <pkg>==<ours>; the recorded editable installs (iqa_cli for five-bench) with --user --no-deps -e $JZ_WORK/code/<dir>; writes the wrapper $JZ_WORK/pyenv/<env>/python (module purge + load arch/h100 <module>, export PYTHONUSERBASE, exec python \"$@\"); then through the wrapper: pip check, a freeze compared with ours (same exclusions), torch version; writes $JZ_WORK/pyenv/<env>/overlay.txt (installed, skipped, remaining mismatches) and prints '[r50_overlay] <env>: installed N, mismatches M, files F' and '[r50_overlay] failures: K'"
    needed_by: [jz-overlay]
    status: pending
  - id: port-script
    content: "jz/r50_port.sh -- reads jz/r50_jz.env and rewrites the R49/R48 job scripts the chain runs (slurm_scripts/five_bench/r49_*.sh, r48_se_stage1.sh, r48_wan_draft.sh, r48_div.sh, r48_wan_render.sh, r50_preflight.sh): REPO= -> ${SLURM_SUBMIT_DIR:-$PWD}; the two env interpreters (~/anaconda3/envs/<env>/bin/python, absolute or ~) -> $JZ_WORK/pyenv/<env>/python; GPU headers --partition=L40S / --exclude=node52 / --mem -> -C h100, -A $JZ_ACCOUNT_GPU, --qos=$JZ_QOS_GPU, --cpus-per-task=$JZ_CPUS_PER_GPU; CPU headers -> --partition=$JZ_CPU_PARTITION, -A $JZ_ACCOUNT_CPU; --time capped at 20:00:00; adds export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 where missing; evaluation/fivebench/config.yaml IQA_PyTorch_model_path -> $JZ_WORK/code/iqa_cli; then greps the same files for /home/ids/skhairi, anaconda3, L40S, node52, --partition=CPU and prints '[r50_port] leftovers: N' (must be 0)"
    needed_by: [jz-port]
    status: pending
  - id: preflight-py
    content: "evaluation/r50_preflight.py -- offline (HF_HUB_OFFLINE=1): prints python, torch, CUDA, transformers, diffusers versions and site.USER_SITE; loads each model R49 uses once and runs one tiny forward pass: DINOv3 ViT-B/16, LPIPS (alex), CLIP ViT-L/14, Qwen2.5-VL-7B processor + model, DINO ViT-B/8 via torch.hub (cached repo), pyiqa NIQE; reports OK/FAIL per model and '[r50_preflight] model failures: N'"
    needed_by: [jz-preflight]
    status: pending
  - id: preflight-sbatch
    content: "slurm_scripts/five_bench/r50_preflight.sh -- one H100 job: nvidia-smi, torch/CUDA through both wrappers, r50_preflight.py (five-bench wrapper), then a one-clip run of the StreamEdit draft (r31_stage1.py, streamgve wrapper) and the Wan-Edit draft (r48_wan_draft.py) on evaluation/r50_preflight_case.json into $D/r50_preflight; prints '[r50_preflight] failures: N'"
    needed_by: [jz-preflight]
    status: pending
  - id: preflight-case-file
    content: "evaluation/r50_preflight_case.json -- one case copied from evaluation/r49_cases.json (0076_A_rabbit, edit 2) for the preflight drafts"
    needed_by: [jz-preflight]
    status: pending
  - id: repro-chain-script
    content: "jz/r50_repro_chain.sh -- on Jean Zay, submits the R49 main path on the 18 clips with afterok: r49_se_stage1, r49_wan_draft -> r49_div -> r49_field -> r49_se_stage3 + r49_wan_render arms (array 0-1) -> r49_eval_all main; no new-clip baselines, reference render or base eval (those reuse R48 renders that stay on this cluster); writes logs/r50_repro_jobs.txt"
    needed_by: [jz-repro]
    status: pending
  - id: compare-script
    content: "evaluation/r50_compare.py -- compares the Jean Zay per-clip CSVs of se_max, wan_max_s9, wan_max_s5 (harness, five_acc, CLIP-D) with ours (shipped in data/evaluation_csv): per metric the mean difference over the 18 clips against fixed tolerances (bg LPIPS 0.01, bg PSNR 0.5 dB, CLIP-D 0.01, FiVE-Acc 0.06 = one clip), plus the 3 clips with the largest deviation; writes evaluation/csv/r50_compare.csv and prints PASS/FAIL"
    needed_by: [jz-compare]
    status: pending
steps:
  - id: fill-config
    type: manual
    check_hint: "cp jz/r50_jz.env.example jz/r50_jz.env and fill JZ_HOST, JZ_WORK, JZ_ACCOUNT_GPU, JZ_ACCOUNT_CPU (and defaults if they differ); git check-ignore jz/r50_jz.env must print the file"
    status: pending
  - id: make-bundle
    type: local
    wait_for: fill-config
    command: bash jz/r50_make_bundle.sh
    status: pending
  - id: push
    type: local
    wait_for: make-bundle
    command: bash jz/r50_push.sh
    sets_status: running
    status: pending
  - id: jz-clone-setup
    type: manual
    wait_for: push
    check_hint: "ON JEAN ZAY frontal: git clone --depth 1 git@github.com:skhairi2k/StreamEdit_new.git $JZ_WORK/StreamEdit_new; cp the pushed r50_jz.env to jz/; bash jz/r50_setup_dirs.sh; bash jz/r50_verify.sh -> '[r50_verify] failures: 0'"
    status: pending
  - id: jz-overlay
    type: manual
    wait_for: jz-clone-setup
    check_hint: "ON JEAN ZAY frontal: bash jz/r50_pip_overlay.sh -> '[r50_overlay] failures: 0', mismatches 0 for both envs; paste the two summary lines and idr_quota_user"
    status: pending
  - id: jz-port
    type: manual
    wait_for: jz-overlay
    check_hint: "ON JEAN ZAY: bash jz/r50_port.sh -> '[r50_port] leftovers: 0'; git diff --stat shows only the job scripts and config.yaml"
    status: pending
  - id: jz-preflight
    type: manual
    wait_for: jz-port
    check_hint: "ON JEAN ZAY: sbatch slurm_scripts/five_bench/r50_preflight.sh; its log ends with '[r50_preflight] failures: 0' (every model OK offline, both one-clip drafts written)"
    status: pending
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

Move R49 to Jean Zay (H100) so its full stage can run there, in one pass that is checked at every step. Code arrives by a shallow clone of the repo. Python comes from the IDRIS H100 PyTorch modules, which match our torch versions, plus a small pinned pip overlay. Weights, data and baseline CSVs are copied byte for byte from this cluster by rsync and verified with SHA-256. One script rewrites the hard-coded paths and Slurm headers from one config file. An offline job confirms every model loads on a compute node, and the R49 main arms are reproduced on the 18 clips and compared with our results.

Done when: on Jean Zay, the checksums verify, the overlay leaves no version mismatches, the port reports no leftovers, the preflight passes offline, and the reproduced R49 main arms (StreamEdit `max`, Wan-Edit `max_s9`, `max_s5`) match our 18-clip metrics within the tolerances. The R49 full stage itself stays in R49.

## Execution steps

| # | Step id | Type | Waits for | Sets status |
|---|---|---|---|---|
| 0 | *(prep)* todos `env-template-file`, `gitignore-edit`, `readme-file`, `bundle-script`, `push-script`, `setup-script`, `verify-script`, `overlay-script`, `port-script`, `preflight-py`, `preflight-sbatch`, `preflight-case-file`, `repro-chain-script`, `compare-script` | /build-step | — | implemented |
| 1 | `fill-config` | manual (this cluster) | — | — |
| 2 | `make-bundle` | local | fill-config | — |
| 3 | `push` | local | make-bundle | running |
| 4 | `jz-clone-setup` | manual (Jean Zay) | push | — |
| 5 | `jz-overlay` | manual (Jean Zay) | jz-clone-setup | — |
| 6 | `jz-port` | manual (Jean Zay) | jz-overlay | — |
| 7 | `jz-preflight` | manual (Jean Zay) | jz-port | — |
| 8 | `jz-repro` | manual (Jean Zay) | jz-preflight | finished |
| 9 | `jz-compare` | manual (Jean Zay) | jz-repro | analyzed |

Invoke: `/run-step R50 fill-config`, `/run-step R50 make-bundle`, `/run-step R50 push`, then the Jean Zay steps in order (`/run-step R50 jz-clone-setup` … `jz-compare`) after running their commands there.

## Decisions

| Item | Choice |
|---|---|
| Target | Jean Zay H100 partition (`-C h100`, `-A <project>@h100`); CPU jobs on the CPU partition with `<project>@cpu` |
| Site values | only in `jz/r50_jz.env` (from `jz/r50_jz.env.example`, git-ignored): host, `$WORK` path, accounts, QOS (default `qos_gpu_h100-t3`), CPU partition (default `cpu_p1`), CPUs per GPU (default 24), the two module names — QOS / partition / CPU defaults to confirm against the IDRIS documentation |
| Code | `git clone --depth 1` of `skhairi2k/StreamEdit_new` on the frontal into `$WORK/StreamEdit_new`; byte-identity of the files R49 runs checked with `CODE.sha256` |
| Python environments | IDRIS modules under `arch/h100`: `pytorch-gpu/py3/2.8.0` (torch 2.8.0, CUDA 12.8) for `streamgve`, `pytorch-gpu/py3/2.5.1` for `five-bench`; every other package that differs from our `pip freeze` is pinned to our version with `pip install --user --no-deps` into `$WORK/pyenv/<env>` (module ships e.g. transformers 4.55.4 / diffusers 0.35.0, ours 5.12.0 / 0.31.0); torch builds come from IDRIS, so they are not byte-identical to ours |
| Why not conda-pack | `$WORK` inode quota: 500k, 253k used; the unpacked envs are 286k files (streamgve 165k, five-bench 121k); modules live on shared storage and cost no user inodes |
| Interpreter | per env a wrapper `$WORK/pyenv/<env>/python` (loads `arch/h100` + the module, sets `PYTHONUSERBASE`, `exec python "$@"`), so job scripts keep one interpreter variable per env and can use both envs in one job |
| Weights | rsync of our own copies (byte-identical, no Hugging Face token needed): Wan2.1-T2V-1.3B (17 GB), Self-Forcing `self_forcing_dmd.pt` (5.3 GB), HF cache entries DINOv3 ViT-B/16 (0.3 GB), CLIP ViT-L/14 (1.6 GB), Qwen2.5-VL-7B-Instruct (16 GB), the torch hub cache (1.2 GB: DINO ViT-B/8, AlexNet/VGG for LPIPS, pyiqa); ~42 GB into `$WORK/weights` and `$WORK/cache` |
| Data | FiVE-Bench `videos`, `images`, `bmasks`, `edit_prompt` (1.3 GB, 17k files), Qwen anchors (229 MB), `evaluation/csv` (56 MB, 7k files, git-ignored), `iqa_cli` (19 MB) |
| Inode budget on `$WORK` | ~246k free: bundle ~25k, overlay ~10–30k (measured by the overlay step), outputs ~500 files per clip-edit (R49 subset: ~9k for 18 clips) |
| Layout on Jean Zay | `$WORK/{StreamEdit_new, pyenv, weights, cache, Data, code}`; `~/Data` and `~/.cache/{huggingface,torch}` are symlinks into `$WORK`, so the Python code runs unchanged; outputs under `$WORK/Data/dataggen/outputs/five_bench` |
| Transfer | direct rsync from this cluster (`jz/r50_push.sh`, resumable, `--partial`), staged in `$LOCAL_STAGE` with hard links so nothing is copied twice here |
| Script port | `jz/r50_port.sh` edits only job scripts and `config.yaml`: repo path, interpreters → wrappers, Slurm headers (`-C h100`, account, QOS, `--cpus-per-task`, no `--exclude`, no `--mem`, `--time` ≤ 20 h), offline flags; must report 0 leftovers |
| Offline check | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; preflight loads every model once on a compute node and runs a one-clip draft on each host |
| Reproduction | R49 main path on the 18 clips (drafts, divergence, field, StreamEdit `max`, Wan-Edit `max_s9` + `max_s5`, eval of those 3 units); baselines and reference arms are not rerun (they reuse R48 renders that stay here) |
| Tolerances | mean over the 18 clips per metric: bg LPIPS ≤ 0.01, bg PSNR ≤ 0.5 dB, CLIP-D ≤ 0.01, FiVE-Acc ≤ 0.06 (one clip); different GPUs and torch builds are not bit-identical |
| Out of scope | the R49 full stage (stays in R49), refactoring or cleaning the method code, a separate repo |

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
bash jz/r50_push.sh
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
grep -E 'failures' logs/r49_*.out
```

### jz-compare
```bash
cd $WORK/StreamEdit_new
$WORK/pyenv/five-bench/python evaluation/r50_compare.py
```

## Pipeline

```mermaid
flowchart LR
  CFG[jz/r50_jz.env] --> MB[r50_make_bundle.sh: weights, caches, data, CSVs, pip freezes + SHA-256]
  MB --> PU[r50_push.sh: rsync to Jean Zay $WORK]
  GIT[GitHub: StreamEdit_new] --> CL[clone --depth 1 on the frontal]
  PU --> SU[r50_setup_dirs.sh + r50_verify.sh]
  CL --> SU
  MOD[IDRIS modules arch/h100 + pytorch-gpu 2.8.0 / 2.5.1] --> OV[r50_pip_overlay.sh: pinned pip --user + wrappers]
  SU --> OV
  OV --> PO[r50_port.sh: paths, Slurm headers, offline flags]
  PO --> PF[r50_preflight.sh: models offline + one-clip drafts]
  PF --> RC[r50_repro_chain.sh: R49 main arms, 18 clips]
  RC --> CO[r50_compare.py vs our CSVs]
  CO --> R49[R49 full stage on Jean Zay]
```

## Code to touch

| File | Change |
|---|---|
| `jz/r50_jz.env.example` | new: commented template of every site value with its default |
| `.gitignore` | add `jz/r50_jz.env` |
| `jz/r50_README.md` | new: ordered procedure, the check after each step, inode budget, recovery notes |
| `jz/r50_make_bundle.sh` | new: stage weights, the three HF cache entries, the torch hub cache, data subset, anchors, `evaluation/csv`, `iqa_cli` under `$LOCAL_STAGE` (hard links via `cp -al` / `rsync --link-dest`); `pip freeze --exclude-editable` and `pip list --editable` of both envs into `freeze/`; `MANIFEST.sha256` over the stage, `CODE.sha256` over the repo files R49 runs (job scripts, `evaluation/r48_*.py`, `r49_*.py`, `r31_*.py`, `r30_b_map.py`, `r33_clip_directional.py`, `fivebench/`, `backbone.py`, case files, the modified `Self-Forcing_StreamEdit` and `wan-edit` files) |
| `jz/r50_push.sh` | new: `rsync -a --partial --info=progress2` of `$LOCAL_STAGE` to `$JZ_HOST:$JZ_WORK/r50_stage/` plus the filled env file |
| `jz/r50_setup_dirs.sh` | new: move stage contents into `$JZ_WORK/{weights,cache,Data,code}`, create the `~/Data`, `~/.cache/*`, `wan_models`, `checkpoints` and `evaluation/csv` links; idempotent |
| `jz/r50_verify.sh` | new: `sha256sum -c` on both manifests, failure count |
| `jz/r50_pip_overlay.sh` | new: per env, module freeze vs ours → `pip install --user --no-deps --no-cache-dir` of every differing package (torch/torchvision/torchaudio/triton/`nvidia-*` excluded) into `$JZ_WORK/pyenv/<env>`, editable installs with `-e`, the `python` wrapper, then `pip check` + freeze re-compare through the wrapper, `overlay.txt`, summary lines with installed / mismatches / file count |
| `jz/r50_port.sh` | new: `sed` rewrites listed in Decisions (Script port) on the R49/R48/R50 job scripts and `evaluation/fivebench/config.yaml`, then the leftover grep and count |
| `evaluation/r50_preflight.py` | new: version report + offline load and one forward pass per model (DINOv3, LPIPS, CLIP ViT-L/14, Qwen2.5-VL-7B, DINO ViT-B/8 via `torch.hub` cached repo, pyiqa NIQE), OK/FAIL table |
| `slurm_scripts/five_bench/r50_preflight.sh` | new: GPU job running both wrappers' torch/CUDA checks, `r50_preflight.py` and one-clip StreamEdit + Wan-Edit drafts into `$D/r50_preflight` |
| `evaluation/r50_preflight_case.json` | new: the `0076_A_rabbit` (edit 2) case from `r49_cases.json` |
| `jz/r50_repro_chain.sh` | new: `afterok` chain of `r49_se_stage1.sh`, `r49_wan_draft.sh`, `r49_div.sh`, `r49_field.sh`, `r49_se_stage3.sh`, `r49_wan_render.sh arms` (array 0-1), `r49_eval_all.sh main`; ids to `logs/r50_repro_jobs.txt` |
| `evaluation/r50_compare.py` | new: per-clip Jean Zay vs local CSVs for the 3 main units (`r48_score` loaders), mean differences vs the tolerances, worst 3 clips, `evaluation/csv/r50_compare.csv`, PASS/FAIL |
