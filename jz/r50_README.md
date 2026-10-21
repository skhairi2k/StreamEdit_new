# R50 — running the Wan-Edit side of R49 on Jean Zay (H100)

**Host split:** every Wan-Edit job (draft, divergence, field, renders, baselines, matched-NFE curves, eval)
runs on Jean Zay H100 in the `five-bench` environment (Wan-Edit is FiVE-Bench's own editor); every
StreamEdit job stays on the lab cluster (L40S) in `streamgve`. Each host's arms, baselines and curves
therefore come from one platform and one environment. Never put Jean Zay and L40S numbers of the same host
in one table or figure.

Ordered procedure. Every step ends with a check line; do not start the next step until that line reads as
below. All site-specific values (host, `$WORK`, optional account, QOS, module name) live in
`jz/r50_jz.env`; nothing else needs editing by hand.

What goes where:

| What | How it reaches Jean Zay |
|---|---|
| Code | `git clone --depth 1` of `skhairi2k/StreamEdit_new` (byte-identity checked with `CODE.sha256`) |
| Python | IDRIS modules `arch/h100` + `pytorch-gpu/py3/2.5.1` (torch 2.5.1, as our `five-bench` env), plus a pinned `pip install --user --no-deps` overlay of every package that differs from our env (incl. flash-attn, which Wan-Edit requires), in `$WORK/pyenv/five-bench` |
| Wan2.1 weights, HF/torch caches, FiVE data, our CSVs (reference), `iqa_cli` | `$LOCAL_STAGE` is a folder of symlinks to our own files; `rsync --copy-unsafe-links` sends them as real files to `$WORK/r50_stage`, checked with `MANIFEST.sha256` |

## Before you start (this cluster)

1. Commit and push every R49/R50 file the chain runs; the clone must contain them or `r50_verify.sh`
   reports `CODE.sha256` failures. `evaluation/csv/` and `evaluation/figures/` stay git-ignored (the
   CSVs travel in the bundle):
   ```bash
   git status --short
   git add evaluation/r49_*.py evaluation/r49_*.json evaluation/r50_*.py evaluation/r50_preflight_case.json
   git add jz/ slurm_scripts/five_bench/r49_*.sh slurm_scripts/five_bench/r50_*.sh .gitignore
   git commit -m "R50 Jean Zay kit" && git push
   ```
2. Check the Jean Zay quota (`idr_quota_user` on the frontal): see **Inode budget** below.
3. Anything that talks to Jean Zay from here (`jz/r50_push.sh`, R49's `wan-full-fetch` rsync) must run on the
   **lab cluster's frontal**: Jean Zay only accepts SSH from registered addresses, and compute nodes such as
   `nodemm06` cannot reach it (`ssh zay` times out). The home directory is shared, so the repo and the stage are
   the same there. Run long transfers inside `tmux` (`tmux new -s r50push`; detach Ctrl+b d; `tmux attach -t r50push`).

## Procedure

| # | Where | Command | Check |
|---|---|---|---|
| 1 | here | `cp jz/r50_jz.env.example jz/r50_jz.env`, fill every `<...>` | `git check-ignore jz/r50_jz.env` prints the file |
| 2 | here | `bash jz/r50_make_bundle.sh` | `[r50_bundle] failures: 0`; size and file count printed; `MANIFEST.sha256`, `CODE.sha256`, `freeze/` written |
| 3 | lab frontal | `bash jz/r50_push.sh 2>&1 \| tee logs/r50_push.log` (inside tmux) | `[r50_push] failures: 0` |
| 4 | JZ frontal | `git clone --depth 1 git@github.com:skhairi2k/StreamEdit_new.git $WORK/StreamEdit_new`<br>`cd $WORK/StreamEdit_new && cp $WORK/r50_stage/r50_jz.env jz/`<br>`bash jz/r50_setup_dirs.sh && bash jz/r50_verify.sh` | `[r50_setup] failures: 0`, then `[r50_verify] failures: 0` (hashing 36 GB takes a few minutes) |
| 5 | JZ frontal (internet) | `bash jz/r50_pip_overlay.sh --dry-run` (plan only), `bash jz/r50_pip_overlay.sh`, then `idr_quota_user` | `--dry-run` first: it prints python / torch / C++ ABI and the flash-attn wheel it needs (`reachable`, `NO SUCH WHEEL` or `NOT reachable` = stop, see Recovery); then `[r50_overlay] failures: 0`, `mismatches 0` |
| 6 | JZ frontal | `bash jz/r50_port.sh --check` (preview), `bash jz/r50_port.sh`, then `git diff --stat` | `[r50_port] leftovers: 0`, `failures: 0`; only job scripts and `evaluation/fivebench/config.yaml` changed; run `sbatch` from the repo root |
| 7 | JZ | `sbatch slurm_scripts/five_bench/r50_preflight.sh` | log ends with `[r50_preflight] failures: 0` |
| 8 | JZ | `bash jz/r50_repro_chain.sh`, wait for `squeue -u $USER` to empty | every log listed in `logs/r50_repro_jobs.txt` ends with `failures: 0` / `unit failures: 0` |
| 9 | JZ | `$WORK/pyenv/five-bench/python evaluation/r50_compare.py` | `PASS`; `evaluation/csv/r50_compare.csv` written |

Only after step 9 passes: run the R49 Wan-Edit full stage on Jean Zay (it stays in the R49 plan; rerun
`bash jz/r50_port.sh` after pulling any new job script).

Steps 1–3 run here, 4–9 on Jean Zay. Record each with `/run-step R50 <step-id>`
(`fill-config`, `make-bundle`, `push`, `jz-clone-setup`, `jz-overlay`, `jz-port`, `jz-preflight`,
`jz-repro`, `jz-compare`), pasting the check line for the Jean Zay steps.

## What step 9 checks

H100 does not reproduce L40S bit for bit (different attention/matmul kernels, bf16 accumulation order,
IDRIS torch build; a token near a rounding boundary can get another start step, and the VLM's yes/no can
flip on borderline clips). So `r50_compare.py` checks what the paper claims, on the 18 clips: the paired
delta `max_s5 − native_s0.92` (adaptive arm minus official Wan-Edit) for FiVE-Acc, CLIP-D, bg LPIPS and
bg PSNR. PASS = every Jean Zay delta lies inside the L40S 95% CI of the same delta. It also reports each
arm's platform shift (Jean Zay − L40S) with its CI.

## Layout on Jean Zay

```
$WORK/StreamEdit_new        repo (clone)
$WORK/pyenv/five-bench/     pip --user overlay + the `python` wrapper (module load + PYTHONUSERBASE)
$WORK/weights/wan_models/   Wan2.1-T2V-1.3B  (linked from the repo's Self-Forcing_StreamEdit/wan_models)
$WORK/cache/{huggingface,torch}   ← ~/.cache/huggingface, ~/.cache/torch are symlinks here
$WORK/Data/                 ← ~/Data is a symlink here (FiVE-Bench, outputs/five_bench)
$WORK/code/iqa_cli          pyiqa editable install (NIQE)
$WORK/r50_ref/evaluation_csv   our CSVs, read-only: the reference r50_compare.py reads (the repo's
                            evaluation/csv is a separate copy that jobs overwrite)
$WORK/r50_stage/            rsync landing area (emptied by r50_setup_dirs.sh)
```

Compute nodes have no internet: every job runs with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`; all
models load from the shipped caches.

## Inode budget

`$WORK` allows 500k files (253k already used at setup time, 2026-10-20); this is why the conda env
(121k files unpacked for `five-bench` alone) is replaced by an IDRIS module, which costs none of your files.

| Item | Files |
|---|---|
| bundle (weights, caches, FiVE data 16k files, 23 reference CSVs, `iqa_cli`) | ~16.4k |
| pip overlay | ~5–15k (printed by step 5) |
| outputs | ~100 PNG frames per clip and render arm, plus drafts, maps and npz |

Check `idr_quota_user` after steps 5 and 8, and before launching the full stage.

## Recovery

| Step | Failure | Fix |
|---|---|---|
| 2 | missing source / broken link | fix the path in `jz/r50_make_bundle.sh` or restore the file; rerun (links are replaced, checksums recomputed) |
| 3 | transfer interrupted | rerun `bash jz/r50_push.sh` (resumable, `--partial`) |
| 4 | `MANIFEST.sha256` failures | rerun step 3 (rsync re-sends differing files), then `bash jz/r50_verify.sh` |
| 4 | `CODE.sha256` failures | the clone differs from this cluster: commit + push here, rerun `bash jz/r50_make_bundle.sh` and `bash jz/r50_push.sh` (seconds), then there: `git checkout -- slurm_scripts evaluation/fivebench/config.yaml` (undo the port edits, if already ported), `git pull`, `bash jz/r50_verify.sh --code-only`, `bash jz/r50_port.sh` |
| 4 | `setup` reports CONFLICT | a target already exists with other content (e.g. a real, non-empty `~/Data`): move it away or merge by hand, rerun; setup never overwrites |
| 5 | a pinned package fails to install or `pip check` complains | read `$WORK/pyenv/five-bench/overlay.txt`, fix the pin, rerun step 5 (reinstalls only what still differs) |
| 5 | flash-attn wheel cannot be downloaded (frontal blocks GitHub) or no wheel matches the module's torch / python | download `flash_attn-<ver>+cu12torch<X.Y>cxx11abi<TRUE\|FALSE>-cp<NNN>-cp<NNN>-linux_x86_64.whl` (the name is printed) elsewhere, copy it to `$WORK/pyenv/wheels/`, rerun; or pick another `JZ_MODULE` |
| 5 | `python ... < 3.10` refusal | the module is too old for transformers 5.x: `module avail pytorch-gpu`, change `JZ_MODULE` in `jz/r50_jz.env` |
| 6 | leftovers > 0 | the printed lines show what the port missed; `git checkout -- slurm_scripts evaluation/fivebench/config.yaml`, fix `jz/r50_port.sh`, rerun |
| 7 | a model fails offline | its cache entry is missing: add it to the bundle (step 2), push, `r50_setup_dirs.sh`, rerun the preflight |
| 8 | one job failed | the R49 scripts are resumable: resubmit that script alone, then its dependents |
| 9 | FAIL | look at the per-arm shifts and the 3 clips that moved most in `evaluation/csv/r50_compare.csv` before running anything else |
