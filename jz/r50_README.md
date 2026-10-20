# R50 — running R49 on Jean Zay (H100)

Ordered procedure for moving R49 to Jean Zay. Every step ends with a check line; do not start the next
step until that line reads as below. All site-specific values (host, `$WORK`, accounts, QOS, module
names) live in `jz/r50_jz.env`; nothing else needs editing by hand.

What goes where:

| What | How it reaches Jean Zay |
|---|---|
| Code | `git clone --depth 1` of `skhairi2k/StreamEdit_new` (byte-identity checked with `CODE.sha256`) |
| Python | IDRIS modules `arch/h100` + `pytorch-gpu/py3/2.8.0` (streamgve) / `2.5.1` (five-bench), plus a pinned `pip install --user --no-deps` overlay of every package that differs from our envs, in `$WORK/pyenv/<env>` |
| Weights, HF/torch caches, FiVE data, anchors, baseline CSVs, `iqa_cli` | rsync of our own copies (`$LOCAL_STAGE` → `$WORK/r50_stage`), checked with `MANIFEST.sha256` |

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

## Procedure

| # | Where | Command | Check |
|---|---|---|---|
| 1 | here | `cp jz/r50_jz.env.example jz/r50_jz.env`, fill every `<...>` | `git check-ignore jz/r50_jz.env` prints the file |
| 2 | here | `bash jz/r50_make_bundle.sh` | prints the stage size and file count; `MANIFEST.sha256`, `CODE.sha256`, `freeze/` written |
| 3 | here | `bash jz/r50_push.sh` | rsync finishes; remote size printed |
| 4 | JZ frontal | `git clone --depth 1 git@github.com:skhairi2k/StreamEdit_new.git $WORK/StreamEdit_new`<br>`cd $WORK/StreamEdit_new && cp $WORK/r50_stage/r50_jz.env jz/`<br>`bash jz/r50_setup_dirs.sh && bash jz/r50_verify.sh` | `[r50_verify] failures: 0` |
| 5 | JZ frontal | `bash jz/r50_pip_overlay.sh` then `idr_quota_user` | `[r50_overlay] failures: 0`, `mismatches 0` for both envs |
| 6 | JZ frontal | `bash jz/r50_port.sh` then `git diff --stat` | `[r50_port] leftovers: 0`; only job scripts and `evaluation/fivebench/config.yaml` changed |
| 7 | JZ | `sbatch slurm_scripts/five_bench/r50_preflight.sh` | log ends with `[r50_preflight] failures: 0` |
| 8 | JZ | `bash jz/r50_repro_chain.sh`, wait for `squeue -u $USER` to empty | every log listed in `logs/r50_repro_jobs.txt` ends with `failures: 0` / `unit failures: 0` |
| 9 | JZ | `$WORK/pyenv/five-bench/python evaluation/r50_compare.py` | `PASS`; `evaluation/csv/r50_compare.csv` written |

Only after step 9 passes: run the R49 full stage on Jean Zay (it stays in the R49 plan).

Steps 1–3 run here, 4–9 on Jean Zay. Record each with `/run-step R50 <step-id>`
(`fill-config`, `make-bundle`, `push`, `jz-clone-setup`, `jz-overlay`, `jz-port`, `jz-preflight`,
`jz-repro`, `jz-compare`), pasting the check line for the Jean Zay steps.

## Layout on Jean Zay

```
$WORK/StreamEdit_new        repo (clone)
$WORK/pyenv/<env>/          pip --user overlay + the `python` wrapper (module load + PYTHONUSERBASE)
$WORK/weights/              Wan2.1-T2V-1.3B, self_forcing_dmd.pt  (linked from the repo)
$WORK/cache/{huggingface,torch}   ← ~/.cache/huggingface, ~/.cache/torch are symlinks here
$WORK/Data/                 ← ~/Data is a symlink here (FiVE-Bench, anchors, outputs/five_bench)
$WORK/code/iqa_cli          pyiqa editable install (NIQE)
$WORK/r50_stage/            rsync landing area (emptied by r50_setup_dirs.sh)
```

Compute nodes have no internet: every job runs with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`; all
models load from the shipped caches.

## Inode budget

`$WORK` allows 500k files (253k already used at setup time, 2026-10-20); this is why the conda envs
(286k files unpacked) are replaced by IDRIS modules, which cost none of your files.

| Item | Files |
|---|---|
| bundle (weights, caches, data, CSVs, `iqa_cli`) | ~25k |
| pip overlay, both envs | ~10–30k (printed by step 5) |
| outputs | ~500 per clip-edit (R49 subset: ~9k for 18 clips) |

Check `idr_quota_user` after steps 5 and 8, and before launching the full stage.

## Recovery

| Step | Failure | Fix |
|---|---|---|
| 2 | missing source path | fix the path in `jz/r50_make_bundle.sh` or restore the file; rerun (hard links, nothing copied twice) |
| 3 | transfer interrupted | rerun `bash jz/r50_push.sh` (resumable, `--partial`) |
| 4 | `MANIFEST.sha256` failures | rerun step 3 (rsync re-sends differing files), then `bash jz/r50_verify.sh` |
| 4 | `CODE.sha256` failures | the clone differs from this cluster: commit + push here, `git pull` there, rerun verify |
| 5 | a pinned package fails to install or `pip check` complains | read `$WORK/pyenv/<env>/overlay.txt`, fix the pin, rerun step 5 (reinstalls only what still differs) |
| 6 | leftovers > 0 | the printed lines show what the port missed; `git checkout -- slurm_scripts evaluation/fivebench/config.yaml`, fix `jz/r50_port.sh`, rerun |
| 7 | a model fails offline | its cache entry is missing: add it to the bundle (step 2), push, `r50_setup_dirs.sh`, rerun the preflight |
| 8 | one job failed | the R49 scripts are resumable: resubmit that script alone, then its dependents |
| 9 | FAIL | look at the 3 worst clips in `evaluation/csv/r50_compare.csv` before running anything else |
