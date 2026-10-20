#!/bin/bash
# R49: submit the whole main chain at once, with afterok dependencies (run locally, not via sbatch):
#
#   newclip_baselines (GPU)  : the new clip under the 10 global Wan-Edit configs               (no dependency)
#   se_stage1, wan_draft (GPU): the two drafts with attention recording
#     -> div (GPU)           : DINO / LPIPS divergence, both hosts            afterok both drafts
#     -> field (CPU)         : q, rho / start steps, reference fields of the new clip   afterok div
#        -> se_stage3 (GPU), wan_render array 0-1 (GPU: max_s9, max_s5), ref_render (GPU)   afterok field
#   eval_base (GPU)          : 12 baseline + reference units                  afterok newclip_baselines, ref_render
#   eval_main (GPU)          : se_max, max_s9, max_s5                         afterok se_stage3, wan_render
#   post (CPU)               : r49_score.py + r49_grids.py                    afterok eval_base, eval_main
#
# 12 queue entries (the wan_render array counts 2). Writes "<stage> <job id>" lines to logs/r49_chain_jobs.txt
# (r49_matched59.sh reads the `post` id). If the submit cap refuses a job, the stage is reported and the jobs already
# submitted are listed so they can be cancelled.
# Usage: bash slurm_scripts/five_bench/r49_chain.sh
set -u
REPO=/home/ids/skhairi/Code/StreamEdit_bigchantier
cd "$REPO" || exit 1
mkdir -p logs
for F in evaluation/r49_cases.json evaluation/r49_new_clip.json; do
  [ -s "$F" ] || { echo "[r49_chain] missing $F"; exit 1; }
done
JOBS=logs/r49_chain_jobs.txt
[ -s "$JOBS" ] && grep -q '^post ' "$JOBS" && { echo "[r49_chain] ${JOBS} already lists a chain ($(paste -sd' ' "$JOBS")) -- move it aside before submitting a new one"; exit 1; }
: > "$JOBS"

submit() {  # $1 = stage name, rest = sbatch args; prints the id, records it, aborts the chain if refused
  local stage=$1; shift
  local out id err
  err=$(mktemp)
  out=$(sbatch --parsable "$@" 2>"$err")                # stdout = "<id>[;cluster]" only; warnings stay on stderr
  id=${out%%;*}
  if ! [[ "$id" =~ ^[0-9]+$ ]]; then
    echo "[r49_chain] REFUSED at stage '${stage}': ${out} $(cat "$err")" >&2
    rm -f "$err"
    echo "[r49_chain] already submitted (cancel with scancel if needed): $(paste -sd' ' "$JOBS")" >&2
    exit 1
  fi
  [ -s "$err" ] && echo "[r49_chain] sbatch note (${stage}): $(cat "$err")" >&2
  rm -f "$err"
  echo "${stage} ${id}" >> "$JOBS"
  echo "[r49_chain] ${stage}: job ${id}" >&2
  echo "$id"
}

NB=$(submit newclip_baselines slurm_scripts/five_bench/r49_newclip_baselines.sh) || exit 1
S1=$(submit se_stage1 slurm_scripts/five_bench/r49_se_stage1.sh) || exit 1
WD=$(submit wan_draft slurm_scripts/five_bench/r49_wan_draft.sh) || exit 1
DV=$(submit div --dependency=afterok:${S1}:${WD} slurm_scripts/five_bench/r49_div.sh) || exit 1
FD=$(submit field --dependency=afterok:${DV} slurm_scripts/five_bench/r49_field.sh) || exit 1
SE=$(submit se_stage3 --dependency=afterok:${FD} slurm_scripts/five_bench/r49_se_stage3.sh) || exit 1
WR=$(submit wan_render --dependency=afterok:${FD} --array=0-1 slurm_scripts/five_bench/r49_wan_render.sh arms) || exit 1
RR=$(submit ref_render --dependency=afterok:${FD} slurm_scripts/five_bench/r49_ref_render.sh) || exit 1
EB=$(submit eval_base --dependency=afterok:${NB}:${RR} slurm_scripts/five_bench/r49_eval_all.sh base) || exit 1
EM=$(submit eval_main --dependency=afterok:${SE}:${WR} slurm_scripts/five_bench/r49_eval_all.sh main) || exit 1
PO=$(submit post --dependency=afterok:${EB}:${EM} slurm_scripts/five_bench/r49_post.sh) || exit 1

echo "[r49_chain] submitted: newclip_baselines ${NB} | se_stage1 ${S1} + wan_draft ${WD} -> div ${DV} -> field ${FD} -> se_stage3 ${SE} + wan_render ${WR} (array 0-1) + ref_render ${RR} -> eval_base ${EB} (after ${NB}, ${RR}) + eval_main ${EM} -> post ${PO}"
echo "[r49_chain] ids in ${JOBS}"
