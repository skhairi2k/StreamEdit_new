#!/bin/bash
# R50 step jz-port (ON JEAN ZAY, in the clone): rewrite the Wan-side job scripts and the FiVE config for Jean Zay H100.
# Idempotent (rerun after every `git pull`, see below). The lab cluster's scripts are never touched: the script refuses to run
# unless $JZ_WORK exists and the repo is outside the lab home.
#
# Rules, applied to the scripts listed in SCRIPTS below (missing ones are skipped):
#   REPO=...                                  -> REPO=<this clone, absolute>                (not the submit directory)
#   ~/anaconda3/envs/five-bench/bin/python    -> $JZ_WORK/pyenv/five-bench/python           (wrapper made by r50_pip_overlay.sh)
#   #SBATCH --output/--error=logs/...         -> <this clone>/logs/...                       (sbatch may be run from any folder)
#   #SBATCH partition / exclude / mem / cpus-per-task / gres / hint / qos / account lines -> one H100 block:
#         -C h100, --gres=gpu:1, --cpus-per-task=$JZ_CPUS_PER_GPU, --hint=nomultithread, --qos=$JZ_QOS_GPU, [-A $JZ_ACCOUNT_GPU]
#     (IDRIS gpu_p6 example: memory follows the CPU count, so no --mem; 'CPU' scripts such as r49_field.sh also run on H100:
#      the project has GPU hours only)
#   #SBATCH --time=HH:MM:SS                   -> capped at 20:00:00 (qos_gpu_h100-t3)
#   export HF_HUB_OFFLINE=1                   -> + TRANSFORMERS_OFFLINE=1 (compute nodes have no internet)
#   evaluation/fivebench/config.yaml          -> IQA_PyTorch_model_path / cotracker_model_path under $JZ_WORK/code
# Then every ported file is syntax-checked (bash -n) and grepped for what is still lab-specific (non-comment lines and #SBATCH
# lines containing /home/ids/skhairi, anaconda3, L40S, node52, --partition, streamgve, --mem): "leftovers" must be 0.
#
# Usage: bash jz/r50_port.sh            port
#        bash jz/r50_port.sh --check    change nothing; say which files would change and list the leftovers
# After a `git pull`:  git checkout -- slurm_scripts evaluation/fivebench/config.yaml && git pull && bash jz/r50_verify.sh --code-only && bash jz/r50_port.sh
# R50_ENV=<file> overrides jz/r50_jz.env (tests). Ends with "[r50_port] leftovers: N" and "[r50_port] failures: N".
set -u

CHECK=0; [ "${1:-}" = "--check" ] && CHECK=1
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CFG=${R50_ENV:-$REPO/jz/r50_jz.env}
[ -f "$CFG" ] || { echo "[r50_port] FATAL: $CFG missing"; exit 1; }
# shellcheck source=/dev/null
source "$CFG"
if set | grep -E '^(JZ_[A-Za-z_]+|LOCAL_STAGE)=' | grep -q '<'; then echo "[r50_port] FATAL: unfilled <...> value in $CFG"; exit 1; fi
WORK=${JZ_WORK:?JZ_WORK not set in $CFG}
QOS=${JZ_QOS_GPU:-qos_gpu_h100-t3}
CPUS=${JZ_CPUS_PER_GPU:-24}
ACCOUNT=${JZ_ACCOUNT_GPU:-}
MAX_HOURS=20
WRAP=$WORK/pyenv/five-bench/python

[ -d "$WORK" ] || { echo "[r50_port] FATAL: $WORK does not exist on $(hostname) -- this script is for Jean Zay only"; exit 1; }
case "$REPO" in
  /home/ids/*|/projects/*|/store/*|/scratch/*) echo "[r50_port] FATAL: $REPO looks like the LAB cluster -- refusing to rewrite its scripts"; exit 1 ;;
esac

cd "$REPO" || exit 1
SCRIPTS=(slurm_scripts/five_bench/{r48_wan_draft,r48_div,r48_wan_render,r49_wan_draft,r49_div,r49_field,r49_wan_render,r49_eval_all,r50_wan_official,r50_preflight,r49_wan_full_chain,r51_field,r51_wan_render,r51_eval,r51_wan_chain,r58_wan_draft,r58_div,r58_field,r58_wan_render,r58_wan_chain,r62_field,r62_wan_render,r62_wan_chain}.sh
         jz/r50_repro_chain.sh)
CONFIG=evaluation/fivebench/config.yaml
FAILED=0; LEFT=0; CHANGED=0
TMP=$(mktemp -d "${TMPDIR:-/tmp}/r50_port.XXXXXX") || exit 1
trap 'rm -rf "$TMP"' EXIT
echo "[r50_port] $([ $CHECK -eq 1 ] && echo CHECK || echo PORT) in $REPO on $(hostname); wrapper=$WRAP qos=$QOS cpus=$CPUS account='${ACCOUNT}' $(date '+%F %T')"
[ -x "$WRAP" ] || echo "[r50_port] WARNING: $WRAP not found yet -- run jz/r50_pip_overlay.sh before submitting anything"

port_text() {  # $1 = in, $2 = out : the line rules, then the #SBATCH header block
  sed -E \
    -e "s|^REPO=.*|REPO=$REPO|" \
    -e "s#(/home/ids/skhairi|~|\\\$HOME)/anaconda3/envs/five-bench/bin/python#$WRAP#g" \
    -e "s|^(#SBATCH[[:space:]]+--(output\|error)=)logs/|\1$REPO/logs/|" \
    -e '/TRANSFORMERS_OFFLINE/! s/HF_HUB_OFFLINE=1/HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1/' \
    "$1" > "$TMP/step1"
  awk -v cap="$MAX_HOURS" -v qos="$QOS" -v cpus="$CPUS" -v acct="$ACCOUNT" '
    function block() {
      print "# [r50_port] Jean Zay H100 header (replaces the lab partition / memory / exclude lines)"
      print "#SBATCH -C h100"; print "#SBATCH --gres=gpu:1"; print "#SBATCH --cpus-per-task=" cpus
      print "#SBATCH --hint=nomultithread"; print "#SBATCH --qos=" qos
      if (acct != "") print "#SBATCH -A " acct
    }
    /^# \[r50_port\]/ { next }
    /^#SBATCH[ \t]/ {
      if ($0 ~ /^#SBATCH[ \t]+(-p|--partition|--exclude|--mem|--mem-per-cpu|--cpus-per-task|--gres|--hint|--qos|-A|--account|-C|--constraint)([= \t]|$)/) next
      if ($0 ~ /^#SBATCH[ \t]+--time=[0-9]+:[0-9]+:[0-9]+$/) { split($0, a, "="); split(a[2], t, ":"); if (t[1] + 0 > cap) $0 = "#SBATCH --time=" cap ":00:00" }
      if (!seen) { block(); seen = 1 }
    }
    { print }' "$TMP/step1" > "$2"
}
port_config() {  # $1 = in, $2 = out
  sed -E -e "s|^(IQA_PyTorch_model_path:).*|\1 $WORK/code/iqa_cli|" \
         -e "s|^(cotracker_model_path:).*|\1 $WORK/code/co-tracker/checkpoints/scaled_offline.pth|" "$1" > "$2"
}
leftovers() {  # $1 = file -> prints the lab-specific lines that are not plain comments
  awk '!/^[[:space:]]*#/ || /^[[:space:]]*#SBATCH/ { print FILENAME ":" NR ": " $0 }' "$1" \
    | grep -E '/home/ids/skhairi|anaconda3|L40S|node52|--partition|streamgve|#SBATCH[[:space:]]+--mem' || true
}
process() {  # $1 = file, $2 = port_text | port_config
  local f=$1 fn=$2
  if [ ! -f "$f" ]; then echo "  not present (skipped): $f"; return; fi
  "$fn" "$f" "$TMP/out" || { echo "  FAILED to rewrite $f"; FAILED=$((FAILED + 1)); return; }
  if cmp -s "$f" "$TMP/out"; then echo "  unchanged: $f"
  else
    CHANGED=$((CHANGED + 1))
    if [ "$CHECK" -eq 1 ]; then echo "  would change: $f"
    else cat "$TMP/out" > "$f" && echo "  ported: $f" || { echo "  FAILED to write $f"; FAILED=$((FAILED + 1)); return; }; fi
  fi
}

for F in "${SCRIPTS[@]}"; do process "$F" port_text; done
process "$CONFIG" port_config

echo "[r50_port] checks"
for F in "${SCRIPTS[@]}" "$CONFIG"; do
  [ -f "$F" ] || continue
  L=$(leftovers "$F"); if [ -n "$L" ]; then LEFT=$((LEFT + $(echo "$L" | wc -l))); echo "$L" | sed 's/^/  leftover: /'; fi
  case "$F" in *.sh) bash -n "$F" 2>/dev/null || { echo "  SYNTAX ERROR: $F"; FAILED=$((FAILED + 1)); } ;; esac
done
echo "[r50_port] files $([ $CHECK -eq 1 ] && echo 'that would change' || echo 'rewritten'): ${CHANGED}"
echo "[r50_port] leftovers: ${LEFT}"
echo "[r50_port] failures: $(( FAILED + (CHECK == 0 ? LEFT : 0) ))"
[ "$CHECK" -eq 0 ] && echo "[r50_port] review: git diff --stat   (only job scripts and evaluation/fivebench/config.yaml should change)"
exit $(( (FAILED + (CHECK == 0 ? LEFT : 0)) > 0 ))
