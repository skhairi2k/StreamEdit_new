#!/bin/bash
# R32 smoke -- blocking gates for the per-head segment-bias path.
# Plan: .claude/plans/r32-headwise-segment-rebalance_c7f3a218.plan.md
#
# No model number from r32_infer.sh is interpreted until every gate here is OK.
#
# GATE1 (HEADAXIS) -- bias head 3 alone; head 3 must move and the other 11 must be
#   BITWISE unchanged. The bias is consumed AFTER attention()'s transpose(1,2), so it
#   lives in [B, Nh, Lq, Lk] where the head axis is dim 1, not dim 2 as in q/k/v. A
#   bias built in q/k/v layout, or shaped [*,1,*,*], would broadcast ONE head's bias
#   across all 12 -- no error, plausible output, and the entire experiment becomes a
#   global reweighting mislabelled as per-head routing. This is the single most
#   likely silent failure, which is why it is gate 1.
#
# GATE2 (SEGALIGN) -- -60 off-segment must reproduce true attention over that segment
#   alone. Catches the bias landing on the wrong token range. The source-current
#   segment length is a background-mask popcount: it varies per clip and per chunk,
#   is absent entirely in the first half of the denoising steps, and is not a
#   multiple of 1560 -- so an off-by-L_src misalignment is easy and invisible.
#
# GATE3 (FIRES) -- zero vs label_b10 must DIFFER above a floor. Catches a table that
#   is loaded, parsed, and then never reaches the softmax (R28's gate-2 failure mode;
#   gates 1 and 2 would both still pass on synthetic input if the BRIDGE dropped it).
#
# GATE4 (BACKEND) -- per-call wall time < 20 ms at production shapes. A non-None
#   attn_mask can silently route to the math backend, which materializes the full
#   [1,12,4680,28080] logit matrix (~3.15 GB per copy) and runs ~25x slower.
#   attention() pins EFFICIENT_ATTENTION so a future shape change raises instead,
#   but the timing check is what catches a pin that stopped holding.
#
# GATE5 (SEGLENS) -- a real 1-clip run with R32_TRACE=1. Segment lengths must sum to
#   the concatenated key length at every capture, and the source-current segment must
#   be ABSENT for current_timestep_index <= total_timestep//2 and PRESENT after.
#   Confirms the layout assumption against the real pipeline, not synthetic tensors.
#
# GATE6 (FLOOR) -- render one clip under `flash` and under `zero` in the same process.
#   These are NOT bit-identical: a biased arm cannot run on FlashAttention, so every
#   arm pays a kernel change, and zeros-bias SDPA differs from attn_mask=None SDPA by
#   ~2.4e-4 per call. The gate RECORDS that delta. It is the noise floor: any arm
#   difference smaller than it is unreadable and must be reported as such. Deliberately
#   NOT compared against a stored baseline render -- R1/R7 were re-rendered after the
#   2026-07-22 seeding fix, so a stored reference risks mixing generations.
#
#SBATCH --job-name=r32_smoke
#SBATCH --partition=L40S,A100
#SBATCH --exclude=node52
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r32_smoke_%j.out
#SBATCH --error=logs/r32_smoke_%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Absolute interpreter path, so no conda activation stacking is possible.
PY=~/anaconda3/envs/streamgve/bin/python

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Triton is broken in this env (FileNotFoundError: /usr/bin/gcc-13; only 14.3.1
# exists). Nothing here compiles, but set CC so any incidental inductor path does
# not die on it. This is also why flex_attention was rejected for the bias.
export CC=/usr/bin/gcc

BIAS=evaluation/r32_seg_bias.pt
WORK=/projects/dataggen/outputs/five_bench/r32_smoke
CASES=evaluation/cases.json
# 0001_bus: 18 latent frames == single window, the configuration Stage 1 runs in.
CLIP=0001_bus

echo "[r32_smoke] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
rm -rf "$WORK" && mkdir -p "$WORK"

# ---------------------------------------------------------------- tables
$PY evaluation/r32_build_seg_bias.py --out "$BIAS" \
  || { echo "GATE0-FAIL could not build $BIAS (sign check?)"; exit 1; }
echo "GATE0-OK tables built (R19 sign convention asserted inside the builder)"

# ---------------------------------------------------------- GATE1..GATE4
echo "=== GATE1-4: bias path on synthetic tensors at production shapes ==="
$PY - "$BIAS" <<'GATE14PY'
import sys, time
sys.path.insert(0, "Self-Forcing_StreamEdit")
import torch
from wan.modules.attention import attention
from wan.modules.causal_model import build_segment_logit_bias

bias_path = sys.argv[1]
dev, dt = "cuda", torch.bfloat16
# Production shapes: Lq = 3 frames x 1560, Lk = 18 latent frames x 1560.
B, Lq, Lk, Nh, D = 1, 4680, 28080, 12, 128
L_PREV = Lk - Lq                      # the segment the bias applies to
torch.manual_seed(0)
q = torch.randn(B, Lq, Nh, D, device=dev, dtype=dt)
k = torch.randn(B, Lk, Nh, D, device=dev, dtype=dt)
v = torch.randn(B, Lk, Nh, D, device=dev, dtype=dt)

zeros = torch.zeros(Nh, device=dev)
b_zero = build_segment_logit_bias(zeros, L_PREV, Lk, device=dev, dtype=dt)
o_zero = attention(q, k, v, attn_bias=b_zero)

# ---- GATE1: head isolation -------------------------------------------------
row = torch.zeros(Nh, device=dev); row[3] = 50.0
b_h3 = build_segment_logit_bias(row, L_PREV, Lk, device=dev, dtype=dt)
o_h3 = attention(q, k, v, attn_bias=b_h3)
moved = [bool((o_h3[:, :, h] != o_zero[:, :, h]).any()) for h in range(Nh)]
bad = [h for h, m in enumerate(moved) if m != (h == 3)]
if bad:
    print(f"GATE1-FAIL headaxis: heads {bad} behaved wrongly; moved={moved}. "
          "One head's bias is reaching others -- the per-head claim is void.")
    raise SystemExit(1)
print("GATE1-OK headaxis: head 3 moved, the other 11 are bitwise unchanged")

# ---- GATE2: segment alignment ---------------------------------------------
row = torch.full((Nh,), 0.0, device=dev)
b_all = build_segment_logit_bias(row, L_PREV, Lk, device=dev, dtype=dt)
b_all[:, :, :, L_PREV:] = -60.0        # suppress everything AFTER the prev segment
o_seg = attention(q, k, v, attn_bias=b_all)
o_ref = attention(q, k[:, :L_PREV], v[:, :L_PREV])
err = (o_seg.float() - o_ref.float()).abs().max().item()
if err > 5e-2:
    print(f"GATE2-FAIL segalign: max err {err:.3e} vs true restricted attention. "
          f"The bias is not landing on [0, {L_PREV}).")
    raise SystemExit(1)
print(f"GATE2-OK segalign: restricting to the prev segment matches true "
      f"restricted attention (max err {err:.2e})")

# ---- GATE3: the real table reaches the softmax ----------------------------
payload = torch.load(bias_path, weights_only=False)
tbl = payload["label"].float().to(dev)
# Pick the layer with the strongest row rather than layer 0: a layer whose heads
# are all MIXED/DENSE has an all-zero row, and this gate would then fail for a
# reason that has nothing to do with the plumbing.
layer = int(tbl.abs().sum(dim=1).argmax())
o_lab = attention(q, k, v,
                  attn_bias=build_segment_logit_bias(tbl[layer], L_PREV, Lk,
                                                     device=dev, dtype=dt))
d = (o_lab.float() - o_zero.float()).abs().max().item()
if d < 1e-4:
    print(f"GATE3-FAIL fires: label_b10 differs from zero by only {d:.3e} -- "
          "the table is being parsed and then discarded.")
    raise SystemExit(1)
print(f"GATE3-OK fires: the real table changes the output (max |delta| {d:.3e})")

# ---- GATE4: backend / timing ----------------------------------------------
def bench(fn, n=12):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1e3

t_flash = bench(lambda: attention(q, k, v))
t_bias = bench(lambda: attention(q, k, v, attn_bias=b_zero))
peak = torch.cuda.max_memory_allocated() / 2**30
print(f"    flash {t_flash:.2f} ms | bias {t_bias:.2f} ms | "
      f"{t_bias/t_flash:.2f}x | peak {peak:.2f} GiB")
if t_bias >= 20.0:
    print(f"GATE4-FAIL backend: {t_bias:.1f} ms/call >= 20 ms -- almost certainly "
          "the math backend; a full run would be ~25x slower and may OOM.")
    raise SystemExit(1)
print(f"GATE4-OK backend: {t_bias:.2f} ms/call on the memory-efficient backend")
GATE14PY
[ $? -eq 0 ] || { echo "GATE1-4 block failed"; exit 1; }

# ---------------------------------------------------------------- GATE5
echo "=== GATE5: segment lengths in a real run ==="
R32_TRACE=1 $PY evaluation/r32_seg_bias_arms.py \
    --cases_json "$CASES" --cases "$CLIP" --arms label_b10 \
    --bias "$BIAS" --out_root "$WORK/g5" --tag g5 \
    > "$WORK/g5_trace.log" 2>&1 \
  || { echo "GATE5-FAIL the traced run itself failed; see $WORK/g5_trace.log"; exit 1; }

$PY - "$WORK/g5_trace.log" <<'GATE5PY'
import re, sys
lines = [l for l in open(sys.argv[1]) if l.startswith("[r32] layer=")]
if not lines:
    print("GATE5-FAIL seglens: no R32_TRACE lines -- the biased branch never ran, "
          "so the arm was NOT what it claimed to be.")
    raise SystemExit(1)
pat = re.compile(r"layer=(\d+) step=(\S+) seg_lens=\[([0-9, ]+)\] Lk=(\d+)")
n_seg2 = n_seg3 = 0
bad_sum, early_src, late_nosrc = [], [], []
steps = set()
for l in lines:
    m = pat.search(l)
    if not m:
        continue
    layer, step, segs, lk = int(m[1]), m[2], [int(x) for x in m[3].split(",")], int(m[4])
    if sum(segs) != lk:
        bad_sum.append((layer, step, segs, lk))
    steps.add(step)
    # 2 segments => the source-current segment is absent; 3 => present.
    if len(segs) == 2:
        n_seg2 += 1
    elif len(segs) == 3:
        n_seg3 += 1
    if step != "None":
        s = int(step)
        if s <= 7 and len(segs) == 3:
            early_src.append((layer, s))
        if s > 7 and len(segs) == 2:
            late_nosrc.append((layer, s))
if bad_sum:
    print(f"GATE5-FAIL seglens: {len(bad_sum)} captures where the segment lengths "
          f"do not sum to Lk, e.g. {bad_sum[:3]}. The bias is misaligned.")
    raise SystemExit(1)
print(f"    {len(lines)} biased captures over steps {sorted(steps, key=str)}; "
      f"2-segment {n_seg2}, 3-segment {n_seg3}")
if early_src or late_nosrc:
    print(f"GATE5-FAIL seglens: source segment present early ({early_src[:3]}) or "
          f"absent late ({late_nosrc[:3]}) -- the t>T//2 condition is not what "
          f"R32 assumes.")
    raise SystemExit(1)
if n_seg2 == 0 or n_seg3 == 0:
    print(f"GATE5-FAIL seglens: only one segment count observed "
          f"(2-seg {n_seg2}, 3-seg {n_seg3}). Both regimes must be exercised or "
          f"the alignment is untested in one of them.")
    raise SystemExit(1)
print("GATE5-OK seglens: lengths sum to Lk everywhere; the source segment is "
      "absent in the first half of the steps and present in the second")
GATE5PY
[ $? -eq 0 ] || { echo "GATE5 block failed"; exit 1; }

# ---------------------------------------------------------------- GATE6
echo "=== GATE6: noise floor (flash vs zero, same process) ==="
$PY evaluation/r32_seg_bias_arms.py \
    --cases_json "$CASES" --cases "$CLIP" --arms flash zero \
    --bias "$BIAS" --out_root "$WORK/g6" --tag g6 \
  || { echo "GATE6-FAIL the flash/zero render failed"; exit 1; }

$PY - "$WORK/g6" "$CLIP" <<'GATE6PY'
import glob, sys
import numpy as np
from PIL import Image
work, clip = sys.argv[1], sys.argv[2]
def frames(arm):
    f = sorted(glob.glob(f"{work}/{arm}/edit*/{clip}/*.png"))
    if not f:
        print(f"GATE6-FAIL floor: no frames for arm '{arm}'"); raise SystemExit(1)
    return np.stack([np.asarray(Image.open(p), dtype=np.float64) for p in f])
a, b = frames("flash"), frames("zero")
if a.shape != b.shape:
    print(f"GATE6-FAIL floor: shape mismatch {a.shape} vs {b.shape}")
    raise SystemExit(1)
d = np.abs(a - b)
mse = float((d ** 2).mean())
psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
print(f"    frames {a.shape[0]}  mean|d| {d.mean():.4f}/255  max|d| {d.max():.0f}/255  "
      f"PSNR {psnr:.2f} dB  pixels differing {float((d > 0).mean()):.3%}")
if psnr < 30.0:
    print(f"GATE6-FAIL floor: PSNR {psnr:.2f} dB < 30 -- the kernel change alone "
          f"moves the render more than an edit would. The SDPA path is not a "
          f"faithful substitute and Axis 1 cannot be measured through it.")
    raise SystemExit(1)
print(f"GATE6-OK floor: flash-vs-zero PSNR {psnr:.2f} dB, mean|d| "
      f"{d.mean():.4f}/255. RECORD THIS -- it is the floor; no arm delta below "
      f"it is readable.")
GATE6PY
[ $? -eq 0 ] || { echo "GATE6 block failed"; exit 1; }

echo
echo "[r32_smoke] ALL GATES OK -- r32_infer.sh may run."
echo "[r32_smoke] Carry GATE6's floor into the analysis; it bounds every arm delta."
