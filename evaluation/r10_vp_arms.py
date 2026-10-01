"""R10 driver: head-gated persistent visual-prompt injection.

Runs each case under several arms that share an identical base configuration
(``blend_off=True``: no self-attention query/key blending, but the background
source-KV injection is KEPT so the target branch stays anchored to the source
video and motion fidelity remains measurable):

  * ``none``     -- no visual prompt at all. Motion ceiling / edit floor; also the
                    plumbing sanity check (broken frames here => flags are wrong
                    and the other arms are uninterpretable).
  * ``all``      -- VP injected into all 360 heads. Expected to reproduce the known
                    static-video failure (prediction P3).
  * ``spatial``  -- VP injected only into the R19 SPATIAL heads (117). Expected:
                    motion returns while the edit persists (P1).
  * ``temporal`` -- VP injected only into the R19 TEMPORAL heads (49). Expected:
                    static AND appearance-drifting (P2, ideas.tex claim 4).
  * ``rand_spatial`` / ``rand_temporal`` -- COUNT-MATCHED random head sets (117
                    and 49) at a fixed seed. The two labelled arms differ in size,
                    so without these a spatial-vs-temporal gap confounds head TYPE
                    with head COUNT. If a random set of the same size reproduces
                    the effect, the labels are not doing the work -- so these are
                    what make the result interpretable, not optional extras.

Gates come from ``evaluation/r19_head_gates.pt`` (R19: equal-budget, disjoint key
sets). The earlier ``r10_head_gates.pt`` is SUPERSEDED -- it derives from key sets
of 1560 vs 18 keys, where a head with no specialisation at all scores margin
-0.97, so its 281/10 split was not a measurement of head type. The driver refuses
a gate file without R19 provenance rather than running on it silently.

Unlike the paper's SS4.5 visual prompting, the anchor is NOT cached as an initial
latent: it lives in a private K/V bank re-roped to the current chunk at every
block, so it never fades as the window advances.

The same seed is reset before every arm, so a difference between arms is
attributable to head gating rather than to sampling noise.

Outputs PNG frames to ``{out_root}/{arm}/edit{T}/{video_name}/`` plus
``_manifest.csv``. The ``edit{T}/`` level is required, not cosmetic: the clip set
contains the same video under two edit types (0011_lucia in edit2 and edit5), so a
flat ``{arm}/{video_name}/`` would have one run silently overwrite the other. It is
also the layout R7 already uses, so ``evaluate.py --tgt_layout edit_video`` scores
both without a new code path.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SF_ROOT = _REPO_ROOT / "Self-Forcing_StreamEdit"

# Single-window inference() holds at most 21 latent frames (max_attention_size
# 32760 / 1560 tokens per frame); 21 latent frames == 81 pixel frames. Longer
# clips overflow the cache (the R9 0034_cows crash), so they are truncated.
MAX_PIXEL_FRAMES = 81

# Arms. `rand_*` are COUNT-MATCHED random head sets at a fixed seed: `spatial`
# (117 heads) and `temporal` (49) differ in size, so a spatial-vs-temporal gap
# would confound head TYPE with head COUNT. If a random set of the same size
# reproduces the effect, the labels are not doing the work -- that control is
# what makes the result interpretable, so do not drop it to save GPU time.
ARMS = ("none", "all", "spatial", "temporal", "mixed", "dense",
        "spatial_core", "temporal_core", "rand_spatial", "rand_temporal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases_json", type=str, default=str(_REPO_ROOT / "evaluation" / "cases.json"))
    parser.add_argument("--cases", type=str, nargs="*", default=None,
                        help="case_ids to run (default: every case in cases.json; the 20 R10 "
                             "clips are listed explicitly in r10_infer.sh)")
    parser.add_argument("--arms", type=str, nargs="*", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--gates", type=str,
                        default=str(_REPO_ROOT / "evaluation" / "r19_head_gates.pt"),
                        help="per-head gates from evaluation/r19_build_head_gates.py. "
                             "NOT r10_head_gates.pt -- those came from the superseded "
                             "1560-vs-18 key sets, where a head with no specialisation "
                             "at all scored margin -0.97.")
    parser.add_argument("--data_root", type=str,
                        default="~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark")
    # Anchors are read as {anchor_root}/edit{T}/{video_name}.png -- the SAME layout and
    # the same files run_fivebench.py uses for the paper's SS4.5 arm (R7), so the R10
    # arms and the SS4.5 comparison row differ only by injection mechanism, never by
    # which anchor image was used.
    parser.add_argument("--anchor_root", type=str,
                        default="/projects/dataggen/outputs/five_bench/anchors",
                        help="Root of Qwen-edited anchors; resolved as {root}/edit{T}/{video}.png")
    parser.add_argument("--out_root", type=str,
                        default="/projects/dataggen/outputs/five_bench/r10_vp_arms")

    parser.add_argument("--fg_boost_factor", type=float, default=4.0, help="omega, CrossAttn boosting")
    parser.add_argument("--blend_power", type=float, default=2.0, help="rho (unused when blend_off)")
    parser.add_argument("--keep_blending", action="store_true", default=False,
                        help="Debug: leave the self-attn blending ON (default: blend_off=True)")

    parser.add_argument("--step", type=int, default=15)
    parser.add_argument("--flow_shift", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sink_size", type=int, default=0)
    parser.add_argument("--config_path", type=str, default="configs/self_forcing_dmd.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/self_forcing_dmd.pt")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--sf_root", type=str, default=str(_DEFAULT_SF_ROOT))
    return parser.parse_args()


def save_frames_png(frames: torch.Tensor, out_dir: Path) -> int:
    """Save a ``[T, C, H, W]`` float tensor in [0, 1] as zero-padded PNG frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = (frames.clamp(0, 1).float().cpu().numpy() * 255.0).round().astype(np.uint8)
    for i in range(arr.shape[0]):
        Image.fromarray(arr[i].transpose(1, 2, 0)).save(out_dir / f"{i:05d}.png")
    return arr.shape[0]


def main() -> None:
    args = parse_args()

    # Resolve every external path BEFORE chdir-ing into the SF build, so relative
    # config/checkpoint paths resolve there while data/output paths stay put.
    sf_root = Path(args.sf_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    cases_json = Path(args.cases_json).expanduser().resolve()
    gates_path = Path(args.gates).expanduser().resolve()

    all_cases = json.loads(cases_json.read_text())
    wanted = args.cases
    if wanted:
        by_id = {c["case_id"]: c for c in all_cases}
        missing = [c for c in wanted if c not in by_id]
        if missing:
            raise SystemExit(f"[r10] case_ids not in {cases_json}: {missing}")
        cases = [by_id[c] for c in wanted]
    else:
        cases = all_cases

    # Resolve anchors from the shared anchor root, mirroring run_fivebench.py's layout
    # so R10 and the SS4.5 (R7) arm consume byte-identical anchor images.
    anchor_root = Path(args.anchor_root).expanduser().resolve()
    for case in cases:
        case["_anchor_abs"] = anchor_root / f"edit{case['edit_type']}" / f"{case['video_name']}.png"
        if not case["_anchor_abs"].exists():
            raise SystemExit(f"[r10] missing anchor for {case['case_id']}: {case['_anchor_abs']}")

    gate_payload = torch.load(gates_path, weights_only=False)
    for arm in args.arms:
        if arm != "none" and arm not in gate_payload:
            raise SystemExit(f"[r10] gate '{arm}' not found in {gates_path}")
    # Refuse the superseded gates outright rather than silently producing arms
    # built on a criterion that mislabels unspecialised heads as spatial.
    if "provenance" not in gate_payload:
        raise SystemExit(
            f"[r10] {gates_path} predates R19 (no provenance field). Rebuild with:\n"
            "  python evaluation/r19_build_head_gates.py")
    print(f"[r10] gates: {gate_payload['provenance']}")
    print(f"[r10]   thresholds {gate_payload['thresholds']} "
          f"over {len(gate_payload['cases'])} videos")

    sys.path.insert(0, str(sf_root))
    os.chdir(sf_root)

    from inference_edit_streamedit import load_pipe, find_closest_num_frame
    from diffusers.utils import load_video

    pipeline, low_memory, device, local_rank = load_pipe(args)

    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    blend_off = not args.keep_blending
    print(f"[r10] cases={len(cases)} arms={args.arms} blend_off={blend_off} "
          f"seed={args.seed} out={out_root}", flush=True)
    print(f"[r10] anchors from {anchor_root}/edit{{T}}/ (shared with the SS4.5/R7 arm)", flush=True)
    for arm in args.arms:
        if arm != "none":
            print(f"[r10]   gate {arm}: {int(gate_payload[arm].sum())}/360 heads", flush=True)

    rows = []
    for case in cases:
        video_name = case["video_name"]
        src_path = data_root / "videos" / f"{video_name}.mp4"

        # ---- encode once per case, reused by every arm ----
        try:
            src_video = load_video(str(src_path))
            new_len = find_closest_num_frame(len(src_video))
            if not new_len:
                raise ValueError(f"video too short: {len(src_video)} frames")
            new_len = min(new_len, MAX_PIXEL_FRAMES)
            src_video = src_video[:new_len]

            src_tensor = torch.stack([transform(img) for img in src_video], dim=1).unsqueeze(0)
            video_latents = pipeline.vae.encode_to_latent(
                src_tensor.to(device=device, dtype=torch.bfloat16)
            ).to(device=device, dtype=torch.bfloat16)
            pipeline.vae.model.clear_cache()

            anchor_img = Image.open(case["_anchor_abs"]).convert("RGB")
            vp_latent = pipeline.vae.encode_to_latent(
                transform(anchor_img).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
            ).to(device=device, dtype=torch.bfloat16)
            pipeline.vae.model.clear_cache()
            print(f"[r10] {video_name}: {new_len} frames -> latents {tuple(video_latents.shape)}, "
                  f"vp {tuple(vp_latent.shape)}", flush=True)
        except Exception as ex:  # encoding failed -> record for every arm and move on
            for arm in args.arms:
                rows.append((case["case_id"], video_name, arm, f"encode_error:{type(ex).__name__}", 0, 0.0))
            print(f"[ERROR] {video_name}: encode failed: {ex}", flush=True)
            continue

        # ---- one run per arm, identical seed ----
        for arm in args.arms:
            out_dir = out_root / arm / f"edit{case['edit_type']}" / video_name
            try:
                # Reset before every arm so arms differ only by head gating, not noise.
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)

                arm_gate = None if arm == "none" else gate_payload[arm].to(device)
                t0 = time.time()
                edit_video = pipeline.rollout_inference(
                    src_video=video_latents,
                    src_prompts=case["src_prompt"],
                    trg_prompts=case["trg_prompt"],
                    src_trigger_words=case["src_word"],
                    trg_trigger_words=case["trg_word"],
                    return_latents=False,
                    wo_video_decode=False,
                    profile=False,
                    low_memory=low_memory,
                    # The VP must NOT be cached as an initial latent -- that is the
                    # fading SS4.5 path. It enters only through vp_latent below.
                    independent_first_frame=False,
                    src_initial_latent=None,
                    trg_initial_latent=None,
                    fg_boost_factor=args.fg_boost_factor,
                    blend_power=args.blend_power,
                    rollout_chunk_size=-1,          # single-window inference() path
                    blend_off=blend_off,
                    vp_latent=None if arm == "none" else vp_latent,
                    vp_head_gate=arm_gate,
                )
                pipeline.vae.model.clear_cache()
                elapsed = time.time() - t0

                n_frames = save_frames_png(edit_video[0], out_dir)
                rows.append((case["case_id"], video_name, arm, "ok", n_frames, round(elapsed, 1)))
                print(f"[ok] {video_name} / {arm}: {n_frames} frames in {elapsed:.1f}s", flush=True)
            except Exception as ex:  # never silently swallow -- record and continue
                rows.append((case["case_id"], video_name, arm, f"error:{type(ex).__name__}", 0, 0.0))
                print(f"[ERROR] {video_name} / {arm}: {ex}", flush=True)

    out_root.mkdir(parents=True, exist_ok=True)
    manifest = out_root / "_manifest.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "video_name", "arm", "status", "n_frames", "seconds"])
        writer.writerows(rows)

    n_ok = sum(1 for r in rows if r[3] == "ok")
    print(f"[r10] done: {n_ok}/{len(rows)} ok -> {manifest}", flush=True)


if __name__ == "__main__":
    main()
