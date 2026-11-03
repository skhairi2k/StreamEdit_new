"""R64: FlowDirector (CVPR 2026, official V1 code @cecd56e, unmodified) on FiVE-Bench-Correct, one model load per process.

Settings = the released ``script-edit-single-gpu.sh``, read from the script itself (so they come from one place), seed
included (its ``--base_seed 42``; paper-wide rule: every method runs its released seed, seed 0 where the release leaves it
random). Two input adaptations, no change to the method:
  * frames: FlowDirector's ``load_video_frames`` VAE-encodes EVERY frame of the mp4 while ``frame_num`` only sizes the
    latents, so a source longer than ``frame_num`` breaks it. We feed the first expected_frames(n) frames (41, or 4k+1 for
    the 7 short videos), the Wan-Edit protocol of r48_wan_common; read / resize / normalisation are FlowDirector's own.
  * trigger words: source_words / target_words = R43's source_object / target_object, passed as the verbatim span of the
    prompt they match case-insensitively on word boundaries (FlowDirector's find_subtokens_range is case-sensitive:
    'A gray rabbit' never matches '...a gray rabbit...'). An empty object (edit5 source, edit6 target) -> None.
Negative prompt = FlowDirector's default (the released script passes none).

Modes:
  render (default)  writes {out_root}/edit{T}/{video}/00000.png ... (the ``edit_video`` layout evaluate.py reads);
                    clips with their expected PNG count are skipped (resumable); --shard/--n_shards split the case list
  --dry_triggers    tokenizer only (CPU): runs FlowDirector's own token matching on every case -> --out_csv
  --missing         prints every case clip without a complete render (post-run guard), nothing else

usage: python evaluation/r64_flowdirector_render.py --cases evaluation/r43_anchor_manifest_correct.json --shard 0 --n_shards 24
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import shlex
import subprocess
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
FD_DIR = Path.home() / "Code/FlowDirector"
sys.path.insert(0, str(FD_DIR))                 # FlowDirector's `wan` package (wan-edit/ has one of the same name)
sys.path.insert(1, str(HERE))
import r48_wan_common as C  # noqa: E402

D = Path.home() / "Data/dataggen/outputs/five_bench"
ANNO = HERE / "five_bench_correct"
RELEASED_SCRIPT = FD_DIR / "script-edit-single-gpu.sh"
PER_CLIP_FLAGS = {"--source_video_path", "--source_prompt", "--prompt", "--source_words", "--target_words",
                  "--ckpt_dir", "--save_file"}      # set per clip / by us, never taken from the released script


def released_flags() -> list[str]:
    """The hyperparameter flags of the released script (seed included), per-clip flags dropped."""
    text = RELEASED_SCRIPT.read_text().replace("\\\n", " ").strip().rstrip("\\")   # upstream file ends on a dangling '\'
    toks = shlex.split(text)
    toks = toks[toks.index("edit.py") + 1:]
    flags: list[str] = []
    i = 0
    while i < len(toks):
        key = toks[i]
        has_val = i + 1 < len(toks) and not toks[i + 1].startswith("--")
        if key not in PER_CLIP_FLAGS:
            flags += [key, toks[i + 1]] if has_val else [key]
        i += 2 if has_val else 1
    return flags


def load_cli():
    """FlowDirector's ``edit.py`` as a module (its CLI is guarded by __main__) + its parsed, validated args."""
    spec = importlib.util.spec_from_file_location("flowdirector_cli", FD_DIR / "edit.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    sys.argv = ["edit.py", *released_flags(), "--ckpt_dir", str(C.CKPT)]
    return cli, cli._parse_args()


def resolve_words(prompt: str, obj: str) -> tuple[str | None, str]:
    """-> (words passed to FlowDirector, status): status in {empty, exact, case_fixed, not_in_prompt}."""
    obj = (obj or "").strip()
    if not obj:
        return None, "empty"
    m = re.search(r"(?<!\w)" + re.escape(obj) + r"(?!\w)", prompt, flags=re.IGNORECASE)
    if m is None:
        return obj, "not_in_prompt"
    return m.group(0), ("exact" if m.group(0) == obj else "case_fixed")


def first_frames_loader(n: int):
    """FlowDirector's WanT2V.load_video_frames, unchanged except that it stops after the first ``n`` frames."""
    import cv2
    import torch

    def load_video_frames(self, video_path, size=(832, 480)):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        frames = []
        while len(frames) < n:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            frame = torch.from_numpy(frame).float().permute(2, 0, 1) / 127.5 - 1.0
            frames.append(frame)
        cap.release()
        if len(frames) != n:
            raise ValueError(f"{video_path}: read {len(frames)} frames, expected {n}")
        frames_tensor = torch.stack(frames).permute(1, 0, 2, 3).to(self.device)
        return self.vae.video_encode(frames_tensor)
    return load_video_frames


def expected(vid: str) -> int:
    return C.expected_frames(C.source_frame_count(vid))


def is_complete(out_dir: Path, want: int) -> bool:
    return out_dir.is_dir() and len(list(out_dir.glob("*.png"))) == want


def dry_triggers(cli, args, cases, out_csv: Path) -> int:
    """FlowDirector's own token matching (edit(): tokenize(..., add_special_tokens=True), drop EOS, find_subtokens_range)."""
    from wan.modules.tokenizers import HuggingfaceTokenizer
    cfg = cli.WAN_CONFIGS[args.task]
    tok = HuggingfaceTokenizer(name=str(C.CKPT / cfg.t5_tokenizer), seq_len=cfg.text_len, clean="whitespace").tokenizer

    def found(prompt, words):
        if words is None:
            return ""
        tk1 = tok.tokenize(prompt, add_special_tokens=True)
        tk2 = tok.tokenize(words, add_special_tokens=True)
        return cli.wan.WanT2V.find_subtokens_range(None, tk1[:-1], tk2[:-1]) is not None

    rows = []
    for c in cases:
        sw, ss = resolve_words(c["source_prompt"], c["source_object"])
        tw, ts = resolve_words(c["target_prompt"], c["target_object"])
        rows.append(dict(edit_type=c["edit_type"], video_name=c["video_name"], source_words=sw or "", target_words=tw or "",
                         src_found=found(c["source_prompt"], sw), tgt_found=found(c["target_prompt"], tw),
                         case_fixed="+".join(k for k, s in (("src", ss), ("tgt", ts)) if s == "case_fixed")))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n_src = sum(r["src_found"] is False for r in rows)
    n_tgt = sum(r["tgt_found"] is False for r in rows)
    print(f"[r64_flowdirector_render] triggers: {len(rows)} cases, src not found {n_src}, tgt not found {n_tgt}, "
          f"src empty {sum(r['src_found'] == '' for r in rows)}, tgt empty {sum(r['tgt_found'] == '' for r in rows)}, "
          f"case-fixed {sum(bool(r['case_fixed']) for r in rows)} -> {out_csv}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, help="json list with video_name / edit_type (e.g. r43_anchor_manifest_correct.json)")
    ap.add_argument("--anno_dir", default=str(ANNO), help="edit{T}_FiVE.json folder (default: FiVE-Bench-Correct)")
    ap.add_argument("--out_root", default=str(D / "r64_flowdirector/correct_full"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--dry_triggers", action="store_true")
    ap.add_argument("--out_csv", default=str(HERE / "csv/r64_triggers.csv"))
    ap.add_argument("--missing", action="store_true")
    a = ap.parse_args()
    if not 0 <= a.shard < a.n_shards:
        raise SystemExit(f"[r64_flowdirector_render] shard {a.shard} not in [0, {a.n_shards})")

    cases = C.load_cases(a.cases, a.anno_dir)[a.shard::a.n_shards]
    out_root = Path(a.out_root).expanduser()

    if a.missing:
        for c in cases:
            if not is_complete(out_root / f"edit{c['edit_type']}" / c["video_name"], expected(c["video_name"])):
                print(f"edit{c['edit_type']}/{c['video_name']}")
        return 0

    cli, args = load_cli()
    if a.dry_triggers:
        return dry_triggers(cli, args, cases, Path(a.out_csv))

    import flash_attn
    import torch
    commit = subprocess.run(["git", "-C", str(FD_DIR), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"[r64_flowdirector_render] FlowDirector @{commit}, torch {torch.__version__}, flash_attn {flash_attn.__version__}, "
          f"gpu {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[r64_flowdirector_render] released flags: {' '.join(released_flags())}", flush=True)
    print(f"[r64_flowdirector_render] shard {a.shard}/{a.n_shards}: {len(cases)} cases, anno={a.anno_dir} -> {out_root}", flush=True)

    pipe = cli.wan.WanT2V(config=cli.WAN_CONFIGS[args.task], checkpoint_dir=str(C.CKPT), device_id=0, rank=0,
                          t5_fsdp=False, dit_fsdp=False, use_usp=False, t5_cpu=False)
    n_fail = n_done = 0
    for c in cases:
        vid, t = c["video_name"], c["edit_type"]
        out_dir = out_root / f"edit{t}" / vid
        want = expected(vid)
        if is_complete(out_dir, want):
            n_done += 1
            continue
        try:
            t0 = time.time()
            sw, ss = resolve_words(c["source_prompt"], c["source_object"])
            tw, ts = resolve_words(c["target_prompt"], c["target_object"])
            pipe.load_video_frames = types.MethodType(first_frames_loader(want), pipe)
            out = pipe.edit(
                c["target_prompt"], size=cli.SIZE_CONFIGS[args.size], frame_num=want, shift=args.sample_shift,
                sample_solver=args.sample_solver, sampling_steps=args.sample_steps, guide_scale=args.sample_guide_scale,
                tar_guide_scale=args.tar_guide_scale, seed=args.base_seed, offload_model=False,
                source_video_path=str(C.FIVE / "videos" / f"{vid}.mp4"), source_prompt=c["source_prompt"],
                nmax_step=args.n_max, nmin_step=args.n_min, n_avg=args.n_avg, worse_avg=args.worse_avg, omega=args.omega,
                source_words=sw, target_words=tw, window_size=args.window_size, decay_factor=args.decay_factor)
            if out is None:
                raise RuntimeError("WanT2V.edit returned None")
            nf = C.save_frames(out, out_dir)
            assert nf == want, f"{nf} frames written, expected {want}"
            n_done += 1
            print(f"[r64_flowdirector_render] ok {t}/{vid}: {nf} frames, src={sw!r} ({ss}), tgt={tw!r} ({ts}), "
                  f"{time.time() - t0:.0f}s", flush=True)
        except Exception as e:                                       # noqa: BLE001 -- report per clip, keep going
            n_fail += 1
            print(f"[r64_flowdirector_render] FAILED {t}/{vid}: {type(e).__name__}: {e}", flush=True)
        torch.cuda.empty_cache()
    print(f"[r64_flowdirector_render] shard {a.shard}: {n_done}/{len(cases)} clips  failures: {n_fail}")
    return int(n_fail > 0)


if __name__ == "__main__":
    sys.exit(main())
