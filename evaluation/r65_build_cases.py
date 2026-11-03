"""R65: case lists for the AnyV2V baseline on FiVE-Bench-Correct.

Writes three files next to this script:

  * ``r65_cases_correct.json`` -- the 415 pairs of ``r43_anchor_manifest_correct.json``: case_id, edit_type,
    video_name, the Correct ``target_prompt``, the Correct Qwen anchor (``anchor_image``, used as AnyV2V's first
    frame) and ``n_frames = min(32, source frame count)`` (FiVE's paper, section 5.1: AnyV2V processes only the first
    32 frames; the only clip shorter than 32, ``0072_dog-agility`` with 25 frames, keeps its own length).
  * ``r65_cases_smoke.json`` -- ``edit1/0001_bus`` plus the ``0072_dog-agility`` pair of lowest edit type (the
    shortest clip, 25 frames), both from the Correct list.
  * ``r65_cases_repro.json`` -- the reproduction gate: official FiVE ``edit1/0001_bus`` and ``edit2/0081_A_horse``
    with the *official* ``target_prompt`` and ``first_frame`` = frame 0 of FiVE's released AnyV2V output
    (FiVE's InstructPix2Pix first frame is not reproducible, so the released one is reused).

Exit 1 if the Correct count is not 415, a source dir / anchor / Correct prompt is missing, or a repro release
clip (or its first-frame image) is missing.

Usage:
    python -I evaluation/r65_build_cases.py
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "evaluation"
MAX_FRAMES = 32  # FiVE paper section 5.1: AnyV2V processes the first 32 frames
EXPECTED_CASES = 415
SMOKE_VIDEO = "0072_dog-agility"
SMOKE_FIRST = "edit1/0001_bus"
REPRO_CASES = ("edit1/0001_bus", "edit2/0081_A_horse")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def _load_prompts(json_dir: Path, edit_type: int) -> Dict[str, dict]:
    path = json_dir / f"edit{edit_type}_FiVE.json"
    return {e["video_name"]: e for e in json.loads(path.read_text())}


def _frame_count(src_root: Path, video_name: str, cache: Dict[str, int]) -> int:
    """Number of source frames of a video (cached: every edit type of a video shares it)."""
    if video_name not in cache:
        src_dir = src_root / video_name
        if not src_dir.is_dir():
            raise FileNotFoundError(f"source frame dir missing: {src_dir}")
        cache[video_name] = sum(1 for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if cache[video_name] == 0:
            raise FileNotFoundError(f"no frames in {src_dir}")
    return cache[video_name]


def build_correct(manifest: Path, correct_dir: Path, src_root: Path) -> List[dict]:
    pairs = json.loads(manifest.read_text())
    if len(pairs) != EXPECTED_CASES:
        raise ValueError(f"{manifest.name} holds {len(pairs)} pairs, expected {EXPECTED_CASES}")
    prompts = {t: _load_prompts(correct_dir, t) for t in range(1, 7)}
    counts: Dict[str, int] = {}
    cases = []
    for p in pairs:
        t, video = int(p["edit_type"]), p["video_name"]
        if video not in prompts[t]:
            raise KeyError(f"{p['case_id']}: not in {correct_dir}/edit{t}_FiVE.json")
        anchor = Path(p["anchor_image"]).expanduser()
        if not anchor.is_file():
            raise FileNotFoundError(f"{p['case_id']}: anchor missing: {anchor}")
        cases.append({
            "case_id": p["case_id"],
            "edit_type": t,
            "video_name": video,
            "target_prompt": prompts[t][video]["target_prompt"],
            "anchor_image": str(anchor),
            "n_frames": min(MAX_FRAMES, _frame_count(src_root, video, counts)),
        })
    return cases


def build_smoke(correct_cases: List[dict]) -> List[dict]:
    by_id = {c["case_id"]: c for c in correct_cases}
    short = sorted((c for c in correct_cases if c["video_name"] == SMOKE_VIDEO), key=lambda c: c["edit_type"])
    if SMOKE_FIRST not in by_id or not short:
        raise KeyError(f"smoke pairs not found: {SMOKE_FIRST} / {SMOKE_VIDEO}")
    return [by_id[SMOKE_FIRST], short[0]]


def build_repro(official_dir: Path, release_root: Path, src_root: Path) -> List[dict]:
    counts: Dict[str, int] = {}
    cases = []
    for case_id in REPRO_CASES:
        edit, video = case_id.split("/")
        t = int(edit.removeprefix("edit"))
        official = _load_prompts(official_dir, t)
        if video not in official:
            raise KeyError(f"{case_id}: not in {official_dir}/edit{t}_FiVE.json")
        clip_dir = release_root / edit / video
        frames = sorted(p for p in clip_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES) if clip_dir.is_dir() else []
        if not frames:
            raise FileNotFoundError(f"FiVE release frames missing for {case_id}: {clip_dir} (frame images expected)")
        cases.append({
            "case_id": case_id,
            "edit_type": t,
            "video_name": video,
            "target_prompt": official[video]["target_prompt"],
            "first_frame": str(frames[0]),
            "n_frames": min(MAX_FRAMES, _frame_count(src_root, video, counts)),
        })
    return cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=str(EVAL_DIR / "r43_anchor_manifest_correct.json"))
    ap.add_argument("--correct_dir", default=str(EVAL_DIR / "five_bench_correct"))
    ap.add_argument("--data_root", default="~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark")
    ap.add_argument("--official_dir", default=None, help="default: <data_root>/edit_prompt")
    ap.add_argument("--release_root", default="~/Data/five_release/AnyV2V")
    ap.add_argument("--out_dir", default=str(EVAL_DIR))
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser()
    src_root = data_root / "images"
    official_dir = Path(args.official_dir).expanduser() if args.official_dir else data_root / "edit_prompt"
    out_dir = Path(args.out_dir)

    try:
        correct = build_correct(Path(args.manifest), Path(args.correct_dir), src_root)
        smoke = build_smoke(correct)
        repro = build_repro(official_dir, Path(args.release_root).expanduser(), src_root)
    except (ValueError, KeyError, FileNotFoundError) as ex:
        print(f"[r65_build_cases] ERROR: {ex}", file=sys.stderr)
        sys.exit(1)

    for name, cases in (("correct", correct), ("smoke", smoke), ("repro", repro)):
        path = out_dir / f"r65_cases_{name}.json"
        path.write_text(json.dumps(cases, indent=1, ensure_ascii=False))
        print(f"[r65_build_cases] {path.relative_to(REPO) if path.is_relative_to(REPO) else path}: {len(cases)} cases")

    short = {c["video_name"]: c["n_frames"] for c in correct if c["n_frames"] < MAX_FRAMES}
    print(f"[r65_build_cases] videos shorter than {MAX_FRAMES} frames (kept at their length): {short}")
    print(f"[r65_build_cases] smoke: {[c['case_id'] for c in smoke]}  repro: {[c['case_id'] for c in repro]}")


if __name__ == "__main__":
    main()
