"""R66: build the ContextFlow case lists for FiVE-Bench-Correct.

Writes
  evaluation/r66_cases_correct.json  the 415 pairs of r43_anchor_manifest_correct.json, with the Correct
                                     target_prompt, the ContextFlow mode / PnP layers / is_delete and the
                                     candidate frame counts (n_src, n81, n41)
  evaluation/r66_cases_smoke.json    the first Correct pair of edit types 2, 3, 5 and 6 whose source has >= 81 frames

Mode mapping (ContextFlow demos): edit types 1-4 -> swap (layers 27 29 31 32), 5 -> insert (6 7 8 9),
6 -> delete (21 26 27 29, is_delete). n{F} = 4 * ((min(F, n_src) - 1) // 4) + 1 (FiVE's Wan rule).

anchor_image and src_first_frame are written relative to the data root ~/Data (e.g.
'dataggen/outputs/five_bench/anchors_r43_correct/edit1/0001_bus.png'), so the same case files work on the lab and on
Jean Zay (~/Data -> $WORK/Data there); readers join them with their --data_dir.

Usage: ~/anaconda3/envs/five-bench/bin/python -I evaluation/r66_build_cases.py
"""
import json
import os
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("R66_DATA_DIR", "~/Data")).expanduser()
DATA_ROOT = DATA_DIR / "FiVE-Fine-Grained-Video-Editing-Benchmark"
MANIFEST = EVAL_DIR / "r43_anchor_manifest_correct.json"
EDIT_JSON_DIR = EVAL_DIR / "five_bench_correct"
OUT_CASES = EVAL_DIR / "r66_cases_correct.json"
OUT_SMOKE = EVAL_DIR / "r66_cases_smoke.json"

EXPECTED_PAIRS = 415
SMOKE_TYPES = (2, 3, 5, 6)
SMOKE_MIN_FRAMES = 81

MODES = {
    "swap": {"pnp_layers": [27, 29, 31, 32], "is_delete": False},
    "insert": {"pnp_layers": [6, 7, 8, 9], "is_delete": False},
    "delete": {"pnp_layers": [21, 26, 27, 29], "is_delete": True},
}
TYPE_TO_MODE = {1: "swap", 2: "swap", 3: "swap", 4: "swap", 5: "insert", 6: "delete"}


def clip_frames(n_src: int, cap: int) -> int:
    """Largest 4k+1 not above min(cap, n_src)."""
    return 4 * ((min(cap, n_src) - 1) // 4) + 1


def data_relative(path: Path) -> str:
    """path relative to DATA_DIR; raises if it lies outside it (it would not exist on Jean Zay)."""
    rel = os.path.relpath(path, DATA_DIR)
    if rel.startswith(".."):
        raise ValueError(f"{path} is not under {DATA_DIR}")
    return rel


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    prompts: dict[tuple[int, str], str] = {}
    for edit_type in range(1, 7):
        for entry in json.loads((EDIT_JSON_DIR / f"edit{edit_type}_FiVE.json").read_text()):
            prompts[(edit_type, entry["video_name"])] = entry["target_prompt"]

    cases, errors = [], []
    for item in manifest:
        edit_type, video = int(item["edit_type"]), item["video_name"]
        frame_dir = DATA_ROOT / "images" / video
        anchor = Path(item["anchor_image"]).expanduser()
        n_src = len(list(frame_dir.glob("*.jpg"))) if frame_dir.is_dir() else 0
        if n_src == 0:
            errors.append(f"{item['case_id']}: no source frames in {frame_dir}")
        if not anchor.is_file():
            errors.append(f"{item['case_id']}: anchor missing {anchor}")
        target_prompt = prompts.get((edit_type, video))
        if not target_prompt:
            errors.append(f"{item['case_id']}: no target_prompt in edit{edit_type}_FiVE.json")
        mode = TYPE_TO_MODE[edit_type]
        cases.append({
            "case_id": item["case_id"],
            "edit_type": edit_type,
            "video_name": video,
            "target_prompt": target_prompt,
            "anchor_image": data_relative(anchor),
            "src_first_frame": data_relative(DATA_ROOT / item["src_first_frame"]),
            "mode": mode,
            "pnp_layers": MODES[mode]["pnp_layers"],
            "is_delete": MODES[mode]["is_delete"],
            "n_src": n_src,
            "n81": clip_frames(n_src, 81) if n_src else 0,
            "n41": clip_frames(n_src, 41) if n_src else 0,
        })

    smoke = []
    for edit_type in SMOKE_TYPES:
        pick = next((c for c in cases if c["edit_type"] == edit_type and c["n_src"] >= SMOKE_MIN_FRAMES), None)
        if pick is None:
            errors.append(f"smoke: no type-{edit_type} pair with >= {SMOKE_MIN_FRAMES} source frames")
        else:
            smoke.append(pick)

    if len(cases) != EXPECTED_PAIRS:
        errors.append(f"expected {EXPECTED_PAIRS} pairs, got {len(cases)}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    OUT_CASES.write_text(json.dumps(cases, indent=1) + "\n")
    OUT_SMOKE.write_text(json.dumps(smoke, indent=1) + "\n")
    print(f"{len(cases)} pairs -> {OUT_CASES}")
    print(f"{len(smoke)} smoke pairs -> {OUT_SMOKE}: {[c['case_id'] for c in smoke]}")
    print(f"pairs with n_src < 81: {sum(c['n_src'] < 81 for c in cases)}; n_src < 49: {sum(c['n_src'] < 49 for c in cases)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
