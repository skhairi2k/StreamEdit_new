"""Build the R7 visual-prompting anchor manifest.

Expands the six FiVE-Bench ``edit_prompt/edit{T}_FiVE.json`` files into a single flat
manifest (one entry per ``(edit_type, video_name)`` pair, 419 total) that
``evaluation/gen_anchors.py`` consumes to produce the oracle Qwen-Image-Edit anchors.

Each manifest entry carries exactly the keys ``gen_anchors.py`` reads (``case_id``,
``src_first_frame``, ``first_frame_edit_prompt``) plus ``edit_type`` for the sbatch-array
``--edit_type`` filter and ``anchor_image`` (the eventual anchor path) for bookkeeping.

Example
-------
    python evaluation/r7_build_anchor_manifest.py \\
        --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \\
        --out evaluation/r7_anchor_manifest.json
"""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", type=str,
                        default="~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark",
                        help="FiVE-Bench root; reads edit_prompt/edit{T}_FiVE.json under it")
    parser.add_argument("--out", type=str, default="evaluation/r7_anchor_manifest.json",
                        help="Output manifest path (list of per-pair anchor cases)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    manifest = []
    for edit_type in range(1, 7):
        edit_json = data_root / "edit_prompt" / f"edit{edit_type}_FiVE.json"
        entries = json.loads(edit_json.read_text())
        for entry in entries:
            video_name = entry["video_name"]
            manifest.append({
                "case_id": f"edit{edit_type}/{video_name}",
                "edit_type": edit_type,
                "video_name": video_name,
                "src_first_frame": f"images/{video_name}/00001.jpg",
                "first_frame_edit_prompt": entry["instruction"],
                "anchor_image": f"evaluation/anchors/edit{edit_type}/{video_name}.png",
            })
        print(f"[r7_manifest] edit{edit_type}: {len(entries)} pairs")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"[r7_manifest] wrote {len(manifest)} entries -> {out_path}")


if __name__ == "__main__":
    main()
