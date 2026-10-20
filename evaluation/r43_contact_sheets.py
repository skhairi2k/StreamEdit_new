"""R43 -- contact sheets: source | R7 anchor (FiVE as released) | R43 anchor (Correct).

Each row shows frame 0, R7's anchor for the same pair, and the new anchor, labelled with
FiVE's instruction and the Correct instruction (the keep-pose suffix is shown as
``+keep-pose``). Cells are 320x185, JPEG quality 85; a missing image is drawn as a grey
"n/a" cell, so a partial run still gives a sheet.

  --smoke   rows = the smoke cases of ``--smoke_manifest`` (anchors read from the manifest's
            ``anchor_image``), one sheet per edit type -> ``r43_smoke_sheet_edit{T}.jpg``
  default   every pair of ``--correct_manifest`` (new anchors under ``--correct_root``/
            edit{T}/{video}.png), one sheet per edit type -> ``r43_sheet_correct_edit{T}.jpg``,
            ``--per_row`` triplets per row

Example
-------
    python evaluation/r43_contact_sheets.py \\
        --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \\
        --r7_manifest evaluation/r7_anchor_manifest.json \\
        --correct_manifest evaluation/r43_anchor_manifest_correct.json \\
        --r7_root ~/Data/dataggen/outputs/five_bench/anchors \\
        --correct_root ~/Data/dataggen/outputs/five_bench/anchors_r43_correct \\
        --out_dir evaluation/figures
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CELL_W, CELL_H = 320, 185
LABEL_LINES, FONT_SIZE, PAD = 3, 12, 4
JPEG_QUALITY = 85
KEEP_POSE_SUFFIX = (" Keep the coarse position, size, facing direction and coarse pose of the "
                    "original object.")
COLUMN_TITLES = ("source", "FiVE / R7", "Correct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", type=str,
                        default="~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark")
    parser.add_argument("--r7_manifest", type=str, default="evaluation/r7_anchor_manifest.json")
    parser.add_argument("--correct_manifest", type=str,
                        default="evaluation/r43_anchor_manifest_correct.json")
    parser.add_argument("--smoke_manifest", type=str,
                        default="evaluation/r43_anchor_manifest_smoke.json")
    parser.add_argument("--r7_root", type=str, default="~/Data/dataggen/outputs/five_bench/anchors")
    parser.add_argument("--correct_root", type=str,
                        default="~/Data/dataggen/outputs/five_bench/anchors_r43_correct")
    parser.add_argument("--out_dir", type=str, default="evaluation/figures")
    parser.add_argument("--smoke", action="store_true", default=False,
                        help="sheets for the smoke manifest only (edit types 1-2)")
    parser.add_argument("--per_row", type=int, default=None,
                        help="triplets per row (default 1 for --smoke, 2 otherwise)")
    return parser.parse_args()


def font() -> ImageFont.ImageFont:
    for cand in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(cand, FONT_SIZE)
        except OSError:
            continue
    return ImageFont.load_default()


def wrap(text: str, fnt, width: int, max_lines: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        cand = f"{cur} {word}".strip()
        if fnt.getlength(cand) <= width:
            cur = cand
            continue
        if cur:
            lines.append(cur)
        cur = word
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and fnt.getlength(last + "…") > width:
            last = last[:-1]
        lines[-1] = last + "…"
    return lines


def cell(path: Path | None, fnt) -> Image.Image:
    """Image fitted into the cell, or a grey cell saying n/a."""
    out = Image.new("RGB", (CELL_W, CELL_H), (160, 160, 160))
    if path is None or not path.is_file():
        ImageDraw.Draw(out).text((PAD, CELL_H // 2 - FONT_SIZE), "n/a", fill="black", font=fnt)
        return out
    im = Image.open(path).convert("RGB")
    im.thumbnail((CELL_W, CELL_H), Image.LANCZOS)
    out.paste(im, ((CELL_W - im.width) // 2, (CELL_H - im.height) // 2))
    return out


def label_text(r7_prompt: str, new_prompt: str) -> str:
    shown = new_prompt.replace(KEEP_POSE_SUFFIX, " +keep-pose")
    return f"FiVE: {r7_prompt}  ||  Correct: {shown}"


def triplet(case: dict, r7_case: dict, r7_root: Path, data_root: Path, new_path: Path,
            fnt) -> Image.Image:
    line_h = FONT_SIZE + 2
    w = 3 * CELL_W + 2 * PAD
    label_h = (LABEL_LINES + 1) * line_h
    block = Image.new("RGB", (w, label_h + CELL_H), "white")
    draw = ImageDraw.Draw(block)
    head = f"{case['case_id']}"
    draw.text((0, 0), head, fill="black", font=fnt)
    body = wrap(label_text(r7_case["first_frame_edit_prompt"], case["first_frame_edit_prompt"]),
                fnt, w, LABEL_LINES)
    for i, line in enumerate(body):
        draw.text((0, (i + 1) * line_h), line, fill="black", font=fnt)
    t, video = case["edit_type"], case["video_name"]
    imgs = (data_root / case["src_first_frame"], r7_root / f"edit{t}" / f"{video}.png", new_path)
    for k, p in enumerate(imgs):
        block.paste(cell(p, fnt), (k * (CELL_W + PAD), label_h))
    return block


def build_sheet(blocks: list[Image.Image], per_row: int) -> Image.Image:
    bw, bh = blocks[0].size
    n_rows = math.ceil(len(blocks) / per_row)
    gap = 3 * PAD
    sheet = Image.new("RGB", (per_row * bw + (per_row + 1) * gap, n_rows * (bh + gap) + gap),
                      (230, 230, 230))
    for i, b in enumerate(blocks):
        r, c = divmod(i, per_row)
        sheet.paste(b, (gap + c * (bw + gap), gap + r * (bh + gap)))
    return sheet


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser()
    r7_root = Path(args.r7_root).expanduser()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r7 = {m["case_id"]: m for m in json.loads(Path(args.r7_manifest).read_text())}
    manifest = json.loads(Path(args.smoke_manifest if args.smoke
                               else args.correct_manifest).read_text())
    per_row = args.per_row or (1 if args.smoke else 2)
    fnt = font()
    by_type = defaultdict(list)
    for m in manifest:
        by_type[m["edit_type"]].append(m)
    for t in sorted(by_type):
        blocks = []
        for m in by_type[t]:
            if m["case_id"] not in r7:
                sys.exit(f"[r43_contact_sheets] {m['case_id']} is not in the R7 manifest")
            new_path = (Path(m["anchor_image"]).expanduser() if args.smoke else
                        Path(args.correct_root).expanduser() / f"edit{t}" / f"{m['video_name']}.png")
            blocks.append(triplet(m, r7[m["case_id"]], r7_root, data_root, new_path, fnt))
        name = f"r43_smoke_sheet_edit{t}.jpg" if args.smoke else f"r43_sheet_correct_edit{t}.jpg"
        path = out_dir / name
        build_sheet(blocks, per_row).save(path, quality=JPEG_QUALITY)
        n_missing = sum(not (Path(m["anchor_image"]).expanduser() if args.smoke else
                             Path(args.correct_root).expanduser() / f"edit{t}" / f"{m['video_name']}.png"
                             ).is_file() for m in by_type[t])
        print(f"[r43_contact_sheets] edit{t}: {len(blocks)} rows ({n_missing} new anchors missing) -> {path}")


if __name__ == "__main__":
    main()
