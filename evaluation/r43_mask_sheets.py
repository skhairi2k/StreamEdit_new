"""R43 -- per-pair mask overlay sheets and mask statistics for the FiVE-Bench mask audit.

For each edit type, writes one JPEG sheet ``--out_dir/r43_mask_edit{T}.jpg`` with one cell per
pair: frame 0 (``images/<video>/00001.jpg``) with ``bmasks/<video>/00001.*`` overlaid in
semi-transparent red, captioned ``case_id: source_object → target_object`` (empty object shown
as ``∅``) followed by the instruction, so each cell shows whether the mask covers
``source_object``. A missing frame or mask is drawn as a grey cell saying so.

Also writes ``--stats_csv`` with one row per video used by any pair: mask area fraction on
frame 0, number of empty mask frames, and the mask frame count compared with ``images/``.

The masks are JPEG-compressed, so "black" pixels carry small non-zero values near edges.
Binarization follows the FiVE harness (``evaluation/fivebench/evaluate.py``: ``mask > 0``),
so the overlay, the area and the empty-frame count show what the metrics actually use.
Only top-level frame files are read (``expanded_mask/`` and ``.ipynb_checkpoints/`` are
ignored, as in the harness).

Example
-------
    python evaluation/r43_mask_sheets.py \\
        --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \\
        --out_dir evaluation/figures \\
        --stats_csv evaluation/csv/r43_mask_stats.csv
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

EDIT_TYPES = range(1, 7)
FRAME_EXTS = (".jpg", ".jpeg", ".png")
FIRST_FRAME = "00001"
OVERLAY_RGB = (255, 0, 0)
OVERLAY_ALPHA = 0.45
CELL_W, IMG_H, CAPTION_LINES, FONT_SIZE, PAD = 320, 180, 4, 12, 4
HEADER_LINES = 2          # caption lines reserved for "case_id: source_object → target_object"
EMPTY_OBJECT = "∅"
N_COLS = 10
JPEG_QUALITY = 85


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", type=str,
                        default="~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark")
    parser.add_argument("--out_dir", type=str, default="evaluation/figures")
    parser.add_argument("--stats_csv", type=str, default="evaluation/csv/r43_mask_stats.csv")
    return parser.parse_args()


def font(size: int) -> ImageFont.FreeTypeFont:
    for cand in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def frame_files(folder: Path) -> list[Path]:
    """Top-level frame files of a video folder, sorted by name."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in FRAME_EXTS)


def first_frame(folder: Path) -> Path | None:
    for p in frame_files(folder):
        if p.stem == FIRST_FRAME:
            return p
    return None


def load_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    """Binary mask (harness convention: any non-zero pixel), optionally resized (nearest)."""
    mask = Image.open(path).convert("L")
    if size is not None and mask.size != size:
        mask = mask.resize(size, Image.NEAREST)
    return np.asarray(mask) > 0


def wrap(text: str, fnt: ImageFont.FreeTypeFont, width: int, max_lines: int) -> list[str]:
    """Greedy word wrap to ``width`` pixels; the last line is ellipsized if text overflows."""
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


def render_cell(data_root: Path, case_id: str, video: str, header: str, instruction: str,
                fnt: ImageFont.FreeTypeFont) -> Image.Image:
    line_h = FONT_SIZE + 2
    cell = Image.new("RGB", (CELL_W, IMG_H + CAPTION_LINES * line_h + 2 * PAD), "white")
    draw = ImageDraw.Draw(cell)

    frame_p = first_frame(data_root / "images" / video)
    mask_p = first_frame(data_root / "bmasks" / video)
    if frame_p is None or mask_p is None:
        draw.rectangle([0, 0, CELL_W - 1, IMG_H - 1], fill=(160, 160, 160))
        missing = "frame 0 missing" if frame_p is None else "mask 0 missing"
        draw.text((PAD, IMG_H // 2 - FONT_SIZE), missing, fill="black", font=fnt)
    else:
        frame = Image.open(frame_p).convert("RGB")
        mask = load_mask(mask_p, frame.size)
        rgb = np.asarray(frame, dtype=np.float32)
        rgb[mask] = (1 - OVERLAY_ALPHA) * rgb[mask] + OVERLAY_ALPHA * np.array(OVERLAY_RGB)
        thumb = Image.fromarray(rgb.astype(np.uint8))
        thumb.thumbnail((CELL_W, IMG_H), Image.LANCZOS)
        cell.paste(thumb, ((CELL_W - thumb.width) // 2, (IMG_H - thumb.height) // 2))

    lines = wrap(header, fnt, CELL_W - 2 * PAD, HEADER_LINES)
    lines += wrap(instruction, fnt, CELL_W - 2 * PAD, CAPTION_LINES - len(lines))
    for i, line in enumerate(lines):
        draw.text((PAD, IMG_H + PAD + i * line_h), line, fill="black", font=fnt)
    return cell


def object_header(case_id: str, item: dict) -> str:
    """``case_id: source_object → target_object`` with empty objects shown as ``∅``."""
    src = item["source_object"].strip() or EMPTY_OBJECT
    tgt = item["target_object"].strip() or EMPTY_OBJECT
    return f"{case_id}: {src} → {tgt}"


def build_sheet(cells: list[Image.Image]) -> Image.Image:
    n_rows = math.ceil(len(cells) / N_COLS)
    n_cols = min(N_COLS, len(cells))
    cw, ch = cells[0].size
    sheet = Image.new("RGB", (n_cols * (cw + PAD) + PAD, n_rows * (ch + PAD) + PAD),
                      (230, 230, 230))
    for i, cell in enumerate(cells):
        r, c = divmod(i, N_COLS)
        sheet.paste(cell, (PAD + c * (cw + PAD), PAD + r * (ch + PAD)))
    return sheet


def video_stats(data_root: Path, video: str, case_ids: list[str]) -> dict:
    frames = frame_files(data_root / "images" / video)
    masks = frame_files(data_root / "bmasks" / video)
    mask0 = first_frame(data_root / "bmasks" / video)
    n_empty = sum(not load_mask(p).any() for p in masks)
    return {
        "video_name": video,
        "case_ids": " ".join(case_ids),
        "area_frac_f0": "" if mask0 is None else f"{load_mask(mask0).mean():.4f}",
        "n_mask_frames": len(masks),
        "n_image_frames": len(frames),
        "frame_count_match": len(masks) == len(frames),
        "n_empty_mask_frames": n_empty,
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fnt = font(FONT_SIZE)

    cases_by_video: dict[str, list[str]] = {}
    for t in EDIT_TYPES:
        path = data_root / "edit_prompt" / f"edit{t}_FiVE.json"
        if not path.is_file():
            sys.exit(f"[r43_mask_sheets] missing annotation file: {path}")
        items = json.loads(path.read_text())
        cells = []
        for item in items:
            case_id = f"edit{t}/{item['video_name']}"
            cases_by_video.setdefault(item["video_name"], []).append(case_id)
            cells.append(render_cell(data_root, case_id, item["video_name"],
                                     object_header(case_id, item), item["instruction"], fnt))
        sheet_path = out_dir / f"r43_mask_edit{t}.jpg"
        build_sheet(cells).save(sheet_path, quality=JPEG_QUALITY)
        print(f"[r43_mask_sheets] edit{t}: {len(cells)} cells -> {sheet_path}")

    rows = [video_stats(data_root, v, cases_by_video[v]) for v in sorted(cases_by_video)]
    stats_path = Path(args.stats_csv)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    n_missing = sum(r["area_frac_f0"] == "" for r in rows)
    n_mismatch = sum(not r["frame_count_match"] for r in rows)
    n_empty = sum(r["n_empty_mask_frames"] > 0 for r in rows)
    print(f"[r43_mask_sheets] {len(rows)} videos -> {stats_path}; "
          f"missing mask 0: {n_missing}, frame-count mismatch: {n_mismatch}, "
          f"videos with empty mask frames: {n_empty}")


if __name__ == "__main__":
    main()
