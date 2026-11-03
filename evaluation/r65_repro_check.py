"""R65: reproduction gate against FiVE's released AnyV2V outputs, plus the smoke contact sheet and timing summary.

For each repro pair (official FiVE ``edit1/0001_bus`` and ``edit2/0081_A_horse``) our ``repro_raw512`` frames are
compared with FiVE's released frames at FiVE's released frame size (our 512x512 frames are resized up with PIL's
default resize), over the first n = min(40, frames in both) frames:

  * mean PSNR and mean LPIPS(alex) between ours and the FiVE release,
  * LPIPS(alex) between the FiVE release and the source frames, as the scale of "a different video".

PASS iff LPIPS(ours, FiVE) <= 0.5 x LPIPS(FiVE, source) on both pairs. The verdict is printed and written to the
CSV; the script itself exits 1 only on missing inputs, so a FAIL still produces the CSV and the sheets that the
manual ``gate`` step reads (FAIL -> stop and decide with the user).

Also writes the contact sheets ``r65_repro_sheet.jpg`` (source | FiVE release | ours, frames 0/10/20/30) and
``r65_smoke_sheet.jpg`` (source | anchor | ours, for the smoke pairs), and prints the smoke manifest's seconds per
video, peak memory and inversion disk.

Usage:
    python -I evaluation/r65_repro_check.py
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import lpips
import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
MAX_FRAMES = 40
PASS_RATIO = 0.5
SHEET_FRAMES = (0, 10, 20, 30)
CELL_WIDTH = 320
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
PAIRS_PER_VIDEO = 415 / 100  # Correct pairs per video, to extrapolate the smoke timing to a full video


def list_frames(directory: Path) -> List[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"frame dir missing: {directory}")
    frames = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not frames:
        raise FileNotFoundError(f"no frames in {directory}")
    return frames


def load(paths: List[Path], size) -> List[Image.Image]:
    """Frames as RGB at ``size`` (PIL default resize when they differ)."""
    out = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        out.append(img if img.size == size else img.resize(size))
    return out


def to_lpips_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None].to(device)


def mean_lpips(model, a: List[Image.Image], b: List[Image.Image], device: torch.device) -> float:
    with torch.no_grad():
        return float(np.mean([model(to_lpips_tensor(x, device), to_lpips_tensor(y, device)).item()
                              for x, y in zip(a, b)]))


def mean_psnr(a: List[Image.Image], b: List[Image.Image]) -> float:
    vals = []
    for x, y in zip(a, b):
        mse = np.mean((np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2)
        vals.append(float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))
    return float(np.mean(vals))


def contact_sheet(rows: List[List[Image.Image]], row_labels: List[str], path: Path) -> None:
    """rows[r][c]: images, each scaled to CELL_WIDTH wide; row label drawn on the first cell."""
    cells = [[im.resize((CELL_WIDTH, round(im.height * CELL_WIDTH / im.width))) for im in row] for row in rows]
    heights = [max(c.height for c in row) for row in cells]
    sheet = Image.new("RGB", (CELL_WIDTH * max(len(r) for r in cells), sum(heights)), "white")
    draw, y = ImageDraw.Draw(sheet), 0
    for row, h, label in zip(cells, heights, row_labels):
        for x, cell in enumerate(row):
            sheet.paste(cell, (x * CELL_WIDTH, y))
        draw.rectangle([0, y, 8 + 7 * len(label), y + 14], fill="black")
        draw.text((4, y + 2), label, fill="white")
        y += h
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


def repro_check(args, model, device) -> bool:
    cases = json.loads(Path(args.repro_cases).read_text())
    rows, sheet_rows, sheet_labels = [], [], []
    for c in cases:
        edit, video = f"edit{c['edit_type']}", c["video_name"]
        release = list_frames(Path(args.release_root).expanduser() / edit / video)
        ours = list_frames(Path(args.repro_raw).expanduser() / edit / video)
        source = list_frames(Path(args.data_root).expanduser() / "images" / video)
        n = min(MAX_FRAMES, len(release), len(ours), len(source))
        size = Image.open(release[0]).size
        rel, our, src = load(release[:n], size), load(ours[:n], size), load(source[:n], size)
        l_ours = mean_lpips(model, our, rel, device)
        l_scale = mean_lpips(model, rel, src, device)
        ok = l_ours <= PASS_RATIO * l_scale
        rows.append({"case_id": c["case_id"], "n_frames": n, "frame_size": f"{size[0]}x{size[1]}",
                     "psnr_ours_vs_five": round(mean_psnr(our, rel), 2),
                     "lpips_ours_vs_five": round(l_ours, 4), "lpips_five_vs_src": round(l_scale, 4),
                     "ratio": round(l_ours / l_scale, 3), "pass": ok})
        idx = [i for i in SHEET_FRAMES if i < n]
        for label, frames in (("source", src), ("FiVE release", rel), ("ours", our)):
            sheet_rows.append([frames[i] for i in idx])
            sheet_labels.append(f"{c['case_id']} {label}")
    contact_sheet(sheet_rows, sheet_labels, Path(args.figures_dir) / "r65_repro_sheet.jpg")

    csv_path = Path(args.csv_dir) / "r65_repro_check.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"[repro] {r['case_id']}: n={r['n_frames']} {r['frame_size']}  PSNR {r['psnr_ours_vs_five']}  "
              f"LPIPS(ours,FiVE) {r['lpips_ours_vs_five']}  LPIPS(FiVE,src) {r['lpips_five_vs_src']}  "
              f"ratio {r['ratio']}  {'PASS' if r['pass'] else 'FAIL'}")
    verdict = all(r["pass"] for r in rows)
    print(f"[repro] gate (ratio <= {PASS_RATIO} on every pair): {'PASS' if verdict else 'FAIL'} -> {csv_path}")
    return verdict


def smoke_sheet(args) -> None:
    cases = json.loads(Path(args.smoke_cases).read_text())
    rows, labels = [], []
    for c in cases:
        edit, video, n = f"edit{c['edit_type']}", c["video_name"], c["n_frames"]
        source = list_frames(Path(args.data_root).expanduser() / "images" / video)
        ours = list_frames(Path(args.smoke_srcres).expanduser() / edit / video)
        size = Image.open(source[0]).size
        idx = [0, len(ours) // 2, len(ours) - 1]
        anchor = Image.open(Path(c["anchor_image"]).expanduser()).convert("RGB").resize(size)
        rows.append([load([source[0]], size)[0], anchor] + load([ours[i] for i in idx], size))
        labels.append(f"{c['case_id']} source | anchor | ours {idx} of {n}")
    contact_sheet(rows, labels, Path(args.figures_dir) / "r65_smoke_sheet.jpg")
    print(f"[smoke] sheet -> {Path(args.figures_dir) / 'r65_smoke_sheet.jpg'}")


def smoke_timing(manifest: Path) -> None:
    """Seconds per video, peak memory and inversion disk from the smoke manifest."""
    per_video: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with open(manifest) as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok":
                print(f"[smoke] {r['case_id']}: status {r['status']}")
                continue
            v = per_video[r["case_id"].split("/")[1]]
            v["frames"] = float(r["n_frames"])
            v["pairs"] += 1
            v["sec_inversion"] += float(r["sec_inversion"])
            v["sec_edit"] += float(r["sec_edit"])
            v["peak_mem_gb"] = max(v["peak_mem_gb"], float(r["peak_mem_gb"]))
            v["inversion_disk_gb"] = max(v["inversion_disk_gb"], float(r["inversion_disk_gb"]))
    for video, v in per_video.items():
        est = v["sec_inversion"] + PAIRS_PER_VIDEO * v["sec_edit"] / v["pairs"]
        print(f"[smoke] {video}: {int(v['frames'])} frames, {int(v['pairs'])} pair(s), inversion {v['sec_inversion']:.0f}s, "
              f"edit {v['sec_edit'] / v['pairs']:.0f}s/pair, peak {v['peak_mem_gb']:.1f} GB, "
              f"inversion disk {v['inversion_disk_gb']:.2f} GB, est. {est:.0f}s for a full video "
              f"({PAIRS_PER_VIDEO:.2f} pairs/video)")


def main() -> None:
    root = Path("~/Data/dataggen/outputs/five_bench/r65_anyv2v").expanduser()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repro_cases", default=str(REPO / "evaluation/r65_cases_repro.json"))
    ap.add_argument("--smoke_cases", default=str(REPO / "evaluation/r65_cases_smoke.json"))
    ap.add_argument("--repro_raw", default=str(root / "repro_raw512"))
    ap.add_argument("--smoke_srcres", default=str(root / "smoke_srcres"))
    ap.add_argument("--smoke_manifest", default=str(root / "smoke_manifest_shard0.csv"))
    ap.add_argument("--release_root", default="~/Data/five_release/AnyV2V")
    ap.add_argument("--data_root", default="~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark")
    ap.add_argument("--csv_dir", default=str(REPO / "evaluation/csv"))
    ap.add_argument("--figures_dir", default=str(REPO / "evaluation/figures"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    try:
        repro_check(args, model, device)
        smoke_sheet(args)
        smoke_timing(Path(args.smoke_manifest))
    except FileNotFoundError as ex:
        print(f"[r65_repro_check] ERROR: {ex}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
