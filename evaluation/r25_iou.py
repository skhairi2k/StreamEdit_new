#!/usr/bin/env python
"""R25 step 1 -- per-case text-grounded object masks and their IoU. No video generation.

Measures, once on clean pixels before any denoising, how much an edit moves the region
the edited content occupies. ONE phrase is grounded per side:

    M_src  = m(src_word, I0)          -- every edit type
    M_edit = m(trg_word, anchor)      -- types 1,2,3,4,5  (swap / colour / material / addition)
           = empty                    -- type 6           (removal: the region is gone)
    IoU    = |M_src & M_edit| / |M_src | M_edit|

WHY ONE PHRASE PER SIDE. The earlier symmetric two-phrase union assumed that a phrase
naming something absent from an image would contribute nothing to that side's union. That
premise is false for GroundingDINO: it never abstains, it returns its best box
unconditionally. Measured on this case set, all 4 groundings fired on 22/22 cases, and the
"absent" phrase scored as high as 0.94 ('A dragon' on the unedited cow) -- higher than most
genuine detections, so no threshold separates them. The union therefore added false
positives on every case and failed outright on the add/remove cases it existed to serve.

type 5 (addition) is now measured the SAME WAY as a swap: M_edit is just the trg_word
grounding on the anchor, e.g. the flamingo alone on 0069_car-turn, not SUV | flamingo.
(Revised 2026-09-01: an earlier version unioned M_edit with M_src so an addition would
read as regional "growth" rather than as a disjoint object -- see the Rejected row in the
plan's Decisions table for why that was reverted.) Only type 6 is still handled by
CONSTRUCTION rather than by detection:

  * type 6 (removal) -- the object is gone, so M_edit is empty and IoU is exactly 0, the
    maximal shape change, routing to tau_max. Nothing is grounded on the anchor at all.
    NOTE this is an ASSUMPTION, not a measurement: the row reads 0 whether or not the
    anchor actually removed the object.

Grounding is GroundingDINO (box, no tuned threshold) refined by SAM2 (extent). Both
checkpoints are read from the local HF cache; run under HF_HUB_OFFLINE=1.

Emits `iou_gt_src` = IoU(M_src, FiVE-Bench GT bmask) per case as the validation column.
The GT mask is used ONLY as a gate, never as one side of the reported IoU -- mixing a GT
mask with a SAM2 mask would bias the ratio by producer. It is meaningful only where
`src_word` names the object FiVE annotated, which is not the case for type 6.

Usage
-----
    HF_HUB_OFFLINE=1 python evaluation/r25_iou.py \\
        --cases evaluation/cases.json \\
        --data_root ~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark \\
        --anchor_root /projects/dataggen/outputs/five_bench/anchors \\
        --box_thr 0.35 \\
        -o evaluation/csv/r25_iou.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

# Edit type whose M_edit is built by construction rather than by grounding.
REMOVAL_EDIT_TYPE: int = 6    # M_edit = empty, IoU = 0, nothing grounded on the anchor


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed every RNG that could touch detection or mask decoding."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resize_nearest(mask: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Resample a bool mask onto the source frame grid with NEAREST.

    The anchor PNG is at the model's working resolution and the source jpg is not, so the
    two sides must be brought onto one grid before they can be intersected. NEAREST keeps
    the mask binary -- any interpolating filter would invent partial-occupancy pixels and
    silently change object extent, which is the quantity being measured.
    """
    h, w = hw
    if mask.shape == (h, w):
        return mask
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
    return np.array(resized) > 127


def load_models(gdino_id: str, sam2_id: str, device: str):
    """Load GroundingDINO + SAM2 once. Both are read from the local HF cache."""
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    gdino_proc = AutoProcessor.from_pretrained(gdino_id)
    gdino = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_id).to(device).eval()
    predictor = SAM2ImagePredictor.from_pretrained(sam2_id, device=device)
    return gdino_proc, gdino, predictor


@torch.no_grad()
def phrase_mask(
    img: Image.Image,
    phrase: str,
    gdino_proc,
    gdino,
    predictor,
    device: str,
    box_thr: float,
) -> Tuple[Optional[np.ndarray], float, Optional[List[float]]]:
    """SAM2 mask for the best GroundingDINO box for `phrase`, or None if it does not fire.

    The phrase is passed verbatim with a trailing period, per the GroundingDINO text
    convention; the processor's BertTokenizer is do_lower_case=True, so casing in
    cases.json is normalized by the tokenizer and needs no handling here.
    """
    inputs = gdino_proc(images=img, text=f"{phrase}.", return_tensors="pt").to(device)
    det = gdino_proc.post_process_grounded_object_detection(
        gdino(**inputs),
        inputs.input_ids,
        threshold=box_thr,
        text_threshold=box_thr,
        target_sizes=[img.size[::-1]],
    )[0]

    if len(det["scores"]) == 0:
        return None, 0.0, None

    i = int(det["scores"].argmax())
    box = det["boxes"][i].cpu().numpy()
    score = float(det["scores"][i])

    predictor.set_image(np.array(img.convert("RGB")))  # SAM2 sets extent, no tuned cutoff
    mask, _, _ = predictor.predict(box=box, multimask_output=False)
    return mask[0].astype(bool), score, [round(float(v), 2) for v in box]


def ground_one(
    img: Image.Image,
    phrase: str,
    hw: Tuple[int, int],
    key: str,
    records: Dict[str, object],
    *,
    gdino_proc,
    gdino,
    predictor,
    device: str,
    box_thr: float,
) -> Optional[np.ndarray]:
    """Ground ONE phrase on ONE image, record the audit fields, return the mask on `hw`."""
    mask, score, box = phrase_mask(
        img, phrase, gdino_proc, gdino, predictor, device, box_thr
    )
    records[f"{key}_fired"] = int(mask is not None)
    records[f"{key}_score"] = round(score, 4)
    records[f"{key}_box"] = "" if box is None else " ".join(str(v) for v in box)
    return None if mask is None else resize_nearest(mask, hw)


def iou_of(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two bool masks on a common grid."""
    return float((a & b).sum()) / float(max((a | b).sum(), 1))


# Tint colours for the audit overlay: the source-side object and the target-side object
# get distinct hues.
COLOR_SRC: Tuple[float, float, float] = (1.0, 0.25, 0.25)   # red
COLOR_TRG: Tuple[float, float, float] = (0.25, 0.55, 1.0)   # blue


def _tint(img: Image.Image, parts: Sequence[Tuple[Optional[np.ndarray], Tuple[float, float, float]]],
          hw: Tuple[int, int], alpha: float = 0.45) -> np.ndarray:
    """Blend each (mask, colour) pair over the image."""
    h, w = hw
    base = np.asarray(img.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    for mask, colour in parts:
        if mask is None:
            continue
        c = np.array(colour, dtype=np.float32)
        base[mask] = (1.0 - alpha) * base[mask] + alpha * c
    return np.clip(base, 0.0, 1.0)


def save_panel(
    out_path: Path,
    src_img: Image.Image,
    anchor_img: Image.Image,
    m_src: Optional[np.ndarray],
    m_trg: Optional[np.ndarray],
    hw: Tuple[int, int],
    row: Dict[str, object],
) -> None:
    """2x2 audit panel: images on top, the two masks below.

    Left is M_src (the src_word grounding). Right is M_edit: the trg_word grounding on
    the anchor for types 1-5; empty by construction for type 6.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = int(row["edit_type"])
    h, w = hw
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.5 * h / max(w, 1) * 2 + 1.6))

    for col, (img, title) in enumerate((
        (src_img, "source $I_0$"),
        (anchor_img, "anchor $I_{0,\\mathrm{edit}}$"),
    )):
        ax = axes[0][col]
        ax.imshow(np.asarray(img.resize((w, h), Image.BILINEAR)))
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    # left: M_src
    ax = axes[1][0]
    ax.imshow(_tint(src_img, [(m_src, COLOR_SRC)], hw))
    ax.set_title("$M_{src}$", fontsize=11)
    ax.set_xlabel(f"src: {row['src_word']!r}  "
                  f"[{'fired ' + str(row['src_on_src_score']) if row['src_on_src_fired'] else 'ABSENT'}]",
                  fontsize=8, family="monospace", loc="left")

    # right: M_edit
    ax = axes[1][1]
    if t == REMOVAL_EDIT_TYPE:
        ax.imshow(_tint(anchor_img, [], hw))
        ax.set_title("$M_{edit}$  (EMPTY — removal, by construction)", fontsize=11)
        ax.set_xlabel("nothing grounded on the anchor for type 6",
                      fontsize=8, family="monospace", loc="left")
    else:
        cap = (f"trg: {row['trg_word']!r}  "
               f"[{'fired ' + str(row['trg_on_edit_score']) if row['trg_on_edit_fired'] else 'ABSENT'}]")
        ax.imshow(_tint(anchor_img, [(m_trg, COLOR_TRG)], hw))
        ax.set_title("$M_{edit}$", fontsize=11)
        ax.set_xlabel(cap, fontsize=8, family="monospace", loc="left")

    for ax in (axes[1][0], axes[1][1]):
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    note = str(row.get("note", "") or "")
    fig.suptitle(
        f"{row['case_id']}  ·  type {row['edit_type']}  ·  "
        f"IoU {row.get('iou', 'nan')}  ·  iou_gt_src {row.get('iou_gt_src', 'nan')}"
        + (f"  ·  {note}" if note else ""),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# per-case measurement
# --------------------------------------------------------------------------------------
def measure_case(
    case: Dict[str, object],
    data_root: Path,
    anchor_root: Path,
    *,
    gdino_proc,
    gdino,
    predictor,
    device: str,
    box_thr: float,
    viz_dir: Optional[Path] = None,
    dump_masks: Optional[Path] = None,
) -> Tuple[Dict[str, object], Optional[str]]:
    """Measure one case. Returns (csv row, hard-failure reason or None)."""
    video_name = str(case["video_name"])
    edit_type = int(case["edit_type"])
    src_word = str(case["src_word"])
    trg_word = str(case["trg_word"])

    src_path = data_root / "images" / video_name / "00001.jpg"
    anchor_path = anchor_root / f"edit{edit_type}" / f"{video_name}.png"
    gt_path = data_root / "bmasks" / video_name / "00001.jpg"
    for p in (src_path, anchor_path, gt_path):
        if not p.exists():
            raise FileNotFoundError(f"{case['case_id']}: missing {p}")

    src_img = Image.open(src_path).convert("RGB")
    anchor_img = Image.open(anchor_path).convert("RGB")
    hw = (src_img.size[1], src_img.size[0])  # (H, W) of the source rendition

    row: Dict[str, object] = {
        "case_id": case["case_id"],
        "video_name": video_name,
        "edit_type": edit_type,
        "src_word": src_word,
        "trg_word": trg_word,
        # audit defaults; the anchor grounding is skipped entirely for removals
        "trg_on_edit_fired": 0, "trg_on_edit_score": 0.0, "trg_on_edit_box": "",
    }
    kw = dict(gdino_proc=gdino_proc, gdino=gdino, predictor=predictor,
              device=device, box_thr=box_thr)

    m_src = ground_one(src_img, src_word, hw, "src_on_src", row, **kw)
    m_trg: Optional[np.ndarray] = None
    failure: Optional[str] = None

    if m_src is None:
        # Nothing to compare against, for any edit type.
        row.update(iou="nan", area_src=0, area_edit="nan", iou_gt_src="nan",
                   note="SRC-NOT-FOUND")
        failure = (f"{case['case_id']}: {src_word!r} did not fire on the SOURCE frame at "
                   f"box_thr={box_thr} -- there is no M_src to compare against")
    else:
        row["area_src"] = int(m_src.sum())

        if edit_type == REMOVAL_EDIT_TYPE:
            # The object is gone: M_edit is empty by construction, IoU is exactly 0.
            # Nothing is grounded on the anchor, so no detector error can enter here.
            m_edit = np.zeros(hw, dtype=bool)
            row["note"] = "TYPE6-EMPTY-BY-CONSTRUCTION"
        else:
            m_trg = ground_one(anchor_img, trg_word, hw, "trg_on_edit", row, **kw)
            if m_trg is None:
                # For a swap or addition this would read IoU = 0 and route to tau_max on
                # a detector miss rather than on the edit. Not a measurement, hard fail.
                row.update(iou="nan", area_edit=0, iou_gt_src="nan", note="TRG-NOT-FOUND")
                failure = (f"{case['case_id']} (type {edit_type}): {trg_word!r} did not "
                           f"fire on the ANCHOR at box_thr={box_thr} -- M_edit is "
                           f"undefined and the case cannot be measured")
                m_edit = None
            else:
                m_edit = m_trg
                row["note"] = ""

        if failure is None:
            row["area_edit"] = int(m_edit.sum())
            row["iou"] = round(iou_of(m_src, m_edit), 6)
            gt = resize_nearest(np.array(Image.open(gt_path).convert("L")) > 127, hw)
            row["iou_gt_src"] = round(iou_of(m_src, gt), 6)

    if dump_masks is not None:
        # M_edit as the IoU actually used it: the trg grounding for types 1-5, empty for
        # type 6. Saving the post-construction masks means the figure draws the same
        # regions the number was computed from, not a re-derivation that could drift.
        if edit_type == REMOVAL_EDIT_TYPE or m_trg is None:
            m_e = np.zeros(hw, dtype=bool)
        else:
            m_e = m_trg
        np.savez_compressed(
            dump_masks / f"{case['case_id']}.npz",
            m_src=np.packbits(m_src if m_src is not None else np.zeros(hw, bool)),
            m_edit=np.packbits(m_e),
            shape=np.array(hw, dtype=np.int32),
            iou=np.array([row.get("iou", float("nan"))], dtype=object),
        )

    if viz_dir is not None:
        save_panel(viz_dir / f"{case['case_id']}.png", src_img, anchor_img,
                   m_src, m_trg, hw, row)

    return row, failure


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def build_fieldnames() -> List[str]:
    return (
        ["case_id", "video_name", "edit_type", "src_word", "trg_word", "iou"]
        + ["area_src", "area_edit", "iou_gt_src"]
        + [f"{k}_{f}" for k in ("src_on_src", "trg_on_edit")
           for f in ("fired", "score", "box")]
        + ["note"]
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", type=Path, default=Path("evaluation/cases.json"),
                   help="Case manifest; src_word/trg_word are the GROUNDING phrases "
                        "(deliberately decoupled from the model's trigger words, which "
                        "the render reads from edit{T}_FiVE.json).")
    p.add_argument("--data_root", type=Path, required=True,
                   help="FiVE-Bench root holding images/ and bmasks/")
    p.add_argument("--anchor_root", type=Path, required=True,
                   help="Anchor root; the anchor is {anchor_root}/edit{T}/{video}.png")
    p.add_argument("-o", "--out", type=Path, default=Path("evaluation/csv/r25_iou.csv"))
    p.add_argument("--box_thr", type=float, default=0.35,
                   help="One threshold for both groundings. Raised from 0.25 to 0.5 on "
                        "2026-08-26: at 0.25 the detector returned a box for every phrase "
                        "on every image, so low-confidence substitutions entered the "
                        "masks. 0.5 was tried and rejected: it discarded a visually "
                        "correct pink-boat anchor scoring 0.45, because detector "
                        "confidence tracks the PHRASING, not the image. A phrase that "
                        "no longer fires is a HARD FAILURE, not a silent empty mask.")
    p.add_argument("--dump_masks", type=Path, default=None,
                   help="If given, save M_src / M_edit per case as "
                        "{dir}/{case_id}.npz (bit-packed bool). Needed by "
                        "r25_figures.py to draw the IoU-overlap column of the "
                        "qualitative grids. Inert by default; the CSV is "
                        "byte-identical either way.")
    p.add_argument("--viz_dir", type=Path, default=None,
                   help="If given, write one 2x2 audit panel per case "
                        "({viz_dir}/{case_id}.png). Off by default; the CSV is "
                        "unaffected either way.")
    p.add_argument("--gdino_id", type=str, default="IDEA-Research/grounding-dino-base")
    p.add_argument("--sam2_id", type=str, default="facebook/sam2-hiera-large")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)

    data_root = args.data_root.expanduser().resolve()
    anchor_root = args.anchor_root.expanduser().resolve()
    viz_dir = args.viz_dir.expanduser() if args.viz_dir else None
    dump_masks = args.dump_masks.expanduser() if args.dump_masks else None
    if dump_masks is not None:
        dump_masks.mkdir(parents=True, exist_ok=True)
    cases = json.loads(args.cases.expanduser().read_text())

    print("=== R25 step 1: single-phrase-per-side IoU ===", flush=True)
    print(f"cases       {args.cases} ({len(cases)} cases)", flush=True)
    print(f"data_root   {data_root}", flush=True)
    print(f"anchor_root {anchor_root}", flush=True)
    print(f"gdino       {args.gdino_id}", flush=True)
    print(f"sam2        {args.sam2_id}", flush=True)
    print(f"box_thr     {args.box_thr}   device {args.device}   seed {args.seed}",
          flush=True)
    print(f"viz_dir     {viz_dir if viz_dir else '(off)'}", flush=True)
    print(f"M_edit      types 1-5: m(trg, anchor) | type 6: empty", flush=True)
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}", flush=True)

    gdino_proc, gdino, predictor = load_models(args.gdino_id, args.sam2_id, args.device)

    rows: List[Dict[str, object]] = []
    failures: List[str] = []
    for case in cases:
        row, failure = measure_case(
            case, data_root, anchor_root,
            gdino_proc=gdino_proc, gdino=gdino, predictor=predictor,
            device=args.device, box_thr=args.box_thr, viz_dir=viz_dir,
            dump_masks=dump_masks,
        )
        rows.append(row)
        if failure is not None:
            failures.append(failure)
            print(f"[FAIL] {failure}", flush=True)
        else:
            print(f"[ok] {row['case_id']:22s} type {row['edit_type']} "
                  f"iou {row['iou']}  iou_gt_src {row['iou_gt_src']}  "
                  f"src {row['src_on_src_score']} trg {row['trg_on_edit_score']}"
                  f"{'  ' + str(row['note']) if row['note'] else ''}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = build_fieldnames()
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"\nwrote {args.out}  ({len(rows)} rows)", flush=True)
    if viz_dir is not None:
        print(f"wrote {len(rows)} audit panels to {viz_dir}", flush=True)

    measured = [float(r["iou"]) for r in rows if r["iou"] not in ("", "nan")]
    if measured:
        print(f"IoU range over {len(measured)} measured cases: "
              f"[{min(measured):.3f}, {max(measured):.3f}]", flush=True)

    print(f"failures: {len(failures)}", flush=True)
    if failures:
        print("\nHard failures -- do NOT proceed to taumap:", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
