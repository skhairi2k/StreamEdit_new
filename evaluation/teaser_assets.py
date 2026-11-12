"""Teaser assets: per example, source / weak end / strong end / EDM-over-source / SNaC frames + colorbar.

Examples (chosen by the user):
  cap:   Wan-Edit, edit6/0017_kid-football ("Remove the red cap from the boy."). R71 (2026-11-12): weak/strong = the
         matched-63 uniform start sweep at s=0.92 (Wan-Edit default control) / s=0.978 (s=0.996, the operating-range end,
         regenerates the whole scene -- pitch, goal, another boy -- so it was rejected as a strawman); SNaC = max_s1. Was R58 matched59 s0.92 / s0.978 and max_s5. Field: R58's q (R71 remaps the
         start step from the same q, gate-checked, so the EDM overlay is unchanged). Frames: cap:2,8 (source frames 8, 32).
  kite:  StreamEdit Self-Forcing (VP), edit2/0028_kite-walk ("Replace the person with a Pikachu.").
         Weak/strong = rho=2 (released default) / rho=50 (rho_max), N=20, matched NFE; SNaC = r59_max_full.
         Field: R59 se_max. Cropped (same box for every column) because the subject is ~12% of the frame.
  tiger, bus: first teaser design (Wan-Edit edit2/0007_guitar-violin; StreamEdit SF VP edit1/0001_bus), kept
         so those assets can be regenerated.
Latent frame k <-> pixel frame 4k (Wan VAE, 4x temporal). The EDM is q = S * div, bilinearly upsampled,
drawn over a dimmed greyscale source frame with alpha growing with q.

Usage:
  python -I evaluation/teaser_assets.py pick OUT_DIR cap kite             # contact sheet of every latent frame
  python -I evaluation/teaser_assets.py make OUT_DIR cap:2,8 kite:3,13    # final assets
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

from PIL import Image, ImageDraw

B = "/scratch/PERSO/skhairi/five_bench"
SRC = os.path.expanduser("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark/images")
W, H = 640, 368   # asset size (source frames are 832x480)

WAN = dict(weak=f"{B}/r71_wan_correct/matched63_s0.92", strong=f"{B}/r71_wan_correct/matched63_s0.978",
           ours=f"{B}/r71_wan_correct/max_s1", field=f"{B}/jz/r58_field/wan_max_s5")   # q only (unchanged by R71)
SF_VP = dict(weak=f"{B}/r59_rho_sweep/r59_n20_rho2_vp", strong=f"{B}/r59_rho_sweep/r59_n20_rho50_vp",
             ours=f"{B}/r59_se_arms/r59_max_full/step14", field=f"{B}/r59_field_full/se_max")

# crop = (centre x, centre y, scale) in frame fractions; the box keeps the frame aspect ratio. None = full frame.
EX = {
    "cap":   dict(t=6, v="0017_kid-football", crop=None, **WAN),
    "kite":  dict(t=2, v="0028_kite-walk", crop=(0.58, 0.50, 0.62), **SF_VP),
    "tiger": dict(t=2, v="0007_guitar-violin", crop=None, **WAN),
    "bus":   dict(t=1, v="0001_bus", crop=None, **SF_VP),
}


def frames(d):
    f = sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg")))
    if not f:
        raise FileNotFoundError(d)
    return f


def load(path):
    return Image.open(path).convert("RGB").resize((W, H), Image.LANCZOS)


def crop(img, box):
    """Same-aspect crop around (cx, cy) at `scale` of the frame, resized back to W x H."""
    if box is None:
        return img
    cx, cy, s = box
    w, h = s * W, s * H
    x0 = min(max(cx * W - w / 2, 0), W - w)
    y0 = min(max(cy * H - h / 2, 0), H - h)
    return img.crop((round(x0), round(y0), round(x0 + w), round(y0 + h))).resize((W, H), Image.LANCZOS)


def edm_overlay(src_img, q):
    """Dimmed greyscale source + magma heatmap whose opacity grows with demand q (0 -> transparent)."""
    base = np.asarray(src_img.convert("L").convert("RGB"), dtype=np.float32) / 255.0 * 0.55
    qu = np.asarray(Image.fromarray((q * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0
    heat = matplotlib.colormaps["magma"](qu)[..., :3]
    alpha = np.clip(qu * 1.25, 0, 0.92)[..., None]
    out = base * (1 - alpha) + heat * alpha
    return Image.fromarray((out * 255).clip(0, 255).astype(np.uint8))


def rows_for(name, k):
    e = EX[name]
    t, v = e["t"], e["v"]
    q = np.load(f"{e['field']}/edit{t}/{v}.npz")["q"]
    ours = frames(f"{e['ours']}/edit{t}/{v}")
    p = min(4 * k, len(ours) - 1)
    src = load(frames(f"{SRC}/{v}")[p])
    full = dict(weak=load(frames(f"{e['weak']}/edit{t}/{v}")[p]), strong=load(frames(f"{e['strong']}/edit{t}/{v}")[p]),
                edm=edm_overlay(src, q[k]), ours=load(ours[p]), src=src)
    return {key: crop(img, e["crop"]) for key, img in full.items()}, p, q.shape[0]


def pick(out_dir, names):
    for name in names:
        e = EX[name]
        L = np.load(f"{e['field']}/edit{e['t']}/{e['v']}.npz")["q"].shape[0]
        sw, sh = W // 2, H // 2
        sheet = Image.new("RGB", (sw * L, sh * 5 + 14), "white")
        dr = ImageDraw.Draw(sheet)
        for k in range(L):
            r, p, _ = rows_for(name, k)
            dr.text((k * sw + 3, 1), f"k={k} / frame {p}", fill="black")
            for i, key in enumerate(("src", "weak", "strong", "edm", "ours")):
                sheet.paste(r[key].resize((sw, sh)), (k * sw, 14 + i * sh))
        sheet.save(os.path.join(out_dir, f"pick_{name}.jpg"), quality=80)
        print("wrote", name, "latent frames", L)


def make(out_dir, specs):
    os.makedirs(out_dir, exist_ok=True)
    for spec in specs:
        name, ks = spec.split(":")
        for col, k in enumerate(int(x) for x in ks.split(",")):
            r, p, _ = rows_for(name, k)
            for key in ("weak", "strong", "edm", "ours", "src"):
                r[key].save(os.path.join(out_dir, f"{name}_{key}_{col}.jpg"), quality=92)
            print(f"{name} col {col}: latent {k} -> frame {p}")
    bar = (matplotlib.colormaps["magma"](np.linspace(0, 1, 512))[None, :, :3] * 255).astype(np.uint8)
    Image.fromarray(np.repeat(bar, 16, axis=0)).save(os.path.join(out_dir, "edm_colorbar.png"))


if __name__ == "__main__":
    if sys.argv[1] == "pick":
        pick(sys.argv[2], sys.argv[3:])
    else:
        make(sys.argv[2], sys.argv[3:])
