"""Assets for the two main-paper qualitative figures (edits = columns, methods = rows, 2 frames per edit).

Every method is sampled at the SAME source frame index (all outputs start at source frame 0 and are consecutive:
Wan-Edit / FlowDirector / UniEdit-Flow / ContextFlow = first 41 frames, AnyV2V = first 32, StreamEdit = full clip).
  text: Wan-Edit figure, frames 4 and 31.  R71 (2026-11-12): SNaC = max_s1 (range-normalized mapping, 63 steps), baseline row =
        Wan-Edit s=0.92 on the matched 63-step grid (main-table setting); was R58 max_s5 / matched59_s0.92. The four picks were
        re-checked on max_s1 (same edits, background kept).
  vp:   StreamEdit (Self-Forcing, visual prompt) figure, frames 4 and 31 (user: same selection as the text figure, 4 and
        32; AnyV2V renders only frames 0-31, so 31 is the closest. Frame 0 is the visual prompt itself; frame 31 is
        AnyV2V's grey frame on 40/415 clips, checked NOT grey on the four picked clips). Baseline row = StreamEdit rho=2, N=20.
        The visual prompt (edited first frame) is saved as <edit>_<video>_prompt.jpg and drawn next to each column
        title (not on the frames).
Frames (user, 2026-11-12): 4 and 28 for every clip of both figures (was 4 and 31; at 31 the butterfly's pale/dark
        wings, faithful to the source shading, read as off at page size; 28 <= 31 keeps AnyV2V in range).
Picks: text (user, 2026-11-11) = edit2/0081 horse->car, edit3/0045 butterfly blue->pink (replaced twirl), edit5/0036 camel+basket,
        edit6/0078 remove toy; all at frames 4/31 (user, 2026-11-11: same frames as the vp figure, no per-clip choice).
        A clip may still override the pair as (edit, video, (f0, f1)).
        vp (draft, pending the user's OK) = edit2/0089 dog->seal (user's choice: it moves; replaces edit2/0075 bicycle->skateboard, static; replaced edit6/0048 hike, kept as fallback; replaced edit1/0001 bus->jeep earlier; replaces edit1/0001
        bus->jeep, where SNaC drops the bus-stop sign; fallbacks edit2/0019 tennis->robot, edit1/0008 snowboarders->robots),
        edit2/0095, edit2/0011, edit5/0007. Column order in the paper: crow, lion, dog, hat (user, 2026-11-11).

Usage: python -I evaluation/qual_assets.py OUT_ROOT     (writes OUT_ROOT/{text,vp}/<edit>_<video>_<row>_<t>.jpg)
"""
import glob
import os
import sys

from PIL import Image

B = "/scratch/PERSO/skhairi/five_bench"
SRC = os.path.expanduser("~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark/images")
ANCH = os.path.expanduser("~/Data/dataggen/outputs/five_bench/anchors_r43_correct")
W, H = 640, 368

FIGS = {
    "text": dict(frames=(4, 28), clips=[(2, "0081_A_horse"), (3, "0045_butterfly"), (5, "0036_camel"),
                                         (6, "0078_A_cat")],
                 rows=[("fd", f"{B}/r64_flowdirector/correct_full"), ("uef", f"{B}/r63_uniedit_flow/full"),
                       ("base", f"{B}/r71_wan_correct/matched63_s0.92"), ("ours", f"{B}/r71_wan_correct/max_s1")]),
    "vp": dict(frames=(4, 28), clips=[(2, "0089_A_dog"), (2, "0095_A_crow"), (2, "0011_lucia"), (5, "0007_guitar-violin")],
               rows=[("anyv2v", f"{B}/r65_anyv2v/correct_srcres"), ("cf", f"{B}/r66_contextflow/correct/n41"),
                     ("base", f"{B}/r59_rho_sweep/r59_n20_rho2_vp"), ("ours", f"{B}/r59_se_arms/r59_max_full/step14")]),
}


def frames(d):
    f = sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg")))
    if not f:
        raise FileNotFoundError(d)
    return f


def main(out_root):
    for fig, cfg in FIGS.items():
        out = os.path.join(out_root, fig)
        os.makedirs(out, exist_ok=True)
        for clip in cfg["clips"]:
            t, v = clip[:2]
            fr = clip[2] if len(clip) > 2 else cfg["frames"]          # optional per-clip frame pair
            rows = [("src", f"{SRC}/{v}")] + [(k, f"{root}/edit{t}/{v}") for k, root in cfg["rows"]]
            for key, d in rows:
                fl = frames(d)
                for i, fi in enumerate(fr):
                    if fi >= len(fl):
                        raise ValueError(f"{d}: frame {fi} >= {len(fl)}")
                    img = Image.open(fl[fi]).convert("RGB").resize((W, H), Image.LANCZOS)
                    img.save(os.path.join(out, f"e{t}_{v}_{key}_{i}.jpg"), quality=90)
            if fig == "vp":
                Image.open(f"{ANCH}/edit{t}/{v}.png").convert("RGB").resize((W, H), Image.LANCZOS) \
                    .save(os.path.join(out, f"e{t}_{v}_prompt.jpg"), quality=90)
            print(f"{fig}: edit{t}/{v} frames {fr} ok")


if __name__ == "__main__":
    main(sys.argv[1])
