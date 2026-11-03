"""R57 debug: why does CF's judge produce NaN/inf probabilities in generate()? Throw-away diagnostic, not part of the suite.

Builds the judge exactly as cf_score.py does (IVEBench evaluator + fps=2.0 metadata fix), with whatever --model_path is
given (the 72B failed; run this with the 7B on one GPU), and wraps ``model.generate`` so that on every call it first
reports
  * whether a plain forward pass over the same inputs gives finite last-token logits (min / max / NaN count),
  * the processor's second_per_grid_ts and input length,
  * what greedy decoding (do_sample=False, 40 tokens) answers,
then runs the original generate call (sampling with the checkpoint's generation_config, as IVEBench does).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cf_score  # noqa: E402
import suite_common as sc  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    sc.add_common_args(p)
    p.add_argument("--ivebench_dir", default="~/Code/IVEBench")
    p.add_argument("--model_path", required=True)
    a = p.parse_args()

    print(f"[cf_debug] transformers {__import__('transformers').__version__}, torch {torch.__version__}, "
          f"gpus {torch.cuda.device_count()}, ffmpeg {cf_score.check_ffmpeg()}", flush=True)
    ev = cf_score.make_evaluator(Path(a.ivebench_dir).expanduser(), Path(a.model_path).expanduser())
    print(f"[cf_debug] model dtype {next(ev.model.parameters()).dtype}, generation_config "
          f"{ev.model.generation_config.to_dict()}", flush=True)

    # per-block probe: first vision block / decoder layer whose output is non-finite, and how large the activations get
    import re
    probe_log = []

    def make_hook(name):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(t) and t.is_floating_point():
                f = t.float()
                probe_log.append((name, str(t.device), float(f.nan_to_num(0.0, 0.0, 0.0).abs().max()),
                                  int((~torch.isfinite(f)).sum())))
        return hook

    for n, mod in ev.model.named_modules():
        if re.search(r"visual\.blocks\.\d+$", n) or re.search(r"language_model\.layers\.\d+$", n):
            mod.register_forward_hook(make_hook(n))

    def report_probe():
        bad = [r for r in probe_log if r[3] > 0]
        print(f"[cf_debug] probe: {len(probe_log)} blocks seen; first non-finite: "
              f"{bad[0] if bad else 'none'} (name, device, max|x|, non-finite count)", flush=True)
        if probe_log:
            step = max(1, len(probe_log) // 12)
            print("[cf_debug] probe max|x| along depth: " +
                  " ".join(f"{r[0].split('.')[-3][0]}{r[0].split('.')[-1]}@{r[1][-1]}:{r[2]:.3g}" for r in probe_log[::step]),
                  flush=True)
        probe_log.clear()

    orig_generate = ev.model.generate

    def probe(**kw):
        inputs = {k: v for k, v in kw.items() if k != "max_new_tokens"}
        print(f"[cf_debug] input_ids {tuple(kw['input_ids'].shape)}, second_per_grid_ts "
              f"{kw.get('second_per_grid_ts').tolist() if kw.get('second_per_grid_ts') is not None else None}, "
              f"video_grid_thw {kw['video_grid_thw'].tolist()}", flush=True)
        with torch.no_grad():
            lg = ev.model(**inputs).logits[0, -1].float()
        report_probe()
        print(f"[cf_debug] forward last-token logits: nan={int(torch.isnan(lg).sum())} inf={int(torch.isinf(lg).sum())} "
              f"min={float(lg[torch.isfinite(lg)].min()) if torch.isfinite(lg).any() else 'n/a'} "
              f"max={float(lg[torch.isfinite(lg)].max()) if torch.isfinite(lg).any() else 'n/a'}", flush=True)
        if not torch.isfinite(lg).all():
            # is the SDPA kernel the culprit? same forward under each attention backend
            from torch.nn.attention import SDPBackend, sdpa_kernel
            for backend in (SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION):
                try:
                    with torch.no_grad(), sdpa_kernel(backend):
                        lg2 = ev.model(**inputs).logits[0, -1].float()
                    probe_log.clear()
                    print(f"[cf_debug] forward under {backend.name}: non-finite logits = "
                          f"{int((~torch.isfinite(lg2)).sum())}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"[cf_debug] forward under {backend.name} failed: {type(e).__name__}: {e}", flush=True)
        try:
            g = orig_generate(**inputs, max_new_tokens=40, do_sample=False)
            txt = ev.processor.batch_decode([g[0][len(inputs["input_ids"][0]):]], skip_special_tokens=True)[0]
            print(f"[cf_debug] greedy answer: {txt!r}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[cf_debug] greedy generate failed: {type(e).__name__}: {e}", flush=True)
        return orig_generate(**kw)

    ev.model.generate = probe

    for q in sc.load_pairs(a.edit_json_dir, a.cases):
        name = f"edit{q['edit_type']}/{q['video_name']}"
        try:
            src = sc.list_images(sc.source_dir(a.data_root, str(q["video_name"])))
            tgt = sc.ensure_render_frames(sc.render_dir(a.root, a.layout, int(q["edit_type"]), str(q["video_name"])))
            with tempfile.TemporaryDirectory(prefix="cf_dbg_") as tmp:
                score, answer = ev.evaluate_video(cf_score.link_frames(src, os.path.join(tmp, "source")),
                                                  cf_score.link_frames(tgt, os.path.join(tmp, "edited")),
                                                  str(q["instruction"]))
            print(f"[cf_debug] {name}: sampled generate OK, score={score}, answer={' '.join(str(answer).split())[:200]!r}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[cf_debug] {name}: sampled generate FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            break      # a device-side assert poisons the CUDA context: nothing after it is meaningful
    return 0


if __name__ == "__main__":
    sys.exit(main())
