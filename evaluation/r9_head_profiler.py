"""R9 -- E0 head-taxonomy existence check: per-head spatial/temporal profiling.

Runs a *degenerate* StreamEdit edit (target == source prompt/trigger) on one
FiVE-Bench case and profiles every self-attention head of the Self-Forcing
generator with the SVG output-reconstruction criterion, computed online inside
a monkey-patched ``attention()`` (wrap-and-restore; no pipeline or model file
is modified).

For every (block, profile step, layer) the SOURCE-branch self-attention call is
intercepted: ~1% of the current block's query rows are sampled (fixed seed), the
full attention output over all visible keys ("golden") is compared against
attention restricted to each candidate key set, and per-head L2 reconstruction
errors are stored. Scalars are collected at ALL blocks x steps {0,7,12,13,14}.

THE KEY SETS
------------
Reconstruction error falls as a key set grows, so the two candidates must offer
the same NUMBER of keys or the comparison measures set size instead of head
type: on random q/k/v, sets of 1560 vs 18 score margin -0.97 for a head with no
specialisation at all. They must also be DISJOINT, or a good temporal score
cannot be attributed to "uses other frames" rather than "uses its own frame".
``build_key_sets`` supplies both properties; ``evaluation/r19_gates.py`` is the
blocking check that they hold. Columns:

``err_spat``
    own frame, minus the query's own token (FRAME_TOKENS - 1 keys).
``err_temp_flat`` / ``err_temp_disk``
    `span = round(L/(N-1))` positions near the query in each of the N-1 OTHER
    frames (~FRAME_TOKENS keys). The two differ only in how those positions are
    picked inside a frame -- 1D (a stripe) vs 2D (isotropic) -- so the band
    shape is settled from data instead of assumed.
``golden_sq``
    mean(golden^2) per head; the scale for the dense-abstain test
    best_rel = sqrt(min(err)/golden_sq), which routes heads that neither pattern
    fits to DENSE instead of forcing them to a side.
``mass_spat`` / ``mass_temp`` (+ log means)
    attention mass on each set -- a budget-free cross-check on the MSE margin.
``n_spat`` / ``n_temp`` / ``span``
    achieved per-block invariants, so the equal-budget claim is checkable after
    the fact rather than taken on trust.

CAVEAT ON READING THE MARGIN
----------------------------
The margin is NOT monotone in temporal mass. Because the restricted softmax
renormalises, a head with only a few percent of its mass on the temporal line
has that mass over-weighted by the temporal set and lands FURTHER from golden
than the spatial set does -- it reads strongly NEGATIVE. Measured on synthetic
heads: 2.6% mass -> margin -0.89, 17% -> -0.60, 77% -> +0.79. So a strongly
negative margin means "spatial OR weakly temporal", which is what best_rel and
the DENSE bucket exist to separate.

All quantitative columns use the same ``N_SAMPLE_ROWS`` random query rows -- 47
rows spanning ~46 distinct in-frame positions over ~29 of the 52 grid columns.

ATTENTION DUMPS
---------------
``--map_mode marginal`` (default, ~5 MB/video) stores a per-frame marginal for
every sampled row plus 30x52 position marginals for the first few -- the full
joint over 47 rows would be ~950 MB. ``--dump_maps`` additionally stores the
legacy full joint for ``MAP_ROWS`` (~81 MB) and is only needed to regenerate
``figures/r9_maps``. Neither flag affects any number above.

Outputs ``{out_root}/{case_id}/r9_scalars.npz``.
"""

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

# Default location of the Self-Forcing StreamEdit build, relative to the repo root.
_DEFAULT_SF_ROOT = Path(__file__).resolve().parent.parent / "Self-Forcing_StreamEdit"

FRAME_TOKENS = 1560            # latent tokens per frame at 480x832 (30 x 52 after 2x2 patch)
MAP_HW = (30, 52)              # (h_l, w_l) token grid, for reshaping dumped prob rows
PROFILE_STEPS = (0, 7, 12, 13, 14)
N_SAMPLE_ROWS = 47             # ~1% of the 4680 current-block query tokens
MAP_ROWS = (100, 1660, 3220, 4000)  # legacy fixed rows for the FULL raw-map dump.
# NB: these four cover only TWO distinct in-frame positions (100 and 880), both at
# grid column 48 of 52 -- the right edge. Fine for eyeballing a picture, unusable as
# a statistical sample. Everything quantitative uses the N_SAMPLE_ROWS random rows.
N_POS_MARGINAL_ROWS = 4        # sampled rows kept as full 30x52 position marginals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=str, required=True,
                        help="case_id from --cases_json (e.g. 0001_bus)")
    parser.add_argument("--cases_json", type=str,
                        default=str(Path(__file__).resolve().parent / "cases.json"))
    parser.add_argument("--data_root", type=str,
                        default="~/Data/FiVE-Fine-Grained-Video-Editing-Benchmark")
    parser.add_argument("--out_root", type=str,
                        default="outputs/five_bench/r9_head_profile")
    parser.add_argument("--dump_maps", action="store_true", default=False,
                        help="Also save the FULL all-head attention-prob rows for the "
                             "legacy MAP_ROWS at the classification point "
                             "(~81 MB/video; only needed to regenerate figures/r9_maps)")
    parser.add_argument("--map_mode", type=str, default="marginal",
                        choices=["none", "marginal"],
                        help="Compact attention dump at the classification point. "
                             "'marginal' (default, ~5 MB/video) keeps a frame marginal "
                             "for ALL sampled rows plus 30x52 position marginals for the "
                             "first few -- the full joint over 47 rows would be ~950 MB. "
                             "'none' stores scalars only (~350 KB/video). The quantitative "
                             "columns do not depend on this flag.")

    # grounding / boosting hyper-parameters (mirror run_fivebench.py defaults)
    parser.add_argument("--fg_boost_factor", type=float, default=4.0)
    parser.add_argument("--blend_power", type=float, default=2.0)

    # sampling / model settings (mirror run_fivebench.py; consumed by load_pipe)
    parser.add_argument("--step", type=int, default=15)
    parser.add_argument("--flow_shift", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sink_size", type=int, default=0)
    parser.add_argument("--config_path", type=str, default="configs/self_forcing_dmd.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/self_forcing_dmd.pt")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--sf_root", type=str, default=str(_DEFAULT_SF_ROOT))
    return parser.parse_args()


def build_key_sets(row_frame: torch.Tensor, row_pos: torch.Tensor, frames_vis: int,
                   shape: str = "flat", grid_hw=MAP_HW):
    """Disjoint, budget-matched spatial/temporal key sets for the SVG probe.

    The probe compares two hypotheses for what a head relies on. They must offer
    the same NUMBER of keys -- restricted-softmax reconstruction error falls as a
    key set grows, so an unequal budget makes the comparison measure set size
    rather than head type (on random q/k/v, sets of 1560 vs 18 score margin -0.97
    for a head with no specialisation at all). They must also be DISJOINT: if the
    temporal set contains part of the query's own frame, a good temporal score no
    longer distinguishes "uses other frames" from "uses its own frame".

        spatial  = the query's own frame, minus the query's own token
                   -> FRAME_TOKENS - 1 keys, constant in N
        temporal = `span` positions near the query, in each of the N-1 OTHER
                   frames, where span = round(FRAME_TOKENS / (N-1))
                   -> span * (N-1) ~= FRAME_TOKENS keys

    The temporal set is a constant-width tube: it does NOT taper with temporal
    distance, so frame f-1 and frame f-17 are treated alike. What shrinks as N
    grows is `span`, purely because the fixed budget is shared over more frames.

    ``shape`` picks how the `span` positions are chosen inside a frame:
      'flat' -- nearest by |p - row_pos| (1D, the svg_routing_strat.md spec).
                In the 30x52 grid this is a horizontal stripe; it never reaches
                the vertical neighbour, which sits 52 away.
      'disk' -- nearest by 2D grid distance. Isotropic, the honest reading of
                "near this spot". SVG's flat band follows from block-sparse
                memory layout, a constraint this probe does not have.

    Both are selected BY RANK (stable argsort, ties broken by index), never by a
    distance threshold -- a threshold over-selects on ties and silently inflates
    the budget. Ranking also handles frame borders for free: a query near an edge
    still gets exactly `span` positions, extending inward.

    Args:
        row_frame: [P] absolute frame index of each sampled query row.
        row_pos:   [P] position within the frame, 0..FRAME_TOKENS-1.
        frames_vis: N, visible frames in the cache at this capture.
        shape: 'flat' or 'disk'.
    Returns:
        (spatial, temporal): bool tensors [P, frames_vis * FRAME_TOKENS].
    """
    if frames_vis < 2:
        raise ValueError(
            f"the temporal axis does not exist at frames_vis={frames_vis}; "
            "no probe can assign a label from a single frame")
    if shape not in ("flat", "disk"):
        raise ValueError(f"shape must be 'flat' or 'disk', got {shape!r}")

    dev = row_frame.device
    s_k = frames_vis * FRAME_TOKENS
    key_idx = torch.arange(s_k, device=dev)
    key_frame = (key_idx // FRAME_TOKENS)[None, :]                    # [1, Sk]
    key_pos = key_idx % FRAME_TOKENS                                  # [Sk]
    own = key_frame == row_frame[:, None]                             # [P, Sk]

    # --- spatial: own frame, minus the query's own token -------------------
    q_abs = row_frame * FRAME_TOKENS + row_pos
    spatial = own & (key_idx[None, :] != q_abs[:, None])

    # --- temporal: `span` nearest positions, in every OTHER frame ----------
    span = int(round(FRAME_TOKENS / (frames_vis - 1)))
    span = max(1, min(span, FRAME_TOKENS))
    pos = torch.arange(FRAME_TOKENS, device=dev)[None, :]              # [1, L]
    if shape == "flat":
        dist = (pos - row_pos[:, None]).abs().float()                  # [P, L]
    else:
        gw = int(grid_hw[1])
        dist = (((pos // gw) - (row_pos[:, None] // gw)) ** 2
                + ((pos % gw) - (row_pos[:, None] % gw)) ** 2).float()
    # Stable sort => ties broken by position index => reproducible across runs.
    keep_pos = torch.argsort(dist, dim=1, stable=True)[:, :span]       # [P, span]
    sel_pos = torch.zeros_like(dist, dtype=torch.bool)
    sel_pos.scatter_(1, keep_pos, True)
    temporal = sel_pos[:, key_pos] & ~own

    return spatial, temporal


class HeadProfiler:
    """Collects per-head SVG reconstruction errors from patched attention calls.

    Layer identity is explicit: each ``CausalWanSelfAttention.forward`` is
    wrapped to scope the layer index, and only the FIRST ``attention()`` call
    inside that scope is captured -- the source call in the bridge path
    (``x_list[0]``) and the dual call (source at batch index 0) in the no-mask
    fallback. Later calls in the same scope (target-branch calls, whose key
    lists can have arbitrary length once gathered background source KV is
    appended at t < t*) are ignored by construction. Cross-attention is
    unaffected: it calls ``attention`` from ``wan.modules.model``'s namespace,
    which is not patched.
    """

    def __init__(self, denoise_values: np.ndarray, n_blocks: int,
                 n_layers: int, n_heads: int, dump_maps: bool,
                 map_mode: str = "marginal"):
        self.map_mode = map_mode
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.step_of_t = {int(v): i for i, v in enumerate(denoise_values)}
        self.step_col = {s: j for j, s in enumerate(PROFILE_STEPS)}
        shape = (n_blocks, len(PROFILE_STEPS), n_layers, n_heads)
        # Reconstruction error under each candidate key set. The two temporal
        # columns differ only in the band SHAPE (see build_key_sets); the budget
        # is matched to the spatial set in both cases.
        self.err_spat = np.full(shape, np.nan, dtype=np.float64)
        self.err_temp_flat = np.full(shape, np.nan, dtype=np.float64)
        self.err_temp_disk = np.full(shape, np.nan, dtype=np.float64)
        # mean(golden^2) per head -- the scale for the dense-abstain test.
        self.golden_sq = np.full(shape, np.nan, dtype=np.float64)
        # Attention MASS on each candidate set, over the same 47 rows as the errors.
        # Mass needs no budget correction: dividing by |set|/S_vis (done in analysis,
        # both sizes are recoverable from svis) makes the two directly comparable.
        # Arithmetic and geometric means are both kept -- enrichment is a ratio, so
        # the geometric mean is the appropriate average, but the arithmetic one says
        # how much attention actually lands there.
        self.mass_spat = np.full(shape, np.nan, dtype=np.float64)
        self.mass_temp = np.full(shape, np.nan, dtype=np.float64)
        self.logmass_spat = np.full(shape, np.nan, dtype=np.float64)
        self.logmass_temp = np.full(shape, np.nan, dtype=np.float64)
        self.svis = np.zeros(n_blocks, dtype=np.int64)
        # Recorded invariants per block: achieved key counts and the band width.
        self.n_spat = np.zeros(n_blocks, dtype=np.int64)
        self.n_temp = np.zeros(n_blocks, dtype=np.int64)
        self.span = np.zeros(n_blocks, dtype=np.int64)
        self.dump_maps = dump_maps
        self.maps = None            # [L, H, len(MAP_ROWS), S_vis] fp16, lazily allocated
        # Marginals over the FULL 47-row sample (see --map_mode): the frame marginal
        # is what the temporal axis needs and costs ~1/90th of a full map dump.
        self.frame_marg = None      # [L, H, N_SAMPLE_ROWS, frames_vis] fp16
        self.pos_marg = None        # [L, H, len(MAP_ROWS), FRAME_TOKENS] fp16
        self._rows_cache = {}
        # per-forward state
        self.active = False
        self.block = -1
        self.step_idx = -1
        self.current_layer = -1
        self.capture_pending = False
        self.captured_count = 0
        self._warned = False

    # ---- forward boundaries (set from the wrapped generator.forward) ----

    def begin_forward(self, block: int, t_value: float) -> None:
        idx = self.step_of_t.get(int(round(t_value)))
        self.active = idx is not None and idx in self.step_col and 0 <= block < self.n_blocks
        self.block = block
        self.step_idx = idx if idx is not None else -1
        self.current_layer = -1
        self.capture_pending = False
        self.captured_count = 0

    def end_forward(self) -> None:
        if self.active and self.captured_count not in (0, self.n_layers) and not self._warned:
            warnings.warn(
                f"[r9] captured {self.captured_count} layers (expected 0 or "
                f"{self.n_layers}) -- capture-scope assumption violated; "
                f"scalars for this point may be incomplete")
            self._warned = True
        self.active = False

    # ---- self-attention scope (set from each wrapped self_attn.forward) ----

    def enter_selfattn(self, layer: int) -> None:
        if self.active:
            self.current_layer = layer
            self.capture_pending = True

    def exit_selfattn(self) -> None:
        self.capture_pending = False
        self.current_layer = -1

    # ---- patched-attention entry point ----

    def on_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        if not (self.active and self.capture_pending):
            return
        self.capture_pending = False              # first call in scope only
        # source q/k are cache-shaped: visual token counts multiple of FRAME_TOKENS
        if q.shape[1] % FRAME_TOKENS != 0 or k.shape[1] % FRAME_TOKENS != 0:
            if not self._warned:
                warnings.warn(f"[r9] unexpected first-call shapes q={tuple(q.shape)} "
                              f"k={tuple(k.shape)} at layer {self.current_layer}; skipped")
                self._warned = True
            return
        self._capture(self.current_layer, q[0], k[0], v[0])
        self.captured_count += 1

    # ---- criterion ----

    def _rows(self, s_q: int) -> np.ndarray:
        if s_q not in self._rows_cache:
            rng = np.random.default_rng(0)
            self._rows_cache[s_q] = np.sort(
                rng.choice(s_q, size=min(N_SAMPLE_ROWS, s_q), replace=False))
        return self._rows_cache[s_q]

    @torch.no_grad()
    def _capture(self, layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        if not (0 <= layer < self.n_layers):
            return
        s_q, s_k = q.shape[0], k.shape[0]
        rows = self._rows(s_q)
        dev = q.device
        rows_t = torch.as_tensor(rows, device=dev)

        qs = q[rows_t].float()                    # [P, H, D]
        kf = k.float()                            # [Sk, H, D]
        vf = v.float()
        logits = torch.einsum("phd,khd->hpk", qs, kf) / math.sqrt(q.shape[-1])
        probs = logits.softmax(-1)                # [H, P, Sk]
        o_full = torch.einsum("hpk,khd->hpd", probs, vf)

        frames_vis = s_k // FRAME_TOKENS
        q_frames = s_q // FRAME_TOKENS
        row_frame = torch.as_tensor(frames_vis - q_frames + rows // FRAME_TOKENS, device=dev)
        row_pos = torch.as_tensor(rows % FRAME_TOKENS, device=dev)
        if frames_vis < 2:
            # The temporal axis does not exist; no probe can assign a label here.
            return

        # Disjoint, budget-matched key sets (see build_key_sets). The temporal set
        # is computed under BOTH band shapes so the choice can be settled from the
        # data rather than assumed.
        spat_mask, temp_flat = build_key_sets(row_frame, row_pos, frames_vis, "flat")
        _, temp_disk = build_key_sets(row_frame, row_pos, frames_vis, "disk")

        def restricted(mask: torch.Tensor) -> torch.Tensor:
            # -inf BEFORE softmax so the kept keys renormalise to sum to 1. Zeroing
            # weights afterwards would leave the output scaled by the retained mass.
            masked = logits.masked_fill(~mask[None], float("-inf"))
            return torch.einsum("hpk,khd->hpd", masked.softmax(-1), vf)

        err_s = ((restricted(spat_mask) - o_full) ** 2).mean(dim=(1, 2))       # [H]
        err_tf = ((restricted(temp_flat) - o_full) ** 2).mean(dim=(1, 2))
        err_td = ((restricted(temp_disk) - o_full) ** 2).mean(dim=(1, 2))
        # Scale for the dense-abstain test: best_rel = sqrt(min(err)/golden_sq) is
        # how badly the BETTER mask still misses, in units of the output itself.
        # Stored raw so the analysis can form it for either band shape.
        golden_sq = (o_full ** 2).mean(dim=(1, 2))                             # [H]

        # ---- attention mass on each set (free: probs/masks already computed) ----
        # Budget-free cross-check on the MSE margin: mass needs no size correction
        # once divided by |set|/S_vis, which the analysis does from svis + n_*.
        mass_s = (probs * spat_mask[None]).sum(-1)                             # [H, P]
        mass_t = (probs * temp_flat[None]).sum(-1)

        col = self.step_col[self.step_idx]
        self.err_spat[self.block, col, layer] = err_s.cpu().numpy()
        self.err_temp_flat[self.block, col, layer] = err_tf.cpu().numpy()
        self.err_temp_disk[self.block, col, layer] = err_td.cpu().numpy()
        self.golden_sq[self.block, col, layer] = golden_sq.cpu().numpy()
        self.mass_spat[self.block, col, layer] = mass_s.mean(1).cpu().numpy()
        self.mass_temp[self.block, col, layer] = mass_t.mean(1).cpu().numpy()
        self.logmass_spat[self.block, col, layer] = \
            mass_s.clamp_min(1e-12).log2().mean(1).cpu().numpy()
        self.logmass_temp[self.block, col, layer] = \
            mass_t.clamp_min(1e-12).log2().mean(1).cpu().numpy()
        self.svis[self.block] = s_k
        # Recorded invariants: the equal-budget claim must be checkable after the
        # fact, not taken on trust.
        self.n_spat[self.block] = int(spat_mask.sum(1).float().mean())
        self.n_temp[self.block] = int(temp_flat.sum(1).float().mean())
        self.span[self.block] = int(round(FRAME_TOKENS / (frames_vis - 1)))

        # ---- compact marginals over ALL sampled rows, at the classification point ----
        # A full [L, H, 47, S_vis] dump is ~950 MB/video. The frame marginal keeps the
        # temporal structure for every sampled row at ~1/90th the size; the position
        # marginal (the 30x52 picture) is kept only for the few MAP_ROWS used in figures.
        if (self.map_mode == "marginal" and self.block == self.n_blocks - 1
                and self.step_idx == PROFILE_STEPS[-1]):
            n_pos = min(N_POS_MARGINAL_ROWS, len(rows))
            if self.frame_marg is None:
                self.frame_marg = np.zeros(
                    (self.n_layers, self.n_heads, len(rows), frames_vis), dtype=np.float16)
                self.pos_marg = np.zeros(
                    (self.n_layers, self.n_heads, n_pos, FRAME_TOKENS), dtype=np.float16)
                self.marg_rows = rows.copy()
            pf = probs.reshape(probs.shape[0], probs.shape[1], frames_vis, FRAME_TOKENS)
            self.frame_marg[layer] = pf.sum(-1).cpu().numpy().astype(np.float16)
            # Position marginal for the first few SAMPLED rows -- not the legacy fixed
            # MAP_ROWS, which are not in the random sample and which turned out to sit
            # at only two grid positions, both in column 48 of 52 (the frame edge).
            self.pos_marg[layer] = pf[:, :n_pos].sum(2).cpu().numpy().astype(np.float16)

        # raw-map dump at the classification point (last block, last profile step)
        if (self.dump_maps and self.block == self.n_blocks - 1
                and self.step_idx == PROFILE_STEPS[-1]):
            map_rows = [r for r in MAP_ROWS if r < s_q]
            mr = torch.as_tensor(map_rows, device=dev)
            ql = q[mr].float()
            l2 = torch.einsum("phd,khd->hpk", ql, kf) / math.sqrt(q.shape[-1])
            if self.maps is None:
                self.maps = np.zeros(
                    (self.n_layers, self.n_heads, len(map_rows), s_k), dtype=np.float16)
            self.maps[layer] = l2.softmax(-1).cpu().numpy().astype(np.float16)  # [H, P, Sk]

    # ---- output ----

    def save(self, out_path: Path, meta: dict) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(
            err_spat=self.err_spat,
            err_temp_flat=self.err_temp_flat,
            err_temp_disk=self.err_temp_disk,
            golden_sq=self.golden_sq,
            n_spat=self.n_spat,
            n_temp=self.n_temp,
            span=self.span,
            mass_spat=self.mass_spat,
            mass_temp=self.mass_temp,
            logmass_spat=self.logmass_spat,
            logmass_temp=self.logmass_temp,
            n_sample_rows=np.asarray(N_SAMPLE_ROWS),
            svis=self.svis,
            profile_steps=np.asarray(PROFILE_STEPS),
            map_rows=np.asarray(MAP_ROWS),
            map_hw=np.asarray(MAP_HW),
            frame_tokens=np.asarray(FRAME_TOKENS),
            meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
        )
        if self.maps is not None:
            arrays["maps"] = self.maps
        if self.frame_marg is not None:
            arrays["frame_marg"] = self.frame_marg
            arrays["pos_marg"] = self.pos_marg
            arrays["marg_rows"] = self.marg_rows
        np.savez_compressed(out_path, **arrays)
        n_pts = int(np.isfinite(self.err_spat[..., 0, 0]).sum())
        print(f"[r9] saved {out_path} ({n_pts} block-step points, "
              f"map_mode={self.map_mode}, "
              f"maps={'yes' if self.maps is not None else 'no'}, "
              f"marginals={'yes' if self.frame_marg is not None else 'no'})", flush=True)


def main() -> None:
    args = parse_args()

    sf_root = Path(args.sf_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    cases_json = Path(args.cases_json).expanduser().resolve()

    # The SF build imports as a top-level package and reads configs relatively.
    sys.path.insert(0, str(sf_root))
    os.chdir(sf_root)

    from inference_edit_streamedit import load_pipe, find_closest_num_frame
    from diffusers.utils import load_video
    import wan.modules.causal_model as cm

    cases = json.loads(cases_json.read_text())
    matches = [c for c in cases if c["case_id"] == args.case]
    if not matches:
        raise SystemExit(f"[r9] case {args.case!r} not in {cases_json} "
                         f"(available: {[c['case_id'] for c in cases]})")
    case = matches[0]

    pipeline, low_memory, device, _ = load_pipe(args)

    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    src_video = load_video(str(data_root / case["src_video"]))
    new_len = find_closest_num_frame(len(src_video))
    if not new_len:
        raise SystemExit(f"[r9] video too short: {len(src_video)} frames")
    # single-window inference caps the KV cache at max_attention_size = 21
    # latent frames (local_attn_size=-1); longer videos overflow the cache
    new_len = min(new_len, 81)                    # 81 pixel = 21 latent frames
    src_video = src_video[:new_len]
    src_tensor = torch.stack([transform(img) for img in src_video], dim=1).unsqueeze(0)
    video_latents = pipeline.vae.encode_to_latent(
        src_tensor.to(device=device, dtype=torch.bfloat16)
    ).to(device=device, dtype=torch.bfloat16)
    pipeline.vae.model.clear_cache()

    n_lat = video_latents.shape[1]
    n_per_block = int(getattr(pipeline, "num_frame_per_block", 3))
    if n_lat % n_per_block != 0:
        raise SystemExit(f"[r9] latent frames {n_lat} not divisible by block size {n_per_block}")
    n_blocks = n_lat // n_per_block

    denoise_values = np.asarray([int(v) for v in pipeline.denoising_step_list])
    prof = HeadProfiler(
        denoise_values=denoise_values,
        n_blocks=n_blocks,
        n_layers=int(pipeline.generator.model.num_layers),
        n_heads=int(pipeline.generator.model.num_heads),
        dump_maps=args.dump_maps,
        map_mode=args.map_mode,
    )
    print(f"[r9] case={args.case} latent_frames={n_lat} blocks={n_blocks} "
          f"steps={denoise_values.tolist()}", flush=True)

    # ---- wrap-and-restore patches ----
    orig_attention = cm.attention
    gen = pipeline.generator
    orig_forward = gen.forward
    block_tokens = n_per_block * FRAME_TOKENS

    def patched_attention(q, k, v, *a, **kw):
        prof.on_attention(q, k, v)
        return orig_attention(q, k, v, *a, **kw)

    def patched_forward(*a, **kw):
        ts = kw.get("timestep")
        cs = int(kw.get("current_start", 0) or 0)
        t_val = float(ts.flatten()[0].item()) if torch.is_tensor(ts) else float(ts or 0)
        prof.begin_forward(cs // block_tokens, t_val)
        try:
            return orig_forward(*a, **kw)
        finally:
            prof.end_forward()

    # explicit layer scoping: wrap every block's self_attn.forward so the
    # patched attention() knows which layer it is in and captures only the
    # first call per scope (the source call in both bridge and fallback modes)
    blocks = gen.model.blocks
    orig_selfattn_fwds = [blk.self_attn.forward for blk in blocks]

    def make_scoped(fwd, layer_idx):
        def scoped(*a, **kw):
            prof.enter_selfattn(layer_idx)
            try:
                return fwd(*a, **kw)
            finally:
                prof.exit_selfattn()
        return scoped

    cm.attention = patched_attention
    gen.forward = patched_forward
    for i, blk in enumerate(blocks):
        blk.self_attn.forward = make_scoped(orig_selfattn_fwds[i], i)
    try:
        pipeline.rollout_inference(
            src_video=video_latents,
            src_prompts=case["src_prompt"],
            trg_prompts=case["src_prompt"],       # degenerate edit: target == source
            src_trigger_words=case["src_word"],
            trg_trigger_words=case["src_word"],
            return_latents=True,
            wo_video_decode=True,                 # no decode needed for profiling
            profile=False,
            low_memory=low_memory,
            fg_boost_factor=args.fg_boost_factor,
            blend_power=args.blend_power,
            rollout_chunk_size=-1,                # single-window inference() path
        )
    finally:
        cm.attention = orig_attention
        gen.forward = orig_forward
        for i, blk in enumerate(blocks):
            blk.self_attn.forward = orig_selfattn_fwds[i]

    prof.save(
        out_root / args.case / "r9_scalars.npz",
        meta=dict(case_id=args.case, video=case["src_video"], n_latent_frames=n_lat,
                  n_blocks=n_blocks, step=args.step, seed=args.seed,
                  fg_boost_factor=args.fg_boost_factor, blend_power=args.blend_power,
                  sample_rows=N_SAMPLE_ROWS, profile_steps=list(PROFILE_STEPS)),
    )


if __name__ == "__main__":
    main()
