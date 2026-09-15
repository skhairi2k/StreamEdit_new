# SVG Head Routing for Block-Causal Video Editing — Implementation Brief

**Audience:** Claude Code attached to the Wan (causal-block) editing repo.
**Goal:** Repurpose Sparse VideoGen's (SVG) spatial/temporal head classifier as a
*calibrated routing signal* that labels each attention head as an **appearance
anchor** (spatial) or a **motion/background carrier** (temporal), for
feature-injection video editing. This is an **interpretability + routing** use,
**not** an inference-speed use.

---

## 1. One-paragraph summary

SVG observes that video-DiT attention heads specialize: *spatial heads* attend
within a frame (appearance/structure), *temporal heads* attend to the same
spatial position across frames (motion/temporal consistency). SVG labels each head
online by comparing, on a few sampled query rows, how well a **spatial mask** vs a
**temporal mask** reconstructs full attention (lower MSE wins). We reuse this
*probe* — but run the real forward pass under **full attention** (we only want the
label), define the two masks so they keep an **equal token budget** (this makes the
decision margin calibrated and stable as the number of frames grows), and turn the
result into a signed routing score used to decide where to inject appearance vs
motion features during editing.

---

## 2. Setup and conventions

- **Latent layout after 3D VAE:** `N` frames, `L` tokens per frame
  (`L = H*W`, the flattened per-frame spatial grid). Total sequence `S = N*L`.
- **Flatten convention (must match the model's):** `token(f, p) = f*L + p`, where
  `p = i*W + j` is the row-major position inside a frame. Confirm this against the
  actual Wan patch/flatten order before trusting any mask — a mismatch silently
  breaks the temporal band.
- **Block-causal visibility:** generation proceeds in blocks of `block` frames
  (here `block = 3`, giving `N = 3, 6, 9, ...`). Attention is **bidirectional
  within the current block** and **causal across blocks**:
  `key_block <= query_block`.
- **Grid picture:** think of the sequence as an `N x L` grid (frame × position).
  Spatial = one **row** (a frame). Temporal = one thin **column-band** (a position
  across all frames). SVG's frame-major layout transform is exactly the transpose
  that maps one onto the other.

---

## 3. The two masks (equal-budget — this is the key decision)

```
spatial(q, k)  = (frame(k) == frame(q))                 # own frame, all positions
                 [optionally OR key in `recent_blocks` previous whole blocks]
temporal(q, k) = (|pos(k) - pos(q)| <= w)               # position-band, ALL frames
                 with band half-width  w = L / (2 * N_visible)
both masks are AND-ed with block-causal visibility (key_block <= query_block).
```

**Budgets:**
- spatial keeps `L` tokens (one full frame). **Fixed in N.**
- temporal keeps `~(2w+1)*N = L` tokens. **Also ~L — equal to spatial, by design.**

So the temporal band **shrinks as N grows** (`w = L/(2N)`), narrowing toward the
pure same-position slash. This is faithful, not a hack: the slash is the temporal
head's real signal; the band is only padding to reach the budget, and you need
less padding per frame as frames accumulate. Each probe always fully covers its
head's *characteristic support* (own-frame L positions for spatial; the N-token
slash for temporal), so the shrinking global density (→ 1/N) does **not** wash out
classification even at large N (verified, §6).

### Why equal budget matters (the point the whole strategy hinges on)

- It is **NOT** needed to get the *sign* of the vote right. For a head with
  genuine structure, the winner is decided by *where the attention mass is*, not by
  how many tokens each mask keeps. Unequal budgets classify structured heads
  correctly at moderate N.
- It **IS** needed for a **calibrated margin**: one threshold across the whole
  model, a magnitude that means "how spatial/temporal," and — critically —
  **stability across blocks** so you can tell a genuinely re-tasking head from
  probe drift. With a *growing* temporal budget, the margin of a *fixed* head
  drifts toward "temporal" as N grows, and a real spatial head can even approach
  the flip line at large N. Equal budget removes this N-dependent bias. (Evidence
  in §6.)

---

## 4. The probe (labels only; forward pass stays dense)

Per `(timestep, layer, head)`:

1. Sample `num_sampled_rows` query rows (default 64). Score them against **all**
   keys → the "golden" full-attention output for those rows is **exact** (no
   approximation on the key side; this is why 64 rows suffice).
2. For each candidate mask, apply it with `-inf` **before** softmax (so kept tokens
   are re-normalized to sum to 1 — do **not** zero weights after softmax), then
   `weights @ V`.
3. `mse_s`, `mse_t` = mean-squared error of spatial / temporal output vs golden.
4. Signed margin `m = (mse_s - mse_t) / (mse_s + mse_t) ∈ [-1, 1]`.
   - `m > 0` → temporal, `m < 0` → spatial.
5. **Abstain / three-way route:**
   - `best_rel = sqrt(min(mse_s, mse_t) / mean(golden^2))`.
     If `best_rel > tau_dense` → **DENSE** (diffuse head; neither pattern fits;
     route dense or to both). This is what replaces "budget fairness for the
     vote": diffuse heads announce themselves via high reconstruction error and
     are opted out, rather than being decided by whichever mask happens to be
     denser.
   - else if `m > tau_route` → **TEMPORAL** (motion/background carrier)
   - else if `m < -tau_route` → **SPATIAL** (appearance anchor)
   - else → **MIXED** (route dense / both). Expect real motion heads to cluster
     near the boundary (see §6), so do **not** force a hard label on small-|m|
     heads.

Suggested starting thresholds: `tau_route = 0.15`, `tau_dense = 0.35`. Tune on the
margin histogram (§7).

---

## 5. The N=1 degeneracy, and where to profile

At a single frame (`N=1`) the temporal axis **does not exist**: spatial = the whole
frame, and the temporal band is a subset of it. The label is genuinely undefined —
no probe can fix this. Consequences:

- **Do not read labels from a single frame.** Classify from context that has
  history: later causal blocks, or — in editing — the **source video** context,
  whose frames are present from the very first generated block (temporal signal is
  "borrowed" from the source, so N is effectively never 1).
- **Head type is largely N-invariant past warm-up.** Profile once where the signal
  is reliable, cache the label by `(timestep, layer, head)`, and reuse it for later
  blocks. Do **not** re-vote per-N and latch onto noisy early labels.
- **If you re-check per block, use hysteresis.** Only honor a flip that is (a)
  decisive (margin past threshold by a margin) and (b) persistent (k consecutive
  blocks) and (c) *lowers* the winner's reconstruction error. A head flickering
  spatial↔temporal block-to-block would itself inject temporal inconsistency — the
  exact artifact temporal heads exist to prevent.

---

## 6. Validation evidence (already run; reproduce as sanity checks)

Two synthetic checks established the strategy. Reproduce them before trusting real
Wan numbers.

**(a) Structured heads classify by mass location, not token count.**
With spatial fixed and temporal growing, a temporal head is labeled temporal even
when it has *fewer* tokens than spatial, and a spatial head stays spatial even when
temporal has 3× more tokens. Coverage bias only appears for *diffuse* heads, and
there the winner's relative error is large → caught by `tau_dense`.

**(b) Equal budget gives an N-stable margin; growing budget does not.**
For a realistic *time-local* temporal head (mass on same position in nearby frames):

```
             UNEQUAL budget      EQUAL budget (w = L/2N)
 N=3            +0.24                 +0.27
 N=6            +0.53                 +0.13
 N=9            +0.68                 +0.20
 N=15           +0.78                 +0.10
 N=30           +0.91                 +0.22
 N=60           +0.96                 +0.21
```

Same head, unchanged mass. Unequal budget: margin quadruples with N. Equal budget:
flat. A genuine spatial head under unequal budget drifts from -0.998 (N=3) toward
-0.28 (N=60) — heading for a false flip — while equal budget holds it near -0.95.
**Conclusion:** use equal budget (`w = L/(2N)`) for the routing margin.

---

## 7. Implementation plan (three stages)

**Stage 1 — Equal-budget probe + margin logger.**
- Hook Wan's attention (mirror `svg.models.wan.inference.replace_wan_attention` in
  the SVG repo; the SVG probe params are `num_sampled_rows`, `sample_mse_max_row`,
  `sparsity`). Keep the forward pass **dense**; add the probe as a side computation.
- Build masks per §3 with `w = L/(2*N_visible)`. **At real sizes, do NOT
  materialize an `S×S` boolean mask** (`S = N*L` is large). Use the block-sparse /
  FlashInfer path SVG already uses, or index the sampled rows only. For the
  temporal head, reuse SVG's frame-major transpose so the band becomes contiguous
  blocks.
- Log per `(step, layer, head, block)`: `mse_s`, `mse_t`, `m`, `best_rel`, `label`.

**Stage 2 — Bimodality + stability figures ("clean classification").**
- Histogram of `m` over heads/steps/layers/prompts, computed **only on queries that
  have history** (later blocks or a full-context reference). "Clean classification"
  = bimodal with a gap around 0, not a smear.
- Margin-vs-block curves per `(layer, head)`: flat = stable head; a genuine step =
  real re-tasking. Report a sign-consistency rate across steps/prompts. This is the
  evidence that importing labels to early blocks is legitimate.

**Stage 3 — Intervention test (the actual contribution; do not skip).**
The margin proves a head's attention is *row-shaped or column-shaped* (geometry).
The editing claim is that row = appearance and column = motion/background
(function). **Geometry ≠ function**; the bridge must be shown by intervention, not
correlation:
- Inject appearance features at **spatial-labeled** heads → appearance changes,
  motion/background preserved.
- Inject at **temporal-labeled** heads → motion/background transfers from source,
  appearance preserved.
- **Control:** inject at a **count-matched random** set of heads. The effect must
  track the label, not just "some heads." This control is what makes the result
  publishable.

---

## 8. Pitfalls / do-nots

- **Do not tie the label to sparsification.** Profiling and masking are
  independent; run full attention, probe for labels only.
- **Do not materialize dense `S×S` masks** at real resolution. Block-sparse / row-
  indexed only.
- **Do not equalize per-query token counts inside the MSE.** Budget balance is set
  once via `w = L/(2N)`; the MSE itself is plain.
- **Do not zero attention weights after softmax.** Mask with `-inf` before softmax.
- **Do not classify at N=1**, and do not re-label per-N without hysteresis.
- **Do not let the clean classifier stand in for the intervention.** Bimodality is
  necessary, not sufficient, for the appearance/motion claim.
- **Verify the flatten order** (`token(f,p)=f*L+p`, `p=i*W+j`) against Wan's actual
  patchifier before trusting the temporal band geometry.

---

## 9. Reference implementation

A validated, framework-agnostic reference (numpy) is provided alongside this brief:
`svg_routing.py`. It contains `build_masks(N, L, block, recent_blocks)` (equal-
budget masks with block-causal visibility) and `route_heads(Q, K, V, ...)` (the
sampled-row probe with signed margin + three-way/abstain routing), plus the §6(b)
stability check as `__main__`. Port to torch by replacing `np.` with `torch.` and
swapping the dense masked-softmax for the model's block-sparse attention path; the
mask definitions, the `w = L/(2N)` budget, the `-inf`-before-softmax rule, and the
margin/abstain logic carry over unchanged.

Key signatures:

```python
build_masks(N, L, block=3, recent_blocks=0) -> (spatial_keep, temporal_keep)
route_heads(Q, K, V, N, L, block=3, recent_blocks=0,
            num_sampled_rows=64, tau_route=0.15, tau_dense=0.35, seed=0)
    -> (labels[H], margins[H])   # labels in {SPATIAL, TEMPORAL, MIXED, DENSE}
```