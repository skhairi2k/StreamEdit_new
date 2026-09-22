Part 1 — Directional CLIP, and why raw CLIP is the wrong tool here
The core mismatch
Editing is a relative operation: "start at A, move to B." Raw CLIPScore is an absolute position measure: "how close is this image to this caption?" It never sees the source. That mismatch is the root of everything below.

What raw CLIP is actually summing
Think of the embeddings as having two parts — the scene, and the object being edited:

$$E_{\text{img}}(\text{render}) \approx \underbrace{[\text{grass, trees, camera framing}]}{\text{large}} + \underbrace{[\text{the dog}]}{\text{small}}$$
$$E_{\text{txt}}(\text{trg}) \approx \underbrace{[\text{"grass, trees, camera stationary"}]}{\text{large}} + \underbrace{[\text{"robot dog"}]}{\text{small}}$$

The score $\cos(E_{\text{img}}, E_{\text{txt}}(\text{trg}))$ is dominated by scene · scene. And that term is:

large — it's most of both embeddings,
constant across every render of that clip — every $b$, every arm,
completely uninformative about whether the edit worked.
The term you actually care about, dog · "robot dog", is the small one. So:

$$\text{raw CLIP} = \underbrace{\text{big constant}}{\text{scene}} + \underbrace{\text{small signal}}{\text{the edit}} + \underbrace{\text{whatever else moves}}_{\text{composition, texture, pose}}$$

And your own point is what makes this fatal: CLIP is source-agnostic, so a regenerated-but-valid tree still scores as a tree. That's correct semantics — but it means the scene term stays pinned at essentially the same value across the entire $\rho$ sweep. It contributes most of the magnitude and none of the gradient. Dead weight that drowns the signal.

What directional CLIP does
$$\Delta_{\text{txt}} = E_{\text{txt}}(\text{trg}) - E_{\text{txt}}(\text{src}) \approx [\text{"robot dog"}] - [\text{"golden retriever"}]$$
$$\Delta_{\text{img}} = E_{\text{img}}(\text{render}) - E_{\text{img}}(\text{source}) \approx [\text{what the edit changed}]$$
$$\text{CLIP-D} = \cos(\Delta_{\text{img}}, \Delta_{\text{txt}})$$

The scene cancels on the text side (both captions contain it) and the untouched content cancels on the image side (both frames contain it). The big constant is gone by construction, not by luck. The edit becomes 100% of what's measured.

Why it's the appropriate choice
raw CLIPScore	CLIP-D
uses the source	no — thrown away	yes (free, already on disk)
can tell "good robot dog" from "was already a robot dog"	no	yes
shared scene content	dominates the score	cancels exactly
scale has a meaningful zero	no — is 31.4 good?	yes: +1 as asked, 0 irrelevant, −1 backwards
two different edits are distinguishable	captions are $\cos=0.85$	directions are $\cos=-0.018$
That last row is the measured payoff: dog→robot dog and bus→jeep are essentially orthogonal as directions, while their full captions were nearly the same point.

It's also the field standard for exactly this reason — StyleGAN-NADA introduced it, InstructPix2Pix uses it as the achievement axis.

⚠️ What CLIP-D does not fix
Being honest, since I've already overclaimed twice in this conversation:

It's untested on your data. We don't yet know $\rho(b, \text{CLIP-D})$. That's precisely what the job will tell us. If it also comes back near zero, the problem is deeper than the axis.
Small-$\Delta_{\text{img}}$ degeneracy. At low $\rho$ the render barely differs from the source, so $\Delta_{\text{img}}$ is a short vector whose direction is poorly determined — CLIP-D may be genuinely noisy at the $b{=}2$ end. This is a known failure mode of directional metrics. I should add $|\Delta_{\text{img}}|$ as an output column so we can see whether we're in that regime rather than guessing.
It measures direction, not quality. An artifact-ridden but directionally-correct robot dog scores well. Read it alongside LPIPS/SSIM (which are well-behaved here) — never alone.
It's still CLIP — same blind spots on counting, spatial relations, fine attributes.
Part 2 — Summary of this discussion
What we started with
Building an HTML report: per-video qualitative grids (source / StreamEdit baseline $b{=}2$ / lpips / dino / normals) with each video's own CLIP-vs-LPIPS Pareto point underneath, instead of the 22-clip average.

What the per-video view exposed
You spotted that CLIP moves the wrong way or non-monotonically on many clips, and that on 0057_dog the dino arm looks dramatically better than baseline but scores slightly lower.

What we tested
test	result
Is the harness's CLIP computation correct?	Yes — I reimplemented CLIPScore and reproduced the stored values exactly (6/8 clips to 0.000)
Does CLIP rank renders by edit strength?	No — $\rho(b,\text{CLIP}) = +0.098$, positive on only 11/22 clips
Do renders degrade as $\rho$ rises?	No — $\rho(b,\text{NIQE}) = -0.530$; quality improves on 18/22
Does CLIP track quality?	No — $\rho(\text{NIQE},\text{CLIP}) = -0.093$
How separable are the two captions?	$\cos = 0.846$ mean, up to 0.970
Does the masked variant agree with your eyes on 0057_dog?	Yes — whole-frame −0.34 vs baseline, edit-part +1.28
Two things found
1. A real bug, one clip. 0040_tennis has 85-token prompts. torchmetrics 1.9.0 raw-slices input_ids[:77], dropping the EOT token that CLIP pools its text embedding at. Its embedding collapses to the BOS position and stops depending on the prompt — src_prompt and trg_prompt return a bit-identical 5.528. With correct truncation: 24.165 / 21.727. Every stored clip_similarity_* number for that clip is measuring nothing.

2. A metric-design failure, all clips. The captions differ by one noun in a ~35-token scene description, so the absolute score is mostly shared, constant content. Raw CLIP is not a usable achievement axis on this benchmark.

Two corrections you made to me, both right
"Noise doesn't explain it — it's already averaged over 8 frames." Correct. The confound is render-level, not frame-level: all frames of one render share its texture/composition statistics, so frame-averaging doesn't touch it.
"Degradation shouldn't lower CLIP — a different tree is still a tree." Correct, and NIQE confirmed it: quality improves with $b$. My "scene parts get worse" mechanism was wrong, and there's no systematic direction at all (11/22 split).
What's built and ready
evaluation/r31_html_report.py → 22 per-video grids + per-video Pareto panels → evaluation/figures/r31_report.html
evaluation/r31_clip_directional.py → CLIP-D over R26's 16 reference renders + R31's 4 arms (440 rows); sanity-checked (reverse direction = exactly −1.0, distinct edits orthogonal); fixes the tennis EOT bug
slurm_scripts/five_bench/r31_clip_directional.sh → GPU job, guards against the unmounted-/projects failure mode
Next
Add the $|\Delta_{\text{img}}|$ column (small, worth having before the run).
Submit the job; check $\rho(b, \text{CLIP-D})$ — the honest pass/fail test for whether the new axis fixes anything.
Rebuild the per-video Pareto panels on CLIP-D and re-check 0057_dog.
Worth noting: the R31 plan's Verdict section ("clean negative result") was read off the old CLIP axis. If CLIP-D behaves better, that verdict needs revisiting — none of those arms were ranked on a working achievement metric.

Want me to add the $|\Delta_{\text{img}}|$ column and submit? And should I record this CLIP finding in the R31 plan file, since its Verdict currently rests on the axis we just invalidated?