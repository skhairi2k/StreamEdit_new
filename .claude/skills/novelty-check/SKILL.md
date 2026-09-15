---
name: novelty-check
description: Critically assess a deep-learning research idea against existing literature — what is genuinely novel, what is recombination, and what the strongest objection would be. Use this skill whenever the user wants a critique, a reality check, a "be honest with me", a reviewer's-eye view, or an assessment of whether an idea is worth pursuing or how it's positioned against prior work. Trigger it for any request to evaluate, stress-test, poke holes in, or judge the contribution of the current proposal — and proactively when an idea seems to be drifting toward something already well-explored, since catching that early saves the user the most time.
---

# Novelty and Critical Analysis

The job is to give the user the critique a sharp, fair reviewer would give — the kind that's hard to hear but saves months. Flattery is the failure mode. The user is developing this idea precisely so that weaknesses surface now, in conversation, rather than later in review or after months of experiments.

Be honest, specific, and constructive. Honest means you state real weaknesses plainly. Specific means you point at the exact claim or mechanism, not vague unease. Constructive means that where you see a problem, you also note what would resolve it or what the strongest version of the idea would be.

## Ground the critique in actual prior work

A novelty assessment is only as good as the literature behind it. If the related-work picture is stale or thin, do a literature search first (the litsearch skill governs how) — you cannot judge novelty against work you haven't looked for. State what you're comparing against.

## Classify the contribution honestly

Locate what kind of contribution the idea actually makes. Most ideas are one or two of these, rarely all:

- **New problem / new framing** — identifies a question the field hasn't posed.
- **New method / mechanism** — a genuinely new way to do something.
- **New combination** — assembles known pieces in a way not done before. This *can* be a real contribution, but only if the combination is non-obvious or yields a non-obvious result. Say which.
- **New analysis / understanding** — explains why something works, or when it fails.
- **New result** — better numbers, new SOTA, or a capability not previously demonstrated.

Name the category plainly. If the honest answer is "this is a recombination of A and B and the combination is fairly natural," say that — and then ask whether the *result* of the combination is surprising enough to carry it, because sometimes it is.

## Distinguish novelty from value

These are different axes and conflating them misleads the user:

- An idea can be **novel but low-value** (no one did it because no one needs it).
- An idea can be **not-novel but still worth doing** (known idea, untested in an important regime; or known but never properly validated).

Be explicit about which axis a concern lives on. "This has been done" and "this isn't worth doing" are different verdicts.

## Steelman the strongest objection

State the single strongest objection a critical reviewer would raise — the one most likely to sink the paper or the project. Make it the *best* version of that objection, not a strawman you can easily knock down. Then assess how answerable it is. This is the most valuable single output of the skill: if the idea has a fatal flaw, the user needs to meet it now.

Common objection families to check against: the contribution collapses to an existing method under a change of notation; the gain is real but trivial or already achievable more simply; the claimed mechanism isn't actually what drives the result; the evaluation that would be needed is infeasible or wouldn't isolate the claim; the assumption the idea rests on doesn't hold in practice.

## Make the novelty claim falsifiable

For each novelty claim in the proposal, ask: what experiment or argument would *establish* it, and what would *refute* it? A claim that can't be tested isn't a contribution yet — it's a hope. Where a claim is currently untestable as stated, help sharpen it into one that is. This connects directly to the "Novelty claims" and "Open questions" sections of `method.tex`.

## Tone

Direct, not harsh; rigorous, not discouraging. The point is to make the idea stronger or to redirect effort early, never to perform skepticism. When the idea is actually strong, say so clearly and specifically — unfounded praise and unfounded dismissal are equally useless. End with the constructive next move: the experiment to run, the claim to sharpen, the related work to differentiate from, or the pivot to consider.
