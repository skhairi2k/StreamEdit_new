---
name: method-doc
description: Maintain the living LaTeX document that describes a deep-learning research method under development. Use this skill whenever the conversation involves writing, updating, revising, or restructuring the method/idea document (method.tex), recording a design decision, or capturing the current state of the research proposal. Trigger it even when the user just says "update the doc", "write this down", "add this to the method", "capture this decision", or describes a change to the idea that should be reflected in the file — not only when they explicitly mention LaTeX or method.tex.
---

# Method Document Maintenance

This skill governs the single LaTeX file (`method.tex`) that is the distilled, current description of a deep-learning research idea being developed across many conversational iterations. The document is the *output* of the discussion; the conversation is the messy reasoning, the document is the clean state.

The most important property of this file is that it must remain **trustworthy as a snapshot**: at any moment, reading `method.tex` should tell you exactly what the current proposal is, what is claimed to be novel, how it relates to prior work, and what is still uncertain. A reader who returns after exploring a dead-end direction should be able to rely on it completely.

## Fixed document structure

Always keep `method.tex` in this section order. Do not invent new top-level sections without telling the user why.

```latex
\documentclass{article}
\usepackage{amsmath,amssymb,hyperref}
\title{<working title>}
\begin{document}
\maketitle

\section{Motivation and problem statement}
% Why this problem matters and what exactly is being solved.
% State the problem precisely: inputs, outputs, the gap in current methods.

\section{Method}
% The core proposal. Be concrete enough that another researcher could
% start implementing. Use equations where they sharpen meaning.

\section{Novelty claims}
% An explicit, ENUMERATED list. Each claim must be falsifiable and specific.
% Bad: "our method is more efficient." Good: "training converges in
% fewer steps than method X because of mechanism Y."

\section{Relation to prior work}
% The closest existing works and how this idea differs from each.
% One short paragraph or entry per closely-related work.

\section{Open questions and risks}
% What is unresolved, what could sink the idea, what needs an experiment.
% This section is allowed to be honest and unflattering.

\section{Changelog}
% Append-only. See rules below.

\end{document}
```

## Editing rules

These rules exist so the file stays a reliable snapshot and so its history stays legible across many iterations.

**Edit incrementally, never rewrite wholesale.** Change only the parts that the current discussion actually changed. Wholesale rewrites destroy the connection between the document and its history and make it impossible to see what a given iteration decided. If a section genuinely needs restructuring, say so explicitly and explain what you're moving and why before doing it.

**Every substantive change appends a Changelog entry.** A substantive change is anything that alters the method, a novelty claim, or the relation to prior work — not fixing a typo. Use this format, newest entry at the bottom:

```latex
\subsection*{<YYYY-MM-DD> — <one-line summary>}
What changed and why. If a direction was abandoned, say so and say why,
because that reason is the single most valuable thing to a future reader
deciding whether to revive it.
```

The changelog is what makes "come back to a previous point" work: when the user abandons a direction, the *reason* lives here, not just in the transcript.

**Keep novelty claims explicit and falsifiable.** This section is the spine of the proposal. Every time the method changes, check whether the novelty claims still hold. If a new paper or a critique undermines a claim, do not silently delete it — note in the changelog that it was weakened or retracted and why.

**Mark uncertainty in place.** If part of the method is speculative or unvalidated, mark it inline with a clear flag rather than presenting it as settled, e.g. `% UNVALIDATED:` or a `\textbf{(speculative)}` tag. A snapshot that hides its own uncertainty is worse than useless.

**Preserve compilability.** The file should always compile. After editing, do a quick mental check that braces, environments, and the preamble are intact. If the user wants it compiled, you can do that in the environment.

## When relating to prior work

Keep the "Relation to prior work" section in sync with what's actually been found via literature search. Each entry should name the work and state the *difference*, not just summarize the work. If the difference is thin, that is important information and the Open questions section should flag it.

## On first creation

If `method.tex` does not exist yet, create it from the structure above using the user's rough idea. It is fine for sections to be sparse or marked as TODO early on — the document grows with the idea. Seed the Changelog with a first entry dated today describing the initial idea.

## What not to do

Do not pad the document with generic background, literature-review prose, or motivation that isn't specific to this idea — it is a method spec, not a paper introduction. Do not let it drift into a transcript of the conversation. Keep it tight; a long method document usually means undistilled thinking.
