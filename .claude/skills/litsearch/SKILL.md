---
name: litsearch
description: Find, verify, and report recent academic work related to a deep-learning research idea. Use this skill whenever the user wants to know what prior or recent work exists near an idea, asks "has this been done?", "what's the latest on X?", "what should I cite?", "find related papers", or wants the related-work picture refreshed during research-idea development. Trigger it for any request to survey, position against, or check the novelty of an idea against the literature — and always use it before asserting that something is novel or that no prior work exists, since that claim is only credible after an actual search.
---

# Literature Search for Research Ideas

The goal is an honest, current picture of what work exists near the idea being developed — good enough to judge novelty and to position the proposal. The two failure modes to avoid above all are (1) inventing or misattributing papers, and (2) declaring something novel without having looked.

## Core principle: never cite what you haven't verified

Model memory of specific papers — exact titles, author lists, years, venues — is unreliable and degrades for recent work. So:

- **Search the web for anything you intend to cite.** Do not produce a citation from memory.
- **Fetch the source** (the arXiv abstract page, the publisher page, the paper PDF) before stating what a paper claims. Snippets are often misleading.
- If you believe a relevant paper probably exists but you cannot confirm it, **say exactly that** — "there is likely work on X; I could not confirm a specific paper" — rather than fabricating a plausible-looking reference. A flagged gap is useful; a fake citation is corrosive.

## How to search

**Start broad, then narrow.** Begin with the core concept in 1–3 words, see what the field calls it, then refine with the field's actual terminology. Ideas are often known under a name you didn't expect.

**Reformulate aggressively.** Each query should be meaningfully different. If "X" returns nothing, try the mechanism, the application, the problem framing, and adjacent subfields. A single search is almost never enough for a novelty judgment.

**Prefer primary sources.** arXiv, OpenReview, and publisher/conference pages over blog aggregators or marketing. For deep learning, arXiv and OpenReview (ICLR/NeurIPS submissions and their reviews) are especially valuable — OpenReview reviews tell you how the community received an idea, which is gold for the critique step.

**Bias toward recency, but include anchors.** Prioritize the last 1–2 years since that's where the model's own knowledge is weakest and where the idea is most likely to have been scooped. But include the seminal/foundational works the idea builds on, even if older.

**Built-in search is enough to start.** claude.ai's web search finds papers well. It returns links, not structured metadata, so for citation counts or "who cites whom" you'd need a dedicated source — only reach for that if a novelty judgment actually hinges on it.

## What to report

For each genuinely related work, report — concisely:

- **Title, authors (or first author et al.), venue, year** — verified, not remembered.
- **What it does** — one or two sentences, in your own words.
- **How it relates to the current idea** — this is the point. State the overlap and, crucially, the *difference*. "This is close because… but differs in…". A list of summaries without positioning is not useful.

Group by relevance: lead with the works that most directly threaten or support the idea's novelty, then adjacent work, then background.

End with a **bottom line**: is the core idea apparently untouched, a variation on existing work, or already done? Be direct. If it looks already-done, say so plainly — that is the most valuable thing a search can return, and it saves the user weeks.

## Honesty discipline

- Distinguish "I found this and verified it" from "this likely exists but I couldn't confirm it" from "I found nothing, but my search may have missed it."
- Never inflate the related-work picture to seem thorough, and never deflate it to make the user's idea look more novel. Either distortion defeats the purpose.
- If results conflict or the field uses contested terminology, surface that rather than papering over it.

## Handing off

When the search changes the picture, the "Relation to prior work" and "Novelty claims" sections of `method.tex` should be updated to match (the method-doc skill governs how). A literature finding that doesn't make it into the document will be lost by the next iteration.
