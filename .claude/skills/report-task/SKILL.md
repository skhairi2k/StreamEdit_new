---
name: report-task
description: >-
  Publishes a custom-designed Claude Artifact reporting a workboard task's
  goal, detailed experimental process, quantitative results, qualitative
  results (figures), and conclusion — read from daily.md and its
  .claude/plans/ plan file. Use when the user says report task, /report-task,
  write up R{N}, or asks for a report / writeup / summary artifact for a
  specific workboard task id.
disable-model-invocation: true
---

# Report a workboard task as an Artifact

## Input

One task identifier (e.g. `R33`). If missing, ask for it and stop.

## Style precedent

Task reports in this project are **custom-designed static HTML pages** in the style of prior reports like "R10 — Does head identity localize appearance editing?" and "R26 Spatial Tau Oracle": a masthead with a one-line thesis and key-facts strip, numbered sections, styled data tables, inline SVG charts, and embedded qualitative image grids — editorial/data-journalism treatment, not a plain memo. Each report gets its own palette and type pairing suited to its subject; do not reuse the same look across tasks.

**Do not build these as Claude Docs pages.** This is a deliberate choice (confirmed with the user 2026-09-22) — even though this repo has a Claude Docs connector attached, and `Artifact(quickstart, intent: "document")` would mandatorily route there, task reports use `intent: "other"` instead, which is unaffected by that override and produces a plain static page.

## Step 1 — Load context (read only)

1. Read `daily.md`; find the workboard row `**<id>**` — Description, Done when, Output, Plan link, Status, Pri.
2. Read the linked plan file (`.claude/plans/<slug>_<hex>.plan.md`) in full: frontmatter (`todos`, `steps` — their `note` fields usually carry the most detailed play-by-play, including detours and fixes) and body (`## Context`, `## Decisions`, `## Pipeline`, `## Code to touch`, and any `## Verdict`-style section some plans add after the six standard ones).
3. If the workboard **Status** is `analyzed` or `dropped`, find this task's bullet under **Progress / outcomes** in `daily.md` — it is usually the best-written synthesis of the final result and anchors the Conclusion.
4. Read every Run log bullet for this task, in order (**Current week**, and **Previous weeks** if the task's week has already been archived) — this is the real narrative of what happened, including pivots, bugs, and retries, not just the intended plan.
5. Note every output file referenced (workboard Output column, `steps[].output_paths`, and any CSV/figure paths named in Run log or plan notes).

If no workboard row matches the id, list the ids that do exist and ask the user to correct.

## Step 2 — Gather the actual data

Do not paraphrase numbers from memory of the plan's prose — **open the files**:

- **Quantitative**: read every CSV named in Output/`output_paths` with a real tool call, and pull the specific numbers that matter (means, counts, deltas — not just column names). If the plan's own notes already state a results table, reproduce and spot-check it against the CSV rather than copying the prose unverified.
- **Qualitative**: locate every figure (PDF/PNG/HTML) named in Output/`output_paths`. These are the images the report should show — a report describing pictures no reader can see is not finished.

## Step 3 — Prepare qualitative assets

Figures in this project are usually `.pdf` (matplotlib); embed images, not PDFs. Convert with PyMuPDF (works with no `pdftoppm` on the node):

```python
import fitz
doc = fitz.open("<fig>.pdf")
doc[0].get_pixmap(dpi=150).save("<fig>.png")
```

Pick 2–6 figures that best represent the qualitative result (a full qualitative grid, a Pareto/trade-off panel, a per-clip visual comparison) — not every figure the task produced. **Embed as base64 `data:` URIs directly in the HTML** (`<img src="data:image/png;base64,...">`), matching the precedent reports — no asset upload or blob step needed for a static page. Keep an eye on total size (16MB page limit, `data:` URIs count toward it): downscale or re-render at a lower DPI if a figure is large, and don't embed more images than the report needs.

## Step 4 — Build the Artifact

1. Call `Artifact` with `action: "quickstart"`, `intent: "other"` — **before** writing any file. This returns the `artifact-design` skill's page-design guidance inline (do not also load `artifact-design` separately) and lists the published Artifact types; none of those types (Design, Design System, Docs, Slides) fit a task report, so build a plain HTML page per that guidance.
2. **Updating an existing report**: if this task's plan frontmatter already has a `report_url` field, this run is an update — `Artifact action: "read"` that URL first, then republish (`url: <report_url>`) instead of creating a new artifact. Read `file_path` from that result and build the new version from it.
3. Sketch a short design plan before writing code (per the design guidance): 4–6 named hex colors, a display/body/utility type pairing (Google Fonts, linked via `<link>`), and a one-to-two-sentence layout concept — specific to this task's subject, not a reused template. This is an **editorial treatment** (see Style precedent): a real hero/masthead is warranted, not just a heading.
4. Structure the page around these five sections, in this order (headings can be titled to fit the report's voice — e.g. "01 · Setup" — they don't need to be the literal words below):

| Section | Content | Source |
|---|---|---|
| **Goal** | What question the task asked and why, 2–4 sentences — good masthead/lede material | Workboard Description + plan `## Context` |
| **Experimental process** | What was actually run, in order, including pivots/bugs/retries that shaped the result — a narrative, not a bare step list | Plan `## Context` / `## Decisions` / `## Pipeline` + Run log bullets, condensed |
| **Quantitative results** | Real numbers from the CSVs (Step 2), as styled tables and/or inline SVG charts (load `dataviz` before writing any chart — see precedent reports for inline-SVG line/bar chart style) | CSVs |
| **Qualitative results** | The embedded figures (Step 3), each with a one-line caption of what it shows | Figures |
| **Conclusion** | The verdict — does the result hold, what it means for the paper/downstream tasks, what's still open — good material for a "bottom line" callout box near the top | Plan `## Verdict`-style prose if present, else the `Progress / outcomes` bullet |

5. Title the artifact `"R{N} — <task title>"`, using the task's canonical title exactly as the workboard row gives it after the em dash (e.g. workboard row `**R33** — Edit-Alignment Metric Validation` → `<title>R33 — Edit-Alignment Metric Validation</title>`). Set this as the page's actual `<title>` tag, not just the publish call's `title` parameter — the parameter is only a fallback used when the file has no `<title>` tag, so an in-file tag is what the artifact's name is actually taken from. This is a deliberate override of the general design guidance's headline-title rule (confirmed with the user 2026-09-22): every task report is named this way, consistently, rather than a one-off headline per report. The masthead `<h1>` inside the page is still a headline/thesis statement — the override is for the `<title>` tag only.
6. Look at the rendered page once before publishing (per the design guidance's write-look-once-publish rule), fix what that pass shows, then publish (or republish to the existing `url`).

## Step 5 — Link it back

1. Write the published URL into the plan file's frontmatter as a top-level `report_url: <url>` field (add if missing, overwrite if present — this is what step 4.2 checks on the next run).
2. Give the user the link in one line. Do not duplicate the report's content in chat.

## Rules

- One task per invocation.
- Never fabricate a number, figure, or verdict not actually present in the CSVs/plan/`daily.md`. If a section has nothing to report yet (task not yet `analyzed`, no figures produced), say so plainly in that section rather than inventing a result.
- Re-running for the same task id updates the same Artifact in place; it does not create a second one.
- This skill only reads `daily.md` and the plan file, except for writing the `report_url` field in Step 5 — it never edits Run log, workboard Status, or any other plan field.
- Do not build these as Claude Docs pages (see Style precedent) — always `intent: "other"`, never `intent: "document"`.
