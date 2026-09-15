---
name: add-task
description: >-
  Adds a new backlog row to the workboard in daily.md. Takes a task
  description from the command argument, asks for a short title and priority,
  auto-assigns the next Rx id, and appends the row with Command/script, Done
  when, Output, and Plan left blank. Use when the user says /add-task or
  "add a task" followed by a description.
disable-model-invocation: true
---

# Add a task to the workboard

## Input

The user provides a **description** of the task as the command argument, e.g.:

```
/add-task evaluate per-category breakdown of the edit types on the benchmark
```

If no description is given, ask for one and stop.

## Step 1 — Load context

1. Read `daily.md`.
2. Scan the workboard table for all existing task ids of the form `**Rn**` (e.g. `**R1**`, `**R8**`).
3. Determine `next_id` = `R{max_n + 1}`.

## Step 2 — Ask clarifying questions

Ask **exactly two questions** before writing anything:

1. **Short title** (3–6 words): generate **3 distinct title suggestions** derived from the description, numbered 1–3. Vary the angle — e.g. one action-focused, one noun-phrase, one outcome-focused. The user can pick a number, tweak one, or provide their own.
   Example format:
   ```
   Title suggestions:
   1. Per-category Edit-type Breakdown
   2. Category-level Metric Comparison
   3. Benchmark Category Analysis
   Pick one (1/2/3) or write your own:
   ```
2. **Priority**: `P0` (before next meeting) · `P1` (rebuttal) · `P2` (nice to have) · `—` (no active work). Match the set defined in the `daily.md` legend.

Wait for answers before Step 3.

## Step 3 — Add the workboard row

Append a new row to the workboard table in `daily.md`:

| Column | Value |
|--------|-------|
| Task | `**<next_id>** — <short title>` |
| Description | The description from the command argument, lightly cleaned |
| Done when | *(empty)* |
| Output | *(empty)* |
| Plan | *(empty)* |
| Pri | `**<priority>**` |
| Status | `` `backlog` `` |

The columns above must match the workboard header in `daily.md` exactly (currently: Task · Description · Done when · Output · Plan · Pri · Status). If the workboard header changes, update this list.

Insert the new row **at the bottom** of the workboard table, before the closing blank line.

Also bump **Last updated** at the top of the file.

## Rules

- **Never** fill in Done when, Output, or Plan — those are set by `/plan-task` and `/run-step`.
- One task per invocation.
- Do not modify the Run log, Previous weeks, or any other section.
- Do not create a plan file — that is `/plan-task`'s job.
