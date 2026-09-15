---
name: plan-task
description: >-
  Plans a research task from daily.md using .claude/base_project_description.md: asks
  clarifying questions, writes a Claude plan under .claude/plans/ with typed
  execution steps, sets status to planned, and links the plan in the workboard
  Plan column. Use when the user says plan task, /plan-task, or provides a
  workboard task id (R1, R4, …).
disable-model-invocation: true
---

# Plan a daily workboard task

## Input

The user provides **one task identifier** (e.g. `R1`, `R4`). If missing, ask for it and stop.

## Step 1 — Load context (read only)

1. Read `.claude/base_project_description.md` and `.claude/ideas.tex`.
2. Read `daily.md`.
3. Find the workboard row whose **Task** column starts with `**<id>**` (e.g. `**R1**`).

If no row matches, list ids from the table and ask the user to correct.

## Step 2 — Clarifying questions

Before planning, ask **3–6 short questions** about scope, done-when, scripts/outputs, blockers, and fit with **Current week**.

Include when relevant:

- Which steps need **Slurm** (`sbatch` / `srun`) vs **local** commands?
- Any **wait** points (array finish, manual runs)?
- What **`sets_status`** transitions apply (`running` → `finished` → `analyzed`)?

Use the AskQuestion tool when available; otherwise numbered questions in chat.

**Wait for answers** before Step 3.

## Step 3 — Write the plan

1. Use Plan mode when appropriate, or write the plan file directly.
2. Create `.claude/plans/<slug>_<8hex>.plan.md` where:
   - `<slug>` = lowercase id + short kebab title (e.g. `r1-tau-sweep-slurm`),
   - `<8hex>` = 8 lowercase hex chars (e.g. `3cf98b04`).
3. **Frontmatter** (required fields):

```yaml
name: ...
overview: ...
task_id: R1          # workboard id
todos:               # human checklist — code, config, analysis
  - id: ...
    content: ...
    status: pending
steps:               # machine-readable — for /run-step
  - id: launch-sbatch
    type: sbatch     # sbatch | local | srun | manual
    command: sbatch slurm_scripts/<benchmark>/r1_<desc>.sh
    # cwd defaults to repo root; set only if the step runs elsewhere
    sets_status: running
    status: pending
  - id: wait-array
    type: manual
    wait_for: launch-sbatch
    check_hint: "sacct -j {job_id}; all array tasks COMPLETED"
    sets_status: finished
    status: pending
  # ... local eval, manual pareto, etc.
isProject: true
```

4. **Body:** write the six sections **in this exact order**:

```
# <Task id>: <Title>

## Context
Task goal + done-when (from workboard). Minimal background only — no scope tables.

## Execution steps
<Table mirroring the YAML steps block, then /run-step invoke commands.>

## Decisions
One flat table: locked Q&A choices (setups, λ, output paths, naming, out-of-scope). Technical constants (latent dims, normalization) go here after Q&A — not in Context.

## Step commands
<One sub-section per step id. Show the exact copy-paste commands that /run-step will execute.>

## Pipeline
Mermaid flowchart only — inputs → scripts → outputs.

## Code to touch
<Repo files to create or modify before runnable steps: paths, one-line change per file, new files, implementation notes / pseudocode when needed.>
```

**Strict — no extra content:** the body has **exactly six** `##` sections, in order, and nothing else. No third-level headings (`###`) except `### <step-id>` under **Step commands**. Do not add rebuttal drafts, narrative placeholders, duplicate tables, on-disk layout blocks, or prose that belongs in `daily.md`.

| Section | Include | Exclude |
|---------|---------|---------|
| **Context** | Task goal, done-when, minimal background | Q&A choice tables, derivations, rebuttal text, scope tables |
| **Execution steps** | Steps table mirroring YAML `steps` + `/run-step` invoke lines; | Commands, file lists, code-only step ids |
| **Decisions** | One flat table summing up key Q&A choices  | `###` subsections, narratives, pseudocode |
| **Step commands** | Copy-paste commands per YAML `step` id | Prep todos, Slurm CONFIGS, implementation notes |
| **Pipeline** | Mermaid flowchart only | Paths, prose, on-disk layout |
| **Code to touch** | Files, one-line changes, pseudocode, new Slurm CONFIGS | Cost/rebuttal commentary, duplicate Decisions |

Each fact belongs in **one** section only. YAML **`steps`** = runnable pipeline after `implemented`  — code prep stays in **`todos`** + **Code to touch**, not in `steps`.

### `todos` vs `steps`

| | `todos` | `steps` |
|--|---------|---------|
| **Audience** | You — planning & implementation | Agent — `/run-step` |
| **Content** | Write code, Slurm scripts, registry, figures | Runnable commands after `implemented` |
| **Types** | Free-text `content` | `sbatch` · `local` · `srun` · `manual` |
| **In plan body** | Listed in **Code to touch**; optional *(prep)* row in Execution steps | Mirrored in Execution steps + **Step commands** |

Every **`steps`** entry should have a clear `id`. Reuse the same `id` as a `todos` row when they are the same milestone (e.g. `launch-sbatch`).

**Schema reference:** [steps-schema.md](../../steps-schema.md)

Summarize the plan for the user. **Do not edit daily.md until they accept** (or say to proceed).

## Step 4 — Update docs (after acceptance)

### `daily.md` (workboard row only)

- Set **Status** to `planned`.
- Set **Plan** to a markdown link with an **explicit descriptive title** (path relative to the repo root):

```markdown
[<link-text>](.claude/plans/<filename>.plan.md)
```

Do not change Run log or other rows. Bump **Last updated** at the top if you edit the file.

## Naming convention

Task-specific output files must be prefixed `r{N}_` where `N` is the workboard task number (e.g. `R7` → `r7_`). `<benchmark>` is the project's evaluation benchmark directory (for this project: `five_bench`). This applies to:

- **Slurm scripts:** `slurm_scripts/<benchmark>/r{N}_<desc>.sh`
- **Python scripts** created specifically for this task: `evaluation/r{N}_<desc>.py`, `experiment_plots/r{N}_<desc>.py`, etc.
- **CSV outputs:** `evaluation/csv/r{N}_<desc>.csv`
- **Figure outputs** (when task-specific): `evaluation/figures/r{N}_<desc>.pdf`

**Shared / reusable files** (e.g. `summarize_results.py`, shared baseline CSVs) are exempt — do not prefix files that are consumed by multiple tasks.

Enforce this when writing commands in **Step commands** and paths in **Decisions** and the **Pipeline** diagram.

## Rules

- No code changes or `sbatch` unless the user asks in the same turn.
- One task per invocation.
- If a plan link already exists, ask whether to replace or extend before overwriting.
- New plans **must** include a `steps` block (at least one executable or manual step).

## Pipeline skills

| Skill | When | Sets status |
|-------|------|-------------|
| `/add-task <description>` | Add a backlog row to the workboard | `backlog` |
| `/plan-task R1` | Scope + plan file + `todos` + `steps` | `planned` |
| `/build-step <step-id> R1` | Implement the code/scripts blocking that step (from `todos` + **Code to touch**) | `implemented` |
| `/run-step R1 [step-id]` | Run sbatch / local / srun / manual verify step | `running` → `finished` → `analyzed` |

`todos` are built by `/build-step`, not by `/run-step`. A step whose scripts do not
exist yet is **not** ready to run — target it with `/build-step` first.
