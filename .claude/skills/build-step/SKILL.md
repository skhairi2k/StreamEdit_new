---
name: build-step
description: >-
  Implements the code and scripts that block a plan step from running.
  Reads the plan's todos and Code-to-touch section, builds any pending
  implementation todos that precede the target step, marks them completed
  in the plan frontmatter, and sets workboard status to `implemented` when
  all pre-launch work is done. Use before /run-step when a step requires
  code changes or new files (e.g. /build-step launch-sbatch R7).
disable-model-invocation: true
---

# Build prerequisites for a plan step

## Input

- **Step id** (required): e.g. `launch-sbatch`
- **Task id** (required): e.g. `R7`

Accepted in any order: `/build-step launch-sbatch R7` or `/build-step R7 launch-sbatch`.

If either is missing, ask and stop.

## Step 1 — Load context

1. Read `daily.md` — find workboard row `**<task-id>**`; extract the Plan link.
2. Read the plan file (path relative to the repo root).
3. Parse YAML frontmatter: `todos`, `steps`.
4. Read the `## Code to touch` body section — this is the authoritative implementation spec.

If the plan has no `todos` block, tell the user to re-run `/plan-task`. If there is no `## Code to touch` section, report it and stop.

## Step 2 — Identify blocking todos

Goal: find the pending `todos` entries whose work is **required to run the target step** — nothing more.

Algorithm:

1. Collect all step ids from the `steps` block → call this `step_ids`.
2. Find the target step entry in `steps` (by `id == <step-id>`). Extract its `command`.
3. Parse the command to identify the **scripts the step directly invokes** — e.g. `python evaluation/summarize_by_category.py …` → `evaluation/summarize_by_category.py`.
4. Walk `todos` in order; collect entries where **all three** conditions hold:
   - `status: pending`
   - `id` is **not** in `step_ids` (it is code-prep, not a runnable step)
   - The todo's `content` (or the matching `## Code to touch` spec) references **one of the scripts identified in step 3** — i.e., the todo creates or modifies a file that this step directly calls.
5. These are the **blocking todos**.

**Do not** collect todos for scripts that are only needed by *later* steps; those will be built when those steps are targeted.

If there are no blocking todos, report "nothing to build — ready to `/run-step`" and stop (do not touch any files).

## Step 3 — Verify what is already done on disk

Before implementing, quickly check each blocking todo to avoid redundant work:

- **New file** (todo describes creating a file): `ls <path>` — if it exists, mark the todo done.
- **Code edit** (todo describes changing an existing file): grep for a key identifier from the `## Code to touch` spec (e.g. `grep "\-\-new_flag" script.py`) — if found, mark the todo done.

Report which todos are already complete and which still need building. If all are already complete, update the plan frontmatter (Step 5) and stop.

## Step 4 — Implement each blocking todo

Process todos in order. For each one that still needs building:

1. Find its implementation spec in the `## Code to touch` section — match by filename or keyword.
2. Read the current file before editing (required before any Edit call).
3. Apply changes:
   - **Existing files**: use Edit with exact old/new strings derived from the spec.
   - **New files**: use Write.
4. After a successful implementation, immediately update `todos[i].status: completed` in the plan frontmatter (write the plan file before moving to the next todo).

If implementation of any todo fails (e.g., the old string to replace is not found): report clearly, do not mark it completed, and stop. Do not attempt later todos.

**When the spec is pseudocode**, translate it faithfully into working code. If a spec decision is genuinely ambiguous, ask the user before proceeding — do not guess.

## Step 5 — Update plan frontmatter

After all todos are processed, write back the plan file with updated `todos[].status` fields.

**Never** mark a `steps` entry as completed — that is `/run-step`'s job. If a todo `id` happens to match a step `id` (e.g. `launch-sbatch`), only update the `todos` entry, not the `steps` entry.

## Step 6 — Update daily.md

1. Bump **Last updated**.
2. Check whether ALL pending implementation todos for this task are now complete — i.e., the only remaining pending items in `todos` are those whose `id` also appears in `steps` (runnable steps, not code prep).
3. If yes: set the workboard **Status** column for this task → `implemented`.

Do **not** append to the Run log — that is `/run-step`'s job.

## Rules

- Do not run `sbatch`, `srun`, or any cluster commands.
- Do not mark `steps` entries as completed.
- Do not edit `daily.md` beyond Last updated and (if warranted) the Status column.
- Implement conservatively: follow the `## Code to touch` spec closely; do not add unrequested features.
- One task + step per invocation.

## Skill pipeline

| Skill | When |
|-------|------|
| `/plan-task R7` | Scope + plan + `steps` + `## Code to touch` |
| `/build-step launch-sbatch R7` | Implement blocking todos; set status → `implemented` |
| `/run-step R7 launch-sbatch` | Execute the step (sbatch / local / manual) |
