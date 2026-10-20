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

- **Step id** (required): the `id` of an entry in the plan's **`steps:`** YAML block — e.g. `launch-sbatch`, `smoke`. **It is never a `todos[].id`.**
- **Task id** (required): e.g. `R7`

Accepted in any order: `/build-step launch-sbatch R7` or `/build-step R7 launch-sbatch`.

If either is missing, ask and stop.

**`steps:` vs `todos:` — do not confuse them.** A plan file has two separate id-bearing lists. `steps:` holds the runnable pipeline (`sbatch`/`local`/`srun`/`manual`); `todos:` holds code-prep work items (write this hook, port this script) that `/build-step` implements *as a side effect* of targeting a step — they are never passed directly as the Step id argument, even though both lists use the key `id` and a todo's name can look exactly like a step name. Concretely: given a plan whose `todos:` lists `se-attn-hook` (a code-prep item) and whose `steps:` starts with `smoke`, the correct invocation is `/build-step smoke R48` — **not** `/build-step se-attn-hook R48`. The latter is invalid input, not a shorthand for "build this one todo."

## Step 1 — Load context

1. Read `daily.md` — find workboard row `**<task-id>**`; extract the Plan link.
2. Read the plan file (path relative to the repo root).
3. Parse YAML frontmatter: `todos`, `steps`.
4. Read the `## Code to touch` body section — this is the authoritative implementation spec.
5. **Validate the given Step id against `steps[].id`.** If it is not present there:
   - If it matches a `todos[].id` instead, STOP. Report: "`<given-id>` is a todo, not a step — `/build-step` targets a step and builds whatever todos block it." Then list the pending `steps[].id` values in order (typically the first one is the right target) and ask the user to re-invoke with one of those.
   - If it matches neither list, STOP and report the id is unknown, listing valid `steps[].id` values.
   - Do not proceed to Step 2 until a valid step id is confirmed.

If the plan has no `todos` block, tell the user to re-run `/plan-task`. If there is no `## Code to touch` section, report it and stop.

## Step 2 — Identify blocking todos

Goal: find the pending `todos` entries whose work is **required to run the target step** — nothing more.

Algorithm:

1. Collect all step ids from the `steps` block → call this `step_ids`.
2. **If the code-prep todos carry `needed_by`** (plans written under the current `/plan-task`): the blocking todos are exactly the pending todos whose `id` is not in `step_ids` and whose `needed_by` contains the target step id. Skip to step 6. No guessing from script names.
3. **Otherwise (older plans without `needed_by`)**, find the target step entry in `steps` (by `id == <step-id>`) and extract its `command`.
4. Parse the command to identify the **scripts the step directly invokes** — e.g. `python evaluation/summarize_by_category.py …` → `evaluation/summarize_by_category.py`. If such a script does not exist yet, also read its `## Code to touch` spec: the files *it* calls count as invoked too (e.g. a smoke script whose gates need a hook).
5. Walk `todos` in order; collect entries where **all three** conditions hold:
   - `status: pending`
   - `id` is **not** in `step_ids` (it is code-prep, not a runnable step)
   - The todo's `content` (or the matching `## Code to touch` spec) references **one of the scripts identified in step 4** — i.e., the todo creates or modifies a file that this step needs.
6. These are the **blocking todos**.

**Bundled todos (older plans).** If a blocking todo bundles several deliverables (a file list or a brace expansion) and only some of them are needed by the target step, split it in the plan frontmatter first: one todo per deliverable, each with `needed_by`, keeping the original `status`. Then build only the parts this step needs. Report the split to the user. Never mark a bundle completed when only part of it is built.

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
