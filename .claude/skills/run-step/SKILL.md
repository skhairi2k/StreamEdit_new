---
name: run-step
description: >-
  Runs one typed execution step from a task plan (.claude/plans/*.plan.md):
  sbatch, local, srun, or manual verify. Appends to daily.md Run log,
  updates workboard status, and updates the Output column with links to any
  output_paths produced. Use when the user says run-step, /run-step, or
  launch step for a workboard id (R1, R4, …).
disable-model-invocation: true
---

# Run one plan step

Execute **one** machine-readable `steps` entry from the task's plan file.

## Input

- **Task id** (required): e.g. `R1`.
- **Step id** (optional): e.g. `launch-sbatch`. If omitted, pick the **first** step where `status: pending` and all `wait_for` steps are `completed`.

If either is missing or ambiguous, ask and stop.

## Step 1 — Load context

1. Read `daily.md` — workboard row `**<id>**`, **Current week → Run log**.
2. Read the plan linked in the **Plan** column (path relative to the repo root).
3. Parse YAML frontmatter: `task_id`, `steps` (see [steps-schema.md](../../steps-schema.md)).
4. Resolve the target step.

If the plan has no `steps` block, tell the user to re-run `/plan-task` or add `steps` manually from the plan body.

## Step 2 — Preflight (read-only)

- Prior `wait_for` steps must be `completed` (with `job_id` recorded for sbatch if relevant).
- Workboard status should allow the step (e.g. do not `sbatch` if already `running` unless user confirms re-run).
- For `sbatch`: ensure `logs/` exists (`mkdir -p logs`).

If preflight fails, report and stop.

## Step 3 — Run by type

| `type` | Action |
|--------|--------|
| `sbatch` | `cd <cwd>` then run `command`; parse job id from `Submitted batch job <id>` |
| `local` | `cd <cwd>` then run `command`; capture exit code |
| `srun` | `cd <cwd>` then `srun <srun_flags> <command>` |
| `manual` | Use `check_hint` (substitute `{job_id}` from the `wait_for` sbatch step). Run read-only checks when possible. Ask user to confirm if unclear. Do not submit new jobs. |

**One step per invocation.** Do not auto-chain cluster steps.

On failure: report stderr or missing outputs; do not mark step completed; do not append success to Run log.

## Step 4 — Update plan frontmatter

On success, set the step:

- `status: completed`
- `completed_at: YYYY-MM-DD` (today)
- `job_id: "<id>"` — sbatch/srun only

If a matching `todos` entry shares the same `id`, set its `status: completed` too.

## Step 5 — Update daily.md

1. Bump **Last updated**.
2. Append **one bullet** to the task's **Run log** sub-section (under **Current week**):

```markdown
- `YYYY-MM-DD` · **<step-id>** · `<type>` · <short detail> · workboard → `<sets_status or unchanged>`
```

**Detail by type:**

- `sbatch`: `job 12345678` · `script_name.sh`
- `local` / `srun`: command summary + main output path if known
- `manual`: what was verified (e.g. `sacct` all COMPLETED)
- Include `→ running` / `→ finished` / `→ analyzed` when the step has `sets_status`

3. If the step has `sets_status`, update the workboard **Status** column for that task id.
4. If the step has `output_paths` (see [steps-schema.md](../../steps-schema.md)), update the workboard **Output** column for that task id:
   - Format each path as a clickable link relative to the repo root: `[filename](path/to/filename)`
   - Only link files that actually exist on disk after the step completes
   - Append to any existing entries; do not overwrite previous outputs
5. If `sets_status: analyzed`, add one bullet under **Progress / outcomes**.

Do not edit other rows or Previous weeks.

## Rules

- Requires explicit user invocation (`/run-step`).
- Cluster access: run `sbatch`/`srun` only when the environment can reach Slurm (user's cluster session).
- Re-run of a completed step: ask for confirmation; append a new Run log line (do not delete old bullets).
- Implementation/code steps stay in `todos`; `/run-step` only runs entries in `steps`.

## Examples

```text
/run-step R1
/run-step R1 launch-sbatch
/run-step R1 wait-array
```

See [steps-schema.md](../../steps-schema.md).
