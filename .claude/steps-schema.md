# Steps schema

Machine-readable `steps` block in plan YAML frontmatter. Used by `/run-step`.

## Top-level plan fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Human-readable plan title |
| `overview` | string | One-sentence summary |
| `task_id` | string | Workboard id (e.g. `R2`) |
| `todos` | list | High-level checklist — implementation tasks and steps; managed alongside `steps` |
| `steps` | list | Ordered execution steps (see below) |
| `isProject` | bool | Set `true` for Claude to treat as a persistent project plan |
| `report_url` | string | Published Claude Artifact URL for this task, set by `/report-task`. Its presence means a re-run of `/report-task` updates this artifact in place instead of publishing a new one. |

---

## Step fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | ✅ | Unique step id within the plan. Should match a `todos[].id` if a todo covers it. |
| `type` | enum | ✅ | `sbatch` · `local` · `srun` · `manual` |
| `command` | string | ✅ (except `manual`) | Shell command(s) to run. Multi-line YAML literal block (`\|`) is recommended. |
| `cwd` | string | | Working directory. Defaults to repo root. |
| `wait_for` | string \| list | | Step `id`(s) that must be `completed` before this step can run. |
| `sets_status` | enum | | Workboard status to set on success: `running` · `finished` · `analyzed`. |
| `output_paths` | list[string] | | Output files produced by this step (paths relative to repo root). Used by `/run-step` to update the workboard **Output** column with clickable links. |
| `status` | enum | | `pending` (default) · `completed` · `failed`. Filled by `/run-step`. |
| `completed_at` | YYYY-MM-DD | | Filled by `/run-step` on success. |
| `job_id` | string | | Slurm job id — `sbatch`/`srun` only. Filled by `/run-step` from `Submitted batch job <id>`. |

### Type-specific fields

#### `sbatch` / `srun`
No extra required fields. `job_id` is auto-filled on success.

#### `srun`
| Field | Type | Description |
|-------|------|-------------|
| `srun_flags` | string | Extra flags appended to `srun` (e.g. `--gres=gpu:1 --time=02:00:00`). |

#### `manual`
| Field | Type | Description |
|-------|------|-------------|
| `check_hint` | string | Read-only shell commands or instructions. `{job_id}` is substituted from the nearest preceding `sbatch` step in `wait_for`. |

---

## Example

`<benchmark>` is the project's evaluation benchmark (e.g. `five_bench` for FiVE-Bench).
`cwd` is omitted here since it defaults to the repo root.

```yaml
steps:
  - id: launch-sbatch
    type: sbatch
    command: sbatch slurm_scripts/<benchmark>/r2_<desc>.sh
    sets_status: running
    status: pending

  - id: wait-array
    type: manual
    wait_for: launch-sbatch
    check_hint: "sacct -j {job_id} --format=JobID,State; ls outputs/r2_<desc>*"
    sets_status: finished
    status: pending

  - id: evaluate
    type: local
    wait_for: wait-array
    command: |
      python evaluation/evaluate_<benchmark>.py \
        --result_path evaluation/csv/r2_<desc>.csv
    output_paths:
      - evaluation/csv/r2_<desc>.csv
    status: pending

  - id: summarize
    type: local
    wait_for: evaluate
    command: |
      python evaluation/summarize_results.py \
        evaluation/csv/r2_<desc>.csv \
        -o evaluation/csv/r2_<desc>_summary.csv
    output_paths:
      - evaluation/csv/r2_<desc>_summary.csv
    sets_status: analyzed
    status: pending
```

---

## Notes

- **`output_paths` drives the workboard Output column.** `/run-step` checks each path exists after the step completes, then writes clickable markdown links into the **Output** column of the workboard row in `docs/daily.md` (paths relative to `docs/` — e.g. `[filename](../evaluation/csv/filename.csv)`).
- **Existing plans without `output_paths`** still work; the Output column is simply not auto-updated and must be filled manually.
- **`todos` vs `steps`**: `todos` covers all work including code/implementation; `steps` covers only runnable execution steps invokable by `/run-step`. Shared ids are kept in sync by `/run-step`.
