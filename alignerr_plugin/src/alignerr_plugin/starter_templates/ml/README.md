# ML Task Template

Use this scaffold for `task_type = "ml"` tasks: the agent produces a data
or model artifact under `/tmp/output/...` that a **continuous** scoring
function grades. It is intentionally minimal; for a complete working
reference (CSV submission loaded via `helpers.load_submission_or_fault`,
FLOOR/REF/PERFECT exponential calibration), read
`examples/mle-tabular-classification/`.

**ML tasks deploy on TPU by default.** This scaffold ships the default TPU tier
`[environment].required_resources = "13vcpu+32gib+tpuv5e1x1"` (TPU is faster on
Taiga; the reference still runs on GPU). Switch to an `h100/*` tier only if the
task genuinely cannot run on a TPU, and to a CPU enum only if it needs no
accelerator at all; note why in either case.

After creating a task, update:

- `instruction.md` with the exact prompt the agent should follow.
- `task.toml` with the task name, resources, timeouts, `[difficulty].license`
  (required for `ml`), and `[[outputs]]`.
- `scorer/compute_score.py` with your grading logic. Read agent submissions
  through `grading.helpers.load_submission_or_fault(...)` (or guard the read
  with `except OSError: raise AgentFault(...)`); never trust stdout.
- `data/` with public files the agent may read.
- `scorer/data/` with private grader fixtures (held-out labels/targets).
- `solution/solve.sh` if you have a reference solution for validation.

Iterate on the reference during development:

```bash
uv run lbx-rl-harness reference --problem-dir problems/<task_id>
```

Run ground truth before PR:

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
```

The grader entrypoint must be:

```python
def compute_score(workspace, trajectory, private):
    ...
```

Return a `float`, a score `dict`, or `RubricBuilder.grade().to_dict()`.
See the repo root `README.md` and `docs/GRADING.md` for the full guide.
