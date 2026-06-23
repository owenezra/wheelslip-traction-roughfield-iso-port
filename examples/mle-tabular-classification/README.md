# MLE Tabular Classification Example

This example adapts the ML_Envs
`examples/dataset-example-task_taiga` task into the new
`lbx-rl-tasks-template` layout. It is intentionally a continuous-scoring
task, not a rubric task.

The underlying task objective, data, metrics, anchors, weights, and
exponential score curve are preserved from ML_Envs. The adaptation is
only structural:

- The old embedded `test_file` is now `scorer/compute_score.py`.
- Public data is copied from `data/` to `/data/`.
- Hidden target data is copied from `scorer/data/` to
  `/mcp_server/data/`.
- The agent writes `/tmp/output/submission.csv`.
- The grader returns a score dict with the same continuous headline
  score plus diagnostic per-target progress values.

Because the submission is a static CSV, the grader does not execute
agent-authored Python during scoring and does not need `PolicyWorker`.
`PolicyWorker` is only for tasks that grade executable artifacts such as
`/tmp/output/policy.py`.

## What The Agent Must Do

The prompt in `instruction.md` asks the agent to train on:

```text
/data/train.parquet
/data/test.parquet
/data/column_mapping.json
```

and write predictions to:

```text
/tmp/output/submission.csv
```

The submission must contain exactly these columns:

```text
t1,t2,label
```

with one row for each row of `/data/test.parquet`, in the same order.

## Why This Is Different From A Rubric Task

Rubric tasks decompose success into criteria such as "file exists",
"model compiles", or "answer is correct". This task is scored like a
classic ML benchmark:

1. Load the submitted predictions.
2. Load hidden ground truth from `scorer/data/test_target.parquet`.
3. Compute raw metrics per target.
4. Convert each raw metric to progress in `[0, 1]` using baseline and
   perfect anchors.
5. Weight-combine target progress values.
6. Map the aggregate through an exponential curve so the reference
   solution lands around `0.5`.

There is no `RubricBuilder` and no LLM judge. The score is a continuous
function of the submission CSV.

## Scoring Details

Per-target metrics and anchors:

| Target | Metric | Direction | Floor | Reference | Perfect | Weight |
| --- | --- | --- | --- | --- | --- | --- |
| `t1` | SRE (`RMSE / std(true)`) | lower is better | `0.2661` | `0.0330` | `0.0` | `0.35` |
| `t2` | SRE (`RMSE / std(true)`) | lower is better | `0.7770` | `0.2107` | `0.0` | `0.35` |
| `label` | binary F1 | higher is better | `0.9231` | `0.9939` | `1.0` | `0.30` |

The final score is:

```text
score = exponential(weighted_progress)
```

where the exponential curve passes through:

```text
(0, 0), (x_ref, 0.5), (1, 1)
```

`x_ref` is the weighted progress achieved by the committed reference
solution.

## Files To Notice

- `task.toml`: declares `/tmp/output/submission.csv`, long ML timeouts,
  dataset difficulty metadata, and the original ML_Envs GPU resource
  request (`12vcpu+100gib+h100/2`).
- `instruction.md`: agent-facing task prompt from the ML_Envs task.
- `environment/Dockerfile`: installs the shared grader package and the
  Python libraries needed by `compute_score.py`.
- `data/`: public parquet files and column mapping exposed to the agent.
- `scorer/data/test_target.parquet`: hidden target data used only by the
  grader.
- `scorer/compute_score.py`: continuous scoring implementation adapted
  from the ML_Envs embedded `test_file`.
- `solution/solution.py`: original reference solution script.
- `solution/submission.csv`: captured reference predictions.
- `solution/solve.sh`: validation-friendly script that writes the captured
  reference submission to `/tmp/output/submission.csv`.
- `baselines/linear/`: original linear/logistic baseline and captured
  baseline submission.

## Measured Scores

The committed outputs from the original ML_Envs task score as follows:

| Submission | Expected score |
| --- | --- |
| `solution/submission.csv` / `solution/solve.sh` reference | about `0.4996` |
| `baselines/linear/submission.csv` | about `0.0000` |

The exact final score is produced by `scorer/compute_score.py`.

## Pattern To Copy

Use this example when migrating ML_Envs-style tasks where the grader is
a continuous scoring function:

- Keep the original metric math and anchors unchanged.
- Move the old embedded `test_file` into `scorer/compute_score.py`.
- Change hard-coded paths into `workspace / "submission.csv"` and
  `private / "test_target.parquet"`.
- Return either a bare float or a score dict. Prefer a score dict when
  you want Boreal and Harbor to retain diagnostic target progress values.
- Do not introduce `RubricBuilder` unless the task is actually rubric
  decomposable.

Run local checks from the repo root:

```bash
uv run lbx-rl-harness reference --problem-dir examples/mle-tabular-classification
uv run lbx-rl-harness run --problem-dir examples/mle-tabular-classification --runtime ground-truth
```

The reference run executes inside the task container (ML auto-selects container)
and should score near `0.5` on the checked-in calibration CSV.
