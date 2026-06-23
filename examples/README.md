# Example Tasks

`examples/` contains template-owned reference tasks. Use these as working
examples when authoring your own task, but put your submitted work under
`problems/<task_id>/`.

## Canonical Examples

The current end-to-end examples are:

- [`mujoco-pendulum`](mujoco-pendulum/): rubric-style deterministic
  scoring with `RubricBuilder`.
- [`mle-tabular-classification`](mle-tabular-classification/):
  ML_Envs-style continuous scoring with no rubric.

### `mujoco-pendulum`

This example demonstrates:

- A complete task directory with `task.toml`, `instruction.md`,
  `environment/Dockerfile`, `scorer/compute_score.py`, `data/`,
  `solution/`, and `baselines/`.
- A deterministic `RubricBuilder` grader that returns
  `rb.grade().to_dict()`.
- Ten equally weighted criteria, including MuJoCo compile checks,
  structural checks, sensor checks, and rollout checks.
- The required `/tmp/output` convention for agent-created artifacts.

### `mle-tabular-classification`

This example demonstrates:

- A complete tabular ML task adapted from
  `ML_Envs/examples/dataset-example-task_taiga`.
- Public parquet train/test data in `data/` and hidden target parquet in
  `scorer/data/`.
- A continuous `compute_score.py` that preserves ML_Envs metrics,
  anchors, target weights, and exponential score mapping.
- A score dict return shape with no `RubricBuilder`, so the headline is
  the original continuous score and subscores are diagnostic target
  progress values.

## How To Use This Example

Try the reference harness against any example (Docker required for ML, CFD,
structures, and hidden-env tasks):

```bash
uv run lbx-rl-harness reference --problem-dir examples/mujoco-pendulum
uv run lbx-rl-harness reference --problem-dir examples/mle-tabular-classification
```

Expected scores on the checked-in references:

| Example | Execution | Expected score |
| --- | --- | --- |
| `mujoco-pendulum` | host | `1.0` |
| `mle-tabular-classification` | container | `~0.5` (continuous) |
| `hidden-env-bandit` | container | `1.0` |
| `openfoam-vortex-suppression` | container | `1.0` |
| `opensees-three-story-retrofit` | container | `1.0` |

Read the closest example before creating your own task:

```bash
ls examples/mujoco-pendulum
ls examples/mle-tabular-classification
```

Then scaffold a new task in `problems/` by copying the starter that matches your
`task_type` (`ml`, `mujoco`, `cfd`, or `structures`):

```bash
mkdir -p problems
cp -R alignerr_plugin/src/alignerr_plugin/starter_templates/mujoco problems/my-task
```

Update `problems/my-task/metadata.json` and `task.toml` after copying the
starter. Copy patterns from the examples, not the example directories themselves.
The `examples/` tree is maintained by the template repo to test the full
pipeline and to document known-good task shapes.

## PR Behavior

The `examples/` tree is maintained by the template repo to test the full
pipeline and to document known-good task shapes. Example changes are typically
made by maintainers via direct commits or internal PRs here, not via labeler
fork PRs.

Labeler task submissions use the fork + trusted CI flow described in
[`../README.md`](../README.md): open a PR in your assigned fork, and trusted CI
posts `trusted-ci/grade` automatically.

## Where To Read More

- [`../README.md`](../README.md): beginner workflow and command reference.
- [`../docs/AUTHORING.md`](../docs/AUTHORING.md): task authoring guide.
- [`../docs/GRADING.md`](../docs/GRADING.md): grader package and
  `compute_score.py` contract.
