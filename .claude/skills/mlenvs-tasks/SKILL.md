---
name: mlenvs-tasks
description: Author ML_Envs-style tasks (dataset/CSV, model module, executable, HDF5, k-fold) in this template. Use when migrating an ML_Envs task or building any continuous-scored task with held-out truth, the submission loaders, FLOOR/REF/PERFECT calibration, dataset licensing, preloaded_files mounts, or cuda-graphics/tpu base flavors.
---

# ML_Envs-style Tasks

A task = a graded submission scored by a continuous, calibrated function of
held-out truth. Runtime contract, file layout, harness, and PR flow are the same
as any task (`docs/AUTHORING.md`); the delta is below. Full guide:
`docs/MLENVS_TASKS.md`. Mitigations every scorer must respect:
`docs/REWARD_HACKING.md`.

## 1. Submission paradigm -> loader

The agent writes an artifact under `/tmp/output` (declare it in `[[outputs]]`);
read it with the sanctioned loader, never by hand, and never trust agent stdout:

- CSV / dataframe -> `helpers.load_submission_or_fault`
- Python policy -> `helpers.run_policy`
- model module (`fit`/`predict` or one callable) -> `helpers.run_model_module`
- executable -> `helpers.run_submitted_executable`
- HDF5 -> `helpers.load_submission_h5_or_fault`
- k-fold CV of a model module -> `score_kfold_cv`

```python
from grading import AgentFault, helpers, calibration

def compute_score(workspace, trajectory, private):
    truth = _load_truth(private)                 # author data: propagate on failure (discard)
    try:
        sub = helpers.load_submission_or_fault(workspace / "submission.csv",
                                               required_columns=["id", "pred"])
    except AgentFault:
        raise                                    # agent fault -> kept 0.0
    rmse = _metric(sub, truth)
    x = calibration.progress_lower_better(rmse, floor=1.0, perfect=0.0)
    return calibration.ExponentialCurve.from_reference(X_REF).score(x)
```

## 2. Calibrate (FLOOR / REF / PERFECT)

Use `grading.calibration`; do not re-derive the curve. `FLOOR` = worst plausible
raw metric, `REF` (reference's metric) -> 0.5, `PERFECT` = optimum. Set
`[difficulty].reward_type = "continuous_scoring_function"`. Ship
`solution/solve.sh` (scores `0.5 ± 0.05`) and `baselines/<name>/` (score `~0`);
the learnability gate fails the task if a baseline reaches the reference.

## 3. Metadata

`[difficulty].task_type = "ml"` (the submission paradigm is the loader you call,
not the task_type -- there is no `dataset` type). `[difficulty].license` is
required for `ml`: a permissive SPDX id or `not_applicable`.

## 4. Scale knobs

- Compute: **ml tasks default to GPU** (`[environment].required_resources =
  "12vcpu+100gib+h100/2"`; the `ml` starter ships this). Keep an H100 enum for
  training, deep models, RAPIDS/cuDF, or batched eval; only pick a CPU enum when
  the task genuinely cannot use one. Do not put `cpus`, `memory_mb`, `gpus`, or
  `gpu_types` in `task.toml`. `base_flavor = "auto"` infers CPU/GPU/TPU from the
  enum.
- Available resource enums:
  - CPU: `2vcpu+6gib`, `4vcpu+16gib`, `6vcpu+32gib`, `8vcpu+64gib`, `16vcpu+64gib`, `16vcpu+128gib`
  - CPU perf: `2vcpu+6gib+perf`, `4vcpu+16gib+perf`, `6vcpu+32gib+perf`, `8vcpu+64gib+perf`, `16vcpu+64gib+perf`, `16vcpu+128gib+perf`
  - TPU: `13vcpu+32gib+tpuv5e1x1`, `16vcpu+64gib+tpuv5e2x2`, `50vcpu+128gib+tpuv5e2x2`
  - H100: `3vcpu+25gib+h100/8`, `6vcpu+50gib+h100/4`, `12vcpu+100gib+h100/2`, `24vcpu+200gib+h100/1`
  - H100 graphics: `3vcpu+25gib+h100/8+graphics`, `6vcpu+50gib+h100/4+graphics`, `12vcpu+100gib+h100/2+graphics`, `24vcpu+200gib+h100/1+graphics`
- `[[preloaded_files]]` mounts large datasets / HF weights read-only at deploy
  time (`scripts/sync_mount.sh` packs/uploads/stamps) instead of baking them in.
- `[environment].base_flavor`: `gpu-openroad` for EDA/OpenROAD,
  `cuda-graphics` for rendering (gVisor blocks hardware GL), `tpu` for `jax[tpu]`.

## 5. Reward-hacking discipline

`raise AgentFault` for agent faults (kept 0.0); let author/infra faults
propagate (discarded). No broad `except: return 0.0`; no exec/pickle of agent
artifacts in the grader. `lbx-rl-template validate` enforces the blocking
`agent_fault` stage.

## Hidden-dynamics environments

If the agent must interact LIVE with an env whose source it cannot read (a
simulator, market, bandit, game opponent), that is a different paradigm: use
`[environment].hidden_env` + the env server. See the `hidden-env-tasks` skill and
`docs/HIDDEN_ENV.md`.

## Verify and submit

While developing the reference (train/solve + grade, cached under
`.alignerr/reference_cache/output`):

```bash
uv run lbx-rl-harness reference --problem-dir problems/<task_id>
uv run lbx-rl-harness reference --problem-dir problems/<task_id> --skip-solve
```

Before PR (updates `.alignerr/build_proof.json`):

```bash
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>
lbx-rl-template validate --problem-dir problems/<task_id>
```

Commit `.alignerr/build_proof.json`, open a fork PR; trusted CI posts `trusted-ci/grade`.
