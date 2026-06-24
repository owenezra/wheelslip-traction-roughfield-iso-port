# Porting notes for legacy task #914

## Structural rewrite

- Replaced legacy `prompt.md`, `test_file.py`, and root metadata with ISO `instruction.md`, `scorer/compute_score.py`, `task.toml`, and ISO `metadata.json`.
- Reclassified `sim_policy` as `[difficulty].task_type = "ml"` with domain `robot_dynamics_system_identification`.
- Declared `/tmp/output/policy.py` under `[[outputs]]`.
- Moved the public simulator to `data/env/practice_env.py` and the held-out evaluator to `scorer/data/wheelslip_eval_protocol.py`.
- Added committed policy baselines in per-baseline subdirectories.
- Marked ground truth as in-container so the oracle solve and grader use the task image's MuJoCo/scikit-learn stack and root-only private fixtures.

## TPU/CPU split

The task requests `13vcpu+32gib+tpuv5e1x1`, following the TPU-first project policy. The unchanged MuJoCo simulation, NumPy feature extraction, and scikit-learn identifier run on the TPU host CPU. The Dockerfile does not install or override JAX, jaxlib, or libtpu. Agent approaches may use the base-provided TPU stack, but no fp64 computation is required or sent to TPU.

The grader, hidden simulator, and submitted policy run on CPU. The submitted policy is sandboxed through `PolicyWorker`; the scorer never directly imports agent-authored Python.

## Reference correction

The legacy reference trained its friction identifier on the private evaluation band. That violates the ISO rule that `solution/solve.sh` use only public information. This port promotes the legacy `broadband_rebuild` method to the reference: 1,800 labeled episodes generated through the public simulator over its disclosed physical support, a deterministic histogram gradient booster, and the unchanged CEM planner.

Its recorded in-image mean quality, `0.6874068017`, is the provisional ISO `X_REF`. The old bespoke score cap was removed; scoring now uses `grading.calibration.PiecewiseLinearCurve`, mapping the reference to 0.5 and theoretical quality 1.0 to score 1.0.

## Required pre-PR commands

From the target repository root:

```bash
uv sync
uv run lbx-rl-harness reference --problem-dir problems/wheelslip-traction-roughfield
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/wheelslip-traction-roughfield
uv run lbx-rl-template validate --problem-dir problems/wheelslip-traction-roughfield
```

Then commit the generated, fresh:

```text
problems/wheelslip-traction-roughfield/.alignerr/build_proof.json
```

If the containerized reference does not reproduce score `0.5 +/- 0.05`, update only after inspecting the raw `mean_quality`: re-bake `X_REF` to the newly measured fixed-reference aggregate, rerun all baselines, and refresh build proof.
