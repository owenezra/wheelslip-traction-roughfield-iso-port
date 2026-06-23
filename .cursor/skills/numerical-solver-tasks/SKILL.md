---
name: numerical-solver-tasks
description: Author non-MuJoCo numerical-solver RL tasks (task_type cfd or structures) using the open-source solver stack baked into the base image. Use when creating tasks that need OpenFOAM, SU2, AeroSandbox/AVL, scikit-fem/PyNite/CalculiX/OpenSeesPy, or when setting [difficulty].domain.
---

# Numerical-Solver Tasks

Non-MuJoCo engineering tasks reuse the same scoring contract as other tasks
(`compute_score(workspace, trajectory, private)`, artifacts to `/tmp/output`,
enum-backed metadata, and reward-type-specific oracle targets). A numerical
solver is the deterministic referee.

References:
- `docs/NUMERICAL_SOLVERS.md` — full engine inventory + how to invoke each.
- `project_guidelines/cfd/cfd_environments.md` — CFD task guide (AVL + OpenFOAM).
- `examples/openfoam-vortex-suppression` — canonical worked example.

## task.toml

- Set `[difficulty].task_type` to `cfd` or `structures`.
- Set `[difficulty].domain` to the scoped enum (`aerodynamics` for CFD,
  `seismic_retrofit` for structures today).
- Set `[difficulty].reward_type = "multi_deterministic_rubrics"` for the
  standard deterministic solver-rubric pattern.

## Engines in the base image

Import directly in `scorer/compute_score.py`:

| Domain | Library (import) |
| --- | --- |
| CFD / aero | `aerosandbox` |
| circuits | `PySpice` (ngspice) |
| structures | `skfem`, `Pynite`, `anastruct`, `openseespy` |
| electromagnetics | `skrf` |
| reacting flow / thermo | `cantera`, `CoolProp` |
| EDA | `klayout.db` |

Heavy engines run via subprocess (they live only in the base image):
- OpenFOAM: `source /etc/solver-envs.d/openfoam.sh` then `blockMesh`/`pimpleFoam`/etc.
- CalculiX `ccx`, `SU2_CFD` — on PATH.
- Meep: `/opt/solver-envs/meep/bin/python`.
- OpenROAD: `openroad` — opt into `base_flavor = "gpu-openroad"` with a non-graphics H100 `required_resources` enum.

Per-task installs (build in `environment/Dockerfile`): AVL, XFOIL, PyNEC, DREAMPlace.

## Determinism (load-bearing)

- Pin the solver version, run **serial**, fixed iteration counts or fixed
  `deltaT` + `endTime` (no adaptive timestep), fixed schemes/seeds.
- Mesh from a **parametric `blockMesh`** or a pre-built mesh — never
  `snappyHexMesh` at grade time. Insert features (e.g. a splitter plate) as a
  `createBaffles` baffle on a fixed mesh rather than re-meshing.
- Grade on tolerances; treat divergence / NaN / non-convergence as `0` (+ penalty).
- Precompute reference values into `scorer/data/` (e.g. a no-control baseline);
  never regenerate them nondeterministically at grade time.

## Runtime budget

The grader runs the solver per hidden case sequentially. Keep
`N_cases * per_case_timeout` under `[verifier].timeout_sec`, use coarse meshes
and low/moderate Reynolds, and a small hidden sweep (2-3 conditions).

## In-container ground truth + reviewer artifact

A grader that calls a base-only engine cannot run on the host. Set
`[ground_truth].in_container = true` so the harness builds the task image and
runs solve + grade + render inside it. The reviewer artifact uses the
generalized render contract (any task type): name it `rendering.mp4`, exactly
`1280x720` h264.

```toml
[difficulty]
task_type = "cfd"
domain = "aerodynamics"
reward_type = "multi_deterministic_rubrics"

[ground_truth]
in_container = true
render_command = "bash solution/render.sh"
render_outputs = [
  { path = "/tmp/output/rendering.mp4", required = true, description = "..." },
]
```

Iterate: `uv run lbx-rl-harness reference --problem-dir problems/<task_id>`

Validate: `uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/<task_id>`
(oracle must match the declared `reward_type` target; commits `.alignerr/ground_truth/rendering.mp4` + build proof).
