# OpenFOAM Vortex-Suppression Example

Canonical reference task for the **CFD / OpenFOAM** domain, the fluids analogue
of [`examples/mujoco-pendulum`](../mujoco-pendulum). It follows
[`project_guidelines/cfd/cfd_environments.md`](../../project_guidelines/cfd/cfd_environments.md)
and reuses the deterministic-scoring contract with a real PDE solver as the
referee. Do not submit work by editing this example.

## What the agent must do

Design a **wake splitter plate** that passively suppresses the von Karman vortex
street behind a 2D cylinder, and write its parameters to:

```text
/tmp/output/control.json      # {"plate_gap": <D>, "plate_length": <D>}
```

See [`instruction.md`](instruction.md) for the full prompt.

## What the grader checks

`scorer/compute_score.py` (a `RubricBuilder` rubric) runs OpenFOAM for each
hidden Reynolds number: it builds a **fixed structured cylinder mesh**
(`blockMesh`, never `snappyHexMesh`), inserts the plate as a **zero-thickness
baffle** (`topoSet` + `createBaffles`) on the wake centreline, runs laminar
`pimpleFoam` **serially** with a **fixed timestep and step count**, and measures
`RMS(Cl)` over a fixed trailing window. 12 deterministic criteria across four
strata:

- Structural: control.json parses; plate within the feasibility shell; plate present.
- Static: mesh + baffle build; solver completes with finite forces; mean-drag cap.
- Solve: `RMS(Cl)` suppression vs the hidden no-plate baseline; low residual shedding.
- Robustness: worst-case suppression across the hidden Reynolds sweep; drag cap and
  convergence for every case; the plate never amplifies shedding.

Numerical instability (divergence / NaN / non-completion) zeroes the case and
triggers a penalty.

## Determinism

`scorer/openfoam_case.py` pins everything that affects the result: OpenFOAM
v2412 (base image), serial execution, fixed mesh, fixed `deltaT`/`endTime`, fixed
schemes, a fixed transverse initial kick to seed the (super-critical) instability
reproducibly, and a fixed analysis window. The hidden no-plate baselines in
`scorer/data/expected.json` are precomputed with the same settings — never
regenerated at grade time.

## Reviewer artifact

`solution/render.sh` -> `solution/render_movie.py` runs the bare cylinder and the
oracle plate, then animates a **side-by-side vorticity wake** to
`/tmp/output/rendering.mp4` (1280x720 h264) showing the street being suppressed.
The ground-truth run commits it to `.alignerr/ground_truth/rendering.mp4`, the
same artifact flow as the MuJoCo example (enabled by the generalized
`[ground_truth].render_outputs` contract + `in_container = true`).

## Files

- `task.toml` — `task_type = "cfd"`, `domain = "vortex_suppression"`, `[ground_truth] in_container=true` + render contract.
- `scorer/compute_score.py` — the rubric.
- `scorer/openfoam_case.py` — deterministic case builder / runner / analyzer.
- `scorer/data/` — hidden Reynolds sweep + precomputed baseline anchors.
- `solution/solve.sh` — oracle (suppressing plate, scores 1.0).
- `solution/render.sh`, `render_movie.py`, `render_config.py` — reviewer video.
- `baselines/naive.sh` — no plate (scores ~0).
- `data/control_template.json` — public starter parameters.

## Reference physics (Re=100, this mesh)

Bare cylinder: `mean Cd ~ 1.40`, `RMS(Cl) ~ 0.25`, `St ~ 0.167` (matches
literature). Oracle plate (`gap=0.5`, `length=2.0`): `RMS(Cl) ~ 0` (suppressed),
`mean Cd ~ 1.14`.
