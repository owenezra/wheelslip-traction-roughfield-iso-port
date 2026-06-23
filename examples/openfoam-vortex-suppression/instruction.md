# Passive Suppression of Vortex Shedding

Uniform flow past a 2D circular cylinder at low Reynolds number sheds an
unsteady **von Karman vortex street**: vortices peel off alternately from the
top and bottom, driving a large oscillating lift force on the body. Your job is
to **kill that oscillation passively** by placing a **splitter plate** in the
wake.

## What to submit

Write a JSON file to:

```text
/tmp/output/control.json
```

with two numbers (both in units of the cylinder diameter `D`):

```json
{
  "plate_gap": 0.5,
  "plate_length": 2.0
}
```

- `plate_gap` — streamwise distance from the **base of the cylinder** to the
  **leading edge** of the plate (`0` = attached to the cylinder).
- `plate_length` — length of the plate along the wake centreline.

The plate is a thin, rigid wall placed on the wake centreline (`y = 0`) directly
behind the cylinder. You are choosing **where** it starts and **how long** it is.

## Required OpenFOAM Diagnostic

Before finalizing `control.json`, you must run a public OpenFOAM diagnostic in
the task container and use its output to sanity-check your design. The diagnostic
does not reveal the hidden scoring cases; it only confirms that the OpenFOAM
stack is usable and that your proposed plate parameters produce a valid public
probe mesh.

The public probe generator is available at:

```text
/data/openfoam_probe.py
```

After writing a draft `control.json`, run commands like:

```bash
python /data/openfoam_probe.py /tmp/output/control.json /tmp/openfoam_probe_case
bash -lc 'source /etc/solver-envs.d/openfoam.sh && blockMesh -case /tmp/openfoam_probe_case && checkMesh -case /tmp/openfoam_probe_case'
```

Your transcript must show OpenFOAM evidence such as sourcing
`/etc/solver-envs.d/openfoam.sh`, running `blockMesh` and `checkMesh`, and
inspecting successful mesh output such as `Mesh OK` before you leave the final
answer in `/tmp/output/control.json`.

## Objective

The hidden scorer runs the full OpenFOAM flow solver and measures the **RMS of
the lift coefficient**, `RMS(Cl)`, over a fixed time window once the wake is
fully developed. Lower is better: a perfectly suppressed wake has
`RMS(Cl) -> 0`. Your score rewards how much you reduce `RMS(Cl)` **relative to
the bare cylinder (no plate)**.

Your design is evaluated across **several hidden Reynolds numbers** and scored on
the **worst case** — a plate tuned to a single condition will not score well.

## Constraints (feasibility)

- `0 <= plate_gap <= 4` (D) and `0 < plate_length <= 8` (D).
- The plate must stay inside the simulation domain.
- **Mean drag is capped.** You cannot "win" by drowning the wake in drag (e.g.
  an enormous plate). Designs whose mean drag coefficient exceeds the cap are
  penalised.
- A simulation that diverges or produces non-physical values scores zero for
  that case.

## Physics hints

- A splitter plate works by interfering with the feedback between the two
  shear layers that sustains shedding. Both the **gap** and the **length**
  matter; classic results show suppression improves up to a plate length of a
  few diameters.
- There is a trade-off: too short or too far downstream and shedding survives;
  unnecessarily large and you pay in drag.

Only `control.json` is graded. Write nothing else to `/tmp/output`.
