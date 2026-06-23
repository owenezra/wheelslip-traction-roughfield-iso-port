# Public Frame Description

This task uses a deterministic OpenSeesPy model inspired by the public
OpenSeesPy three-story steel frame example, "Three story steel building with
rigid beam-column connections and W-section". The original example builds a
2D steel frame with W-section fiber members, fixed bases, gravity loading, and
a displacement-controlled pushover analysis.

## Geometry

- Number of stories: 3
- Number of bays: 2
- Story height: 10 ft = 120 in
- Bay width: 16 ft = 192 in
- Units: kip, inch, second
- Base column nodes are fixed.
- Floor nodes are located at each column line and each story level.

Story numbering:

- Story 1 spans from the base to the first floor.
- Story 2 spans from the first floor to the second floor.
- Story 3 spans from the second floor to the roof.

Bay numbering:

- Bay 1 is between the left and middle column lines.
- Bay 2 is between the middle and right column lines.

## Baseline Member Properties

The hidden verifier uses a frame based on the OpenSeesPy example's steel
material, W-section dimensions, gravity loads, and pushover force pattern. The
public task does not require you to edit or submit an OpenSees model. You only
submit a retrofit design JSON file.

## Retrofit Device Interpretation

Each submitted device is modeled by the verifier as an equivalent diagonal
retrofit link in the selected story and bay.

- `steel_brace` devices add high axial stiffness through a diagonal steel link.
- `viscous_damper` devices add a lower equivalent static stiffness link used by
  the deterministic static analyses in this task.

Both device types are simplified for the task reward. The task is about choosing
effective locations and magnitudes under a budget, not writing OpenSees syntax.

## Cost Formula

```text
viscous_damper cost = 4.0 + 0.55 * coefficient
steel_brace cost = 6.0 + 8.0 * area
```

The total cost must be at most `100.0`.

## Scoring Summary

The hidden verifier runs fixed OpenSeesPy analyses and reports a continuous
score in `[0, 1]`. Invalid or over-budget designs receive `0.0`. Valid designs
are scored against deterministic baseline and oracle anchors for:

- service-load interstory drift,
- service-load roof displacement,
- pushover base shear capacity,
- cost efficiency,
- analysis convergence.

The final score is not a hand-graded rubric. It is computed from deterministic
OpenSeesPy results and hidden fixed cases.
