# OpenSees Three-Story Steel Frame Retrofit

Design a supplemental retrofit layout for a 2D three-story, two-bay steel moment
frame inspired by the OpenSeesPy `ThreeStorySteel` example. The frame uses
10 ft story heights, 16 ft bay widths, fixed base nodes, W-section steel members,
gravity loading, and pushover-style lateral loading.

Your final answer must be one JSON file:

```text
/tmp/output/retrofit_design.json
```

Do not put the final answer anywhere else.

## Public Files

The following files are available in `/data`:

- `/data/frame_description.md`: structural layout, node/story/bay convention,
  design rules, and scoring summary.
- `/data/design_schema.json`: JSON schema for the required submission format.
- `/data/public_model_summary.py`: lightweight public helper with geometry,
  cost, and schema constants.
- `/data/opensees_probe.py`: public OpenSeesPy diagnostic model that imports
  `openseespy.opensees as ops`, builds a small frame, calls `ops.analyze`, and
  prints response quantities for your draft design.

## Required OpenSeesPy Diagnostic

Before finalizing `retrofit_design.json`, you must run OpenSeesPy in the task
container and use the numerical output to sanity-check your design. The public
probe is intentionally smaller than the hidden verifier model, but it exercises
the same OpenSeesPy stack and responds to your submitted device layout.

After writing a draft design, run:

```bash
/mcp_server/.venv/bin/python /data/opensees_probe.py /tmp/output/retrofit_design.json
```

Your transcript must show OpenSeesPy evidence such as importing
`openseespy.opensees as ops`, creating the model with calls like `ops.model`,
`ops.node`, and `ops.element`, running `ops.analyze`, and inspecting printed
outputs such as `analyze_code`, `roof_disp`, `max_drift_ratio`, or
`base_shear_proxy`. Use those diagnostics to refine or confirm the final JSON.

## Design Goal

Choose supplemental devices that improve the frame's deterministic OpenSeesPy
response while staying within the retrofit budget. The hidden verifier compares
your design against an unretrofitted baseline and a reference oracle over fixed
analysis cases.

The score rewards:

- lower peak interstory drift ratio,
- lower roof displacement under a fixed service lateral load,
- higher stable base shear capacity during pushover,
- lower cost relative to the reference design,
- successful OpenSees analysis convergence.

## Required JSON Format

The top-level JSON object must contain exactly one key named `devices`.
`devices` must be a list with 0 to 6 entries.

Each entry must be one of these two shapes:

```json
{
  "story": 1,
  "bay": 1,
  "type": "viscous_damper",
  "coefficient": 45.0
}
```

or:

```json
{
  "story": 2,
  "bay": 2,
  "type": "steel_brace",
  "area": 2.0
}
```

Valid values:

- `story` must be an integer: `1`, `2`, or `3`.
- `bay` must be an integer: `1` or `2`.
- `type` must be either `"viscous_damper"` or `"steel_brace"`.
- `coefficient` is required for `"viscous_damper"` and must be a number from
  `10.0` to `90.0`, inclusive.
- `area` is required for `"steel_brace"` and must be a number from `0.5` to
  `4.0`, inclusive.
- Do not include both `coefficient` and `area` on the same device.
- Do not repeat the same `(story, bay, type)` combination.
- Use at most 6 devices total.

## Cost Constraint

Your total cost must be at most `100.0`.

Cost is computed as:

```text
viscous_damper cost = 4.0 + 0.55 * coefficient
steel_brace cost = 6.0 + 8.0 * area
```

For example, this device:

```json
{"story": 1, "bay": 1, "type": "viscous_damper", "coefficient": 40.0}
```

costs:

```text
4.0 + 0.55 * 40.0 = 26.0
```

If the submitted design is invalid or over budget, it receives score `0.0`.

## Story And Bay Convention

Stories are numbered from bottom to top:

- `story = 1`: base to first floor.
- `story = 2`: first floor to second floor.
- `story = 3`: second floor to roof.

Bays are numbered from left to right:

- `bay = 1`: between the left and middle column lines.
- `bay = 2`: between the middle and right column lines.

The verifier models each device as an equivalent diagonal retrofit link in the
selected story and bay. This is intentional: submit the design variables above,
not an OpenSees model file.

## Example Valid Submission

```json
{
  "devices": [
    {"story": 1, "bay": 1, "type": "viscous_damper", "coefficient": 55.0},
    {"story": 1, "bay": 2, "type": "steel_brace", "area": 2.5},
    {"story": 2, "bay": 1, "type": "viscous_damper", "coefficient": 35.0}
  ]
}
```

Remember: write only the final JSON file to
`/tmp/output/retrofit_design.json`.
