# OpenSees Three-Story Retrofit

This task asks an agent to design a budget-constrained supplemental retrofit for
a three-story, two-bay steel moment frame inspired by the public OpenSeesPy
`ThreeStorySteel` example.

The submitted artifact is `/tmp/output/retrofit_design.json`. It contains a list
of simplified equivalent diagonal devices: `viscous_damper` and `steel_brace`.
The verifier validates the JSON, checks the budget, runs deterministic
OpenSeesPy static analyses, and returns a continuous score in `[0, 1]`.

The task intentionally keeps the submitted artifact declarative. Agents do not
submit OpenSees code; the hidden scorer owns the model and fixed analysis cases.
This makes the reward reproducible and keeps private fixtures separate from the
public task prompt.

Ground-truth rendering uses `opsvis` and matplotlib to produce
`/tmp/output/rendering.mp4`, plus auxiliary PNG plots for local review.
