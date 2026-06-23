# Reference solution

`solve.sh` trains a deterministic public-only friction identifier from episodes
produced by `/data/env/practice_env.py`, then stages `policy.py` and its sibling
`policy.pt` in `/tmp/output`.

The reference uses the simulator's disclosed physical friction support, a
histogram gradient boosting regressor, and a self-contained physics-informed
CEM planner. It does not import or read scorer fixtures.
