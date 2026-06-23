# ML_Envs Migration

The first migration goal is to express existing `ML_Envs` tasks in the universal task layout.

Mapping:

- `problems-metadata.json` prompt -> `instruction.md`
- `test_file.compute_score()` -> `scorer/compute_score.py`
- `data/public` -> `data/`
- `data/private` -> `scorer/data/`
- `required_resources` -> `task.toml [environment].required_resources`
- ML_Envs base/rubric server -> `base/cpu` and `base/gpu`

The Taiga exporter reconstructs `problems-metadata.json` with a small `test_file` shim that imports the image-baked grader.

Open migration tasks:

- port the rubric MCP server from `ML_Envs/shared/rubric`,
- port `taiga-core` from `ML_Envs/shared/taiga-core`,
- replace `ML_Envs/scripts/build_and_push.sh` with the WIF/GAR workflow in this repo,
- adapt Taiga polling from `ML_Envs/scripts/poll_taiga_runs.py`.
