#!/usr/bin/env bash
set -euo pipefail
# In-image smoke test: grade whatever is in /tmp/output with the real scorer
# (which invokes OpenFOAM). Used for manual debugging inside the task image.
mkdir -p /logs/verifier
/mcp_server/.venv/bin/python - <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, "/mcp_server")
sys.path.insert(0, "/mcp_server/grader")
from grader.compute_score import compute_score

result = compute_score(Path("/tmp/output"), None, Path("/mcp_server/data"))
Path("/logs/verifier/reward.json").write_text(json.dumps(result, indent=2))
print("score:", result.get("score"))
PY
