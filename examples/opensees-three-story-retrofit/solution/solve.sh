#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/output
if [[ -d /tmp/verifier ]]; then
  exec > >(tee -a /tmp/verifier/transcript.txt) 2> >(tee -a /tmp/verifier/transcript.txt >&2)
fi
cat > /tmp/output/retrofit_design.json <<'JSON'
{
  "devices": [
    {"story": 1, "bay": 1, "type": "steel_brace", "area": 2.3},
    {"story": 1, "bay": 2, "type": "steel_brace", "area": 2.1},
    {"story": 2, "bay": 1, "type": "viscous_damper", "coefficient": 34.0},
    {"story": 2, "bay": 2, "type": "steel_brace", "area": 1.3},
    {"story": 3, "bay": 1, "type": "viscous_damper", "coefficient": 17.0}
  ]
}
JSON

echo "Running OpenSeesPy probe: /mcp_server/.venv/bin/python /data/opensees_probe.py /tmp/output/retrofit_design.json"
/mcp_server/.venv/bin/python /data/opensees_probe.py /tmp/output/retrofit_design.json
