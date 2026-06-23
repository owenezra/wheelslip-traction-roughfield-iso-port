#!/usr/bin/env bash
set -euo pipefail
# Naive baseline: no plate at all -> the bare cylinder sheds at full amplitude,
# so suppression is zero and the score should be near 0.
OUT="${LBT_OUTPUT_DIR:-/tmp/output}"
mkdir -p "${OUT}"
cat > "${OUT}/control.json" <<'JSON'
{
  "plate_gap": 0.0,
  "plate_length": 0.0
}
JSON
