#!/usr/bin/env bash
set -euo pipefail
# Reviewer artifact: animated side-by-side vorticity wake (bare cylinder vs the
# oracle splitter plate) -> /tmp/output/rendering.mp4 (1280x720 h264).
# Runs inside the task image where OpenFOAM + matplotlib + ffmpeg are available.
cd "$(dirname "$0")/.."
PY=/mcp_server/.venv/bin/python
[ -x "$PY" ] || PY=python3
"$PY" solution/render_movie.py
