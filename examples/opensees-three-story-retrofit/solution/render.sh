#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
mkdir -p "${OUT_DIR}"
export MPLCONFIGDIR="${OUT_DIR}/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="uv run python"
fi

${PYTHON_BIN} scorer/render_review.py \
  --design "${OUT_DIR}/retrofit_design.json" \
  --out-dir "${OUT_DIR}"
