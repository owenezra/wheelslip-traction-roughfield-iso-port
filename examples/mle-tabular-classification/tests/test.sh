#!/usr/bin/env bash
set -euo pipefail

exec /runtime/run_grader.py \
  --workspace /tmp/output \
  --grader-dir /mcp_server/grader \
  --private-dir /mcp_server/data \
  --output-dir /logs/verifier
