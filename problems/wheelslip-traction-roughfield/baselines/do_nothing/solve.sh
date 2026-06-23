#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p /tmp/output
cp "$SCRIPT_DIR/policy.py" /tmp/output/policy.py
