#!/usr/bin/env bash
set -euo pipefail
# Oracle: a splitter plate close behind the cylinder, ~2D long, which fully
# suppresses the von Karman street (RMS(Cl) -> 0) while reducing base drag.
OUT="${LBT_OUTPUT_DIR:-/tmp/output}"
mkdir -p "${OUT}"
if [[ -d /tmp/verifier ]]; then
  exec > >(tee -a /tmp/verifier/transcript.txt) 2> >(tee -a /tmp/verifier/transcript.txt >&2)
fi
cat > "${OUT}/control.json" <<'JSON'
{
  "plate_gap": 0.5,
  "plate_length": 2.0
}
JSON

PROBE_CASE="/tmp/openfoam_probe_case"
rm -rf "${PROBE_CASE}"
python /data/openfoam_probe.py "${OUT}/control.json" "${PROBE_CASE}"
echo "Running OpenFOAM probe: source /etc/solver-envs.d/openfoam.sh && blockMesh -case ${PROBE_CASE} && checkMesh -case ${PROBE_CASE}"
bash -lc "source /etc/solver-envs.d/openfoam.sh && blockMesh -case ${PROBE_CASE} && checkMesh -case ${PROBE_CASE}"
