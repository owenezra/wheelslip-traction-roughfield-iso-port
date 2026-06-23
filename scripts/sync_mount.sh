#!/usr/bin/env bash
# Pack a task's declared [[preloaded_files]] into content-addressed squashfs
# objects, upload them once to the shared cache, and stamp the deploy manifest
# (.alignerr/preloaded_files.json). The exporter then emits that manifest so the
# platform mounts each object read-only at deploy time -- instead of baking large
# datasets / model weights into the per-task image.
#
# Usage: scripts/sync_mount.sh <problem-dir>
# Requires: mksquashfs, jq, curl, python3; TAIGA_ENV_ID + TAIGA_TOKEN for upload.
# Ported/adapted from ML_Envs/scripts/sync_mount.sh.
set -euo pipefail

PROBLEM_DIR="${1:?usage: sync_mount.sh <problem-dir>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_ID="$(basename "$PROBLEM_DIR")"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Build the mount list as TSV: source<TAB>hf_repo<TAB>hf_revision<TAB>mount_path<TAB>read_only.
# Trusted CI auto-mounts the conventional dataset dirs for known task types (e.g.
# ml: data/ -> /data, scorer/data -> /mcp_server/data) so the raw dataset is
# mounted read-only instead of baked into the image, plus any explicitly declared
# [[preloaded_files]] (HF repos / extra trees). A declared mount_path wins over
# an auto one at the same path.
ENTRY_SEP=$'\x1f'
mapfile -t ENTRIES < <(cd "$REPO_ROOT" && python3 -c "
import sys, tomllib
from pathlib import Path
sys.path.insert(0, 'alignerr_plugin/src')
from alignerr_plugin.preloaded import auto_mount_entries
sep = '\x1f'
pd = Path('$PROBLEM_DIR')
data = tomllib.load(open(pd / 'task.toml', 'rb'))
task_type = (data.get('difficulty', {}) or {}).get('task_type', '')
declared = data.get('preloaded_files', []) or []
declared_paths = {e['mount_path'] for e in declared}
for source_rel, mount_path in auto_mount_entries(pd, task_type):
    if mount_path in declared_paths:
        continue
    print(sep.join([source_rel, '', '', mount_path, 'true']))
for e in declared:
    print(sep.join([
        e.get('source',''), e.get('hf_repo',''), e.get('hf_revision',''),
        e['mount_path'], 'true' if e.get('read_only', True) else 'false',
    ]))
")

if [[ "${#ENTRIES[@]}" -eq 0 ]]; then
  echo ":: $TASK_ID has no datasets to mount (no auto dirs, no preloaded_files)"
  exit 0
fi

STAMP_ARGS=()
idx=0
for line in "${ENTRIES[@]}"; do
  IFS="${ENTRY_SEP}" read -r source hf_repo hf_revision mount_path read_only <<< "$line"
  idx=$((idx + 1))
  name="m${idx}"

  staging="${WORK}/${name}"
  mkdir -p "$staging"
  if [[ -n "$source" ]]; then
    cp -a "${PROBLEM_DIR}/${source}/." "$staging/"
  else
    echo ":: fetching HF repo ${hf_repo}@${hf_revision:-main}"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='${hf_repo}', revision='${hf_revision}' or None, local_dir='${staging}')
"
  fi

  address="$(cd "$REPO_ROOT" && python3 -c "
import sys; sys.path.insert(0, 'alignerr_plugin/src')
from pathlib import Path
from alignerr_plugin.preloaded import content_address
print(content_address(Path('${staging}')))
")"
  remote_name="$(cd "$REPO_ROOT" && python3 -c "
import sys; sys.path.insert(0, 'alignerr_plugin/src')
from alignerr_plugin.preloaded import remote_squashfs_name
print(remote_squashfs_name('${TASK_ID}', '${name}', '${address}'))
")"

  squashfs="${WORK}/${name}-${address}.squashfs"
  echo ":: packing ${name} (${address}) -> ${squashfs}"
  mksquashfs "$staging" "$squashfs" -noappend -quiet

  remote_path="$(bash "${REPO_ROOT}/scripts/upload_squashfs.sh" "$squashfs" "$remote_name")"
  STAMP_ARGS+=(--entry "${mount_path}::${remote_path}::${read_only}")
done

python3 "${REPO_ROOT}/scripts/stamp_preloaded_files.py" \
  --problem-dir "$PROBLEM_DIR" "${STAMP_ARGS[@]}"
echo ":: synced ${idx} mount(s) for ${TASK_ID}"
