#!/usr/bin/env bash
set -euo pipefail
trap 'echo "install-common.sh failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

export DEBIAN_FRONTEND=noninteractive
export UV_PYTHON_INSTALL_DIR=/opt/uv-python

echo ":: install-common: installing apt packages"
apt-get update
apt-get install -y --no-install-recommends \
  bash \
  build-essential \
  ca-certificates \
  curl \
  ffmpeg \
  git \
  graphviz \
  libgl1 \
  libglib2.0-0 \
  libgomp1 \
  libngspice0-dev \
  libsndfile1 \
  ngspice \
  pkg-config \
  tmux
rm -rf /var/lib/apt/lists/*

# CPU and GPU images run the Taiga runtime on Python 3.13; the TPU flavor pins
# 3.12 (jax[tpu]/jaxlib wheels target 3.12). Override with PYTHON_VERSION.
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"
echo ":: install-common: installing Python ${PYTHON_VERSION}"
mkdir -p "${UV_PYTHON_INSTALL_DIR}"
uv python install "${PYTHON_VERSION}"
python_bin="$(uv python find "${PYTHON_VERSION}")"

mkdir -p /mcp_server /tmp/output /data /grader/data /workdir /runtime /tmp/hf-cache/hub
# Taiga tools run as a non-root uid. Keep public work/output directories
# writable even when task Dockerfiles do not repair ownership themselves.
# /tmp/hf-cache is HF_HOME (see base Dockerfiles): agent-writable cache and the
# parent for deploy-time read-only Hugging Face weight mounts (preloaded_files),
# so from_pretrained resolves mounted weights without a network fetch.
chmod 0777 /tmp/output /workdir /tmp/hf-cache /tmp/hf-cache/hub

# Unprivileged account that the rubric server's agent-facing tools and the
# grader's PolicyWorker drop to. Real /etc/passwd entry + writable home so
# tools that read $HOME (numpy/mujoco/pip caches, shell history, …) work.
echo ":: install-common: creating agent user (uid 1000)"
groupadd -g 1000 agent
useradd -u 1000 -g 1000 -m -d /home/agent -s /bin/bash agent

echo ":: install-common: creating runtime venv"
uv venv --python "${python_bin}" /mcp_server/.venv

runtime_constraints=()
if [[ -f /tmp/base/requirements-runtime.txt ]]; then
  echo ":: install-common: installing pinned rubric/grader runtime dependencies"
  uv pip install --python /mcp_server/.venv/bin/python --no-cache -r /tmp/base/requirements-runtime.txt
  runtime_constraints=(-c /tmp/base/requirements-runtime.txt)
fi

echo ":: install-common: installing rubric package"
uv pip install --python /mcp_server/.venv/bin/python --no-cache --no-deps -e /mcp_server

if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  echo ":: install-common: installing torch packages from ${TORCH_INDEX_URL}"
  read -r -a torch_packages <<< "${TORCH_PACKAGES:-torch torchvision torchaudio}"
  # CUDA wheels use local versions like +cu124; uv's first-index safety can hide
  # them once PyPI has a plain `torch` release, so CUDA bases opt into best-match.
  torch_index_strategy="${TORCH_INDEX_STRATEGY:-first-index}"
  uv pip install \
    --python /mcp_server/.venv/bin/python \
    --no-cache \
    --index-url "${TORCH_INDEX_URL}" \
    --extra-index-url https://pypi.org/simple \
    --index-strategy "${torch_index_strategy}" \
    "${torch_packages[@]}"
fi

echo ":: install-common: installing common requirements"
uv pip install --python /mcp_server/.venv/bin/python --no-cache "${runtime_constraints[@]}" -r /tmp/base/requirements-common.txt

if [[ -f /tmp/base/requirements-solvers.txt ]]; then
  echo ":: install-common: installing numerical-solver stack"
  uv pip install --python /mcp_server/.venv/bin/python --no-cache "${runtime_constraints[@]}" -r /tmp/base/requirements-solvers.txt
fi

if [[ -f /tmp/base/install-solvers-heavy.sh ]]; then
  echo ":: install-common: installing heavy binary solver engines"
  bash /tmp/base/install-solvers-heavy.sh
fi

if [[ -n "${BASE_EXTRA_REQUIREMENTS:-}" && -f "${BASE_EXTRA_REQUIREMENTS}" ]]; then
  echo ":: install-common: installing extra requirements from ${BASE_EXTRA_REQUIREMENTS}"
  uv pip install --python /mcp_server/.venv/bin/python --no-cache "${runtime_constraints[@]}" -r "${BASE_EXTRA_REQUIREMENTS}"
fi

if [[ -f /runtime/grading/pyproject.toml ]]; then
  echo ":: install-common: installing grading runtime"
  uv pip install --python /mcp_server/.venv/bin/python --no-cache --no-deps -e /runtime/grading
fi

# Taiga tools and task scripts expect these names on PATH. Use a wrapper for
# python: CPython follows symlinks to the base interpreter, which bypasses the
# venv and drops baked packages like numpy from sys.path.
rm -f /usr/local/bin/python /usr/local/bin/python3
cat > /usr/local/bin/python <<'PYTHON_WRAPPER'
#!/bin/sh
exec /mcp_server/.venv/bin/python "$@"
PYTHON_WRAPPER
chmod 0755 /usr/local/bin/python
ln -sf /usr/local/bin/python /usr/local/bin/python3
ln -sf /mcp_server/.venv/bin/pip /usr/local/bin/pip
ln -sf /mcp_server/.venv/bin/rubric /usr/local/bin/rubric

# Taiga's Bash tool is launched as a non-root user and initializes via
# `python`. Keep the shared runtime traversable; task images should lock
# only private grader directories such as /mcp_server/data and
# /mcp_server/grader.
chmod -R a+rX "${UV_PYTHON_INSTALL_DIR}"
chmod 0755 /mcp_server /mcp_server/.venv /mcp_server/.venv/bin

cat > /runtime/run_grader.py <<'PY'
#!/usr/bin/env python
from grader_runner.run_grader import main

raise SystemExit(main())
PY
chmod +x /runtime/run_grader.py
