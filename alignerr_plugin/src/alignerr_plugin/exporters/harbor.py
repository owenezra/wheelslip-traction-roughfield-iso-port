"""Export a task directory in self-contained Harbor task format.

The export intentionally does not depend on Boreal Artifact Registry images.
It writes an ``environment/Dockerfile`` plus the grader/scorer/runtime sources
needed to build the task image from the exported directory itself.

Customer flow once exported::

    harbor run -p ./harbor_task -a my-agent -m my-model
    # then
    reward = json.loads("logs/verifier/reward.json")["score"]
    details = json.loads("logs/verifier/reward-details.json")
    for sub in details["structured_subscores"]:
        log_to_wandb(sub["name"], sub["score"], sub["weight"])

The generated ``tests/test.sh`` invokes ``/runtime/run_grader.py`` inside the
self-contained image. ``task.toml`` is not stamped with a remote docker image;
Harbor users can build the image from ``environment/Dockerfile``.
"""

from __future__ import annotations

import shutil
import tomllib
from pathlib import Path

import tomli_w

from alignerr_plugin.runtime_notices import runtime_notices_for_resources
from alignerr_plugin.utils import load_task_toml

# Repo-root-relative paths to runtime sources we bundle into the Harbor export.
# This file is at: <repo>/alignerr_plugin/src/alignerr_plugin/exporters/harbor.py
# So parents[4] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_GRADER_DIR = _REPO_ROOT / "grader"
_GRADING_SRC = _GRADER_DIR / "src" / "grading"
_RUN_GRADER_SRC = _GRADER_DIR / "src" / "grader_runner" / "run_grader.py"
_RUBRIC_DIR = _REPO_ROOT / "taiga_runtime" / "rubric"
_BASE_DIR = _REPO_ROOT / "base"

_SELF_CONTAINED_DOCKERFILE = """\
FROM python:3.13-slim

ENV UV_SYSTEM_PYTHON=1
ENV PATH="/mcp_server/.venv/bin:/usr/local/bin:${PATH}"
ENV TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV BASE_EXTRA_REQUIREMENTS=/tmp/base/requirements-cpu.txt

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /mcp_server

COPY taiga_runtime/rubric/ /mcp_server/
COPY grader/ /runtime/grading/
COPY base/requirements-runtime.txt base/requirements-common.txt base/requirements-cpu.txt /tmp/base/
COPY --chmod=0755 base/install-common.sh /tmp/base/install-common.sh
RUN /tmp/base/install-common.sh

COPY --chown=root:root scorer/data/ /mcp_server/data/
COPY --chown=root:root scorer/ /mcp_server/grader/
COPY data/ /workspace/data/
RUN ln -s /workspace/data /data
COPY task.toml instruction.md /task/
RUN rm -rf /mcp_server/grader/data \
    && chown -R root:root /mcp_server/data /mcp_server/grader \
    && chmod 0755 /mcp_server \
    && find /mcp_server/data /mcp_server/grader -type d -exec chmod 0700 {} + \
    && find /mcp_server/data /mcp_server/grader -type f -exec chmod 0600 {} +

CMD ["/bin/bash"]
"""


_TEST_SH_DEFAULT = """\
#!/bin/bash
set -euo pipefail

# Invoke the grader runner built into the self-contained Harbor image.
exec /runtime/run_grader.py \\
    --workspace /app \\
    --grader-dir /mcp_server/grader \\
    --private-dir /mcp_server/data \\
    --output-dir /logs/verifier
"""

_TEST_SH_STANDALONE = """\
#!/bin/bash
set -euo pipefail

# Prefer the image-baked runner, fall back to the bundled copy when the
# image is not built FROM lbx-tasks-base. The bundled copy lives at
# /tests/_runtime/ and is byte-identical to what the base image ships.
if [ -x /runtime/run_grader.py ]; then
    exec /runtime/run_grader.py \\
        --workspace /app \\
        --grader-dir /mcp_server/grader \\
        --private-dir /mcp_server/data \\
        --output-dir /logs/verifier
fi

export PYTHONPATH="/tests/_runtime${PYTHONPATH:+:${PYTHONPATH}}"
exec python /tests/_runtime/run_grader.py \\
    --workspace /app \\
    --grader-dir /mcp_server/grader \\
    --private-dir /mcp_server/data \\
    --output-dir /logs/verifier
"""


def _grader_uses_llm(problem_dir: Path) -> bool:
    """Best-effort static check: does the task scorer call into the LLM judge?"""
    grader_path = problem_dir / "scorer" / "compute_score.py"
    if not grader_path.exists():
        return False
    try:
        text = grader_path.read_text()
    except OSError:
        return False
    return "llm_criterion" in text or "LLMJudge" in text


def _ensure_verifier_env_for_llm(task_toml_path: Path) -> None:
    """Add ``ANTHROPIC_API_KEY`` to ``[verifier.env]`` so ``harbor run``
    requests it from the user. Caller decides whether to invoke this."""
    data = tomllib.loads(task_toml_path.read_text())
    verifier = data.setdefault("verifier", {})
    env = verifier.setdefault("env", [])
    if isinstance(env, list):
        if "ANTHROPIC_API_KEY" not in env:
            env.append("ANTHROPIC_API_KEY")
    elif isinstance(env, dict):
        env.setdefault("ANTHROPIC_API_KEY", "${ANTHROPIC_API_KEY}")
    task_toml_path.write_text(tomli_w.dumps(data))


def _stamp_image_ref(task_toml_path: Path, image_ref: str | None) -> None:
    if not image_ref:
        return
    data = tomllib.loads(task_toml_path.read_text())
    environment = data.setdefault("environment", {})
    environment["docker_image"] = image_ref
    task_toml_path.write_text(tomli_w.dumps(data))


def _notice_resource_for_problem(problem_dir: Path) -> str:
    """Return the Taiga resource tier for runtime notice generation."""
    task_toml = load_task_toml(problem_dir)
    return task_toml.environment.required_resources


def _stamp_runtime_notices(task_toml_path: Path, problem_dir: Path) -> None:
    resource = _notice_resource_for_problem(problem_dir)
    data = tomllib.loads(task_toml_path.read_text())
    metadata = data.setdefault("metadata", {})
    notices = runtime_notices_for_resources(resource, metadata.get("runtime_notices"))
    if notices:
        metadata["runtime_notices"] = notices
    else:
        metadata.pop("runtime_notices", None)
    task_toml_path.write_text(tomli_w.dumps(data))


def _copy_standalone_runtime(tests_dir: Path) -> None:
    """Bundle a copy of the shared `grader/` package and `run_grader.py` under tests/_runtime/.

    Used by ``--standalone`` so the Harbor task runs even when the image is
    not built from `lbx-tasks-base`.
    """
    runtime_dir = tests_dir / "_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    # Copy the grading package so `from grading import ...` resolves.
    # Exclude __pycache__ so the export is reproducible across machines.
    grading_dst = runtime_dir / "grading"
    if grading_dst.exists():
        shutil.rmtree(grading_dst)
    shutil.copytree(
        _GRADING_SRC,
        grading_dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    shutil.copy2(_RUN_GRADER_SRC, runtime_dir / "run_grader.py")
    (runtime_dir / "run_grader.py").chmod(0o755)


def _copytree_clean(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )


def _write_self_contained_environment(problem_dir: Path, output_dir: Path) -> None:
    environment_dir = output_dir / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    (environment_dir / "Dockerfile").write_text(_SELF_CONTAINED_DOCKERFILE)

    for name in ("task.toml", "instruction.md"):
        source = problem_dir / name
        if source.exists():
            shutil.copy2(source, environment_dir / name)

    for name in ("data", "scorer"):
        source = problem_dir / name
        destination = environment_dir / name
        if source.exists():
            _copytree_clean(source, destination)
        else:
            destination.mkdir(parents=True, exist_ok=True)

    _copytree_clean(_GRADER_DIR, environment_dir / "grader")
    _copytree_clean(_RUBRIC_DIR, environment_dir / "taiga_runtime" / "rubric")
    _copytree_clean(_BASE_DIR, environment_dir / "base")

    original_environment = problem_dir / "environment"
    if original_environment.exists():
        _copytree_clean(original_environment, environment_dir / "source_environment")


def export_harbor(
    problem_dir: Path,
    output_dir: Path,
    *,
    image_ref: str | None = None,
    standalone: bool = True,
    include_runtime_notices: bool = True,
) -> Path:
    """Write a Harbor-format copy of the task into ``output_dir``.

    Files copied or generated:

      * ``task.toml``            (``ANTHROPIC_API_KEY`` added to ``[verifier].env`` if
                                 the grader uses LLM judging; optional runtime
                                 notice metadata stamped)
      * ``instruction.md``
      * ``environment/``         (self-contained Dockerfile + runtime/grader/scorer files)
      * ``solution/``            (optional Oracle solver)
      * ``tests/test.sh``        (shim that calls /runtime/run_grader.py)
    """
    _ = image_ref, standalone
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for name in ["task.toml", "instruction.md", "solution"]:
        source = problem_dir / name
        destination = output_dir / name
        if source.is_dir():
            _copytree_clean(source, destination)
        elif source.exists():
            shutil.copy2(source, destination)

    _write_self_contained_environment(problem_dir, output_dir)

    tests_dir = output_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_sh = tests_dir / "test.sh"
    test_sh.write_text(_TEST_SH_DEFAULT)
    test_sh.chmod(0o755)

    task_toml_path = output_dir / "task.toml"
    if task_toml_path.exists():
        if _grader_uses_llm(problem_dir):
            _ensure_verifier_env_for_llm(task_toml_path)
        if include_runtime_notices:
            _stamp_runtime_notices(task_toml_path, problem_dir)
        shutil.copy2(task_toml_path, output_dir / "environment" / "task.toml")

    return output_dir
