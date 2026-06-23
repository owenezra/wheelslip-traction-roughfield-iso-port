"""Tests for the Harbor exporter."""

from __future__ import annotations

import shutil
import tomllib
from pathlib import Path

from alignerr_plugin.exporters.harbor import export_harbor


def test_export_default_mode(template_examples: Path, tmp_path: Path) -> None:
    out = tmp_path / "harbor"
    export_harbor(
        template_examples / "mujoco-pendulum",
        out,
        image_ref="gcr.io/example/p@sha256:abc",
    )

    assert (out / "task.toml").exists()
    assert (out / "instruction.md").exists()
    assert (out / "environment" / "Dockerfile").exists()
    assert (out / "environment" / "scorer" / "compute_score.py").exists()
    assert (out / "environment" / "data").exists()
    assert (out / "environment" / "grader" / "pyproject.toml").exists()
    assert (out / "environment" / "base" / "install-common.sh").exists()
    assert (
        out / "environment" / "taiga_runtime" / "rubric" / "pyproject.toml"
    ).exists()
    assert (out / "solution" / "solve.sh").exists()

    test_sh = (out / "tests" / "test.sh").read_text()
    dockerfile = (out / "environment" / "Dockerfile").read_text()
    assert "COPY data/ /workspace/data/" in dockerfile
    assert "base/requirements-runtime.txt" in dockerfile
    assert "ln -s /workspace/data /data" in dockerfile
    assert "COPY --chown=root:root scorer/data/ /mcp_server/data/" in dockerfile
    assert "COPY --chown=root:root scorer/ /mcp_server/grader/" in dockerfile
    assert "rm -rf /mcp_server/grader/data" in dockerfile
    assert (
        "find /mcp_server/data /mcp_server/grader -type d -exec chmod 0700"
        in dockerfile
    )
    assert (
        "find /mcp_server/data /mcp_server/grader -type f -exec chmod 0600"
        in dockerfile
    )
    assert "/runtime/run_grader.py" in test_sh
    assert "--workspace /app" in test_sh
    assert "--grader-dir /mcp_server/grader" in test_sh
    assert "--private-dir /mcp_server/data" in test_sh
    assert "--output-dir /logs/verifier" in test_sh

    assert sorted(p.name for p in (out / "tests").iterdir()) == ["test.sh"]

    assert not (out / "scorer").exists()


def test_export_does_not_stamp_remote_image_ref(
    template_examples: Path, tmp_path: Path
) -> None:
    out = tmp_path / "harbor"
    export_harbor(
        template_examples / "mujoco-pendulum",
        out,
        image_ref="gcr.io/example/p@sha256:abc123",
    )
    text = (out / "task.toml").read_text()
    assert "docker_image" not in text


def test_export_gpu_task_writes_runtime_notice_metadata(
    template_examples: Path, tmp_path: Path
) -> None:
    problem_dir = template_examples / "mle-tabular-classification"
    original_instruction = (problem_dir / "instruction.md").read_text()
    out = tmp_path / "harbor"

    export_harbor(problem_dir, out)

    assert (out / "instruction.md").read_text() == original_instruction
    root_toml = tomllib.loads((out / "task.toml").read_text())
    image_toml = tomllib.loads((out / "environment" / "task.toml").read_text())
    expected = [
        {
            "kind": "accelerator_availability",
            "accelerator": "gpu",
            "text": "A GPU may be available.",
            "default_enabled": True,
        }
    ]
    assert root_toml["metadata"]["runtime_notices"] == expected
    assert image_toml["metadata"]["runtime_notices"] == expected


def test_export_tpu_task_writes_runtime_notice_metadata(
    template_examples: Path, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "mle-tpu"
    shutil.copytree(template_examples / "mle-tabular-classification", problem_dir)
    task_toml = problem_dir / "task.toml"
    text = task_toml.read_text()
    text = text.replace(
        'required_resources = "12vcpu+100gib+h100/2"',
        'required_resources = "13vcpu+32gib+tpuv5e1x1"',
    )
    task_toml.write_text(text)
    out = tmp_path / "harbor"

    export_harbor(problem_dir, out)

    root_toml = tomllib.loads((out / "task.toml").read_text())
    assert root_toml["metadata"]["runtime_notices"] == [
        {
            "kind": "accelerator_availability",
            "accelerator": "tpu",
            "text": "A TPU may be available.",
            "default_enabled": True,
        }
    ]


def test_export_preserves_custom_runtime_notices(
    template_examples: Path, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "mle-custom-notice"
    shutil.copytree(template_examples / "mle-tabular-classification", problem_dir)
    task_toml = problem_dir / "task.toml"
    task_toml.write_text(task_toml.read_text() + """

[metadata]
runtime_notices = [
  { kind = "custom_hint", text = "Use the provided cached dataset." },
]
""")
    out = tmp_path / "harbor"

    export_harbor(problem_dir, out)

    root_toml = tomllib.loads((out / "task.toml").read_text())
    image_toml = tomllib.loads((out / "environment" / "task.toml").read_text())
    expected = [
        {"kind": "custom_hint", "text": "Use the provided cached dataset."},
        {
            "kind": "accelerator_availability",
            "accelerator": "gpu",
            "text": "A GPU may be available.",
            "default_enabled": True,
        },
    ]
    assert root_toml["metadata"]["runtime_notices"] == expected
    assert image_toml["metadata"]["runtime_notices"] == expected


def test_export_removes_stale_accelerator_runtime_notice_for_cpu_task(
    template_examples: Path, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "cpu-stale-notice"
    shutil.copytree(template_examples / "mujoco-pendulum", problem_dir)
    task_toml = problem_dir / "task.toml"
    task_toml.write_text(task_toml.read_text() + """

[metadata]
runtime_notices = [
  { kind = "custom_hint", text = "Keep the simulator deterministic." },
  { kind = "accelerator_availability", accelerator = "gpu", text = "A GPU may be available.", default_enabled = true },
]
""")
    out = tmp_path / "harbor"

    export_harbor(problem_dir, out)

    root_toml = tomllib.loads((out / "task.toml").read_text())
    image_toml = tomllib.loads((out / "environment" / "task.toml").read_text())
    expected = [{"kind": "custom_hint", "text": "Keep the simulator deterministic."}]
    assert root_toml["metadata"]["runtime_notices"] == expected
    assert image_toml["metadata"]["runtime_notices"] == expected


def test_export_can_disable_runtime_notice_metadata(
    template_examples: Path, tmp_path: Path
) -> None:
    out = tmp_path / "harbor"

    export_harbor(
        template_examples / "mle-tabular-classification",
        out,
        include_runtime_notices=False,
    )

    root_toml = tomllib.loads((out / "task.toml").read_text())
    assert "runtime_notices" not in root_toml.get("metadata", {})
