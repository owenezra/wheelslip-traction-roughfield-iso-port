from __future__ import annotations

import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from alignerr_plugin.local_cli import app


def test_export_taiga_cli_writes_metadata_and_sidecar(
    template_examples: Path, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "mujoco-pendulum"
    shutil.copytree(template_examples / "mujoco-pendulum", problem_dir)
    output = tmp_path / "problems-metadata.json"

    result = CliRunner().invoke(
        app,
        [
            "export-taiga",
            "--problem-dir",
            str(problem_dir),
            "--out",
            str(output),
            "--image",
            "local:test",
        ],
    )

    assert result.exit_code == 0, result.output
    metadata = json.loads(output.read_text())
    assert metadata["problem_set"]["problems"][0]["id"] == "mujoco-pendulum"

    sidecar = json.loads((problem_dir / ".taiga_submit.json").read_text())
    assert sidecar["task_id"] == "mujoco-pendulum"
    assert sidecar["image"] == "local:test"


def test_validate_cli_exits_nonzero_for_invalid_task(tmp_path: Path) -> None:
    problem_dir = tmp_path / "broken-task"
    problem_dir.mkdir()

    result = CliRunner().invoke(
        app,
        ["validate", "--problem-dir", str(problem_dir)],
    )

    assert result.exit_code == 1


def test_export_harbor_cli_writes_harbor_task(
    template_examples: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "harbor"

    result = CliRunner().invoke(
        app,
        [
            "export-harbor",
            "--problem-dir",
            str(template_examples / "mujoco-pendulum"),
            "--out",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "task.toml").exists()
    assert (output_dir / "environment" / "Dockerfile").exists()
