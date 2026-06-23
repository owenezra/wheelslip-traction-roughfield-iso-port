from __future__ import annotations

import json
from pathlib import Path

from alignerr_plugin.exporters.harbor import export_harbor
from alignerr_plugin.exporters.taiga import export_taiga

from lbx_rl_tasks_harness.formats.harbor import load_harbor_dir
from lbx_rl_tasks_harness.formats.problem_dir import load_problem_dir
from lbx_rl_tasks_harness.formats.taiga import load_taiga_metadata
from lbx_rl_tasks_harness.runtimes.deepagents import _extra_fields_for_mcp

ROOT = Path(__file__).resolve().parents[2]
MUJOCO = ROOT / "examples" / "mujoco-pendulum"
OPENSEES = ROOT / "examples" / "opensees-three-story-retrofit"
OPENFOAM = ROOT / "examples" / "openfoam-vortex-suppression"


def test_load_problem_dir() -> None:
    problem = load_problem_dir(MUJOCO)
    assert problem.id == "mujoco-pendulum"
    assert problem.source_format == "problem-dir"
    assert problem.outputs[0].path == "/tmp/output/model.xml"
    assert problem.grader_dir == MUJOCO / "scorer"
    assert problem.taiga_problem is not None
    assert (
        problem.taiga_problem["startup_command"] == "/mcp_server/.venv/bin/rubric mcp"
    )
    fields = _extra_fields_for_mcp(problem, "local:test")
    assert "task_prompt" in fields
    assert "test_file" in fields


def test_load_problem_dir_adds_structures_preprompt() -> None:
    problem = load_problem_dir(OPENSEES)
    fields = _extra_fields_for_mcp(problem, "local:test")

    assert "<!-- lbx-task-type-preprompt -->" not in problem.prompt
    assert problem.prompt.startswith("## Required OpenSeesPy Use")
    assert "## Required OpenSeesPy Use" in problem.prompt
    assert fields["task_prompt"] == problem.prompt


def test_export_taiga_adds_structures_preprompt(tmp_path: Path) -> None:
    metadata = tmp_path / "problems-metadata.json"
    export_taiga(OPENSEES, metadata, image_ref="local:test")

    data = json.loads(metadata.read_text())
    problem = data["problem_set"]["problems"][0]

    assert "<!-- lbx-task-type-preprompt -->" not in problem["task_prompt"]
    assert problem["task_prompt"].startswith("## Required OpenSeesPy Use")
    assert "## Required OpenSeesPy Use" in problem["task_prompt"]


def test_load_problem_dir_adds_cfd_preprompt() -> None:
    problem = load_problem_dir(OPENFOAM)
    fields = _extra_fields_for_mcp(problem, "local:test")

    assert "<!-- lbx-task-type-preprompt -->" not in problem.prompt
    assert problem.prompt.startswith("## Required OpenFOAM Use")
    assert "## Required OpenFOAM Use" in problem.prompt
    assert fields["task_prompt"] == problem.prompt


def test_export_taiga_adds_cfd_preprompt(tmp_path: Path) -> None:
    metadata = tmp_path / "problems-metadata.json"
    export_taiga(OPENFOAM, metadata, image_ref="local:test")

    data = json.loads(metadata.read_text())
    problem = data["problem_set"]["problems"][0]

    assert "<!-- lbx-task-type-preprompt -->" not in problem["task_prompt"]
    assert problem["task_prompt"].startswith("## Required OpenFOAM Use")
    assert "## Required OpenFOAM Use" in problem["task_prompt"]


def test_load_problem_dir_does_not_add_solver_preprompt_for_mujoco() -> None:
    problem = load_problem_dir(MUJOCO)

    assert "<!-- lbx-task-type-preprompt -->" not in problem.prompt


def test_load_taiga_metadata_with_source_problem(tmp_path: Path) -> None:
    metadata = tmp_path / "problems-metadata.json"
    export_taiga(MUJOCO, metadata, image_ref="local:test")

    problem = load_taiga_metadata(metadata, MUJOCO)

    assert problem.source_format == "taiga"
    assert problem.id == "mujoco-pendulum"
    assert problem.image == "local:test"
    assert problem.grader_dir == MUJOCO / "scorer"
    assert problem.metadata["grading_strategy"] == [{"type": "mcp", "weight": 1.0}]
    assert problem.taiga_problem is not None
    assert problem.taiga_problem["image"] == "local:test"


def test_load_taiga_metadata_recovers_exported_outputs(tmp_path: Path) -> None:
    metadata = tmp_path / "problems-metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "problem_set": {
                    "owner": "labelbox",
                    "name": "test",
                    "description": "test",
                    "problems": [
                        {
                            "id": "output-fixture",
                            "image": "local:test",
                            "startup_command": "rubric mcp",
                            "required_tools": ["bash"],
                            "task_prompt": "Write /tmp/output/model.xml.",
                            "outputs": [
                                {
                                    "path": "/tmp/output/model.xml",
                                    "required": True,
                                    "description": "MJCF XML model",
                                }
                            ],
                        }
                    ],
                }
            }
        )
        + "\n"
    )

    problem = load_taiga_metadata(metadata)

    assert [output.path for output in problem.outputs] == ["/tmp/output/model.xml"]
    assert problem.outputs[0].required is True
    assert problem.outputs[0].description == "MJCF XML model"


def test_load_harbor_dir_with_source_problem(tmp_path: Path) -> None:
    harbor_dir = tmp_path / "harbor"
    export_harbor(MUJOCO, harbor_dir, image_ref="local:test")

    problem = load_harbor_dir(harbor_dir, MUJOCO)

    assert problem.source_format == "harbor"
    assert problem.id == "mujoco-pendulum"
    assert problem.grader_dir == MUJOCO / "scorer"
    assert (harbor_dir / "tests" / "test.sh").exists()
    assert problem.taiga_problem is not None
