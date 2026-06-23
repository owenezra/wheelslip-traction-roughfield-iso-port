from __future__ import annotations

import asyncio

from lbx_rl_tasks_harness import auto_qa
from lbx_rl_tasks_harness.message_text import message_text
from lbx_rl_tasks_harness.models import HarnessProblem, OutputSpec


class FakeModel:
    prompt = ""
    reasoning = None

    async def ainvoke(self, prompt: str):
        self.prompt = prompt

        class Message:
            content = """{
              "overall_assessment": "needs_changes",
              "summary": "Task is close but needs stronger rollout checks.",
              "is_solvable": {"value": true, "evidence": "The prompt declares /tmp/output/model.xml."},
              "scientifically_correct": {"value": false, "evidence": "No robustness perturbations are present."},
              "confidence": "high",
              "blocking_issues": ["Missing robustness checks in scorer/compute_score.py"],
              "non_blocking_feedback": ["Add a naive baseline."],
              "checks": [
                {
                  "check": "Prompt clarity",
                  "status": "pass",
                  "reason": "Deliverables are explicit.",
                  "recommendation": ""
                }
              ]
            }"""

        return Message()


def _problem(tmp_path):
    repo = tmp_path
    problem_dir = repo / "problems" / "demo"
    (repo / "docs").mkdir(parents=True)
    (repo / "project_guidelines").mkdir()
    (repo / "docs" / "RUBRIC_GUIDANCE.md").write_text("Rubric guidance")
    (repo / "docs" / "GRADING.md").write_text("Grading guidance")
    (repo / "project_guidelines" / "mujoco_environments.md").write_text(
        "MuJoCo guidance"
    )
    (repo / "pyproject.toml").write_text("[project]\nname='demo'\n")
    (problem_dir / "scorer").mkdir(parents=True)
    (problem_dir / "instruction.md").write_text("Write /tmp/output/model.xml")
    (problem_dir / "task.toml").write_text("[task]\nname='demo'\n")
    (problem_dir / "metadata.json").write_text(
        '{"benchmark": "taiga_task", "problem_data": {"instance_id": "demo"}}'
    )
    (problem_dir / "scorer" / "compute_score.py").write_text(
        "def compute_score(workspace, trajectory, private): return 1.0"
    )
    return HarnessProblem(
        id="demo",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/model.xml")],
        source_problem_dir=problem_dir,
    )


def _cfd_problem(tmp_path):
    repo = tmp_path
    problem_dir = repo / "problems" / "demo-cfd"
    (repo / "docs").mkdir(parents=True)
    (repo / "project_guidelines" / "cfd").mkdir(parents=True)
    (repo / "docs" / "RUBRIC_GUIDANCE.md").write_text("Rubric guidance")
    (repo / "docs" / "GRADING.md").write_text("Grading guidance")
    (repo / "project_guidelines" / "cfd" / "cfd_environments.md").write_text(
        "CFD guidance requires agent-side OpenFOAM evidence."
    )
    (repo / "pyproject.toml").write_text("[project]\nname='demo'\n")
    (problem_dir / "scorer").mkdir(parents=True)
    (problem_dir / "data").mkdir()
    (problem_dir / "instruction.md").write_text(
        "Run OpenFOAM with blockMesh and checkMesh before writing /tmp/output/design.json."
    )
    (problem_dir / "task.toml").write_text(
        "[task]\nname='demo-cfd'\n[difficulty]\ntask_type='cfd'\n"
    )
    (problem_dir / "metadata.json").write_text(
        '{"benchmark": "taiga_task", "problem_data": {"instance_id": "demo-cfd"}}'
    )
    (problem_dir / "data" / "openfoam_probe.py").write_text("blockMesh checkMesh")
    (problem_dir / "scorer" / "compute_score.py").write_text(
        "def compute_score(workspace, trajectory, private): return 1.0"
    )
    return HarnessProblem(
        id="demo-cfd",
        source_format="problem-dir",
        prompt="demo",
        outputs=[OutputSpec(path="/tmp/output/design.json")],
        source_problem_dir=problem_dir,
    )


def test_run_auto_qa_check_uses_autoqa_env_and_normalizes(monkeypatch, tmp_path):
    fake_model = FakeModel()
    monkeypatch.setenv("LBX_RL_AUTOQA_MODEL", "openai:gpt-5.5")
    monkeypatch.setenv("LBX_RL_AUTOQA_REASONING_EFFORT", "xhigh")
    monkeypatch.setenv("LBX_RL_HARNESS_MODEL", "claude-opus-4-7")
    monkeypatch.setattr(auto_qa, "has_key_for_model", lambda _model: True)
    monkeypatch.setattr(
        auto_qa,
        "_build_autoqa_model",
        lambda model, *, reasoning_effort: fake_model,
    )

    result = asyncio.run(
        auto_qa.run_auto_qa_check(
            _problem(tmp_path),
            proof={"harness_result": {"score": 0.2}},
            grade_payload={"score": 0.2},
            rubric_quality={"status": "completed", "checks": []},
        )
    )

    assert result["status"] == "completed"
    assert result["model"] == "openai:gpt-5.5"
    assert result["reasoning_effort"] == "xhigh"
    assert result["overall_assessment"] == "needs_changes"
    assert result["is_solvable"]["value"] is True
    assert result["scientifically_correct"]["value"] is False
    assert {check["check"] for check in result["checks"]} == set(auto_qa.AUTOQA_CHECKS)
    assert "SUBMITTED BUILD PROOF" in fake_model.prompt
    assert "SUBMITTED RUBRIC QUALITY REVIEW" in fake_model.prompt


def test_autoqa_prompt_for_cfd_includes_solver_contract_guidance(tmp_path):
    prompt = auto_qa._autoqa_prompt(
        _cfd_problem(tmp_path),
        proof={},
        grade_payload={"score": 0.3},
        rubric_quality={"status": "completed", "checks": []},
    )

    assert "Task type: cfd" in prompt
    assert "Solver usage contract" in prompt
    assert "Public solver evidence path" in prompt
    assert "agent-side OpenFOAM" in prompt
    assert "cfd_environments.md" in prompt


def test_build_autoqa_model_sets_openai_reasoning(monkeypatch):
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "langchain_openai.ChatOpenAI",
        FakeChatOpenAI,
    )

    auto_qa._build_autoqa_model("openai:gpt-5.5", reasoning_effort="xhigh")

    assert captured["model"] == "gpt-5.5"
    assert captured["reasoning"] == {"effort": "xhigh"}
    assert captured["max_tokens"] is None


def test_run_auto_qa_check_skips_when_openai_key_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(auto_qa, "has_key_for_model", lambda _model: False)

    result = asyncio.run(
        auto_qa.run_auto_qa_check(
            _problem(tmp_path),
            proof={},
            grade_payload={"score": 0.0},
            rubric_quality={"status": "completed"},
            model_name="openai:gpt-5.5",
            reasoning_effort="xhigh",
        )
    )

    assert result["status"] == "skipped"
    assert result["model"] == "openai:gpt-5.5"
    assert result["reasoning_effort"] == "xhigh"
    assert "OPENAI_API_KEY" in result["reason"]


def test_message_text_ignores_reasoning_blocks_before_json() -> None:
    class Message:
        content = [
            {"type": "reasoning", "reasoning": "considering {not valid json}"},
            {"type": "text", "text": '{"overall_assessment": "pass"}'},
        ]

    assert message_text(Message()) == '{"overall_assessment": "pass"}'
