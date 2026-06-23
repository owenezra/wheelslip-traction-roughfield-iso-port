from __future__ import annotations

import importlib.util
from pathlib import Path

from alignerr_plugin import ground_truth

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github" / "scripts" / "template_pr_check.py"


def load_template_pr_check():
    spec = importlib.util.spec_from_file_location("template_pr_check", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_template_qa_comment():
    script = ROOT / ".github" / "scripts" / "template_qa_comment.py"
    spec = importlib.util.spec_from_file_location("template_qa_comment", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_success_body_uses_ground_truth_score_before_validator_sample_score(
    monkeypatch, tmp_path
) -> None:
    template_pr_check = load_template_pr_check()
    problem_dir = tmp_path / "problems" / "example"
    artifact = problem_dir / ".alignerr" / "ground_truth" / "rendering.mp4"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"video")
    (problem_dir / "task.toml").write_text(
        "[task]\nname = 'demo'\n\n[difficulty]\ntask_type = 'mujoco'\ndomain = 'model_environment_construction'\nreward_type = 'multi_deterministic_rubrics'\n"
    )
    monkeypatch.setattr(ground_truth, "video_dimensions", lambda _path: (1280, 720))
    body = template_pr_check.success_body(
        problem_dir,
        {
            "status": "valid",
            "metadata": {"return_shape": "rubric_grade", "sample_score": 0.0},
            "stages": {"local_build_proof": {"passed": True, "issues": []}},
        },
        {
            "score": 0.75,
            "structured_subscores": [
                {"name": "criterion", "score": 0.75, "weight": 1.0}
            ],
        },
        proof={
            "built_at": "2026-05-13T15:43:48+00:00",
            "task_dir_sha256": "abcdef1234567890",
            "image_digest": "sha256:0123456789abcdef",
            "ground_truth_result": {
                "score": 1.0,
                "review_artifacts": [
                    {
                        "path": ".alignerr/ground_truth/rendering.mp4",
                        "logical_path": "/tmp/output/rendering.mp4",
                        "sha256": ground_truth.sha256_file(artifact),
                        "bytes": 5,
                        "width": 1280,
                        "height": 720,
                    }
                ],
            },
        },
        head_sha="fedcba9876543210",
        score_source="`solution/solve.sh` reference solution",
    )

    assert "## Template Validation: Ground Truth Summary (Score: 0.750)" in body
    assert f"- Task: `{problem_dir.as_posix()}`" in body
    assert "- Validation score: `0.750`" in body
    assert "- Score source: `solution/solve.sh` reference solution" in body
    assert "- PR head SHA: `fedcba987654`" in body
    assert "- Build proof task hash: `abcdef123456`" in body
    assert "- Ground truth score: `1.000`" in body
    assert "`.alignerr/ground_truth/rendering.mp4`" in body
    assert "| criterion | 0.750 | 1.000 |" in body


def test_success_body_preserves_recomputed_zero_score() -> None:
    template_pr_check = load_template_pr_check()
    body = template_pr_check.success_body(
        Path("problems/example"),
        {
            "status": "valid",
            "metadata": {"return_shape": "score_dict", "sample_score": 0.9},
            "stages": {},
        },
        {"score": 0.0},
    )

    assert "## Template Validation: Ground Truth Summary (Score: 0.000)" in body
    assert "- Validation score: `0.000`" in body


def test_failure_body_instructs_ground_truth_verification() -> None:
    template_pr_check = load_template_pr_check()
    body = template_pr_check.failure_body(Path("problems/example"), ["missing proof"])

    assert "ground-truth build proof" in body
    assert (
        "uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/example"
        in body
    )
    assert "AI agent harness, rubric QA, Auto QA, and Boreal submission" in body


def test_template_qa_comment_ignores_continue_on_error_failures() -> None:
    template_qa_comment = load_template_qa_comment()

    failed_key, _step = template_qa_comment.first_failed_step(
        {
            "upload_gcs_ground_truth": {
                "outcome": "failure",
                "conclusion": "success",
            },
            "dispatch_submission": {
                "outcome": "failure",
                "conclusion": "failure",
            },
        }
    )

    assert failed_key == "dispatch_submission"


def test_template_qa_comment_renders_rubric_quality_table() -> None:
    template_qa_comment = load_template_qa_comment()

    body = template_qa_comment.success_body(
        Path("problems/example"),
        {
            "status": "qa_completed",
            "problem_id": "example",
            "score": 0.0,
            "required_resources": "4vcpu+16gib",
            "ground_truth_summary": {
                "score": 1.0,
                "runtime": "solution",
                "review_artifacts": [{"path": ".alignerr/ground_truth/rendering.mp4"}],
            },
        },
        harness_summary={
            "actions_run_url": "https://github.com/org/repo/actions/runs/1",
            "score": 0.25,
            "runtime": "claude-code",
            "model": "claude-opus-4-7",
            "grade": {
                "score": 0.25,
                "structured_subscores": [
                    {"name": "policy.py exists", "score": 1.0, "weight": 0.2},
                    {"name": "stable rollout", "score": 0.0, "weight": 0.8},
                ],
            },
            "rubric_quality": {
                "status": "completed",
                "model": "claude-opus-4-7",
                "summary": "Rubric is measurable but overlaps.",
                "checks": [
                    {
                        "check": "Logical independence",
                        "status": "warning",
                        "reason": "task_completion overlaps with position | progress.",
                        "recommendation": "Remove duplicate weighting.\nKeep bottleneck only once.",
                    }
                ],
            },
        },
    )

    assert "### QA Scores" in body
    assert (
        "| Ground truth | `1.000` | runtime `solution`; 1 review artifact(s) |" in body
    )
    assert (
        "| Agent harness | `0.250` | runtime `claude-code`; model `claude-opus-4-7` |" in body
    )
    assert "### Agent Harness Rubric Scores" in body
    assert "| policy.py exists | `1.000` | `0.200` |" in body
    assert "| stable rollout | `0.000` | `0.800` |" in body
    assert "### Rubric QA Results" in body
    assert "- Status: `completed`" in body
    assert "- Model: `claude-opus-4-7`" in body
    assert "| Check | Status | Reason | Recommendation |" in body
    assert (
        "| Logical independence | warning | task_completion overlaps with position \\| progress. | "
        "Remove duplicate weighting.<br>Keep bottleneck only once. |" in body
    )
    assert "- Rubric quality:" not in body


def test_template_qa_comment_ready_for_review_label_threshold() -> None:
    template_qa_comment = load_template_qa_comment()

    assert template_qa_comment.should_add_ready_for_review_label(
        {
            "status": "qa_completed",
            "score": 0.4,
            "ground_truth_summary": {"score": 1.0},
        }
    )
    assert template_qa_comment.should_add_ready_for_review_label(
        {
            "status": "qa_completed",
            "score": 0.1,
        },
        harness_summary={
            "score": 0.25,
            "ground_truth": {"score": 1.0},
        },
    )


def test_template_qa_comment_ready_for_review_label_requires_low_agent_score_and_ground_truth() -> (
    None
):
    template_qa_comment = load_template_qa_comment()

    assert not template_qa_comment.should_add_ready_for_review_label(
        {
            "status": "qa_completed",
            "score": 0.401,
            "ground_truth_summary": {"score": 1.0},
        }
    )
    assert not template_qa_comment.should_add_ready_for_review_label(
        {
            "status": "qa_completed",
            "score": 0.4,
            "ground_truth_summary": {"score": 0.99},
        }
    )
    assert not template_qa_comment.should_add_ready_for_review_label(
        {
            "status": "pending_review",
            "score": 0.4,
            "ground_truth_summary": {"score": 1.0},
        }
    )


def test_template_qa_comment_renders_auto_qa_details() -> None:
    template_qa_comment = load_template_qa_comment()

    body = template_qa_comment.success_body(
        Path("problems/example"),
        {
            "status": "qa_completed",
            "problem_id": "example",
            "score": 0.0,
            "required_resources": "4vcpu+16gib",
        },
        auto_qa={
            "status": "completed",
            "model": "openai:gpt-5.5",
            "reasoning_effort": "xhigh",
            "overall_assessment": "needs_changes",
            "confidence": "high",
            "summary": "Needs stronger robustness checks.",
            "is_solvable": {"value": True, "evidence": "Output paths exist."},
            "scientifically_correct": {
                "value": False,
                "evidence": "Contact dynamics are under-specified.",
            },
            "blocking_issues": ["Missing perturbation rollout."],
            "non_blocking_feedback": ["Add a naive baseline."],
            "checks": [
                {
                    "check": "MuJoCo physics validity",
                    "status": "fail",
                    "reason": "No perturbations.",
                    "recommendation": "Add fixed perturbations.",
                }
            ],
        },
    )

    assert "### AutoQA Results" in body
    assert "- Overall: `needs_changes`" in body
    assert "- Model: `openai:gpt-5.5`" in body
    assert "- Reasoning effort: `xhigh`" in body
    assert "- Solvable: `yes`" in body
    assert "- Scientifically correct: `no`" in body
    assert "Missing perturbation rollout." in body
    assert (
        "| MuJoCo physics validity | fail | No perturbations. | Add fixed perturbations. |"
        in body
    )


def test_ground_truth_required_body_instructs_verification() -> None:
    template_pr_check = load_template_pr_check()
    body = template_pr_check.ground_truth_required_body(
        Path("problems/example"),
        {"task_dir_sha256": "abcdef1234567890"},
        "fedcba9876543210",
    )

    assert "## Template Validation: Ground Truth Verification Required" in body
    assert (
        "uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/example"
        in body
    )
    assert "- PR head SHA: `fedcba987654`" in body
    assert "- Build proof task hash: `abcdef123456`" in body


def test_ground_truth_required_body_explains_low_score() -> None:
    template_pr_check = load_template_pr_check()
    body = template_pr_check.ground_truth_required_body(
        Path("problems/example"),
        {"task_dir_sha256": "abcdef1234567890"},
        errors=[
            "`ground_truth_result.score` is `0.186047`, but template validation "
            "requires `1.000000` within epsilon `1e-09`."
        ],
    )

    assert "Why this failed:" in body
    assert "`ground_truth_result.score` is `0.186047`" in body
    assert "requires `1.000000`" in body


def test_ground_truth_from_build_proof_requires_score_and_artifacts(
    monkeypatch, tmp_path
) -> None:
    template_pr_check = load_template_pr_check()
    problem_dir = tmp_path / "problems" / "example"
    artifact = problem_dir / ".alignerr" / "ground_truth" / "rendering.mp4"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"video")
    sha = ground_truth.sha256_file(artifact)
    monkeypatch.setattr(ground_truth, "video_dimensions", lambda _path: (1280, 720))

    assert (
        template_pr_check.ground_truth_from_build_proof(
            {
                "ground_truth_result": {
                    "score": 1.0,
                    "review_artifacts": [
                        {
                            "path": ".alignerr/ground_truth/rendering.mp4",
                            "logical_path": "/tmp/output/rendering.mp4",
                            "sha256": sha,
                            "bytes": 5,
                            "width": 1280,
                            "height": 720,
                        }
                    ],
                }
            },
            problem_dir,
        )
        is not None
    )
    assert (
        template_pr_check.ground_truth_from_build_proof(
            {
                "ground_truth_result": {
                    "score": 0.99,
                    "review_artifacts": [
                        {
                            "path": ".alignerr/ground_truth/rendering.mp4",
                            "logical_path": "/tmp/output/rendering.mp4",
                            "sha256": sha,
                            "bytes": 5,
                            "width": 1280,
                            "height": 720,
                        }
                    ],
                }
            },
            problem_dir,
        )
        is None
    )
    low_score_errors = template_pr_check.ground_truth_validation_errors(
        {
            "ground_truth_result": {
                "score": 0.99,
                "review_artifacts": [
                    {
                        "path": ".alignerr/ground_truth/rendering.mp4",
                        "logical_path": "/tmp/output/rendering.mp4",
                        "sha256": sha,
                        "bytes": 5,
                        "width": 1280,
                        "height": 720,
                    }
                ],
            }
        },
        problem_dir,
    )
    assert "`ground_truth_result.score` is `0.990000`" in low_score_errors[0]
    assert (
        template_pr_check.ground_truth_from_build_proof(
            {"ground_truth_result": {"score": 1.0, "review_artifacts": []}}
        )
        is None
    )
    assert (
        template_pr_check.ground_truth_from_build_proof(
            {
                "ground_truth_result": {
                    "score": 1.0,
                    "review_artifacts": [
                        {
                            "path": ".alignerr/ground_truth/rendering.mp4",
                            "logical_path": "/tmp/output/rendering.mp4",
                            "sha256": "b" * 64,
                            "bytes": 5,
                            "width": 1280,
                            "height": 720,
                        }
                    ],
                }
            },
            problem_dir,
        )
        is None
    )


def test_ground_truth_from_build_proof_uses_task_epsilon(monkeypatch, tmp_path) -> None:
    template_pr_check = load_template_pr_check()
    problem_dir = tmp_path / "problems" / "example"
    artifact = problem_dir / ".alignerr" / "ground_truth" / "rendering.mp4"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"video")
    (problem_dir / "task.toml").write_text(
        "[task]\nname = 'demo'\n\n"
        "[ground_truth]\nscore_epsilon = 0.01\n"
        "\n[difficulty]\ntask_type = 'mujoco'\ndomain = 'model_environment_construction'\nreward_type = 'multi_deterministic_rubrics'\n"
    )
    monkeypatch.setattr(ground_truth, "video_dimensions", lambda _path: (1280, 720))

    proof = {
        "ground_truth_result": {
            "score": 0.995,
            "review_artifacts": [
                {
                    "path": ".alignerr/ground_truth/rendering.mp4",
                    "logical_path": "/tmp/output/rendering.mp4",
                    "sha256": ground_truth.sha256_file(artifact),
                    "bytes": 5,
                    "width": 1280,
                    "height": 720,
                }
            ],
        }
    }

    assert (
        template_pr_check.ground_truth_from_build_proof(proof, problem_dir) is not None
    )


def test_ground_truth_from_build_proof_allows_ml_without_video(tmp_path) -> None:
    template_pr_check = load_template_pr_check()
    problem_dir = tmp_path / "problems" / "example"
    problem_dir.mkdir(parents=True)
    (problem_dir / "task.toml").write_text(
        "[task]\nname = 'demo'\n\n[difficulty]\ntask_type = 'ml'\ndomain = 'scientific_discovery_computational_science'\nreward_type = 'continuous_scoring_function'\nlicense = 'MIT'\n"
    )
    proof = {"ground_truth_result": {"score": 1.0, "review_artifacts": []}}

    assert (
        template_pr_check.ground_truth_from_build_proof(proof, problem_dir) is not None
    )


def test_success_body_renders_ml_ground_truth_without_video(tmp_path) -> None:
    template_pr_check = load_template_pr_check()
    problem_dir = tmp_path / "problems" / "example"
    problem_dir.mkdir(parents=True)
    (problem_dir / "task.toml").write_text(
        "[task]\nname = 'demo'\n\n[difficulty]\ntask_type = 'ml'\ndomain = 'scientific_discovery_computational_science'\nreward_type = 'continuous_scoring_function'\nlicense = 'MIT'\n"
    )

    body = template_pr_check.success_body(
        problem_dir,
        {
            "status": "valid",
            "metadata": {"return_shape": "score_dict", "sample_score": 0.499607},
            "stages": {},
        },
        {"score": 0.499607},
        proof={"ground_truth_result": {"score": 1.0, "review_artifacts": []}},
    )

    assert "- Ground truth score: `1.000`" in body
    assert "Ground truth review artifacts" not in body


def test_success_body_renders_rubric_table_from_subscores_and_weights() -> None:
    template_pr_check = load_template_pr_check()
    body = template_pr_check.success_body(
        Path("problems/example"),
        {
            "status": "valid",
            "metadata": {"return_shape": "rubric_grade", "sample_score": 0.0},
            "stages": {},
        },
        {
            "score": 0.273,
            "subscores": {
                "model.xml exists": 1.0,
                "End-effector moves closer to the target": 0.0,
            },
            "weights": {
                "model.xml exists": 0.018,
                "End-effector moves closer to the target": 0.218,
            },
        },
        score_source="build_proof.json harness_result",
    )

    assert "### Rubric Scores" in body
    assert "| model.xml exists | 1.000 | 0.018 |" in body
    assert "| End-effector moves closer to the target | 0.000 | 0.218 |" in body
    assert "Continuous reward functions" not in body


def test_success_body_renders_rubric_quality_review() -> None:
    template_pr_check = load_template_pr_check()
    body = template_pr_check.success_body(
        Path("problems/example"),
        {
            "status": "valid",
            "metadata": {"return_shape": "rubric_grade"},
            "stages": {},
        },
        {"score": 1.0},
        proof={
            "harness_result": {
                "rubric_quality": {
                    "status": "completed",
                    "model": "claude-opus-4-7",
                    "summary": "Rubric is mostly measurable.",
                    "checks": [
                        {
                            "check": "Measurability",
                            "status": "pass",
                            "reason": "Criteria use code predicates.",
                            "recommendation": "",
                        }
                    ],
                }
            }
        },
    )

    assert "### Rubric Quality Review" in body
    assert "- Status: `completed`" in body
    assert "- Model: `claude-opus-4-7`" in body
    assert "| Measurability | pass | Criteria use code predicates. |  |" in body


def test_success_body_renders_auto_qa_review() -> None:
    template_pr_check = load_template_pr_check()
    body = template_pr_check.success_body(
        Path("problems/example"),
        {
            "status": "valid",
            "metadata": {"return_shape": "rubric_grade"},
            "stages": {},
        },
        {"score": 1.0},
        proof={
            "harness_result": {
                "auto_qa": {
                    "status": "completed",
                    "model": "openai:gpt-5.5",
                    "reasoning_effort": "xhigh",
                    "overall_assessment": "needs_changes",
                    "confidence": "high",
                    "summary": "Task needs stronger robustness checks.",
                    "is_solvable": {"value": True, "evidence": "Output path exists."},
                    "scientifically_correct": {
                        "value": False,
                        "evidence": "Rollout is under-specified.",
                    },
                    "blocking_issues": ["Missing perturbation rollout."],
                    "non_blocking_feedback": ["Add a naive baseline."],
                    "checks": [
                        {
                            "check": "MuJoCo physics validity",
                            "status": "fail",
                            "reason": "No perturbations.",
                            "recommendation": "Add fixed perturbations.",
                        }
                    ],
                }
            }
        },
    )

    assert "### Auto QA" in body
    assert "- Overall: `needs_changes`" in body
    assert "- Model: `openai:gpt-5.5`" in body
    assert "- Reasoning effort: `xhigh`" in body
    assert "- Solvable: `yes`" in body
    assert "Missing perturbation rollout." in body
    assert (
        "| MuJoCo physics validity | fail | No perturbations. | Add fixed perturbations. |"
        in body
    )
