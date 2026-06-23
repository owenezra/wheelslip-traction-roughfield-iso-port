"""Tests for the Boreal submission payload builder.

The negative tests pin our "no worldsim CU/split-strategy fields ever
leak into the payload" invariant. If a future port accidentally
re-introduces `interaction_mode`, `cua_*`, `agentic_grader`, or top-level
`rubric` items, these fail loudly.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

# conftest already adds REPO_ROOT to sys.path so `from alignerr_plugin import ...` resolves.
from alignerr_plugin.ground_truth import sha256_file
from alignerr_plugin.exporters.taiga import (
    _derive_resources_from_toml,
    _max_required_resources,
    _taiga_container_runtime,
    build_batch_job_payload,
    build_job_payload,
    derive_taiga_resources,
    export_taiga,
)
from alignerr_plugin.schemas import (
    Difficulty,
    EnvironmentSection,
    RunnerConfig,
    TaskToml,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Helpers ───────────────────────────────────────────────────────


_FORBIDDEN_FRAGMENTS = (
    "interaction_mode",
    "cua_",
    "agentic_grader",
    "task_rubric",
    "task_penalties",
    "penalty_avoidance",
)


def _assert_no_worldsim_leak(payload: dict, label: str) -> None:
    flat = json.dumps(payload)
    for forbidden in _FORBIDDEN_FRAGMENTS:
        assert (
            forbidden not in flat
        ), f"{label} payload leaked worldsim-only field: {forbidden!r}"


# ── Tests ─────────────────────────────────────────────────────────


def _resource_task_toml(
    required_resources: str,
    *,
    base_flavor: str = "auto",
) -> TaskToml:
    return TaskToml.model_construct(
        environment=EnvironmentSection(
            required_resources=required_resources,
            base_flavor=base_flavor,
        ),
        runner=RunnerConfig(),
        difficulty=Difficulty.model_construct(),
    )


def test_derive_resources_cpu_default(template_examples: Path) -> None:
    res = derive_taiga_resources(template_examples / "mujoco-pendulum")
    assert res["base_flavor"] == "cpu"
    assert res["base_image"].endswith("/lbx-tasks-base")
    assert res["base_image_ref"] == f"{res['base_image']}:{res['base_tag']}"
    assert res["api_model_name"] == "claude-fable-5"
    assert res["required_resources"] == "4vcpu+16gib"


def test_derive_resources_uses_required_resources_enum_verbatim() -> None:
    assert (
        _derive_resources_from_toml(_resource_task_toml("4vcpu+16gib"))[
            "required_resources"
        ]
        == "4vcpu+16gib"
    )
    assert (
        _derive_resources_from_toml(_resource_task_toml("12vcpu+100gib+h100/2"))[
            "base_flavor"
        ]
        == "gpu"
    )
    assert (
        _derive_resources_from_toml(
            _resource_task_toml("12vcpu+100gib+h100/2+graphics")
        )["base_flavor"]
        == "cuda-graphics"
    )
    assert (
        _derive_resources_from_toml(_resource_task_toml("13vcpu+32gib+tpuv5e1x1"))[
            "base_flavor"
        ]
        == "tpu"
    )


def test_derive_resources_openroad_gpu_flavor(
    template_examples: Path, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "mle-openroad"
    shutil.copytree(template_examples / "mle-tabular-classification", problem_dir)
    task_toml = problem_dir / "task.toml"
    text = task_toml.read_text().replace(
        'required_resources = "12vcpu+100gib+h100/2"\n',
        'required_resources = "12vcpu+100gib+h100/2"\nbase_flavor = "gpu-openroad"\n',
    )
    task_toml.write_text(text)

    res = derive_taiga_resources(problem_dir)
    assert res["base_flavor"] == "gpu-openroad"
    assert res["base_image"].endswith("/lbx-tasks-base-gpu-openroad")
    assert res["base_image_ref"] == f"{res['base_image']}:{res['base_tag']}"
    assert res["required_resources"] == "12vcpu+100gib+h100/2"


def test_container_runtime_helper_cpu_vs_accelerator() -> None:
    assert _taiga_container_runtime("4vcpu+16gib") == "firecracker"
    assert _taiga_container_runtime("12vcpu+100gib+h100/2") == "gvisor"
    assert _taiga_container_runtime("12vcpu+100gib+h100/2+graphics") == "gvisor"
    assert _taiga_container_runtime("13vcpu+32gib+tpuv5e1x1") == "gvisor"


def test_batch_max_required_resources_uses_capacity_not_openapi_order() -> None:
    assert (
        _max_required_resources(
            [
                "16vcpu+64gib+tpuv5e2x2",
                "50vcpu+128gib+tpuv5e2x2",
            ]
        )
        == "50vcpu+128gib+tpuv5e2x2"
    )


def test_export_cpu_task_runs_under_firecracker(template_examples: Path) -> None:
    p = build_job_payload(
        template_examples / "mujoco-pendulum",
        image_ref="gcr.io/example/img@sha256:abc",
    )
    problem_set = p["problems_metadata"]["problem_set"]
    problem = problem_set["problems"][0]
    assert problem_set["container_runtime"] == "firecracker"
    assert problem["container_runtime"] == "firecracker"
    assert problem["metadata"]["base_image"]["flavor"] == "cpu"
    assert problem["metadata"]["base_image"]["ref"].endswith(
        f":{problem['metadata']['base_image']['tag']}"
    )


def test_export_gpu_task_runs_under_gvisor(template_examples: Path) -> None:
    # ml tasks default to a GPU tier; accelerator tasks must export gVisor, not
    # the task.toml firecracker default (which would fail on Taiga's GPU nodes).
    p = build_job_payload(
        template_examples / "mle-tabular-classification",
        image_ref="gcr.io/example/img@sha256:abc",
    )
    problem_set = p["problems_metadata"]["problem_set"]
    problem = problem_set["problems"][0]
    assert problem_set["container_runtime"] == "gvisor"
    assert problem["container_runtime"] == "gvisor"


def test_export_cpu_task_injects_tmux_notice_only(template_examples: Path) -> None:
    p = build_job_payload(
        template_examples / "mujoco-pendulum",
        image_ref="gcr.io/example/img@sha256:abc",
    )
    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    prompt = problem["task_prompt"]
    assert "dedicated tmux tool" in prompt
    assert "not tmux inside the bash tool" in prompt
    assert "A GPU may be available." not in prompt
    assert "A TPU may be available." not in prompt


def test_export_gpu_task_injects_gpu_and_tmux_notices(
    template_examples: Path, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "mle-no-runtime-guidance"
    shutil.copytree(template_examples / "mle-tabular-classification", problem_dir)
    (problem_dir / "instruction.md").write_text("Write /tmp/output/submission.csv.\n")
    p = build_job_payload(problem_dir, image_ref="gcr.io/example/img@sha256:abc")
    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    prompt = problem["task_prompt"]
    assert "A GPU may be available." in prompt
    assert "dedicated tmux tool" in prompt
    assert "A TPU may be available." not in prompt


def test_export_tpu_task_injects_tpu_notice(
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

    p = build_job_payload(problem_dir, image_ref="gcr.io/example/img@sha256:abc")
    problem_set = p["problems_metadata"]["problem_set"]
    assert problem_set["required_resources"] == "13vcpu+32gib+tpuv5e1x1"
    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    assert problem["required_resources"] == "13vcpu+32gib+tpuv5e1x1"
    assert problem["container_runtime"] == "gvisor"
    prompt = problem["task_prompt"]
    assert "A TPU may be available." in prompt
    assert "A GPU may be available." not in prompt


def test_export_tpu_task_rejects_unsupported_resource_request(
    template_examples: Path, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "mle-tpu-too-large"
    shutil.copytree(template_examples / "mle-tabular-classification", problem_dir)
    task_toml = problem_dir / "task.toml"
    text = task_toml.read_text()
    text = text.replace(
        'required_resources = "12vcpu+100gib+h100/2"',
        'required_resources = "13vcpu+32gib+tpu-v5e/8"',
    )
    task_toml.write_text(text)

    with pytest.raises(ValueError, match="required_resources must be one of"):
        derive_taiga_resources(problem_dir)


def test_export_accelerator_notice_appends_despite_authored_guidance(
    template_examples: Path, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "mle-with-guidance"
    shutil.copytree(template_examples / "mle-tabular-classification", problem_dir)
    (problem_dir / "instruction.md").write_text(
        "Use the GPU and keep long runs in tmux. Write /tmp/output/submission.csv.\n"
    )
    p = build_job_payload(problem_dir, image_ref="gcr.io/example/img@sha256:abc")
    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    prompt = problem["task_prompt"]
    assert prompt.count("A GPU may be available.") == 1
    assert prompt.count("dedicated tmux tool") == 0


def test_export_tpu_notice_not_suppressed_by_gpu_wording(
    template_examples: Path, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "mle-tpu-with-gpu-word"
    shutil.copytree(template_examples / "mle-tabular-classification", problem_dir)
    (problem_dir / "instruction.md").write_text(
        "The starter mentions GPU in passing. Write /tmp/output/submission.csv.\n"
    )
    task_toml = problem_dir / "task.toml"
    text = task_toml.read_text()
    text = text.replace(
        'required_resources = "12vcpu+100gib+h100/2"',
        'required_resources = "13vcpu+32gib+tpuv5e1x1"',
    )
    task_toml.write_text(text)

    p = build_job_payload(problem_dir, image_ref="gcr.io/example/img@sha256:abc")
    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    prompt = problem["task_prompt"]
    assert "A TPU may be available." in prompt


def test_export_gpu_notice_not_suppressed_by_tpu_wording(
    template_examples: Path, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "mle-gpu-with-tpu-word"
    shutil.copytree(template_examples / "mle-tabular-classification", problem_dir)
    (problem_dir / "instruction.md").write_text(
        "This is not a TPU task. Write /tmp/output/submission.csv.\n"
    )
    p = build_job_payload(problem_dir, image_ref="gcr.io/example/img@sha256:abc")
    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    prompt = problem["task_prompt"]
    assert "A GPU may be available." in prompt


def test_ml_starter_boilerplate_does_not_suppress_prompt_notices(
    tmp_path: Path,
) -> None:
    src = (
        REPO_ROOT
        / "alignerr_plugin"
        / "src"
        / "alignerr_plugin"
        / "starter_templates"
        / "ml"
    )
    problem_dir = tmp_path / "ml-starter"
    shutil.copytree(src, problem_dir)
    p = build_job_payload(problem_dir, image_ref="gcr.io/example/img@sha256:abc")
    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    prompt = problem["task_prompt"]
    # The ml starter now deploys on TPU by default (Manu's launch directive).
    assert "A TPU may be available." in prompt
    assert "dedicated tmux tool" in prompt


def test_export_task_toml_hints_as_native_taiga_hints(
    template_examples: Path, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "mujoco-with-hint"
    shutil.copytree(template_examples / "mujoco-pendulum", problem_dir)
    task_toml = problem_dir / "task.toml"
    task_toml.write_text(
        task_toml.read_text()
        + '\n[[hint]]\ntext = "Start with a proportional controller."\n'
        + "enabled = false\nspoiler_level = 0.25\n"
    )

    p = build_job_payload(problem_dir, image_ref="gcr.io/example/img@sha256:abc")
    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    assert problem["hints"] == [
        {
            "message": "Start with a proportional controller.",
            "enabled": False,
            "spoiler_level": 0.25,
        }
    ]
    assert "hints" not in problem["extra_fields"]


@pytest.mark.parametrize(
    "starter_name",
    [
        "ml",
        "mujoco",
        "cfd",
        "structures",
    ],
)
def test_export_task_toml_outputs_for_all_task_types(
    tmp_path: Path, starter_name: str
) -> None:
    src = (
        REPO_ROOT
        / "alignerr_plugin"
        / "src"
        / "alignerr_plugin"
        / "starter_templates"
        / starter_name
    )
    problem_dir = tmp_path / f"{starter_name}-starter"
    shutil.copytree(src, problem_dir)
    p = build_job_payload(
        problem_dir,
        image_ref="gcr.io/example/img@sha256:abc",
    )
    problem = p["problems_metadata"]["problem_set"]["problems"][0]

    assert problem["outputs"]
    assert problem["outputs"] == problem["extra_fields"]["task_metadata"]["outputs"]
    for output in problem["outputs"]:
        assert set(output) == {"path", "required", "description"}
        assert output["path"].startswith("/tmp/output/")
        assert isinstance(output["required"], bool)
        assert isinstance(output["description"], str)


def test_build_job_payload_default_shape(template_examples: Path) -> None:
    p = build_job_payload(
        template_examples / "mujoco-pendulum",
        image_ref="gcr.io/example/img@sha256:abc",
    )

    _assert_no_worldsim_leak(p, "single-problem")

    problem_set = p["problems_metadata"]["problem_set"]
    assert problem_set["name"] == "lbx_rl_tasks_mujoco"
    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    # The grader is image-baked. Per-criterion detail flows back via the
    # in-image MCP `Grade.metadata.structured_subscores`. So the Boreal
    # payload should always carry an empty rubric and a single mcp strategy.
    assert problem["rubric"] == []
    assert problem["grading_strategy"] == [{"type": "mcp", "weight": 1.0}]
    shim = problem["extra_fields"]["test_file"]
    assert "/runtime/grading/src" in shim
    assert "/mcp_server/grader/compute_score.py" in shim
    assert 'sys.path.insert(0, "/mcp_server")' not in shim
    # The shim forwards the agent transcript (injected as a TRANSCRIPT global)
    # instead of the old hardcoded empty trajectory, so transcript-based
    # anti-cheat checks are reachable.
    assert 'trajectory=globals().get("TRANSCRIPT") or []' in shim
    assert "trajectory=[]" not in shim
    # Startup exec's the prebuilt venv binary directly (no per-start uv resolve
    # that blew the 120s MCP init timeout under load).
    assert problem["startup_command"] == "/mcp_server/.venv/bin/rubric mcp"
    assert problem["enable_anthropic_api"] is False

    # Per-problem classification metadata for Boreal UI.
    md = problem["metadata"]
    assert "difficulty" not in md
    assert "task_type" in md
    assert "interaction_mode" not in md  # CU/CUA never appears

    # Defaults for runner config
    assert p["api_model_name"] == "claude-fable-5"
    assert p["n_attempts_per_problem"] == 3  # task.toml [runner] default
    assert p["max_ctx"] == 1_000_000
    assert p["priority"] == "high"
    assert p["iteration_order"] == "problems_first"


def test_build_job_payload_preserves_explicit_anthropic_api_opt_in(
    template_examples: Path,
) -> None:
    p = build_job_payload(
        template_examples / "opensees-three-story-retrofit",
        image_ref="gcr.io/example/img@sha256:abc",
    )
    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    assert problem["enable_anthropic_api"] is True


def test_build_job_payload_includes_redacted_ground_truth_evidence(
    template_examples: Path,
) -> None:
    problem_dir = template_examples / "mujoco-pendulum"
    p = build_job_payload(
        problem_dir,
        image_ref="gcr.io/example/img@sha256:abc",
    )

    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    evidence = problem["extra_fields"]["ground_truth_evidence"]
    proof_path = problem_dir / ".alignerr" / "build_proof.json"

    assert set(evidence) == {
        "schema_version",
        "source",
        "build_proof_sha256",
        "task_type",
        "domain",
        "reward_type",
        "expected_score",
        "score_epsilon",
        "passed",
        "score",
        "runtime",
        "graded_at",
        "review_artifacts",
    }
    assert evidence["schema_version"] == 1
    assert evidence["source"] == ".alignerr/build_proof.json"
    assert evidence["build_proof_sha256"] == sha256_file(proof_path)
    assert evidence["task_type"] == "mujoco"
    assert evidence["domain"] == "model_environment_construction"
    assert evidence["reward_type"] == "multi_deterministic_rubrics"
    assert evidence["expected_score"] == 1.0
    assert evidence["passed"] is True
    assert evidence["score"] == 1.0
    assert evidence["runtime"] == "solution"

    artifact = evidence["review_artifacts"][0]
    assert set(artifact) == {
        "path",
        "logical_path",
        "sha256",
        "bytes",
        "width",
        "height",
    }
    assert artifact["path"] == ".alignerr/ground_truth/rendering.mp4"
    assert artifact["logical_path"] == "/tmp/output/rendering.mp4"
    assert artifact["width"] == 1280
    assert artifact["height"] == 720

    flat_evidence = json.dumps(evidence)
    assert "run_dir" not in flat_evidence
    assert "reward_path" not in flat_evidence
    assert "details_path" not in flat_evidence
    assert "metadata" not in flat_evidence
    assert "structured_subscores" not in flat_evidence


def test_build_job_payload_omits_ground_truth_evidence_without_proof(
    template_examples: Path,
) -> None:
    p = build_job_payload(
        template_examples / "mle-tabular-classification",
        image_ref="gcr.io/example/img@sha256:abc",
    )

    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    assert "ground_truth_evidence" not in problem["extra_fields"]


def test_build_job_payload_uses_reward_type_for_continuous_ground_truth(
    template_examples: Path, tmp_path: Path
) -> None:
    problem_dir = tmp_path / "mle-tabular-classification"
    shutil.copytree(template_examples / "mle-tabular-classification", problem_dir)
    proof_dir = problem_dir / ".alignerr"
    proof_dir.mkdir(exist_ok=True)
    (proof_dir / "build_proof.json").write_text(
        json.dumps(
            {
                "ground_truth_result": {
                    "score": 0.0,
                    "runtime": "solution",
                    "graded_at": "2026-05-31T00:00:00+00:00",
                }
            }
        )
    )

    p = build_job_payload(
        problem_dir,
        image_ref="gcr.io/example/img@sha256:abc",
    )

    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    evidence = problem["extra_fields"]["ground_truth_evidence"]
    assert evidence["score"] == 0.0
    assert evidence["runtime"] == "solution"
    assert evidence["task_type"] == "ml"
    assert evidence["domain"] == "scientific_discovery_computational_science"
    assert evidence["reward_type"] == "continuous_scoring_function"
    assert evidence["expected_score"] == 0.5
    assert evidence["score_epsilon"] == 0.05
    assert evidence["passed"] is False


def test_build_job_payload_caller_overrides_runner_defaults(
    template_examples: Path,
) -> None:
    p = build_job_payload(
        template_examples / "mujoco-pendulum",
        image_ref="gcr.io/example/img@sha256:abc",
        n_attempts=5,
        turn_limit=2000,
        max_ctx=500_000,
        model="claude-sonnet-4-5",
    )
    assert p["n_attempts_per_problem"] == 5
    assert p["turn_limit"] == 2000
    assert p["max_ctx"] == 500_000
    assert p["api_model_name"] == "claude-sonnet-4-5"


def test_build_job_payload_ships_runner_timeouts(template_examples: Path) -> None:
    p = build_job_payload(
        template_examples / "mujoco-pendulum",
        image_ref="gcr.io/example/img@sha256:abc",
    )
    problem = p["problems_metadata"]["problem_set"]["problems"][0]
    assert problem["setup_timeout_seconds"] == 600  # default
    assert problem["grading_timeout_seconds"] == 600
    assert problem["tool_timeout_seconds"] == 120


def test_batch_payload_rejects_missing_image_ref(template_examples: Path) -> None:
    with pytest.raises(ValueError, match="image_refs missing entry"):
        build_batch_job_payload(
            [template_examples / "mujoco-pendulum"],
            image_refs={},  # no entries
        )


def test_batch_payload_applies_container_runtime_override(
    template_examples: Path,
) -> None:
    p = build_batch_job_payload(
        [template_examples / "mujoco-pendulum"],
        image_refs={"mujoco-pendulum": "gcr.io/example/img@sha256:abc"},
        overrides={
            "required_resources": "24vcpu+200gib+h100/1",
            "container_runtime": "custom-runtime",
        },
    )
    ps = p["problems_metadata"]["problem_set"]
    assert ps["container_runtime"] == "custom-runtime"
    assert ps["problems"][0]["container_runtime"] == "custom-runtime"


def test_export_taiga_writes_legacy_problems_metadata(
    template_examples: Path, tmp_path: Path
) -> None:
    out = tmp_path / "problems-metadata.json"
    sidecar = export_taiga(
        template_examples / "mujoco-pendulum",
        out,
        image_ref="gcr.io/example/img@sha256:abc",
    )
    assert out.exists()
    data = json.loads(out.read_text())
    pset = data["problem_set"]
    assert pset["owner"] == "labelbox"
    assert pset["name"] == "lbx_rl_tasks_mujoco"
    assert pset["problems"][0]["id"] == "mujoco-pendulum"
    assert "difficulty" not in pset["problems"][0]["metadata"]
    assert pset["problems"][0]["extra_fields"]["ground_truth_evidence"]["score"] == 1.0

    # Sidecar carries the resource derivation that the workflow reads
    assert sidecar["task_id"] == "mujoco-pendulum"
    assert sidecar["base_flavor"] == "cpu"
    assert sidecar["base_image"].endswith("/lbx-tasks-base")
    assert sidecar["base_image_ref"] == f"{sidecar['base_image']}:{sidecar['base_tag']}"
