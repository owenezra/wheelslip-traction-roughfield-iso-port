from __future__ import annotations

from pathlib import Path

from alignerr_plugin.exporters.taiga import build_job_payload
from alignerr_plugin.utils import load_metadata, load_task_toml, task_id

from lbx_rl_tasks_harness.models import GroundTruthSpec, HarnessProblem, OutputSpec, ReferenceSpec
from lbx_rl_tasks_harness.prompts import prompt_with_task_type_preprompt


def load_problem_dir(problem_dir: Path) -> HarnessProblem:
    problem_dir = problem_dir.resolve()
    metadata = load_metadata(problem_dir)
    task_toml = load_task_toml(problem_dir)
    task_type = str(task_toml.difficulty.task_type)
    prompt = prompt_with_task_type_preprompt(
        (problem_dir / "instruction.md").read_text(), task_type
    )
    scorer_dir = problem_dir / "scorer"
    taiga_problem = build_job_payload(problem_dir, image_ref="LOCAL_IMAGE")[
        "problems_metadata"
    ]["problem_set"]["problems"][0]
    _apply_prompt_to_taiga_problem(taiga_problem, prompt)

    return HarnessProblem(
        id=task_id(problem_dir),
        source_format="problem-dir",
        prompt=prompt,
        outputs=[
            OutputSpec(
                path=out.path, required=out.required, description=out.description
            )
            for out in task_toml.outputs
        ],
        ground_truth=GroundTruthSpec(
            render_command=task_toml.ground_truth.render_command,
            render_outputs=[
                OutputSpec(
                    path=out.path,
                    required=out.required,
                    description=out.description,
                )
                for out in task_toml.ground_truth.render_outputs
            ],
            score_epsilon=task_toml.ground_truth.score_epsilon,
            continuous_score_epsilon=task_toml.ground_truth.continuous_score_epsilon,
            in_container=task_toml.ground_truth.in_container,
        ),
        reference=ReferenceSpec(
            execution=task_toml.reference.execution,
            cache_dir=task_toml.reference.cache_dir,
            entrypoint=task_toml.reference.entrypoint,
            proof_mode=task_toml.reference.proof_mode,
        ),
        task_dir=problem_dir,
        source_problem_dir=problem_dir,
        grader_dir=scorer_dir,
        private_dir=scorer_dir / "data",
        required_tools=list(task_toml.runner.required_tools),
        metadata={
            "benchmark": metadata.benchmark,
            "task": task_toml.task.model_dump(),
            "agent": task_toml.agent.model_dump(),
            "verifier": task_toml.verifier.model_dump(),
            "environment": task_toml.environment.model_dump(),
            "runner": task_toml.runner.model_dump(),
            "difficulty": task_toml.difficulty.model_dump(),
        },
        taiga_problem=taiga_problem,
    )


def _apply_prompt_to_taiga_problem(taiga_problem: dict, prompt: str) -> None:
    taiga_problem["prompt"] = prompt
    extra_fields = taiga_problem.get("extra_fields")
    if isinstance(extra_fields, dict):
        extra_fields["task_prompt"] = prompt
    else:
        taiga_problem["extra_fields"] = {"task_prompt": prompt}
