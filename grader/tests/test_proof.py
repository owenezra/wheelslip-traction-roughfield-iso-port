from __future__ import annotations

import json
from pathlib import Path

from alignerr_plugin.proof import update_build_proof_result, write_build_proof


def test_update_build_proof_result_records_harness_score(tmp_path: Path) -> None:
    problem_dir = tmp_path / "problems" / "demo"
    problem_dir.mkdir(parents=True)
    (problem_dir / "task.toml").write_text("[task]\nname = 'demo'\n")

    proof_path = write_build_proof(
        problem_dir,
        image_digest="sha256:image",
        base_image_ref="lbx-tasks-base:runtime",
        platform="linux/amd64",
        alignerr_cli_version="0.1.0",
        duration_seconds=1.25,
    )

    updated = update_build_proof_result(
        problem_dir,
        runtime="deepagents",
        grade_payload={
            "score": 0.290909,
            "subscores": {"criterion": 0.0},
            "weights": {"criterion": 1.0},
            "structured_subscores": [
                {"name": "criterion", "score": 0.0, "weight": 1.0}
            ],
        },
        run_dir=Path(".harness-runs/demo"),
        reward_path=Path(".harness-runs/demo/verifier/reward.json"),
        details_path=Path(".harness-runs/demo/verifier/reward-details.json"),
        rubric_quality={
            "status": "completed",
            "checks": [{"check": "Measurability", "status": "pass"}],
        },
    )

    assert updated == proof_path
    proof = json.loads(proof_path.read_text())
    assert proof["harness_result"]["runtime"] == "deepagents"
    assert proof["harness_result"]["score"] == 0.290909
    assert proof["harness_result"]["subscores"] == {"criterion": 0.0}
    assert proof["harness_result"]["weights"] == {"criterion": 1.0}
    assert proof["harness_result"]["structured_subscores"] == [
        {"name": "criterion", "score": 0.0, "weight": 1.0}
    ]
    assert proof["harness_result"]["rubric_quality"] == {
        "status": "completed",
        "checks": [{"check": "Measurability", "status": "pass"}],
    }
