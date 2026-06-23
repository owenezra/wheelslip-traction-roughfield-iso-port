"""Utility helpers for task discovery, hashing, and config loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import tomllib

from alignerr_plugin.schemas import ProblemMetadata, TaskToml

IGNORED_HASH_PARTS = {".git", ".alignerr", "__pycache__", ".taiga_submit.json"}


def load_task_toml(problem_dir: Path) -> TaskToml:
    """Load and validate task.toml."""
    with (problem_dir / "task.toml").open("rb") as handle:
        return TaskToml.model_validate(tomllib.load(handle))


def load_metadata(problem_dir: Path) -> ProblemMetadata:
    """Load Alignerr metadata.json."""
    return ProblemMetadata.model_validate(
        json.loads((problem_dir / "metadata.json").read_text())
    )


def task_id(problem_dir: Path) -> str:
    """Return the problem instance id from metadata.json."""
    metadata = load_metadata(problem_dir)
    instance_id = metadata.problem_data.get("instance_id") or metadata.problem_data.get(
        "id"
    )
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError(
            "metadata.json problem_data must include a non-empty instance_id"
        )
    return instance_id


def task_dir_sha256(problem_dir: Path) -> str:
    """Compute a deterministic hash over task files, excluding local state."""
    digest = hashlib.sha256()
    for path in sorted(problem_dir.rglob("*")):
        rel = path.relative_to(problem_dir)
        if any(part in IGNORED_HASH_PARTS for part in rel.parts):
            continue
        if path.is_dir():
            continue
        digest.update(str(rel).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON object with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
