from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from alignerr_plugin.preloaded import (
    PRELOADED_MANIFEST_PATH,
    auto_mount_entries,
    content_address,
    load_preloaded_manifest,
    manifest_entry,
    mount_notice,
    remote_squashfs_name,
)
from alignerr_plugin.schemas import PreloadedFile
from alignerr_plugin.validators.task import validator as validator_module

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── schema ────────────────────────────────────────────────────────────────


def test_preloaded_file_accepts_source_only() -> None:
    pf = PreloadedFile(source="data/big", mount_path="/data/big")
    assert pf.read_only is True


def test_preloaded_file_accepts_hf_repo_only() -> None:
    PreloadedFile(hf_repo="org/model", mount_path="/tmp/hf-cache")


def test_preloaded_file_requires_exactly_one_source() -> None:
    with pytest.raises(ValueError):
        PreloadedFile(mount_path="/data/big")  # neither
    with pytest.raises(ValueError):
        PreloadedFile(source="data/big", hf_repo="org/model", mount_path="/data/big")


def test_preloaded_file_mount_path_must_be_absolute() -> None:
    with pytest.raises(ValueError):
        PreloadedFile(source="data/big", mount_path="relative/path")


# ── content addressing ─────────────────────────────────────────────────────


def test_content_address_stable_and_identical_for_equal_trees(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    for root in (a, b):
        (root / "sub").mkdir(parents=True)
        (root / "sub" / "x.bin").write_bytes(b"hello")
        (root / "y.txt").write_text("world")
    addr_a1 = content_address(a)
    assert addr_a1 == content_address(a)  # stable
    assert addr_a1 == content_address(b)  # identical trees -> same address
    assert len(addr_a1) == 16


def test_content_address_changes_on_content_change(tmp_path: Path) -> None:
    root = tmp_path / "a"
    root.mkdir()
    (root / "x.txt").write_text("one")
    before = content_address(root)
    (root / "x.txt").write_text("two")
    assert content_address(root) != before


def test_remote_squashfs_name_is_content_addressed() -> None:
    name = remote_squashfs_name("my-task", "m1", "abcd1234abcd1234")
    assert name == "my-task/m1-abcd1234abcd1234.squashfs"


# ── manifest load / notice ─────────────────────────────────────────────────


def test_load_preloaded_manifest_absent_is_empty(tmp_path: Path) -> None:
    assert load_preloaded_manifest(tmp_path) == []


def test_load_preloaded_manifest_reads_stamped_file(tmp_path: Path) -> None:
    out = tmp_path / PRELOADED_MANIFEST_PATH
    out.parent.mkdir(parents=True)
    out.write_text(
        json.dumps(
            {"preloaded_files": [manifest_entry("t/m1-abc.squashfs", "/data/big")]}
        )
    )
    manifest = load_preloaded_manifest(tmp_path)
    assert manifest == [
        {
            "remote_path": "t/m1-abc.squashfs",
            "local_path": "/data/big",
            "is_read_only": True,
        }
    ]


def test_mount_notice_lists_paths() -> None:
    notice = mount_notice([manifest_entry("r", "/data/big"), manifest_entry("r2", "/opt/w")])
    assert "/data/big" in notice and "/opt/w" in notice
    assert mount_notice([]) == ""


# ── declaration validation (static) ─────────────────────────────────────────


_PRELOAD_TASK_TOML = (
    "[task]\nname = 'x'\n\n"
    "[difficulty]\ntask_type = 'mujoco'\ndomain = 'locomotion'\n"
    "reward_type = 'multi_deterministic_rubrics'\n\n"
    "[[preloaded_files]]\nsource = 'data/big'\nmount_path = '/data/big'\n"
)

_PRELOAD_TASK_TOML_HF = (
    "[task]\nname = 'x'\n\n"
    "[difficulty]\ntask_type = 'mujoco'\ndomain = 'locomotion'\n"
    "reward_type = 'multi_deterministic_rubrics'\n\n"
    "[[preloaded_files]]\nhf_repo = 'org/model'\nmount_path = '/tmp/hf-cache'\n"
)


def test_preloaded_validation_flags_missing_source(tmp_path: Path) -> None:
    # Trusted CI packs/uploads/stamps at submit time, so no committed stamp is
    # required -- but a declared local source that does not exist is a typo we
    # still catch in the sandbox.
    problem_dir = tmp_path / "task"
    problem_dir.mkdir()
    (problem_dir / "task.toml").write_text(_PRELOAD_TASK_TOML)
    issues = validator_module._preloaded_files_issues(problem_dir)
    assert any("does not exist" in issue for issue in issues)


def test_preloaded_validation_passes_when_source_exists(tmp_path: Path) -> None:
    problem_dir = tmp_path / "task"
    (problem_dir / "data" / "big").mkdir(parents=True)
    (problem_dir / "data" / "big" / "x.bin").write_bytes(b"payload")
    (problem_dir / "task.toml").write_text(_PRELOAD_TASK_TOML)
    assert validator_module._preloaded_files_issues(problem_dir) == []


def test_preloaded_validation_passes_for_hf_repo(tmp_path: Path) -> None:
    problem_dir = tmp_path / "task"
    problem_dir.mkdir()
    (problem_dir / "task.toml").write_text(_PRELOAD_TASK_TOML_HF)
    assert validator_module._preloaded_files_issues(problem_dir) == []


# ── auto mounts ──────────────────────────────────────────────────────────────


def test_auto_mount_entries_ml_includes_populated_dirs(tmp_path: Path) -> None:
    pd = tmp_path / "task"
    (pd / "data").mkdir(parents=True)
    (pd / "data" / "train.csv").write_text("a,b\n1,2\n")
    (pd / "scorer" / "data").mkdir(parents=True)
    (pd / "scorer" / "data" / "truth.csv").write_text("y\n1\n")
    assert auto_mount_entries(pd, "ml") == [
        ("data", "/data"),
        ("scorer/data", "/mcp_server/data"),
    ]


def test_auto_mount_entries_skips_empty_and_gitkeep(tmp_path: Path) -> None:
    pd = tmp_path / "task"
    (pd / "data").mkdir(parents=True)
    (pd / "data" / ".gitkeep").write_text("")  # placeholder only -> not mounted
    assert auto_mount_entries(pd, "ml") == []


def test_auto_mount_entries_noop_for_non_ml(tmp_path: Path) -> None:
    pd = tmp_path / "task"
    (pd / "data").mkdir(parents=True)
    (pd / "data" / "x.bin").write_bytes(b"x")
    assert auto_mount_entries(pd, "mujoco") == []


def test_preloaded_validation_noop_without_declaration(tmp_path: Path) -> None:
    problem_dir = tmp_path / "task"
    problem_dir.mkdir()
    (problem_dir / "task.toml").write_text(
        "[task]\nname = 'x'\n\n"
        "[difficulty]\ntask_type = 'mujoco'\ndomain = 'locomotion'\n"
        "reward_type = 'multi_deterministic_rubrics'\n"
    )
    assert validator_module._preloaded_files_issues(problem_dir) == []


def test_stamp_script_writes_manifest(tmp_path: Path) -> None:
    problem_dir = tmp_path / "task"
    problem_dir.mkdir()
    remote_path = (
        "gs://anthropic-argonrl-dog-bowl-us-central1-0/biome/"
        "environment_files/env-123/task/m1-abc.squashfs"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "stamp_preloaded_files.py"),
            "--problem-dir",
            str(problem_dir),
            "--entry",
            f"/data/big::{remote_path}::true",
            "--entry",
            "/opt/w::task/m2-def.squashfs::false",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = load_preloaded_manifest(problem_dir)
    assert len(manifest) == 2
    assert manifest[0]["local_path"] == "/data/big"
    assert manifest[0]["remote_path"] == remote_path
    assert manifest[0]["is_read_only"] is True
    assert manifest[1]["is_read_only"] is False
