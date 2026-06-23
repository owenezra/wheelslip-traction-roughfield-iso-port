"""Tests for shared runtime hardening helpers."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from grading.runtime_hardening import (
    prepare_grader_cache,
    lock_down_grader_private,
    pre_grade_cleanup,
    scrub_escaping_symlinks,
    scrub_nonregular_files,
)


def test_scrub_escaping_symlinks_removes_private_redirect(tmp_path: Path) -> None:
    output = tmp_path / "output"
    private = tmp_path / "private"
    output.mkdir()
    private.mkdir()
    secret = private / "truth.csv"
    secret.write_text("answer")
    link = output / "submission.csv"
    os.symlink(secret, link)

    assert scrub_escaping_symlinks(output) == 1
    assert not link.exists()


def test_scrub_escaping_symlinks_keeps_internal_links(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    real = output / "real.csv"
    real.write_text("ok")
    link = output / "submission.csv"
    os.symlink(real, link)

    assert scrub_escaping_symlinks(output) == 0
    assert link.exists()


def test_scrub_nonregular_files_removes_fifo(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    fifo = output / "submission.csv"
    os.mkfifo(fifo)

    assert scrub_nonregular_files(output) == 1
    assert not fifo.exists()


def test_pre_grade_cleanup_runs_scrubs(tmp_path: Path) -> None:
    output = tmp_path / "output"
    private = tmp_path / "private"
    output.mkdir()
    private.mkdir()
    secret = private / "truth.csv"
    secret.write_text("answer")
    os.symlink(secret, output / "submission.csv")

    result = pre_grade_cleanup(output)

    assert result["removed_symlinks"] == 1
    assert result["removed_nonregular"] == 0
    assert result["killed_processes"] >= 0


def test_lock_down_grader_private_removes_group_other_bits(tmp_path: Path) -> None:
    private = tmp_path / "grader"
    private.mkdir()
    secret = private / "compute_score.py"
    secret.write_text("x = 1\n")
    os.chmod(private, 0o755)
    os.chmod(secret, 0o644)

    lock_down_grader_private((private,))

    assert stat.S_IMODE(private.stat().st_mode) & 0o077 == 0
    assert stat.S_IMODE(secret.stat().st_mode) & 0o077 == 0


def test_prepare_grader_cache_omits_cache_env_when_all_roots_fail(
    monkeypatch,
) -> None:
    class FailingPath(type(Path())):
        def mkdir(self, *args, **kwargs):
            raise OSError("no writable cache")

    monkeypatch.setattr("grading.runtime_hardening.Path", FailingPath)

    env = prepare_grader_cache("/nope/.grader_cache")

    assert env["PYTHONSAFEPATH"] == "1"
    assert "PYTHONPATH" not in env
    for name in (
        "XDG_CACHE_HOME",
        "MPLCONFIGDIR",
        "NUMBA_CACHE_DIR",
        "TORCH_HOME",
        "HF_HOME",
    ):
        assert name not in env
