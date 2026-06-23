from __future__ import annotations

import os
import stat
from pathlib import Path

import numpy as np
import pytest

from grading import helpers, score_kfold_cv
from grading.faults import AgentFault
from grading.policy_runner import PolicyWorker


def _write_script(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


# ── run_submitted_executable ──────────────────────────────────────────────


def test_executable_strips_rubric_score_lines(tmp_path: Path) -> None:
    exe = _write_script(
        tmp_path / "run.sh",
        "#!/bin/sh\necho hello\necho 'RUBRIC_SCORE=1.0'\necho world\n",
    )
    result = helpers.run_submitted_executable(exe, timeout_s=10)
    assert result.returncode == 0
    assert "hello" in result.stdout
    assert "world" in result.stdout
    assert "RUBRIC_SCORE=" not in result.stdout


def test_executable_passes_args_and_returncode(tmp_path: Path) -> None:
    exe = _write_script(
        tmp_path / "run.sh",
        '#!/bin/sh\necho "$1"\nexit 3\n',
    )
    result = helpers.run_submitted_executable(exe, args=["payload"], timeout_s=10)
    assert result.returncode == 3
    assert result.stdout.strip() == "payload"


def test_executable_missing_raises_agent_fault(tmp_path: Path) -> None:
    with pytest.raises(AgentFault):
        helpers.run_submitted_executable(tmp_path / "nope.sh")


def test_executable_not_marked_executable_raises_agent_fault(tmp_path: Path) -> None:
    plain = tmp_path / "plain.sh"
    plain.write_text("#!/bin/sh\necho hi\n")
    os.chmod(plain, 0o644)
    with pytest.raises(AgentFault):
        helpers.run_submitted_executable(plain)


def test_executable_timeout_raises_agent_fault(tmp_path: Path) -> None:
    exe = _write_script(tmp_path / "slow.sh", "#!/bin/sh\nsleep 5\n")
    with pytest.raises(AgentFault):
        helpers.run_submitted_executable(exe, timeout_s=0.3)


def test_executable_stdout_over_cap_raises_agent_fault(tmp_path: Path) -> None:
    exe = _write_script(
        tmp_path / "loud.sh",
        "#!/bin/sh\nhead -c 4096 /dev/zero | tr '\\0' 'x'\n",
    )
    with pytest.raises(AgentFault):
        helpers.run_submitted_executable(exe, timeout_s=10, max_output_bytes=1024)


def test_executable_unbounded_stdout_is_capped_not_buffered(tmp_path: Path) -> None:
    # An infinite stdout producer must be killed by the streaming cap (raising
    # the "produced more than" fault), not buffered whole nor left to run until
    # the timeout. With timeout_s=30, a timeout-based failure would report
    # "timed out"; asserting the cap message proves the cap fired first.
    exe = _write_script(
        tmp_path / "flood.sh",
        "#!/bin/sh\nwhile true; do echo xxxxxxxxxxxxxxxx; done\n",
    )
    with pytest.raises(AgentFault) as excinfo:
        helpers.run_submitted_executable(
            exe, timeout_s=30, max_output_bytes=1_000_000
        )
    assert "produced more than" in str(excinfo.value)


def test_executable_stderr_flood_is_discarded(tmp_path: Path) -> None:
    # A submission can flood stderr without counting against the stdout cap and
    # without accumulating in grader memory (stderr -> /dev/null). With a tiny
    # cap, a 64 KiB stderr burst must still succeed while stdout stays clean.
    exe = _write_script(
        tmp_path / "noisy.sh",
        "#!/bin/sh\nhead -c 65536 /dev/zero | tr '\\0' 'x' 1>&2\necho ok\n",
    )
    result = helpers.run_submitted_executable(exe, timeout_s=10, max_output_bytes=1024)
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


# ── load_submission_h5_or_fault ───────────────────────────────────────────


def test_h5_reads_regular_dataset(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    sub = tmp_path / "submission.h5"
    with h5py.File(sub, "w") as f:
        f.create_dataset("preds", data=np.arange(4, dtype=float))
    out = helpers.load_submission_h5_or_fault(sub, datasets=["preds"])
    assert np.array_equal(out["preds"], np.arange(4, dtype=float))


def test_h5_rejects_external_link(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    secret = tmp_path / "truth.h5"
    with h5py.File(secret, "w") as f:
        f.create_dataset("secret", data=np.ones(4))
    sub = tmp_path / "submission.h5"
    with h5py.File(sub, "w") as f:
        f["leak"] = h5py.ExternalLink("truth.h5", "secret")
    with pytest.raises(AgentFault):
        helpers.load_submission_h5_or_fault(sub, datasets=["leak"])


def test_h5_rejects_virtual_dataset(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    layout = h5py.VirtualLayout(shape=(4,), dtype="f")
    layout[:] = h5py.VirtualSource("source.h5", "data", shape=(4,))
    sub = tmp_path / "submission.h5"
    with h5py.File(sub, "w") as f:
        f.create_virtual_dataset("vds", layout)
    with pytest.raises(AgentFault):
        helpers.load_submission_h5_or_fault(sub, datasets=["vds"])


def test_h5_missing_file_raises_agent_fault(tmp_path: Path) -> None:
    with pytest.raises(AgentFault):
        helpers.load_submission_h5_or_fault(tmp_path / "nope.h5", datasets=["x"])


def test_h5_reads_all_top_level_datasets_when_unspecified(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    sub = tmp_path / "submission.h5"
    with h5py.File(sub, "w") as f:
        f.create_dataset("a", data=np.arange(3, dtype=float))
        f.create_dataset("b", data=np.ones(2, dtype="int64"))
    out = helpers.load_submission_h5_or_fault(sub)
    assert set(out) == {"a", "b"}
    assert np.array_equal(out["a"], np.arange(3, dtype=float))
    assert np.array_equal(out["b"], np.ones(2, dtype="int64"))


def test_h5_reads_large_multidim_dataset_via_file_transfer(tmp_path: Path) -> None:
    """Datasets cross the privilege boundary as .npy files, not a single RPC
    frame, so a read is not bounded by the policy protocol's 1 GiB wire limit.
    A few-MiB multi-dim array round-trips exactly (dtype + shape preserved)."""
    h5py = pytest.importorskip("h5py")
    sub = tmp_path / "submission.h5"
    arr = np.arange(256 * 1024, dtype="float64").reshape(512, 512)
    with h5py.File(sub, "w") as f:
        f.create_dataset("big", data=arr)
    out = helpers.load_submission_h5_or_fault(sub, datasets=["big"])
    assert out["big"].dtype == np.dtype("float64")
    assert out["big"].shape == (512, 512)
    assert np.array_equal(out["big"], arr)


def test_h5_corrupt_file_raises_agent_fault_not_crash(tmp_path: Path) -> None:
    """A crafted/corrupt .h5 must surface as an AgentFault in the parent, not a
    crash: the libhdf5 parse runs in the worker, so a malformed file that kills
    the parse is attributed to the agent (kept 0.0) and the grader survives."""
    pytest.importorskip("h5py")
    sub = tmp_path / "submission.h5"
    # HDF5 magic prefix then garbage -> libhdf5 errors while parsing the body.
    sub.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00\xff" * 4096)
    with pytest.raises(AgentFault):
        helpers.load_submission_h5_or_fault(sub, datasets=["preds"])


@pytest.mark.skipif(
    os.geteuid() != 0, reason="privilege drop only activates when grader is root"
)
def test_h5_read_runs_unprivileged_not_as_root(tmp_path: Path) -> None:
    """When the grader is root, the libhdf5 parse runs as the unprivileged agent
    account, not in the root process. Proof: a valid .h5 readable ONLY by root
    (0600 root:root) is unreadable to the uid-1000 worker, so the read fails as
    an AgentFault -- whereas an in-process root read would have succeeded.
    Mirrors test_policy_runner_sandbox.test_root_policy_cannot_read_root_only_*.
    """
    h5py = pytest.importorskip("h5py")
    sub = tmp_path / "root_only.h5"
    with h5py.File(sub, "w") as f:
        f.create_dataset("preds", data=np.arange(4, dtype=float))
    os.chmod(sub, 0o600)  # root:root 0600 -- unreadable to the agent uid
    with pytest.raises(AgentFault):
        helpers.load_submission_h5_or_fault(sub, datasets=["preds"])


# ── load_submission_h5ad_or_fault ─────────────────────────────────────────


def test_h5ad_sanitizer_accepts_valid_anndata(tmp_path: Path) -> None:
    anndata = pytest.importorskip("anndata")
    pytest.importorskip("h5py")
    src = tmp_path / "submission.h5ad"
    anndata.AnnData(X=np.eye(3, dtype="float32")).write_h5ad(src)
    out = tmp_path / "clean.h5ad"
    result = helpers.load_submission_h5ad_or_fault(src, out_path=out)
    assert result == out
    assert out.exists() and out.stat().st_size > 0
    # The sanitized copy round-trips back through anndata with the same X.
    assert np.array_equal(anndata.read_h5ad(out).X, np.eye(3, dtype="float32"))


def test_h5ad_sanitizer_rejects_malformed(tmp_path: Path) -> None:
    pytest.importorskip("anndata")
    src = tmp_path / "broken.h5ad"
    src.write_bytes(b"this is not an HDF5 / AnnData file at all")
    with pytest.raises(AgentFault):
        helpers.load_submission_h5ad_or_fault(src, out_path=tmp_path / "out.h5ad")


def test_h5ad_sanitizer_missing_file_raises_agent_fault(tmp_path: Path) -> None:
    with pytest.raises(AgentFault):
        helpers.load_submission_h5ad_or_fault(
            tmp_path / "nope.h5ad", out_path=tmp_path / "out.h5ad"
        )


# ── score_kfold_cv ────────────────────────────────────────────────────────


_MEAN_MODEL = """
_state = {}


def fit(X, y):
    _state["mean"] = sum(y) / len(y) if y else 0.0


def predict(X):
    return [_state.get("mean", 0.0)] * len(X)
"""

# Tries to persist a marker across folds; with the cross-fold reset it never
# observes a previous fold (predicts 0.0 every fold).
_LEAK_MODEL = """
from pathlib import Path


def fit(X, y):
    saw_prev = Path("leak_marker.txt").exists()
    Path("saw_prev.txt").write_text("1" if saw_prev else "0")
    Path("leak_marker.txt").write_text("seen")


def predict(X):
    marker = Path("saw_prev.txt")
    saw = marker.read_text().strip() if marker.exists() else "0"
    return [float(saw)] * len(X)
"""

_BAD_SHAPE_MODEL = """
def fit(X, y):
    pass


def predict(X):
    return [0.0]
"""


def test_kfold_runs_across_folds(tmp_path: Path) -> None:
    model = tmp_path / "model.py"
    model.write_text(_MEAN_MODEL)
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)
    result = score_kfold_cv(
        model, X, y, metric=lambda yt, yp: float(np.mean(yp)), n_folds=5, timeout_s=20
    )
    assert 0.0 <= result <= 1.0


def test_kfold_resets_tree_between_folds(tmp_path: Path) -> None:
    model = tmp_path / "model.py"
    model.write_text(_LEAK_MODEL)
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.zeros(10, dtype=float)
    # If the disk channel leaked, later folds would see the marker and predict
    # 1.0 -> mean > 0. The reset keeps every fold's prediction at 0.0.
    result = score_kfold_cv(
        model, X, y, metric=lambda yt, yp: float(np.mean(yp)), n_folds=5, timeout_s=20
    )
    assert result == 0.0


def test_kfold_invalid_predictions_raise_agent_fault(tmp_path: Path) -> None:
    model = tmp_path / "model.py"
    model.write_text(_BAD_SHAPE_MODEL)
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.zeros(10, dtype=float)
    with pytest.raises(AgentFault):
        score_kfold_cv(
            model, X, y, metric=lambda yt, yp: 0.0, n_folds=5, timeout_s=20
        )


def test_kfold_missing_model_raises_agent_fault(tmp_path: Path) -> None:
    with pytest.raises(AgentFault):
        score_kfold_cv(
            tmp_path / "nope.py",
            np.zeros((10, 2)),
            np.zeros(10),
            metric=lambda yt, yp: 0.0,
        )


# ── PolicyWorker unshare_ipc opt-in ───────────────────────────────────────


def test_policy_worker_accepts_unshare_ipc(tmp_path: Path) -> None:
    policy = tmp_path / "policy.py"
    policy.write_text("def act(obs):\n    return obs\n")
    with PolicyWorker(policy, unshare_ipc=True, timeout_s=15) as worker:
        assert worker.act(7) == 7
