"""K-fold cross-validation scoring for submitted model modules, with the
cross-fold target-leakage closure.

A k-fold task asks the agent to submit a model module exposing ``fit(X, y)`` and
``predict(X)``; the grader fits on each train fold and scores the held-out fold.
The attack this closes: a model that, having seen a validation fold's labels in
one fold, *persists* them and replays them in a later fold to inflate its score.
Such state can survive across folds through three channels, all closed here:

* **process memory** -- a fresh :class:`~grading.policy_runner.PolicyWorker`
  (separate process) is spawned per fold;
* **SysV / POSIX shared memory** -- each worker runs with ``CLONE_NEWIPC``
  (``unshare_ipc=True``), so IPC segments do not survive into the next fold;
* **disk** -- the agent-writable tree is rebuilt from a root-only pristine
  snapshot between folds, so files written during one fold are gone in the next.

The model runs in the non-root sandbox (it cannot read the held-out truth under
``/mcp_server``); agent-controlled failures raise :class:`AgentFault`.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from grading.faults import AgentFault
from grading.policy_runner import PolicyWorker, PolicyWorkerError


def _kfold_indices(n: int, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return np.array_split(rng.permutation(n), n_folds)


def _snapshot_tree(root: Path) -> Path:
    """Copy ``root`` into a fresh root-only pristine store; return its path."""
    pristine = Path(tempfile.mkdtemp(prefix="kfold-pristine-"))
    if root.exists():
        shutil.copytree(root, pristine / "root", symlinks=False, dirs_exist_ok=True)
    return pristine


def _restore_tree(root: Path, pristine: Path) -> None:
    """Wipe everything under ``root`` and restore the pristine snapshot."""
    if root.exists():
        for child in root.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except OSError:
                    pass
    source = pristine / "root"
    if source.exists():
        shutil.copytree(source, root, symlinks=False, dirs_exist_ok=True)


def score_kfold_cv(
    model_path: str | Path,
    X: Any,
    y: Any,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_folds: int = 5,
    fit_method: str = "fit",
    predict_method: str = "predict",
    seed: int = 0,
    timeout_s: float = 120.0,
    agent_writable_root: str | Path | None = None,
) -> float:
    """Cross-validate a submitted model module and return the mean fold metric.

    The module at ``model_path`` must expose ``fit(X, y)`` and ``predict(X)``
    (override names via ``fit_method`` / ``predict_method``). Each fold fits on
    the other folds and scores the held-out fold with ``metric(y_true, y_pred)``.
    Cross-fold leakage is closed on the memory, IPC, and disk channels (see the
    module docstring). Agent-controlled failures (bad module, wrong-shaped or
    non-finite predictions, timeout) raise :class:`AgentFault`.

    ``agent_writable_root`` is the tree reset between folds; it defaults to the
    submitted module's parent directory.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise AgentFault(f"submitted model module not found at {model_path}")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = len(X)
    if n < n_folds:
        raise AgentFault(f"not enough rows ({n}) for {n_folds}-fold CV")

    folds = _kfold_indices(n, n_folds, seed)
    root = (
        Path(agent_writable_root)
        if agent_writable_root is not None
        else model_path.parent
    )
    pristine = _snapshot_tree(root)
    scores: list[float] = []
    try:
        for fold_i, val_idx in enumerate(folds):
            if fold_i > 0:
                _restore_tree(root, pristine)
            train_idx = np.concatenate(
                [folds[j] for j in range(n_folds) if j != fold_i]
            )
            x_train, y_train = X[train_idx], y[train_idx]
            x_val = X[val_idx]
            try:
                with PolicyWorker(
                    model_path,
                    unshare_ipc=True,
                    timeout_s=timeout_s,
                    first_call_timeout_s=timeout_s,
                    cwd=root,
                ) as worker:
                    worker.call(fit_method, x_train.tolist(), y_train.tolist())
                    raw = worker.call(predict_method, x_val.tolist())
            except (PolicyWorkerError, TimeoutError) as exc:
                raise AgentFault(
                    f"submitted model failed on fold {fold_i}: {exc}"
                ) from exc
            preds = np.asarray(raw, dtype=float).reshape(-1)
            if preds.shape[0] != len(val_idx) or not np.isfinite(preds).all():
                raise AgentFault(
                    f"submitted model returned invalid predictions on fold {fold_i} "
                    f"(expected {len(val_idx)} finite values)"
                )
            scores.append(float(metric(y[val_idx], preds)))
    finally:
        shutil.rmtree(pristine, ignore_errors=True)
    return float(np.mean(scores)) if scores else 0.0
