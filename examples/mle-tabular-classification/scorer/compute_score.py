"""Continuous ML_Envs-style scoring for the tabular classification example.

This is a direct adaptation of ML_Envs'
`examples/dataset-example-task_taiga` scorer. The underlying objective,
metrics, anchors, weights, and exponential score curve are unchanged.

Unlike rubric-based tasks, this scorer does not define criteria or use
`RubricBuilder`. It computes one continuous headline score from raw ML
metrics and returns a score dict so the new grader package can preserve
diagnostic per-target progress values while keeping `score`
authoritative.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from grading import AgentFault, calibration

# ---------------------------------------------------------------------
# Per-target anchors (floor = baseline, ref = reference, perfect = ceiling)
# ---------------------------------------------------------------------
# Measured against the committed baseline (linear/logistic on raw
# features) and reference (domain-aware linear + XGBoost on engineered
# features). Re-measure if you change either solution or the data
# generator.
T1_FLOOR = 0.2661
T1_REF = 0.0330
T1_PERFECT = 0.0

T2_FLOOR = 0.7770
T2_REF = 0.2107
T2_PERFECT = 0.0

LABEL_FLOOR = 0.9231
LABEL_REF = 0.9939
LABEL_PERFECT = 1.0

# Per-target weights. Split weight across regression and classification:
# the two continuous targets carry the bulk, and the label rounds out.
W_T1 = 0.35
W_T2 = 0.35
W_LABEL = 0.30
assert abs(W_T1 + W_T2 + W_LABEL - 1.0) < 1e-9


def _sre(pred: Any, true: Any) -> float:
    """Standardized RMSE: RMSE(pred, true) / std(true)."""
    import numpy as np

    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    denom = float(np.std(true, ddof=0))
    return rmse / denom if denom > 0 else rmse


# Per-target progress + the exponential anchor curve come from the shared
# calibration toolkit (grading.calibration) so the methodology is identical
# across tasks and centrally tested.
_progress_higher_better = calibration.progress_higher_better
_progress_lower_better = calibration.progress_lower_better


def _reference_aggregate_x() -> float:
    xt1 = _progress_lower_better(T1_REF, T1_FLOOR, T1_PERFECT)
    xt2 = _progress_lower_better(T2_REF, T2_FLOOR, T2_PERFECT)
    xlb = _progress_higher_better(LABEL_REF, LABEL_FLOOR, LABEL_PERFECT)
    return W_T1 * xt1 + W_T2 * xt2 + W_LABEL * xlb


_CURVE = calibration.ExponentialCurve.from_reference(_reference_aggregate_x())


def _exponential_score(x: float) -> float:
    return _CURVE.score(x)


def _load_submission(workspace: Path) -> Any:
    # Use the sanctioned CSV loader so the submission gets the symlink/FIFO,
    # size, schema, and non-finite guards (and AgentFault on any of them) for
    # free -- never read the agent artifact by hand.
    from grading import helpers

    return helpers.load_submission_or_fault(
        workspace / "submission.csv",
        required_columns=["t1", "t2", "label"],
        allow_extra_columns=True,
    )


def _load_truth(private: Path) -> Any:
    import pandas as pd

    return pd.read_parquet(private / "test_target.parquet")


def compute_score(
    workspace: Path, trajectory: list[dict[str, Any]] | None, private: Path
) -> dict[str, Any]:
    """Score `/tmp/output/submission.csv` against hidden test targets.

    The `trajectory` argument is unused for ML_Envs-style continuous
    scoring; the score is a deterministic function of the submitted CSV
    and hidden target parquet.
    """
    _ = trajectory

    # Hidden truth parquet is author data: a load failure is infra, not the
    # agent's fault, so let it propagate (env_internal_failure -> discard).
    truth = _load_truth(private)

    try:
        sub = _load_submission(workspace)
    except AgentFault:
        raise
    except Exception as exc:  # noqa: BLE001 - submission boundary
        raise AgentFault(
            f"could not load submission: {type(exc).__name__}: {exc}"
        ) from exc

    if len(sub) != len(truth):
        # Wrong row count is an agent fault -> typed AgentFault (kept 0.0), not a
        # homebrew score dict, so it carries env_internal_failure=False.
        raise AgentFault(f"row count mismatch (sub={len(sub)}, truth={len(truth)})")

    # sklearn is an infra dependency: an import failure should propagate.
    from sklearn.metrics import f1_score

    # Hidden truth columns are author data: read them OUTSIDE the agent try so a
    # broken target (missing column / bad dtype) propagates as infra (discard),
    # not as an AgentFault kept 0.0.
    truth_t1 = truth["t1"].to_numpy()
    truth_t2 = truth["t2"].to_numpy()
    truth_label = truth["label"].astype(int).to_numpy()

    try:
        t1_sre = _sre(sub["t1"].to_numpy(), truth_t1)
        t2_sre = _sre(sub["t2"].to_numpy(), truth_t2)
        label_pred = sub["label"].astype(int).clip(0, 1).to_numpy()
        label_f1 = float(f1_score(truth_label, label_pred, average="binary"))
    except AgentFault:
        raise
    except Exception as exc:  # noqa: BLE001 - submission boundary
        raise AgentFault(
            f"could not compute metrics from submission: {type(exc).__name__}: {exc}"
        ) from exc

    xt1 = _progress_lower_better(t1_sre, T1_FLOOR, T1_PERFECT)
    xt2 = _progress_lower_better(t2_sre, T2_FLOOR, T2_PERFECT)
    xlb = _progress_higher_better(label_f1, LABEL_FLOOR, LABEL_PERFECT)
    x_agg = W_T1 * xt1 + W_T2 * xt2 + W_LABEL * xlb
    final = _exponential_score(x_agg)

    print("=" * 64)
    print("Per-target raw metrics:")
    print(
        f"  t1     SRE = {t1_sre:7.4f}  (floor={T1_FLOOR}, ref={T1_REF}, perfect={T1_PERFECT})"
    )
    print(
        f"  t2     SRE = {t2_sre:7.4f}  (floor={T2_FLOOR}, ref={T2_REF}, perfect={T2_PERFECT})"
    )
    print(
        f"  label  F1  = {label_f1:7.4f}  "
        f"(floor={LABEL_FLOOR}, ref={LABEL_REF}, perfect={LABEL_PERFECT})"
    )
    print("Per-target progress x_i in [0, 1]:")
    print(f"  xt1 = {xt1:.4f}  w={W_T1}")
    print(f"  xt2 = {xt2:.4f}  w={W_T2}")
    print(f"  xlb = {xlb:.4f}  w={W_LABEL}")
    print(f"Aggregate x     = {x_agg:.4f}  (x_ref = {_CURVE.x_ref:.4f})")
    print(f"Exponential (a, b) = ({_CURVE.a:.6g}, {_CURVE.b:.6g})")
    print(f"Final score     = {final:.4f}")
    print("=" * 64)

    return {
        "score": final,
        "subscores": {
            "t1_progress": xt1,
            "t2_progress": xt2,
            "label_progress": xlb,
        },
        "weights": {
            "t1_progress": W_T1,
            "t2_progress": W_T2,
            "label_progress": W_LABEL,
        },
        "metadata": {
            "return_shape": "continuous_score_dict",
            "source": "ML_Envs/examples/dataset-example-task_taiga",
            "raw_metrics": {
                "t1_sre": t1_sre,
                "t2_sre": t2_sre,
                "label_f1": label_f1,
            },
            "anchors": {
                "t1": {"floor": T1_FLOOR, "reference": T1_REF, "perfect": T1_PERFECT},
                "t2": {"floor": T2_FLOOR, "reference": T2_REF, "perfect": T2_PERFECT},
                "label": {
                    "floor": LABEL_FLOOR,
                    "reference": LABEL_REF,
                    "perfect": LABEL_PERFECT,
                },
            },
            "aggregate_progress": x_agg,
            "reference_aggregate_progress": _CURVE.x_ref,
            "exponential_constants": {"a": _CURVE.a, "b": _CURVE.b},
        },
    }
