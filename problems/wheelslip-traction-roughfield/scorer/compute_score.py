"""Deterministic ISO scorer for the rough-field wheelslip policy task."""
from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from grading import AgentFault, PolicyWorker, calibration

# The hidden MuJoCo simulator and submitted policy execute on the host CPU.
# These defaults make native math behavior stable and prevent oversubscription.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

# Public-only reference pipeline, measured in the legacy evaluation image on
# the unchanged fixed protocol. Re-bake after any simulator, dependency,
# reference, or hidden-fixture change.
X_REF = 0.6874068017
QUALITY_FLOOR = 0.0
QUALITY_PERFECT = 1.0
POLICY_CALL_TIMEOUT_S = 300.0


def _load_private_protocol(private: Path) -> ModuleType:
    """Load the author-owned private protocol from its exact root-only path."""
    private_dir = Path(private).resolve()
    protocol_file = private_dir / "wheelslip_eval_protocol.py"
    if not protocol_file.is_file():
        raise FileNotFoundError(f"missing private evaluation protocol: {protocol_file}")

    module_name = "wheelslip_eval_protocol"
    sys.modules.pop(module_name, None)
    sys.path.insert(0, str(private_dir))
    try:
        protocol = importlib.import_module(module_name)
    finally:
        try:
            sys.path.remove(str(private_dir))
        except ValueError:
            pass

    loaded_from = Path(protocol.__file__).resolve()
    if loaded_from != protocol_file:
        raise RuntimeError(
            f"private protocol resolved to unexpected path: {loaded_from}"
        )
    return protocol


def compute_score(
    workspace: Path,
    trajectory: list[dict[str, Any]] | None,
    private: Path,
) -> dict[str, Any]:
    """Roll out the submitted policy and return calibrated mean quality."""
    _ = trajectory

    # Private/author data loading deliberately sits outside agent-fault logic.
    protocol = _load_private_protocol(Path(private))
    seeds = tuple(int(seed) for seed in protocol.EVAL_SEEDS)
    if not seeds or len(seeds) != int(protocol.N_EVAL):
        raise RuntimeError("private evaluation seed configuration is invalid")

    policy_path = Path(workspace) / "policy.py"
    if not policy_path.is_file():
        raise AgentFault("missing required output: /tmp/output/policy.py")

    qualities: list[float] = []
    settled_count = 0

    # Submitted code runs only in PolicyWorker's privilege-dropped child.
    with PolicyWorker(
        policy_path,
        timeout_s=POLICY_CALL_TIMEOUT_S,
        factory_name="load_policy",
    ) as policy:
        try:
            has_reset = policy.has("reset")
        except Exception as exc:
            raise AgentFault(
                f"could not inspect submitted policy reset hook: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        for seed in seeds:
            if has_reset:
                try:
                    policy.call("reset")
                except Exception as exc:
                    raise AgentFault(
                        f"policy reset() failed at seed={seed}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
            try:
                result = protocol.run_episode(policy, seed)
            except protocol.PolicyContractError as exc:
                raise AgentFault(str(exc)) from exc

            quality = float(result["quality"])
            if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
                raise RuntimeError(
                    f"private evaluator produced invalid quality {quality!r} "
                    f"for seed {seed}"
                )
            qualities.append(quality)
            settled_count += int(bool(result["settled"]))

    if len(qualities) != len(seeds):
        raise RuntimeError("evaluation ended without one quality per hidden seed")

    mean_quality = math.fsum(qualities) / len(qualities)
    progress = calibration.progress_higher_better(
        mean_quality,
        floor=QUALITY_FLOOR,
        perfect=QUALITY_PERFECT,
    )
    curve = calibration.PiecewiseLinearCurve.from_reference(X_REF)
    final = float(curve.score(progress))
    if not math.isfinite(final):
        raise RuntimeError("calibration produced a non-finite score")
    final = min(1.0, max(0.0, final))

    return {
        "score": final,
        "subscores": {"mean_quality_progress": float(progress)},
        "weights": {"mean_quality_progress": 1.0},
        "metadata": {
            "mean_quality": float(mean_quality),
            "settled_rate": settled_count / len(seeds),
            "n_eval": len(seeds),
        },
    }
