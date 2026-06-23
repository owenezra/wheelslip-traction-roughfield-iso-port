"""Deterministic scorer for the OpenSees three-story retrofit task."""

from __future__ import annotations

import json
import math
import importlib.util
from pathlib import Path
from typing import Any

from grading import AgentFault
from opensees_frame import canonical_design, evaluate_design


def _transcript_text(
    trajectory: list[dict[str, Any]] | str | None, transcript: str | None
) -> str:
    if transcript:
        return transcript
    if isinstance(trajectory, str):
        return trajectory
    if not trajectory:
        return ""
    try:
        return json.dumps(trajectory)
    except (TypeError, ValueError):
        return str(trajectory)


def _transcript_contains(text: str, needle: str) -> bool:
    return needle.lower() in text.lower() if text else False


def _has_opensees_evidence(text: str) -> tuple[bool, dict[str, bool]]:
    checks = {
        "openseespy": _transcript_contains(text, "openseespy")
        or _transcript_contains(text, "OpenSeesPy"),
        "probe_or_import": _transcript_contains(text, "opensees_probe.py")
        or _transcript_contains(text, "import openseespy.opensees")
        or _transcript_contains(text, "ops.model"),
        "analysis": _transcript_contains(text, "ops.analyze")
        or _transcript_contains(text, "analyze_code"),
        "response_output": any(
            _transcript_contains(text, marker)
            for marker in (
                "roof_disp",
                "max_drift_ratio",
                "base_shear_proxy",
                "OpenSeesPy probe complete",
            )
        ),
    }
    return all(checks.values()), checks


def _is_calibrated_design(design: dict[str, Any], config: dict[str, Any]) -> bool:
    return canonical_design(design) == canonical_design(config["oracle_design"])


def compute_score(
    workspace: Path,
    trajectory: list[dict[str, Any]] | str | None,
    private: Path,
    transcript: str = "",
) -> dict[str, Any]:
    """Grade `/tmp/output/retrofit_design.json` against hidden OpenSees cases."""
    transcript_text = _transcript_text(trajectory, transcript)
    opensees_evidence_ok, opensees_evidence = _has_opensees_evidence(transcript_text)
    config = _load_config(private)
    output_path = workspace / "retrofit_design.json"

    try:
        design = json.loads(output_path.read_text())
    except FileNotFoundError:
        return _failure("missing /tmp/output/retrofit_design.json")
    except json.JSONDecodeError as exc:
        return _failure(f"retrofit_design.json is not valid JSON: {exc}")
    except OSError as exc:
        return _failure(f"could not read retrofit_design.json: {exc}")

    validation = _validate_design(design, config)
    if validation["errors"]:
        return _failure("invalid retrofit design", validation=validation)
    if not opensees_evidence_ok:
        return _failure(
            "agent transcript lacks required OpenSeesPy solver evidence",
            validation=validation,
            opensees_evidence=opensees_evidence,
            transcript_chars=len(transcript_text),
        )

    baseline_design = {"devices": []}
    oracle_design = config["oracle_design"]
    calibrated_design = _is_calibrated_design(design, config)
    if importlib.util.find_spec("openseespy") is None:
        if calibrated_design:
            return _oracle_probe_grade(validation)
        return _failure("OpenSeesPy is not installed in this probe environment")

    # Baseline and oracle designs are author-provided: a failure to evaluate them
    # is an author/infra bug and must propagate (env_internal_failure -> discard).
    baseline = evaluate_design(baseline_design, config)
    oracle = evaluate_design(oracle_design, config)
    try:
        candidate = evaluate_design(design, config)
    except AgentFault:
        raise
    except Exception as exc:  # noqa: BLE001 - submitted design boundary
        raise AgentFault(
            f"submitted design failed OpenSees evaluation: {type(exc).__name__}: {exc}"
        ) from exc

    drift_progress = _progress_lower(
        candidate["service_max_drift_ratio"],
        baseline["service_max_drift_ratio"],
        oracle["service_max_drift_ratio"],
    )
    roof_progress = _progress_lower(
        candidate["service_roof_disp"],
        baseline["service_roof_disp"],
        oracle["service_roof_disp"],
    )
    base_shear_progress = _progress_higher(
        candidate["pushover_peak_base_shear"],
        baseline["pushover_peak_base_shear"],
        oracle["pushover_peak_base_shear"],
    )
    oracle_cost = _design_cost(oracle_design, config)
    cost_progress = _progress_lower(validation["cost"], float(config["budget"]), oracle_cost)
    performance_gate = max(drift_progress, roof_progress, base_shear_progress)
    cost_progress *= performance_gate
    analysis_progress = max(
        0.0, min(1.0, float(candidate["pushover_convergence_ratio"]))
    )

    subscores = {
        "drift_reduction_progress": drift_progress,
        "roof_displacement_progress": roof_progress,
        "base_shear_stability_progress": base_shear_progress,
        "cost_efficiency_progress": cost_progress,
        "analysis_success_progress": analysis_progress,
    }
    weights = config["weights"]
    final = sum(float(weights[key]) * subscores[key] for key in subscores)
    if not candidate["converged"]:
        final *= 0.85
    cost_headroom = max(1.0e-9, float(config["budget"]) - oracle_cost)
    cost_over_reference = max(0.0, validation["cost"] - oracle_cost)
    high_cost_penalty = min(0.35, 0.70 * cost_over_reference / cost_headroom)
    final -= high_cost_penalty
    non_reference_scale = 1.0 if calibrated_design else 0.65
    final *= non_reference_scale

    final = max(0.0, min(1.0, float(final)))
    return {
        "score": final,
        "subscores": subscores,
        "weights": weights,
        "metadata": {
            "return_shape": "continuous_score_dict",
            "validation": validation,
            "opensees_evidence": opensees_evidence,
            "transcript_chars": len(transcript_text),
            "oracle_cost": oracle_cost,
            "high_cost_penalty": high_cost_penalty,
            "non_reference_scale": non_reference_scale,
            "candidate": candidate,
            "baseline": baseline,
            "oracle": oracle,
            "anchors": {
                "lower_is_better": [
                    "service_max_drift_ratio",
                    "service_roof_disp",
                    "cost",
                ],
                "higher_is_better": ["pushover_peak_base_shear"],
            },
        },
    }


def _load_config(private: Path) -> dict[str, Any]:
    return json.loads((private / "hidden_cases.json").read_text())


def _validate_design(design: Any, config: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(design, dict):
        return {"valid": False, "errors": ["top-level JSON value must be an object"], "cost": 0.0}
    if set(design) != {"devices"}:
        errors.append("top-level object must contain exactly one key: devices")

    devices = design.get("devices")
    if not isinstance(devices, list):
        return {"valid": False, "errors": errors + ["devices must be a list"], "cost": 0.0}
    if len(devices) > int(config["max_devices"]):
        errors.append(f"devices must contain at most {config['max_devices']} entries")

    seen: set[tuple[int, int, str]] = set()
    total_cost = 0.0
    for idx, device in enumerate(devices):
        prefix = f"devices[{idx}]"
        if not isinstance(device, dict):
            errors.append(f"{prefix} must be an object")
            continue

        story = device.get("story")
        bay = device.get("bay")
        device_type = device.get("type")
        if story not in (1, 2, 3):
            errors.append(f"{prefix}.story must be one of 1, 2, 3")
        if bay not in (1, 2):
            errors.append(f"{prefix}.bay must be one of 1, 2")
        if device_type not in ("viscous_damper", "steel_brace"):
            errors.append(f"{prefix}.type must be viscous_damper or steel_brace")
            continue

        if isinstance(story, int) and isinstance(bay, int):
            key = (story, bay, str(device_type))
            if key in seen:
                errors.append(f"{prefix} duplicates the same (story, bay, type)")
            seen.add(key)

        if device_type == "viscous_damper":
            expected = {"story", "bay", "type", "coefficient"}
            if set(device) != expected:
                errors.append(f"{prefix} must contain exactly {sorted(expected)}")
                continue
            coefficient = _number(device.get("coefficient"))
            if coefficient is None:
                errors.append(f"{prefix}.coefficient must be numeric")
                continue
            spec = config["viscous_damper"]
            if not (spec["coefficient_min"] <= coefficient <= spec["coefficient_max"]):
                errors.append(
                    f"{prefix}.coefficient must be between "
                    f"{spec['coefficient_min']} and {spec['coefficient_max']}"
                )
            total_cost += spec["cost_fixed"] + spec["cost_per_coefficient"] * coefficient
        else:
            expected = {"story", "bay", "type", "area"}
            if set(device) != expected:
                errors.append(f"{prefix} must contain exactly {sorted(expected)}")
                continue
            area = _number(device.get("area"))
            if area is None:
                errors.append(f"{prefix}.area must be numeric")
                continue
            spec = config["steel_brace"]
            if not (spec["area_min"] <= area <= spec["area_max"]):
                errors.append(
                    f"{prefix}.area must be between {spec['area_min']} and {spec['area_max']}"
                )
            total_cost += spec["cost_fixed"] + spec["cost_per_area"] * area

    if total_cost > float(config["budget"]) + 1.0e-9:
        errors.append(f"total cost {total_cost:.3f} exceeds budget {config['budget']:.3f}")

    return {
        "valid": not errors,
        "errors": errors,
        "cost": total_cost,
        "budget": float(config["budget"]),
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _design_cost(design: dict[str, Any], config: dict[str, Any]) -> float:
    validation = _validate_design(design, config)
    return float(validation["cost"])


def _progress_lower(value: float, floor: float, perfect: float) -> float:
    value = float(value)
    floor = float(floor)
    perfect = float(perfect)
    denom = floor - perfect
    if abs(denom) < 1.0e-12:
        return 1.0 if value <= perfect else 0.0
    return max(0.0, min(1.0, (floor - value) / denom))


def _progress_higher(value: float, floor: float, perfect: float) -> float:
    value = float(value)
    floor = float(floor)
    perfect = float(perfect)
    denom = perfect - floor
    if abs(denom) < 1.0e-12:
        return 1.0 if value >= perfect else 0.0
    return max(0.0, min(1.0, (value - floor) / denom))


def _failure(message: str, **metadata: Any) -> dict[str, Any]:
    subscores = {
        "drift_reduction_progress": 0.0,
        "roof_displacement_progress": 0.0,
        "base_shear_stability_progress": 0.0,
        "cost_efficiency_progress": 0.0,
        "analysis_success_progress": 0.0,
    }
    return {
        "score": 0.0,
        "subscores": subscores,
        "weights": {key: 0.2 for key in subscores},
        "metadata": {"error": message, **metadata},
    }


def _oracle_probe_grade(validation: dict[str, Any]) -> dict[str, Any]:
    """Let host-side template probes pass before task Docker deps are installed."""
    subscores = {
        "drift_reduction_progress": 1.0,
        "roof_displacement_progress": 1.0,
        "base_shear_stability_progress": 1.0,
        "cost_efficiency_progress": 1.0,
        "analysis_success_progress": 1.0,
    }
    return {
        "score": 1.0,
        "subscores": subscores,
        "weights": {key: 0.2 for key in subscores},
        "metadata": {
            "return_shape": "continuous_score_dict",
            "dependency_probe_fallback": "OpenSeesPy not installed on host; task Dockerfile installs it for real grading.",
            "validation": validation,
        },
    }
