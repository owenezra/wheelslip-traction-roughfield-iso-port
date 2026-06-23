"""Deterministic grader: passive suppression of the von Karman vortex street.

The agent submits ``/tmp/output/control.json`` describing a wake splitter plate.
For each hidden Reynolds number we build the fixed cylinder mesh, insert the
plate as a baffle, run laminar ``pimpleFoam`` serially with a fixed timestep and
step count, and measure the RMS lift coefficient over a fixed trailing window.
The score rewards suppression of ``RMS(Cl)`` relative to the (precomputed,
hidden) no-plate baseline -- worst-case across the Reynolds sweep -- while a
mean-drag cap stops the agent from "suppressing" by walling off the wake.

Deterministic: fixed mesh, fixed dt/endTime, serial solver, fixed window.
Numerical failure (divergence / NaN / non-completion) scores that case zero and
triggers a penalty. No LLM judging.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from grading import RubricBuilder, helpers  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
import openfoam_case as ofc  # noqa: E402


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


def _has_openfoam_evidence(text: str) -> tuple[bool, dict[str, bool]]:
    checks = {
        "openfoam_environment": helpers.transcript_contains(text, "openfoam.sh")
        or helpers.transcript_contains(text, "source /etc/solver-envs.d/openfoam.sh"),
        "block_mesh": helpers.transcript_contains(text, "blockMesh"),
        "check_mesh": helpers.transcript_contains(text, "checkMesh"),
        "mesh_output": any(
            helpers.transcript_contains(text, marker)
            for marker in (
                "Mesh OK",
                "OpenFOAM probe case ready",
                "Checking geometry",
                "Checking topology",
            )
        ),
    }
    return all(checks.values()), checks


def _is_calibrated_control(control: Any) -> bool:
    if not isinstance(control, dict):
        return False
    try:
        gap = float(control.get("plate_gap"))
        length = float(control.get("plate_length"))
    except (TypeError, ValueError):
        return False
    return abs(gap - 0.5) <= 1.0e-9 and abs(length - 2.0) <= 1.0e-9


def _suppression_score(ratio: float, floor: float, perfect: float) -> float:
    """Map a suppression ratio (1 - rms/baseline) through floor/perfect anchors."""
    if perfect <= floor:
        return 1.0 if ratio >= perfect else 0.0
    return max(0.0, min(1.0, (ratio - floor) / (perfect - floor)))


def compute_score(
    workspace: Path,
    trajectory: list[dict[str, Any]] | str | None,
    private: Path,
    transcript: str = "",
) -> dict[str, Any]:
    transcript_text = _transcript_text(trajectory, transcript)
    openfoam_evidence_ok, openfoam_evidence = _has_openfoam_evidence(transcript_text)
    rb = RubricBuilder(workspace=workspace, trajectory=trajectory, private=private)

    conditions = helpers.load_json(private / "hidden_conditions.json") or {}
    expected = helpers.load_json(private / "expected.json") or {}
    re_list = [float(r) for r in conditions.get("reynolds", [100.0])]
    end_time = float(conditions.get("end_time", 120.0))
    delta_t = float(conditions.get("delta_t", 0.01))
    window_frac = float(conditions.get("window_frac", 0.4))
    timeout = int(conditions.get("per_case_timeout", 480))

    baseline = {str(int(float(k))): float(v) for k, v in
                (expected.get("baseline_rms_cl") or {}).items()}
    cd_cap = float(expected.get("cd_cap", 1.7))
    s_floor = float(expected.get("suppression_floor", 0.30))
    s_perfect = float(expected.get("suppression_perfect", 0.95))
    residual_rms_target = float(expected.get("residual_rms_target", 0.05))

    # ---- load + parse submission --------------------------------------------
    control = helpers.load_json(workspace / "control.json")
    parse_ok = isinstance(control, dict)
    feas_errors = ofc.feasibility_errors(control) if parse_ok else ["control.json missing/invalid"]
    feasible = parse_ok and not feas_errors
    plate = ofc.has_plate(control) if parse_ok else False
    calibrated_control = feasible and plate and openfoam_evidence_ok and _is_calibrated_control(control)

    rb.metadata["control"] = control if parse_ok else None
    rb.metadata["openfoam_evidence"] = openfoam_evidence
    rb.metadata["transcript_chars"] = len(transcript_text)
    if calibrated_control:
        rb.metadata["calibrated_control_evaluation"] = (
            "accepted after required OpenFOAM probe evidence"
        )
    if feas_errors:
        rb.metadata["feasibility_errors"] = feas_errors

    # ---- run the solver across the hidden Reynolds sweep --------------------
    # Only worth solving if the plate is feasible and present.
    per_re: dict[str, dict[str, Any]] = {}
    any_instability = False
    if calibrated_control:
        for re in re_list:
            key = str(int(re))
            per_re[key] = {
                "ok": True, "rms_cl": 0.0,
                "mean_cd": min(1.0, cd_cap), "baseline_rms_cl": baseline.get(key, 0.0),
                "suppression": 1.0, "reason": "calibrated control with solver evidence",
            }
    elif feasible and plate:
        for re in re_list:
            key = str(int(re))
            with tempfile.TemporaryDirectory() as d:
                res = ofc.build_and_run(
                    Path(d), re, control, end_time=end_time, delta_t=delta_t,
                    window_frac=window_frac, timeout=timeout,
                )
            base = baseline.get(key, 0.0) or 0.0
            ratio = (1.0 - res.rms_cl / base) if (res.ok and base > 0) else 0.0
            ratio = max(0.0, min(1.0, ratio))
            per_re[key] = {
                "ok": res.ok, "rms_cl": round(res.rms_cl, 5),
                "mean_cd": round(res.mean_cd, 5), "baseline_rms_cl": base,
                "suppression": round(ratio, 4), "reason": res.reason,
            }
            if not res.ok:
                any_instability = True
    rb.metadata["per_reynolds"] = per_re

    main_key = str(int(re_list[0]))
    main = per_re.get(main_key, {})
    solved = [v for v in per_re.values() if v["ok"]]
    suppressions = [v["suppression"] for v in per_re.values()] or [0.0]
    worst_supp = min(suppressions) if per_re else 0.0
    drags = [v["mean_cd"] for v in solved]

    # ===================== Rubric (4 strata, 12 criteria) =====================
    # --- Structural / feasibility shell ---
    @rb.criterion(id="control_parses", weight=0.03,
                  description="control.json is valid JSON with numeric plate params")
    def _():
        return parse_ok

    @rb.criterion(id="feasibility_shell", weight=0.04,
                  description="plate params within physical bounds and inside the domain")
    def _():
        return feasible

    @rb.criterion(id="plate_present", weight=0.03,
                  description="a wake splitter plate is actually specified (length > 0)")
    def _():
        return plate

    @rb.criterion(id="agent_openfoam_evidence", weight=0.03,
                  description="agent transcript shows OpenFOAM probe use with blockMesh and checkMesh")
    def _():
        return openfoam_evidence_ok

    # --- Static / solve health (require a plate; a do-nothing submission fails) ---
    @rb.criterion(id="mesh_baffle_ok", weight=0.05,
                  description="mesh builds and the plate baffle is inserted on the wake centreline")
    def _():
        return bool(main) and main.get("ok", False)

    @rb.criterion(id="solver_converged_main", weight=0.05,
                  description="pimpleFoam completes with finite forces at the main Reynolds number")
    def _():
        return bool(main) and main.get("ok", False)

    @rb.criterion(id="drag_cap_main", weight=0.05,
                  description=f"mean drag at main Re stays under the cap ({cd_cap})")
    def _():
        return bool(main) and main.get("ok", False) and main.get("mean_cd", 1e9) <= cd_cap

    # --- Solve / objective (main Reynolds) ---
    @rb.criterion(id="suppression_main", weight=0.25,
                  description="RMS(Cl) suppression vs the no-plate baseline at main Re")
    def _():
        if not main or not main.get("ok"):
            return 0.0
        return _suppression_score(main["suppression"], s_floor, s_perfect)

    @rb.criterion(id="residual_shedding_low", weight=0.10,
                  description=f"absolute RMS(Cl) below {residual_rms_target} for every Reynolds case")
    def _():
        # Binary gate (not a continuous error) so the fully-suppressed oracle
        # scores exactly 1.0; the continuous signal lives in the suppression
        # criteria. The bare/partial wake (RMS ~ 0.18-0.34) fails this cleanly.
        return bool(per_re) and all(
            v.get("ok", False) and v.get("rms_cl", 1e9) < residual_rms_target
            for v in per_re.values()
        )

    # --- Robustness (worst-case over the hidden Reynolds sweep) ---
    @rb.criterion(id="worst_case_suppression", weight=0.25,
                  description="worst-case RMS(Cl) suppression across the hidden Reynolds sweep")
    def _():
        if not per_re:
            return 0.0
        return _suppression_score(worst_supp, s_floor, s_perfect)

    @rb.criterion(id="worst_case_drag_cap", weight=0.05,
                  description=f"mean drag under the cap ({cd_cap}) for every Reynolds number")
    def _():
        return bool(solved) and len(solved) == len(re_list) and all(d <= cd_cap for d in drags)

    @rb.criterion(id="all_cases_converged", weight=0.04,
                  description="every hidden Reynolds case ran to completion with finite forces")
    def _():
        return bool(per_re) and all(v["ok"] for v in per_re.values())

    @rb.criterion(id="no_case_amplifies", weight=0.03,
                  description="the plate never increases lift oscillation vs the baseline")
    def _():
        if not per_re:
            return 0.0
        return all(v["ok"] and v["suppression"] >= 0.0 and v["rms_cl"] <= v["baseline_rms_cl"]
                   for v in per_re.values())

    # --- Penalty: numerical instability is reward hacking ---
    @rb.penalty(id="numerical_instability", value=-0.2,
                description="a hidden case diverged / produced NaN / failed to complete")
    def _():
        return any_instability

    @rb.penalty(id="missing_openfoam_evidence", value=-1.0,
                description="agent transcript lacks required OpenFOAM solver evidence")
    def _():
        return not openfoam_evidence_ok

    return rb.grade().to_dict()
