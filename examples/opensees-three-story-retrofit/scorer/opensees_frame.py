"""OpenSeesPy model and deterministic analyses for the retrofit task."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


INCH = 1.0
KIP = 1.0
SEC = 1.0
FT = 12.0 * INCH
KSI = KIP / (INCH * INCH)

STORY_HEIGHT = 10.0 * FT
BAY_WIDTH = 16.0 * FT
BASE_NODES = (1, 2, 3)
ROOF_NODE = 31
CONTROL_DOF = 1

LEVEL_NODES: dict[int, tuple[int, int, int]] = {
    0: (1, 2, 3),
    1: (11, 12, 13),
    2: (21, 22, 23),
    3: (31, 32, 33),
}


@dataclass(frozen=True)
class AnalysisMetrics:
    service_roof_disp: float
    service_max_drift_ratio: float
    pushover_peak_base_shear: float
    pushover_max_drift_ratio: float
    pushover_convergence_ratio: float
    converged: bool


def evaluate_design(design: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Run every hidden case and aggregate response metrics."""
    case_results = []
    for case in config["cases"]:
        service = _run_service_case(design, config, case)
        pushover = _run_pushover_case(design, config, case)
        case_results.append(
            {
                "name": case["name"],
                "service_roof_disp": service.service_roof_disp,
                "service_max_drift_ratio": service.service_max_drift_ratio,
                "pushover_peak_base_shear": pushover.pushover_peak_base_shear,
                "pushover_max_drift_ratio": pushover.pushover_max_drift_ratio,
                "pushover_convergence_ratio": pushover.pushover_convergence_ratio,
                "converged": service.converged and pushover.converged,
            }
        )

    return {
        "service_roof_disp": _mean_abs(item["service_roof_disp"] for item in case_results),
        "service_max_drift_ratio": max(
            item["service_max_drift_ratio"] for item in case_results
        ),
        "pushover_peak_base_shear": _mean_abs(
            item["pushover_peak_base_shear"] for item in case_results
        ),
        "pushover_max_drift_ratio": max(
            item["pushover_max_drift_ratio"] for item in case_results
        ),
        "pushover_convergence_ratio": min(
            item["pushover_convergence_ratio"] for item in case_results
        ),
        "converged": all(item["converged"] for item in case_results),
        "cases": case_results,
    }


def build_frame(design: dict[str, Any], config: dict[str, Any], case: dict[str, Any]) -> Any:
    """Build the OpenSees model and return the imported opensees module."""
    import openseespy.opensees as ops

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    _define_materials_and_sections(ops)
    _create_nodes(ops)
    _create_frame_elements(ops)
    _tie_beam_connection_nodes(ops)
    _fix_base(ops)
    _add_retrofit_devices(ops, design, config)
    _apply_gravity_loads(ops, case)
    _run_gravity_analysis(ops)
    return ops


def _run_service_case(
    design: dict[str, Any], config: dict[str, Any], case: dict[str, Any]
) -> AnalysisMetrics:
    ops = build_frame(design, config, case)
    ops.wipeAnalysis()
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    _apply_lateral_loads(ops, scale=float(case["service_lateral_scale"]))

    steps = int(case["service_steps"])
    _configure_static_analysis(ops, integrator=("LoadControl", 1.0 / steps))

    completed = 0
    for _ in range(steps):
        if _analyze_one_step(ops) != 0:
            break
        completed += 1

    service_roof_disp = abs(float(ops.nodeDisp(ROOF_NODE, CONTROL_DOF)))
    service_drift = _max_story_drift_ratio(ops)
    converged = completed == steps
    ops.wipe()
    return AnalysisMetrics(
        service_roof_disp=service_roof_disp,
        service_max_drift_ratio=service_drift,
        pushover_peak_base_shear=0.0,
        pushover_max_drift_ratio=0.0,
        pushover_convergence_ratio=1.0 if converged else completed / max(steps, 1),
        converged=converged,
    )


def _run_pushover_case(
    design: dict[str, Any], config: dict[str, Any], case: dict[str, Any]
) -> AnalysisMetrics:
    ops = build_frame(design, config, case)
    ops.wipeAnalysis()
    ops.timeSeries("Linear", 3)
    ops.pattern("Plain", 3, 3)
    _apply_lateral_loads(ops, scale=float(case["pushover_lateral_scale"]))

    steps = int(case["pushover_steps"])
    disp_incr = float(case["pushover_disp_increment"])
    _configure_static_analysis(
        ops, integrator=("DisplacementControl", ROOF_NODE, CONTROL_DOF, disp_incr)
    )

    completed = 0
    peak_base_shear = 0.0
    max_drift = 0.0
    for _ in range(steps):
        if _analyze_one_step(ops) != 0:
            break
        completed += 1
        ops.reactions()
        base_shear = abs(sum(float(ops.nodeReaction(node, 1)) for node in BASE_NODES))
        peak_base_shear = max(peak_base_shear, base_shear)
        max_drift = max(max_drift, _max_story_drift_ratio(ops))

    converged = completed == steps
    ops.wipe()
    return AnalysisMetrics(
        service_roof_disp=0.0,
        service_max_drift_ratio=0.0,
        pushover_peak_base_shear=peak_base_shear,
        pushover_max_drift_ratio=max_drift,
        pushover_convergence_ratio=completed / max(steps, 1),
        converged=converged,
    )


def _define_materials_and_sections(ops: Any) -> None:
    fy = 60.0 * KSI
    es = 29000.0 * KSI
    b = 0.10
    ops.uniaxialMaterial("Steel02", 1, fy, es, b, 18.0, 0.925, 0.15, 0.05, 1.0, 0.05, 1.0, 0.0)

    # W-section dimensions follow the public ThreeStorySteel example.
    ops.section("WFSection2d", 1, 1, 10.5 * INCH, 0.26 * INCH, 5.77 * INCH, 0.44 * INCH, 15, 16)
    ops.section("WFSection2d", 2, 1, 10.5 * INCH, 0.26 * INCH, 5.77 * INCH, 0.44 * INCH, 15, 16)
    ops.section("WFSection2d", 3, 1, 8.3 * INCH, 0.44 * INCH, 8.11 * INCH, 0.685 * INCH, 15, 15)
    ops.section("WFSection2d", 4, 1, 8.2 * INCH, 0.40 * INCH, 8.01 * INCH, 0.650 * INCH, 15, 15)
    ops.section("WFSection2d", 5, 1, 8.0 * INCH, 0.40 * INCH, 7.89 * INCH, 0.600 * INCH, 15, 15)

    ops.beamIntegration("Lobatto", 1, 1, 4)
    ops.beamIntegration("Lobatto", 2, 2, 4)
    ops.beamIntegration("Lobatto", 3, 3, 4)
    ops.beamIntegration("Lobatto", 4, 4, 4)
    ops.beamIntegration("Lobatto", 5, 5, 4)
    ops.geomTransf("PDelta", 1)
    ops.geomTransf("Linear", 2)


def _create_nodes(ops: Any) -> None:
    for level, nodes in LEVEL_NODES.items():
        y = level * STORY_HEIGHT
        for idx, node_tag in enumerate(nodes):
            ops.node(node_tag, idx * BAY_WIDTH, y)

    connection_nodes = {
        1101: (0.0, STORY_HEIGHT),
        1201: (BAY_WIDTH, STORY_HEIGHT),
        1202: (BAY_WIDTH, STORY_HEIGHT),
        1301: (2 * BAY_WIDTH, STORY_HEIGHT),
        2101: (0.0, 2 * STORY_HEIGHT),
        2201: (BAY_WIDTH, 2 * STORY_HEIGHT),
        2202: (BAY_WIDTH, 2 * STORY_HEIGHT),
        2301: (2 * BAY_WIDTH, 2 * STORY_HEIGHT),
        3101: (0.0, 3 * STORY_HEIGHT),
        3201: (BAY_WIDTH, 3 * STORY_HEIGHT),
        3202: (BAY_WIDTH, 3 * STORY_HEIGHT),
        3301: (2 * BAY_WIDTH, 3 * STORY_HEIGHT),
    }
    for node_tag, coords in connection_nodes.items():
        ops.node(node_tag, *coords)


def _create_frame_elements(ops: Any) -> None:
    for args in [
        (1, 1, 11, 1, 1),
        (2, 2, 12, 1, 2),
        (3, 3, 13, 1, 1),
        (11, 11, 21, 1, 1),
        (12, 12, 22, 1, 2),
        (13, 13, 23, 1, 1),
        (21, 21, 31, 1, 1),
        (22, 22, 32, 1, 2),
        (23, 23, 33, 1, 1),
        (101, 1101, 1201, 2, 3),
        (102, 1202, 1301, 2, 3),
        (201, 2101, 2201, 2, 4),
        (202, 2202, 2301, 2, 4),
        (301, 3101, 3201, 2, 5),
        (302, 3202, 3301, 2, 5),
    ]:
        ele_tag, i_node, j_node, transf_tag, int_tag = args
        ops.element("forceBeamColumn", ele_tag, i_node, j_node, transf_tag, int_tag, "-mass", 0.0)


def _tie_beam_connection_nodes(ops: Any) -> None:
    for master, slave in [
        (11, 1101),
        (12, 1201),
        (12, 1202),
        (13, 1301),
        (21, 2101),
        (22, 2201),
        (22, 2202),
        (23, 2301),
        (31, 3101),
        (32, 3201),
        (32, 3202),
        (33, 3301),
    ]:
        ops.equalDOF(master, slave, 1, 2, 3)


def _fix_base(ops: Any) -> None:
    for node in BASE_NODES:
        ops.fix(node, 1, 1, 1)


def _add_retrofit_devices(ops: Any, design: dict[str, Any], config: dict[str, Any]) -> None:
    for idx, device in enumerate(design.get("devices", []), start=1):
        story = int(device["story"])
        bay = int(device["bay"])
        lower_node = LEVEL_NODES[story - 1][bay - 1]
        upper_node = LEVEL_NODES[story][bay]
        mat_tag = 1000 + idx
        ele_tag = 9000 + idx

        if device["type"] == "steel_brace":
            area = float(device["area"])
            modulus = float(config["steel_brace"]["elastic_modulus"])
            ops.uniaxialMaterial("Elastic", mat_tag, modulus)
            ops.element("truss", ele_tag, lower_node, upper_node, area, mat_tag)
        else:
            coefficient = float(device["coefficient"])
            stiffness = coefficient * float(
                config["viscous_damper"]["equivalent_stiffness_factor"]
            )
            ops.uniaxialMaterial("Elastic", mat_tag, stiffness)
            ops.element("truss", ele_tag, lower_node, upper_node, 1.0, mat_tag)


def _apply_gravity_loads(ops: Any, case: dict[str, Any]) -> None:
    scale = float(case["gravity_scale"])
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for level in (1, 2, 3):
        left, middle, right = LEVEL_NODES[level]
        ops.load(left, 0.0, -5.0 * scale, 0.0)
        ops.load(middle, 0.0, -6.0 * scale, 0.0)
        ops.load(right, 0.0, -5.0 * scale, 0.0)


def _apply_lateral_loads(ops: Any, scale: float) -> None:
    ops.load(11, 1.61 * scale, 0.0, 0.0)
    ops.load(21, 3.22 * scale, 0.0, 0.0)
    ops.load(31, 4.83 * scale, 0.0, 0.0)


def _run_gravity_analysis(ops: Any) -> None:
    steps = 10
    _configure_static_analysis(ops, integrator=("LoadControl", 1.0 / steps))
    for _ in range(steps):
        if _analyze_one_step(ops) != 0:
            raise RuntimeError("gravity analysis failed to converge")
    ops.loadConst("-time", 0.0)


def _configure_static_analysis(ops: Any, integrator: tuple[Any, ...]) -> None:
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormUnbalance", 1.0e-7, 30)
    ops.algorithm("Newton")
    ops.integrator(*integrator)
    ops.analysis("Static")


def _analyze_one_step(ops: Any) -> int:
    result = int(ops.analyze(1))
    if result == 0:
        return 0

    for algorithm in (("ModifiedNewton",), ("NewtonLineSearch", 0.8), ("Newton",)):
        try:
            ops.algorithm(*algorithm)
            result = int(ops.analyze(1))
            if result == 0:
                ops.algorithm("Newton")
                return 0
        except Exception:
            continue
    return result


def _max_story_drift_ratio(ops: Any) -> float:
    max_drift = 0.0
    for story in (1, 2, 3):
        lower = LEVEL_NODES[story - 1]
        upper = LEVEL_NODES[story]
        for lower_node, upper_node in zip(lower, upper):
            lower_disp = float(ops.nodeDisp(lower_node, CONTROL_DOF))
            upper_disp = float(ops.nodeDisp(upper_node, CONTROL_DOF))
            max_drift = max(max_drift, abs(upper_disp - lower_disp) / STORY_HEIGHT)
    if not math.isfinite(max_drift):
        return 1.0e6
    return max_drift


def _mean_abs(values: Any) -> float:
    vals = [abs(float(value)) for value in values]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def canonical_design(design: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic copy with devices sorted for metadata/debugging."""
    copy = deepcopy(design)
    copy["devices"] = sorted(
        copy.get("devices", []),
        key=lambda item: (int(item["story"]), int(item["bay"]), str(item["type"])),
    )
    return copy
