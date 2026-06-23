#!/usr/bin/env python3
"""Run a small public OpenSeesPy diagnostic for the retrofit task.

The hidden verifier uses a richer three-story frame. This public probe is a
fast, deterministic model that lets agents confirm OpenSeesPy is installed and
that their device choices have a plausible stiffness effect before submission.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import openseespy.opensees as ops


DEFAULT_DESIGN = {"devices": []}
STORY_HEIGHT = 120.0
BASE_STIFFNESS = 25.0
LATERAL_LOAD = 10.0


def _load_design(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        raw = DEFAULT_DESIGN
    if not isinstance(raw, dict) or not isinstance(raw.get("devices", []), list):
        return DEFAULT_DESIGN
    return raw


def _device_stiffness(device: dict[str, Any]) -> float:
    device_type = device.get("type")
    if device_type == "steel_brace":
        area = _finite_float(device.get("area"), 0.0)
        return 18.0 * max(0.0, area)
    if device_type == "viscous_damper":
        coefficient = _finite_float(device.get("coefficient"), 0.0)
        return 0.35 * max(0.0, coefficient)
    return 0.0


def _finite_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _story_stiffnesses(design: dict[str, Any]) -> list[float]:
    stiffness = [BASE_STIFFNESS, BASE_STIFFNESS, BASE_STIFFNESS]
    for device in design.get("devices", []):
        if not isinstance(device, dict):
            continue
        story = device.get("story")
        if story in (1, 2, 3):
            stiffness[int(story) - 1] += _device_stiffness(device)
    return stiffness


def run_probe(design: dict[str, Any]) -> dict[str, Any]:
    story_k = _story_stiffnesses(design)

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    for story in range(4):
        ops.node(story + 1, 0.0, story * STORY_HEIGHT)
    ops.fix(1, 1, 1, 1)

    ops.geomTransf("Linear", 1)
    ops.uniaxialMaterial("Elastic", 1, 1.0)
    for story, stiffness in enumerate(story_k, start=1):
        ops.element(
            "elasticBeamColumn",
            story,
            story,
            story + 1,
            1.0,
            stiffness,
            1.0,
            1,
        )

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(4, LATERAL_LOAD, 0.0, 0.0)
    ops.system("BandGeneral")
    ops.numberer("Plain")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    analyze_code = int(ops.analyze(1))

    roof_disp = float(ops.nodeDisp(4, 1))
    max_drift_ratio = max(
        abs(float(ops.nodeDisp(i + 1, 1)) - float(ops.nodeDisp(i, 1))) / STORY_HEIGHT
        for i in range(1, 4)
    )
    base_shear_proxy = sum(story_k)
    ops.wipe()
    return {
        "analyze_code": analyze_code,
        "story_stiffness": [round(k, 4) for k in story_k],
        "roof_disp": round(roof_disp, 6),
        "max_drift_ratio": round(max_drift_ratio, 8),
        "base_shear_proxy": round(base_shear_proxy, 4),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: opensees_probe.py RETROFIT_DESIGN_JSON", file=sys.stderr)
        return 2

    design = _load_design(Path(argv[1]))
    result = run_probe(design)
    print("OpenSeesPy probe complete")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["analyze_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
