"""Public constants for the OpenSees three-story retrofit task.

This helper intentionally contains no hidden analysis cases or scoring anchors.
It exists so agents can inspect the design space without reverse-engineering the
grader.
"""

from __future__ import annotations

from typing import Any

STORIES = (1, 2, 3)
BAYS = (1, 2)
MAX_DEVICES = 6
BUDGET = 100.0

VISCOUS_DAMPER_COEFFICIENT_RANGE = (10.0, 90.0)
STEEL_BRACE_AREA_RANGE = (0.5, 4.0)


def device_cost(device: dict[str, Any]) -> float:
    """Return the public deterministic cost of one submitted device."""
    if device.get("type") == "viscous_damper":
        return 4.0 + 0.55 * float(device["coefficient"])
    if device.get("type") == "steel_brace":
        return 6.0 + 8.0 * float(device["area"])
    raise ValueError(f"unknown device type: {device.get('type')!r}")


def total_cost(design: dict[str, Any]) -> float:
    """Return the public deterministic cost of a submitted design."""
    return sum(device_cost(device) for device in design.get("devices", []))
