#!/usr/bin/env python3
"""Generate a tiny public OpenFOAM probe case for the splitter-plate task.

The hidden grader owns the production cylinder-and-plate case. This public probe
exists only so agents can exercise the installed OpenFOAM stack and inspect
`blockMesh` / `checkMesh` output before finalizing a design.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


DEFAULT_CONTROL = {"plate_gap": 1.0, "plate_length": 1.0}


def _load_control(path: Path) -> dict[str, float]:
    try:
        raw: Any = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        raw = DEFAULT_CONTROL

    if not isinstance(raw, dict):
        raw = DEFAULT_CONTROL

    gap = _finite_float(raw.get("plate_gap"), DEFAULT_CONTROL["plate_gap"])
    length = _finite_float(raw.get("plate_length"), DEFAULT_CONTROL["plate_length"])
    return {
        "plate_gap": max(0.0, min(4.0, gap)),
        "plate_length": max(0.1, min(8.0, length)),
    }


def _finite_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _foam_header(class_name: str, object_name: str) -> str:
    return (
        "FoamFile\n"
        "{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {class_name};\n"
        f"    object      {object_name};\n"
        "}\n"
    )


def _block_mesh_dict(control: dict[str, float]) -> str:
    # Vary the streamwise extent with the proposed plate so the probe is tied to
    # the submitted control values while remaining a very fast rectangular mesh.
    x_max = 6.0 + control["plate_gap"] + control["plate_length"]
    nx = max(12, int(round(3.0 * x_max)))
    return f"""{_foam_header("dictionary", "blockMeshDict")}
scale 1;

vertices
(
    (0 -1 -0.05)
    ({x_max:.6f} -1 -0.05)
    ({x_max:.6f} 1 -0.05)
    (0 1 -0.05)
    (0 -1 0.05)
    ({x_max:.6f} -1 0.05)
    ({x_max:.6f} 1 0.05)
    (0 1 0.05)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} 8 1) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces ((0 4 7 3));
    }}
    outlet
    {{
        type patch;
        faces ((1 2 6 5));
    }}
    walls
    {{
        type wall;
        faces ((0 1 5 4) (3 7 6 2));
    }}
    frontAndBack
    {{
        type empty;
        faces ((0 3 2 1) (4 5 6 7));
    }}
);

mergePatchPairs
(
);
"""


def _control_dict() -> str:
    return f"""{_foam_header("dictionary", "controlDict")}
application     checkMesh;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""


def _fv_schemes() -> str:
    return f"""{_foam_header("dictionary", "fvSchemes")}
ddtSchemes
{{
    default         steadyState;
}}

gradSchemes
{{
    default         Gauss linear;
}}

divSchemes
{{
    default         none;
}}

laplacianSchemes
{{
    default         Gauss linear corrected;
}}

interpolationSchemes
{{
    default         linear;
}}

snGradSchemes
{{
    default         corrected;
}}
"""


def _fv_solution() -> str:
    return f"""{_foam_header("dictionary", "fvSolution")}
solvers
{{
}}
"""


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "usage: openfoam_probe.py CONTROL_JSON OUT_CASE_DIR",
            file=sys.stderr,
        )
        return 2

    control_path = Path(argv[1])
    case_dir = Path(argv[2])
    control = _load_control(control_path)

    (case_dir / "system").mkdir(parents=True, exist_ok=True)
    (case_dir / "constant").mkdir(parents=True, exist_ok=True)
    (case_dir / "0").mkdir(parents=True, exist_ok=True)
    (case_dir / "system" / "blockMeshDict").write_text(_block_mesh_dict(control))
    (case_dir / "system" / "controlDict").write_text(_control_dict())
    (case_dir / "system" / "fvSchemes").write_text(_fv_schemes())
    (case_dir / "system" / "fvSolution").write_text(_fv_solution())

    print(
        "OpenFOAM probe case ready: "
        f"plate_gap={control['plate_gap']:.3f}, "
        f"plate_length={control['plate_length']:.3f}, "
        f"case={case_dir}"
    )
    print(
        "Run: bash -lc 'source /etc/solver-envs.d/openfoam.sh && "
        f"blockMesh -case {case_dir} && checkMesh -case {case_dir}'"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
