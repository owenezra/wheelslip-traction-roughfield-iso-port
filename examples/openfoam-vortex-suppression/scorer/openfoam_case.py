"""Deterministic OpenFOAM case builder/runner for the vortex-suppression task.

The cylinder mesh is a FIXED, fully-structured full-domain O-grid (pure
``blockMesh``; never ``snappyHexMesh``), so von Karman shedding is captured
without a symmetry plane suppressing it. The agent's only influence is a wake
**splitter plate**, inserted as a zero-thickness baffle (``topoSet`` +
``createBaffles``) along the downstream centreline y=0 -- this changes the
baffle placement, not the mesh topology, keeping everything deterministic.

All solves are serial, fixed-``deltaT``, fixed-``endTime`` (fixed step count),
laminar ``pimpleFoam``. Metrics come from a fixed trailing window of the lift
coefficient time series. Divergence / NaN / non-completion are treated as
failures by the caller.

This module is import-safe on hosts without OpenFOAM; it only shells out to the
solver when ``build_and_run`` is called (inside the task image).
"""

from __future__ import annotations

import math
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

# ---- fixed geometry / domain / numerics (pinned for determinism) -----------
D = 1.0
R = D / 2.0
R_OUT = 2.0
X_UP, X_DOWN = -8.0, 25.0
Y_TOP, Y_BOT = 8.0, -8.0
Z0, Z1 = -0.5, 0.5
U0 = 1.0
N_R, N_C, N_OUT = 24, 14, 28
RADIAL_GRADE, OUT_GRADE = 8, 12
IC_KICK = 0.1  # deterministic transverse kick to seed the (Re>Re_c) instability

OF_ENV = "source /etc/solver-envs.d/openfoam.sh >/dev/null 2>&1"

# ---- control-parameter feasibility bounds (the "feasibility shell") --------
PLATE_GAP_MIN, PLATE_GAP_MAX = 0.0, 4.0      # in D, cylinder base -> plate LE
PLATE_LEN_MIN, PLATE_LEN_MAX = 0.0, 8.0      # in D (strictly > 0 for valid designs)
PLATE_LE_MAX_X = X_DOWN - 3.0                 # keep plate well inside the domain


@dataclass
class CaseResult:
    ok: bool
    mean_cd: float = 0.0
    rms_cl: float = 0.0
    amp_cl: float = 0.0
    strouhal: float = 0.0
    n_window: int = 0
    reason: str = ""


def feasibility_errors(control: dict) -> list[str]:
    """Return human-readable reasons the control params are out of bounds."""
    errs: list[str] = []
    try:
        gap = float(control["plate_gap"])
        length = float(control["plate_length"])
    except (KeyError, TypeError, ValueError):
        return ["control.json must provide numeric 'plate_gap' and 'plate_length'"]
    if not (PLATE_GAP_MIN <= gap <= PLATE_GAP_MAX):
        errs.append(f"plate_gap {gap} outside [{PLATE_GAP_MIN}, {PLATE_GAP_MAX}] D")
    if not (PLATE_LEN_MIN < length <= PLATE_LEN_MAX):
        errs.append(
            f"plate_length {length} outside ({PLATE_LEN_MIN}, {PLATE_LEN_MAX}] D"
        )
    if length > 0 and (R + gap * D + length * D) > PLATE_LE_MAX_X:
        errs.append("plate extends too far downstream (must stay inside the domain)")
    return errs


def has_plate(control: dict | None) -> bool:
    if not control:
        return False
    try:
        return float(control.get("plate_length", 0.0)) > 0.0
    except (TypeError, ValueError):
        return False


# ---- mesh + case generation ------------------------------------------------
def _vertices() -> list[tuple[float, float]]:
    angles = [math.radians(45 * k) for k in range(8)]
    inner = [(R * math.cos(a), R * math.sin(a)) for a in angles]
    mid = [(R_OUT * math.cos(a), R_OUT * math.sin(a)) for a in angles]
    box = [
        (X_DOWN, 0.0), (X_DOWN, Y_TOP), (0.0, Y_TOP), (X_UP, Y_TOP),
        (X_UP, 0.0), (X_UP, Y_BOT), (0.0, Y_BOT), (X_DOWN, Y_BOT),
    ]
    return inner + mid + box


def _block_mesh_dict() -> str:
    plane = _vertices()
    sz = len(plane)

    def g(layer, k, z):
        base = {"i": 0, "m": 8, "o": 16}[layer] + k
        return base + (sz if z else 0)

    vlines = []
    for z in (Z0, Z1):
        for (x, y) in plane:
            vlines.append(f"    ({x:.6f} {y:.6f} {z:.6f})")

    blocks, edges = [], []
    for k in range(8):
        kn = (k + 1) % 8
        ik, ikn = g("i", k, 0), g("i", kn, 0)
        mk, mkn = g("m", k, 0), g("m", kn, 0)
        ok, okn = g("o", k, 0), g("o", kn, 0)
        blocks.append(
            f"    hex ({ik} {mk} {mkn} {ikn} {ik+sz} {mk+sz} {mkn+sz} {ikn+sz}) "
            f"({N_R} {N_C} 1) simpleGrading ({RADIAL_GRADE} 1 1)"
        )
        blocks.append(
            f"    hex ({mk} {ok} {okn} {mkn} {mk+sz} {ok+sz} {okn+sz} {mkn+sz}) "
            f"({N_OUT} {N_C} 1) simpleGrading ({OUT_GRADE} 1 1)"
        )
    for z, zc in ((0, Z0), (1, Z1)):
        for k in range(8):
            kn = (k + 1) % 8
            edges.append(f"    arc {g('i',k,z)} {g('i',kn,z)} origin (0 0 {zc:.3f})")
            edges.append(f"    arc {g('m',k,z)} {g('m',kn,z)} origin (0 0 {zc:.3f})")

    def face(a, b):
        return f"        ({a} {b} {b+sz} {a+sz})"

    cyl = [face(g("i", k, 0), g("i", (k + 1) % 8, 0)) for k in range(8)]

    def obox(k):
        return face(g("o", k, 0), g("o", (k + 1) % 8, 0))

    inlet = [obox(3), obox(4)]
    outlet = [obox(7), obox(0)]
    top = [obox(1), obox(2)]
    bottom = [obox(5), obox(6)]
    nl = "\n"
    return f"""FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
scale 1;
vertices
(
{nl.join(vlines)}
);
blocks
(
{nl.join(blocks)}
);
edges
(
{nl.join(edges)}
);
boundary
(
    cylinder {{ type wall; faces ( {nl}{nl.join(cyl)} ); }}
    inlet {{ type patch; faces ( {nl.join(inlet)} ); }}
    outlet {{ type patch; faces ( {nl.join(outlet)} ); }}
    top {{ type symmetryPlane; faces ( {nl.join(top)} ); }}
    bottom {{ type symmetryPlane; faces ( {nl.join(bottom)} ); }}
);
defaultPatch {{ name frontAndBack; type empty; }}
mergePatchPairs ();
"""


def _fh(cls, obj):
    return f"FoamFile {{ version 2.0; format ascii; class {cls}; object {obj}; }}"


def write_case(case_dir: Path, reynolds: float, control: dict | None,
               *, end_time: float, delta_t: float, field_write_interval: int = 0) -> None:
    """Write a complete OpenFOAM case (fixed mesh + per-case nu / plate).

    ``field_write_interval`` > 0 writes U/p/vorticity every N steps (for
    rendering); 0 (default, for grading) writes only the final step.
    """
    case_dir = Path(case_dir)
    for sub in ("system", "constant", "0"):
        (case_dir / sub).mkdir(parents=True, exist_ok=True)
    nu = U0 * D / reynolds
    plate = has_plate(control)

    (case_dir / "system/blockMeshDict").write_text(_block_mesh_dict())
    (case_dir / "constant/transportProperties").write_text(
        _fh("dictionary", "transportProperties")
        + f"\ntransportModel Newtonian;\nnu {nu:.10f};\n"
    )
    (case_dir / "constant/turbulenceProperties").write_text(
        _fh("dictionary", "turbulenceProperties") + "\nsimulationType laminar;\n"
    )
    (case_dir / "0/U").write_text(
        _fh("volVectorField", "U") + f"""
dimensions [0 1 -1 0 0 0 0];
internalField uniform ({U0} {IC_KICK * U0} 0);
boundaryField
{{
    inlet {{ type fixedValue; value uniform ({U0} 0 0); }}
    outlet {{ type inletOutlet; inletValue uniform (0 0 0); value uniform ({U0} 0 0); }}
    cylinder {{ type noSlip; }}
    top {{ type symmetryPlane; }}
    bottom {{ type symmetryPlane; }}
    frontAndBack {{ type empty; }}
}}
"""
    )
    (case_dir / "0/p").write_text(
        _fh("volScalarField", "p") + """
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    cylinder { type zeroGradient; }
    top { type symmetryPlane; }
    bottom { type symmetryPlane; }
    frontAndBack { type empty; }
}
"""
    )
    force_patches = "(cylinder plate)" if plate else "(cylinder)"
    write_interval = field_write_interval if field_write_interval > 0 else 1000000
    vorticity_fo = ""
    if field_write_interval > 0:
        vorticity_fo = f"""
    vorticity
    {{
        type vorticity; libs (fieldFunctionObjects);
        writeControl timeStep; writeInterval {field_write_interval};
    }}"""
    (case_dir / "system/controlDict").write_text(
        _fh("dictionary", "controlDict") + f"""
application pimpleFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {end_time};
deltaT {delta_t}; writeControl timeStep; writeInterval {write_interval};
purgeWrite 0; writeFormat ascii; writePrecision 8; timeFormat general;
timePrecision 8; runTimeModifiable false; adjustTimeStep no;
functions
{{
    forceCoeffs
    {{
        type forceCoeffs; libs (forces); writeControl timeStep; writeInterval 1;
        patches {force_patches};
        rho rhoInf; rhoInf 1; liftDir (0 1 0); dragDir (1 0 0);
        CofR (0 0 0); pitchAxis (0 0 1);
        magUInf {U0}; lRef {D}; Aref {D};
    }}{vorticity_fo}
}}
"""
    )
    (case_dir / "system/fvSchemes").write_text(
        _fh("dictionary", "fvSchemes") + """
ddtSchemes { default backward; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) Gauss linearUpwind grad(U);
             div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""
    )
    (case_dir / "system/fvSolution").write_text(
        _fh("dictionary", "fvSolution") + """
solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.01; }
    pFinal { $p; relTol 0; }
    "(U|UFinal)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0; }
}
PIMPLE { nOuterCorrectors 1; nCorrectors 2; nNonOrthogonalCorrectors 1; }
"""
    )
    if plate:
        gap = float(control["plate_gap"])
        length = float(control["plate_length"])
        x0 = R + gap * D
        x1 = x0 + length * D
        (case_dir / "system/topoSetDict").write_text(
            _fh("dictionary", "topoSetDict") + f"""
actions
(
    {{ name plateFaces; type faceSet; action new;
      source boxToFace; box ({x0:.6f} -1e-4 -1) ({x1:.6f} 1e-4 1); }}
    {{ name plateZone; type faceZoneSet; action new;
      source setToFaceZone; faceSet plateFaces; }}
);
"""
        )
        (case_dir / "system/createBafflesDict").write_text(
            _fh("dictionary", "createBafflesDict") + """
internalFacesOnly true;
fields true;
baffles
{
    plate
    {
        type faceZone; zoneName plateZone;
        patches { master { name plate; type wall; } slave { name plate; type wall; } }
    }
}
"""
        )


def _run(case_dir: Path, cmd: str, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-lc", f"{OF_ENV}; cd {shlex.quote(str(case_dir))}; {cmd}"],
        capture_output=True, text=True, timeout=timeout,
    )


def build_and_run(case_dir: Path, reynolds: float, control: dict | None, *,
                  end_time: float, delta_t: float, window_frac: float = 0.4,
                  timeout: int = 480, field_write_interval: int = 0) -> CaseResult:
    """Generate the case, run it serially, and return shedding metrics."""
    write_case(case_dir, reynolds, control, end_time=end_time, delta_t=delta_t,
               field_write_interval=field_write_interval)
    plate = has_plate(control)
    r = _run(case_dir, "blockMesh", 120)
    if r.returncode != 0:
        return CaseResult(False, reason=f"blockMesh failed: {r.stderr[-300:]}")
    if plate:
        r = _run(case_dir, "topoSet", 120)
        if r.returncode != 0:
            return CaseResult(False, reason=f"topoSet failed: {r.stderr[-300:]}")
        # baffle insertion must actually find centreline faces
        if "plateFaces now size 0" in (r.stdout + r.stderr):
            return CaseResult(False, reason="plate placed off the wake centreline (0 faces)")
        r = _run(case_dir, "createBaffles -overwrite", 120)
        if r.returncode != 0:
            return CaseResult(False, reason=f"createBaffles failed: {r.stderr[-300:]}")
        _run(case_dir, 'foamDictionary -entry boundaryField/plate -set "{type noSlip;}" 0/U', 60)
        _run(case_dir, 'foamDictionary -entry boundaryField/plate -set "{type zeroGradient;}" 0/p', 60)
    r = _run(case_dir, "pimpleFoam", timeout)
    blob = r.stdout + r.stderr
    if r.returncode != 0:
        return CaseResult(False, reason=f"pimpleFoam failed: {blob[-300:]}")
    # Word-boundary match so we don't trip on tokens like "rhoInf" in the
    # forceCoeffs banner; analyze() also rejects non-finite force data.
    if "FOAM FATAL" in blob or re.search(r"\b(nan|inf|-nan)\b", blob.lower()):
        return CaseResult(False, reason="solver instability (NaN/inf/fatal)")
    dat = case_dir / "postProcessing/forceCoeffs/0/coefficient.dat"
    if not dat.exists():
        return CaseResult(False, reason="no force coefficient output produced")
    return analyze(dat, window_frac=window_frac)


def analyze(dat: Path, *, window_frac: float = 0.4) -> CaseResult:
    """Parse coefficient.dat -> mean Cd, RMS Cl, amplitude, Strouhal."""
    import numpy as np

    t, cd, cl = [], [], []
    columns: dict[str, int] = {}
    for line in Path(dat).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            header = line.lstrip("#").strip().split()
            if header and header[0] == "Time":
                columns = {name: i for i, name in enumerate(header)}
            continue
        p = line.split()
        try:
            # OpenFOAM forceCoeffs column order can differ between versions and
            # configs. Prefer the header names; fall back to the v2412 default:
            # Time Cm Cd Cl Cl(f) Cl(r) ...
            time_i = columns.get("Time", 0)
            cd_i = columns.get("Cd", 2)
            cl_i = columns.get("Cl", 3)
            t.append(float(p[time_i]))
            cd.append(float(p[cd_i]))
            cl.append(float(p[cl_i]))
        except (IndexError, ValueError):
            continue
    if len(t) < 50:
        return CaseResult(False, reason="force record too short")
    t = np.asarray(t); cd = np.asarray(cd); cl = np.asarray(cl)
    if not (np.isfinite(cd).all() and np.isfinite(cl).all()):
        return CaseResult(False, reason="non-finite force coefficients")
    i0 = int((1.0 - window_frac) * len(t))
    tw, clw, cdw = t[i0:], cl[i0:], cd[i0:]
    rms_cl = float(np.sqrt(np.mean((clw - clw.mean()) ** 2)))
    amp_cl = float((clw.max() - clw.min()) / 2.0)
    mean_cd = float(cdw.mean())
    dt = float(np.median(np.diff(tw)))
    sig = clw - clw.mean()
    freqs = np.fft.rfftfreq(len(sig), d=dt)
    power = np.abs(np.fft.rfft(sig)) ** 2
    st = float(freqs[1 + int(np.argmax(power[1:]))] * D / U0) if len(power) > 2 else 0.0
    return CaseResult(True, mean_cd=mean_cd, rms_cl=rms_cl, amp_cl=amp_cl,
                      strouhal=st, n_window=len(tw))
