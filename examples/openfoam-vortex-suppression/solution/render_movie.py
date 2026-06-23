"""Render a deterministic side-by-side vorticity animation to rendering.mp4.

Top panel: the bare cylinder (full von Karman street). Bottom panel: the oracle
splitter plate (suppressed wake). Both run with the pinned mesh/dt; we animate a
fixed developed-wake window. Output is exactly 1280x720 h264 (browser-playable),
which the harness ground-truth render contract requires.

Runs inside the task image (OpenFOAM + matplotlib + ffmpeg all present).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scorer"))
import openfoam_case as ofc  # noqa: E402
from render_config import RENDER  # noqa: E402

OUT = Path(os.environ.get("LBT_OUTPUT_DIR", "/tmp/output"))
_NUM = re.compile(r"[-+0-9.eE]+")


def _read_vector_internal(path: Path) -> np.ndarray:
    """Parse an OpenFOAM ascii volVectorField internalField -> (N,3) array."""
    txt = path.read_text()
    m = re.search(r"internalField\s+nonuniform\s+List<vector>\s*\d*\s*\((.*?)\)\s*;",
                  txt, re.S)
    if not m:
        # uniform field fallback
        mu = re.search(r"internalField\s+uniform\s*\(([^)]*)\)", txt)
        if mu:
            v = [float(x) for x in _NUM.findall(mu.group(1))]
            return np.array([v])
        raise ValueError(f"could not parse vector field {path}")
    body = m.group(1)
    vals = [float(x) for x in _NUM.findall(body)]
    return np.array(vals).reshape(-1, 3)


def _time_dirs(case: Path, start: float) -> list[tuple[float, Path]]:
    out = []
    for d in case.iterdir():
        if not d.is_dir():
            continue
        try:
            t = float(d.name)
        except ValueError:
            continue
        if t >= start - 1e-9 and (d / "vorticity").exists():
            out.append((t, d))
    return sorted(out)


def _prepare_case(tmp: Path, control: dict | None) -> tuple[Path, np.ndarray, list]:
    case = tmp / ("oracle" if control else "baseline")
    case.mkdir(parents=True, exist_ok=True)
    res = ofc.build_and_run(
        case, RENDER["reynolds"], control,
        end_time=RENDER["end_time"], delta_t=RENDER["delta_t"],
        window_frac=0.4, timeout=900,
        field_write_interval=RENDER["field_write_interval"],
    )
    if not res.ok:
        raise RuntimeError(f"render solve failed ({'oracle' if control else 'baseline'}): {res.reason}")
    ofc._run(case, "postProcess -func writeCellCentres -time 0 >/dev/null 2>&1", 120)
    cc = case / "0" / "C"
    centres = _read_vector_internal(cc)
    frames = _time_dirs(case, RENDER["window_start"])
    return case, centres, frames


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _, c_base, f_base = _prepare_case(tmp, None)
        _, c_orc, f_orc = _prepare_case(tmp, RENDER["oracle_control"])
        n = min(len(f_base), len(f_orc))
        if n == 0:
            raise RuntimeError("no vorticity frames were written")

        tri_base = mtri.Triangulation(c_base[:, 0], c_base[:, 1])
        tri_orc = mtri.Triangulation(c_orc[:, 0], c_orc[:, 1])
        levels = np.linspace(RENDER["vmin"], RENDER["vmax"], 41)
        frames_dir = tmp / "frames"
        frames_dir.mkdir()

        for i in range(n):
            wz_b = _read_vector_internal(f_base[i][1] / "vorticity")[:, 2]
            wz_o = _read_vector_internal(f_orc[i][1] / "vorticity")[:, 2]
            fig, axes = plt.subplots(
                2, 1, figsize=RENDER["figsize"], dpi=RENDER["dpi"], constrained_layout=True
            )
            for ax, tri, wz, title in (
                (axes[0], tri_base, wz_b, "Bare cylinder - von Karman shedding"),
                (axes[1], tri_orc, wz_o, "With splitter plate - suppressed wake"),
            ):
                ax.tricontourf(tri, wz, levels=levels, cmap="RdBu_r", extend="both")
                circ = plt.Circle((0, 0), ofc.R, color="0.2", zorder=5)
                ax.add_patch(circ)
                ax.set_xlim(RENDER["xlim"]); ax.set_ylim(RENDER["ylim"])
                ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(title, fontsize=11)
            # draw the oracle plate
            g = RENDER["oracle_control"]["plate_gap"]
            length = RENDER["oracle_control"]["plate_length"]
            x0 = ofc.R + g * ofc.D
            axes[1].plot([x0, x0 + length * ofc.D], [0, 0], color="k", lw=2.5, zorder=6)
            fig.suptitle(f"Re={int(RENDER['reynolds'])}  t={f_base[i][0]:.1f}", fontsize=12)
            fig.savefig(frames_dir / f"f{i:04d}.png", dpi=RENDER["dpi"])
            plt.close(fig)

        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        cmd = [
            ffmpeg, "-y", "-framerate", str(RENDER["fps"]),
            "-i", str(frames_dir / "f%04d.png"),
            "-vf", "scale=1280:720:flags=bicubic,format=yuv420p",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            str(OUT / "rendering.mp4"),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {r.stderr[-400:]}")
    print(f"wrote {OUT / 'rendering.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
