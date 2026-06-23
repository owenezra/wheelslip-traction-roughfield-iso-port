"""Render reviewer artifacts for the OpenSees retrofit oracle."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from opensees_frame import BAY_WIDTH, LEVEL_NODES, STORY_HEIGHT, build_frame, evaluate_design


WIDTH = 1280
HEIGHT = 720


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((Path(__file__).parent / "data" / "hidden_cases.json").read_text())
    design = json.loads(args.design.read_text())

    try:
        _render_opsvis_pngs(design, config, args.out_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"opsvis render skipped: {type(exc).__name__}: {exc}")

    _render_mp4(design, args.out_dir / "rendering.mp4")


def _render_opsvis_pngs(design: dict[str, Any], config: dict[str, Any], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import opsvis as opsv

    ops = build_frame(design, config, config["cases"][0])
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    opsv.plot_model()
    plt.title("OpenSees frame with oracle retrofit")
    plt.tight_layout()
    fig.savefig(out_dir / "review_frame.png")
    plt.close(fig)
    ops.wipe()

    baseline = evaluate_design({"devices": []}, config)
    oracle = evaluate_design(design, config)
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    ax.bar(
        ["Baseline", "Oracle"],
        [baseline["pushover_peak_base_shear"], oracle["pushover_peak_base_shear"]],
        color=["#777777", "#1f77b4"],
    )
    ax.set_ylabel("Peak base shear (kip)")
    ax.set_title("Pushover base shear comparison")
    ax.grid(axis="y", linestyle=":")
    fig.tight_layout()
    fig.savefig(out_dir / "review_pushover.png")
    plt.close(fig)


def _render_mp4(design: dict[str, Any], output: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to render /tmp/output/rendering.mp4")

    with tempfile.TemporaryDirectory() as td:
        frame_dir = Path(td)
        for idx in range(48):
            phase = idx / 47.0
            _write_ppm_frame(frame_dir / f"frame_{idx:03d}.ppm", design, phase)

        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                "12",
                "-i",
                str(frame_dir / "frame_%03d.ppm"),
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "format=yuv420p",
                str(output),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _write_ppm_frame(path: Path, design: dict[str, Any], phase: float) -> None:
    pixels = bytearray([255, 255, 255] * WIDTH * HEIGHT)
    _draw_grid(pixels)

    amp = math.sin(phase * math.pi) * 34.0
    _draw_frame(pixels, x_offset=0.0, color=(150, 150, 150), thickness=3)
    _draw_frame(pixels, x_offset=amp, color=(30, 90, 170), thickness=5)
    _draw_devices(pixels, design, x_offset=amp)

    with path.open("wb") as handle:
        handle.write(f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii"))
        handle.write(pixels)


def _draw_grid(pixels: bytearray) -> None:
    for y in range(90, HEIGHT - 80, 60):
        _line(pixels, 80, y, WIDTH - 80, y, (235, 235, 235), 1)
    for x in range(110, WIDTH - 80, 80):
        _line(pixels, x, 90, x, HEIGHT - 80, (245, 245, 245), 1)


def _draw_frame(
    pixels: bytearray, *, x_offset: float, color: tuple[int, int, int], thickness: int
) -> None:
    coords = _screen_coords(x_offset)
    for level in range(4):
        nodes = LEVEL_NODES[level]
        for left, right in zip(nodes[:-1], nodes[1:]):
            _line(pixels, *coords[left], *coords[right], color, thickness)
    for col in range(3):
        for level in range(3):
            lower = LEVEL_NODES[level][col]
            upper = LEVEL_NODES[level + 1][col]
            _line(pixels, *coords[lower], *coords[upper], color, thickness)


def _draw_devices(pixels: bytearray, design: dict[str, Any], *, x_offset: float) -> None:
    coords = _screen_coords(x_offset)
    for device in design.get("devices", []):
        story = int(device["story"])
        bay = int(device["bay"])
        lower = LEVEL_NODES[story - 1][bay - 1]
        upper = LEVEL_NODES[story][bay]
        color = (210, 70, 45) if device["type"] == "steel_brace" else (40, 150, 90)
        _line(pixels, *coords[lower], *coords[upper], color, 6)


def _screen_coords(x_offset: float) -> dict[int, tuple[int, int]]:
    left = 310
    bottom = 610
    sx = 2.2
    sy = 1.38
    coords = {}
    for level, nodes in LEVEL_NODES.items():
        y = bottom - int(level * STORY_HEIGHT * sy)
        story_offset = int(x_offset * level / 3.0)
        for idx, node in enumerate(nodes):
            x = left + int(idx * BAY_WIDTH * sx) + story_offset
            coords[node] = (x, y)
    return coords


def _line(
    pixels: bytearray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        _rect(
            pixels,
            x - thickness // 2,
            y - thickness // 2,
            x + thickness // 2,
            y + thickness // 2,
            color,
        )
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def _rect(
    pixels: bytearray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    x0 = max(0, min(WIDTH - 1, x0))
    x1 = max(0, min(WIDTH - 1, x1))
    y0 = max(0, min(HEIGHT - 1, y0))
    y1 = max(0, min(HEIGHT - 1, y1))
    for y in range(min(y0, y1), max(y0, y1) + 1):
        for x in range(min(x0, x1), max(x0, x1) + 1):
            pos = (y * WIDTH + x) * 3
            pixels[pos : pos + 3] = bytes(color)


if __name__ == "__main__":
    main()
