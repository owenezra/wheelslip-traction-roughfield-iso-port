"""Reviewer-render configuration for the vortex-suppression task.

Tunes the deterministic side-by-side vorticity animation (bare cylinder vs the
oracle's suppressed wake). Output must be exactly 1280x720 h264.
"""

RENDER = {
    "reynolds": 100.0,
    "end_time": 120.0,
    "delta_t": 0.01,
    "field_write_interval": 25,   # write fields every 25 steps (=0.25 s)
    "window_start": 2.0,          # animate from just after the initial kick: both
                                  # wakes start perturbed, then the top develops
                                  # the street while the plate damps the bottom
    "fps": 20,
    "oracle_control": {"plate_gap": 0.5, "plate_length": 2.0},
    "vmin": -3.0,
    "vmax": 3.0,
    "xlim": [-3.0, 14.0],
    "ylim": [-4.0, 4.0],
    "dpi": 100,
    "figsize": [12.8, 7.2],       # x dpi=100 -> 1280x720
}
