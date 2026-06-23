"""Practice environment for the rover traction-programming task.

A planar two-wheel rover (MuJoCo physics) drives on rough ground whose
wheel-ground friction coefficient mu is different in every episode. Each
episode has two phases:

  Phase 1 (probe, 6 s at 50 Hz):  the rover executes a scripted, per-episode
    randomized probe drive (launch / brake / coast / ramp / multisine /
    step-reverse) on a rough test strip. You observe wheel encoder angles,
    chassis pitch and pitch rate, and a body-mounted accelerometer that is
    corrupted by an unknown per-episode bias and scale factor plus noise.
    There is NO position/velocity odometry channel.

  Phase 2 (blind, 2.7 s at 50 Hz):  the rover is placed at the bottom of a
    smooth uphill slope (angle alpha, disclosed) and must stop AND settle at a
    target distance s_star up the slope (disclosed). The phase is open-loop:
    you commit ONE 14-knot torque program before it starts and receive no
    observations during execution. Episode quality rewards an accurate,
    settled, and fast stop (see ``score_summary`` below for the exact metric).

Everything about the vehicle and the simulation is disclosed: the full MuJoCo
XML (printed in this file), motor gain ``G_GAIN``, axle damping ``B_DAMP``,
geometry, the probe script generator (``probe_commands``, openly seeded by the
episode's ``probe_seed``), the rough-strip terrain generator, the sensor noise
levels, and the distributions of the per-episode nuisance parameters (terrain
amplitude band, accelerometer bias/scale ranges). The quantities hidden from
you within an episode are the episode's draws: the friction coefficient mu,
the terrain profile realization and amplitude, the accelerometer bias/scale,
and the sensor noise realization.

``make_train_episode(seed)`` is the practice episode generator: it draws mu
from the practice range ``BAND_TRAIN``. Evaluation episodes are produced by a
private generator whose parameter ranges are not disclosed and need not match
the practice range; the simulator clips/supports friction values in
``MU_PHYSICAL`` = [0.30, 1.50]. You may also build episodes with any explicit
parameter values you like through ``make_episode`` / the override arguments of
``rollout_window`` -- the simulator itself is fully open.

Submission interface (see the task prompt): your policy receives one
observation dict per episode,

    {"phi_f", "phi_r", "pitch", "dpitch", "ax", "az", "u"}  (each (300,) float)
    + "alpha" (slope, degrees), "s_star" (target, m), "probe_seed" (int)

and returns the 14-knot blind-phase torque program (values clipped to
[-1, 1]; each knot is held for ~0.193 s, see ``knots_to_u``).

Usage example:

    import numpy as np
    from practice_env import (make_train_episode, rollout_window,
                              rollout_blind, score_blind, N_KNOTS)
    ep = make_train_episode(0)
    obs = rollout_window(ep)["obs"]              # phase-1 observations
    program = np.zeros(N_KNOTS)                  # your program here
    xs, vs = rollout_blind(ep["mu"], ep["alpha"], program)
    q = score_blind(xs, vs, ep["s_star"], ep["alpha"], ep["mu"])
"""
from __future__ import annotations

import numpy as np
import mujoco

# ---------- timing ----------
DT = 0.002
CTRL_HZ = 50
NSUB = int(round(1.0 / (CTRL_HZ * DT)))      # 10 substeps per control tick
T_SETTLE = 0.4
N_SETTLE = int(T_SETTLE / DT)
T1 = 6.0
N1 = int(T1 * CTRL_HZ)                       # 300 probe-phase ticks
T2 = 2.7
N2 = int(T2 * CTRL_HZ)                       # 135 blind-phase ticks
N_KNOTS = 14

# ---------- geometry / disclosed constants ----------
R_W = 0.12
WHEEL_DX = 0.28
WHEEL_DZ = -0.10
X0 = 4.0
HF_NCOL = 401
HF_SX = 20.0
HF_X0 = 10.0
AMP_MAX = 0.16        # XML z-extent; per-episode data scaled to the episode amplitude
HF_SMOOTH = 2.2
G_GAIN = 36.0         # DISCLOSED motor gain (Nm at u=1)
B_DAMP = 0.015        # DISCLOSED axle viscous damping

# ---------- per-episode nuisance distributions (distributions open, draws hidden) ----------
A_LO, A_HI = 0.07, 0.15          # terrain amplitude band (m)
ACC_BIAS_SD = 0.30               # accelerometer bias, N(0, ACC_BIAS_SD) per axis
ACC_SCALE_RANGE = (0.93, 1.07)   # accelerometer scale factor, uniform

# ---------- probe ----------
U_PROBE = 0.55
SLEW = 10.0

# ---------- observation noise (1-sigma) ----------
NOISE = {"phi": 0.005, "pitch": 0.004, "dpitch": 0.01, "acc": 0.25}

# ---------- friction ranges ----------
BAND_TRAIN = (0.95, 1.35)        # practice episode generator's mu range
MU_PHYSICAL = (0.30, 1.50)       # range supported by the simulator/episode builders

# ---------- disclosed per-episode task parameters ----------
ALPHA_RANGE = (8.0, 10.5)        # slope angle (degrees)
SSTAR_RANGE = (3.5, 4.4)         # stop target distance (m)

# ---------- episode quality metric (fully disclosed; see score_summary) ----------
SETTLE_X = 0.30       # settled iff |x - s_star| < SETTLE_X and |v| < SETTLE_V
SETTLE_V = 0.30
P0 = 0.50             # position-accuracy scale (m)
V0 = 1.00             # final-speed scale (m/s)
C_LO, C_HI = 1.10, 2.10   # q_time = clip((C_HI - t_s/t_opt)/(C_HI - C_LO), 0, 1)
W_POS, W_VEL, W_TIME = 0.25, 0.05, 0.70
MISS_CREDIT = 0.25    # unsettled episodes: velocity-gated partial position credit

_XML_COMMON = f"""
    <body name="chassis" pos="0 0 0">
      <joint name="jx" type="slide" axis="1 0 0"/>
      <joint name="jz" type="slide" axis="0 0 1"/>
      <joint name="jp" type="hinge" axis="0 1 0" limited="true" range="-0.6 0.6"/>
      <geom name="hull" type="box" size="0.35 0.18 0.06" density="800"/>
      <body name="wf" pos="{WHEEL_DX} 0 {WHEEL_DZ}">
        <joint name="jwf" type="hinge" axis="0 1 0"/>
        <geom name="gwf" type="sphere" size="{R_W}" density="600" friction="1.0 0.005 0.0001"/>
      </body>
      <body name="wr" pos="-{WHEEL_DX} 0 {WHEEL_DZ}">
        <joint name="jwr" type="hinge" axis="0 1 0"/>
        <geom name="gwr" type="sphere" size="{R_W}" density="600" friction="1.0 0.005 0.0001"/>
      </body>
    </body>
"""

_XML_ACT = f"""
  <actuator>
    <general name="mf" joint="jwf" gaintype="fixed" gainprm="{G_GAIN} 0 0" ctrlrange="-1 1"/>
    <general name="mr" joint="jwr" gaintype="fixed" gainprm="{G_GAIN} 0 0" ctrlrange="-1 1"/>
  </actuator>
"""

_XML_PAIRS = """
  <contact>
    <pair name="pf" geom1="gwf" geom2="ground" friction="1.0 1.0 0.005 0.0001 0.0001"/>
    <pair name="pr" geom1="gwr" geom2="ground" friction="1.0 1.0 0.005 0.0001 0.0001"/>
  </contact>
"""

XML_ROUGH = f"""
<mujoco model="ws_rough">
  <option timestep="{DT}" gravity="0 0 -9.81" integrator="implicitfast"/>
  <asset>
    <hfield name="rough" nrow="2" ncol="{HF_NCOL}" size="{HF_SX} 1.0 {AMP_MAX} 0.5"/>
  </asset>
  <worldbody>
    <geom name="ground" type="hfield" hfield="rough" pos="{HF_X0} 0 0" friction="1.0 0.005 0.0001"/>
    {_XML_COMMON}
  </worldbody>
  {_XML_PAIRS}
  {_XML_ACT}
</mujoco>
"""

XML_SLOPE = f"""
<mujoco model="ws_slope">
  <option timestep="{DT}" gravity="0 0 -9.81" integrator="implicitfast"/>
  <worldbody>
    <geom name="ground" type="plane" size="80 2 0.1" pos="0 0 0" friction="1.0 0.005 0.0001"/>
    {_XML_COMMON}
  </worldbody>
  {_XML_PAIRS}
  {_XML_ACT}
</mujoco>
"""


class Worlds:
    def __init__(self):
        self.mr = mujoco.MjModel.from_xml_string(XML_ROUGH)
        self.ms = mujoco.MjModel.from_xml_string(XML_SLOPE)
        self.dr = mujoco.MjData(self.mr)
        self.ds = mujoco.MjData(self.ms)
        self.hf_adr = self.mr.hfield_adr[0]
        self.hf_n = self.mr.hfield_nrow[0] * self.mr.hfield_ncol[0]
        # wheel rotational inertia about its hinge (derivable from the open XML)
        wfb = mujoco.mj_name2id(self.mr, mujoco.mjtObj.mjOBJ_BODY, "wf")
        self.I_W = float(self.mr.body_inertia[wfb][1])
        self.M_TOT = float(self.mr.body_mass.sum())

    def apply_mu(self, m, mu):
        m.pair_friction[:, 0] = mu
        m.pair_friction[:, 1] = mu

    def set_terrain(self, data01_scaled):
        self.mr.hfield_data[self.hf_adr:self.hf_adr + self.hf_n] = data01_scaled.ravel()


_W = None


def worlds() -> Worlds:
    global _W
    if _W is None:
        _W = Worlds()
    return _W


# ---------- episode factory ----------

def make_episode(seed: int, band, amp_band=None) -> dict:
    """Build an episode dict with all per-episode draws derived from ``seed``.

    ``band`` is the (lo, hi) range the friction coefficient is drawn from;
    ``amp_band`` optionally overrides the terrain amplitude band. Every field
    of the returned dict may also be overwritten manually -- the simulator is
    fully open for practice and experimentation.
    """
    rng = np.random.default_rng(seed)
    ab = amp_band if amp_band is not None else (A_LO, A_HI)
    return {
        "seed": seed,
        "mu": float(rng.uniform(*band)),
        "terrain_seed": int(rng.integers(0, 2**31 - 1)),
        "amp": float(rng.uniform(*ab)),
        "probe_seed": int(rng.integers(0, 2**31 - 1)),
        "noise_seed": int(rng.integers(0, 2**31 - 1)),
        "bx": float(rng.normal(0, ACC_BIAS_SD)),
        "bz": float(rng.normal(0, ACC_BIAS_SD)),
        "ksc": float(rng.uniform(*ACC_SCALE_RANGE)),
        "alpha": float(rng.uniform(*ALPHA_RANGE)),
        "s_star": float(rng.uniform(*SSTAR_RANGE)),
    }


def make_train_episode(seed: int) -> dict:
    """The practice episode generator (mu drawn from ``BAND_TRAIN``)."""
    return make_episode(seed, BAND_TRAIN)


def terrain_profile(seed: int) -> np.ndarray:
    """Seeded smooth random rough-strip profile in [0,1], shape (HF_NCOL,)."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(HF_NCOL)
    k = np.arange(-8, 9)
    ker = np.exp(-0.5 * (k / HF_SMOOTH) ** 2)
    ker /= ker.sum()
    sm = np.convolve(raw, ker, mode="same")
    lo, hi = sm.min(), sm.max()
    return (sm - lo) / max(hi - lo, 1e-9)


def terrain_height(prof01: np.ndarray, amp: float, x: float) -> float:
    xs = np.linspace(HF_X0 - HF_SX, HF_X0 + HF_SX, HF_NCOL)
    return float(np.interp(x, xs, prof01) * amp)


# ---------- probe (open script, randomized per episode by probe_seed) ----------

def probe_commands(probe_seed: int) -> np.ndarray:
    """The scripted probe drive for phase 1. The schedule (segment timings,
    amplitudes, ramp slope, multisine phases) is randomized per episode from
    the disclosed ``probe_seed``, so the script for any episode is exactly
    reproducible from its observation dict."""
    rng = np.random.default_rng(probe_seed)
    t = np.arange(N1) / CTRL_HZ
    u = np.zeros(N1)
    seg = lambda a, b: (t >= a) & (t < b)
    # jittered segment boundaries
    d_launch = rng.uniform(0.55, 1.0)
    d_brake = rng.uniform(0.35, 0.65)
    d_coast = rng.uniform(0.3, 0.6)
    d_ramp = rng.uniform(0.7, 1.1)
    d_rev = rng.uniform(0.35, 0.6)
    a_launch = rng.uniform(0.85, 1.0) * U_PROBE
    a_brake = rng.uniform(0.85, 1.0) * U_PROBE
    a_ramp = rng.uniform(0.85, 1.0) * U_PROBE
    a_rev = rng.uniform(0.5, 0.75) * U_PROBE
    e0 = 0.0
    u[seg(e0, e0 + d_launch)] = a_launch
    e0 += d_launch
    u[seg(e0, e0 + d_brake)] = -a_brake
    e0 += d_brake
    e0 += d_coast                                     # coast (u stays 0)
    m_ramp = seg(e0, e0 + d_ramp)
    u[m_ramp] = np.linspace(0.0, a_ramp, m_ramp.sum())
    e0 += d_ramp
    u[seg(e0, e0 + d_rev)] = -a_rev
    e0 += d_rev
    # multisine block until the final step/reverse pair
    t_tail = 6.0 - 1.0 - rng.uniform(0.0, 0.25)
    m1 = seg(e0, t_tail)
    fr = np.array([0.7, 1.6, 3.1]) * rng.uniform(0.85, 1.2)
    ph = rng.uniform(0, 2 * np.pi, fr.size)
    u[m1] = 0.45 * U_PROBE + 0.5 * U_PROBE * np.sum(
        np.sin(2 * np.pi * fr[None] * t[m1, None] + ph[None]), 1) / fr.size * 1.6
    d_step = rng.uniform(0.4, 0.55)
    u[seg(t_tail, t_tail + d_step)] = U_PROBE
    u[seg(t_tail + d_step, min(t_tail + d_step + 0.5, 5.55))] = -U_PROBE
    # final coast tail (>= 0.45 s)
    u[seg(5.55, 6.0)] = 0.0
    out = np.empty_like(u)
    cur = 0.0
    dmax = SLEW / CTRL_HZ
    for i, ui in enumerate(u):
        cur += np.clip(ui - cur, -dmax, dmax)
        out[i] = cur
    return np.clip(out, -1.0, 1.0)


# ---------- rollouts ----------

def rollout_window(ep: dict, noisy: bool = True, mu_override=None,
                   terrain: str = "true", terrain_seed=None, amp_override=None,
                   corrupt_override=None):
    """Phase-1 rollout on the rough strip.

    terrain: "true" (the episode's own profile), "flat", or "draw" (a fresh
    profile from ``terrain_seed`` with amplitude ``amp_override``).
    mu_override: simulate at a friction value of your choice.
    corrupt_override: dict(bx, bz, ksc) accelerometer corruption to apply
    instead of the episode's own draws.

    Returns a dict with the noise-free state traces ``q``/``v`` (use them for
    your own experiments only -- the grading rollout exposes just ``obs``) and,
    if ``noisy``, the observation dict ``obs`` with keys phi_f, phi_r, pitch,
    dpitch, ax, az, u (each shape (N1,)).
    """
    W = worlds()
    mu = float(mu_override) if mu_override is not None else ep["mu"]
    W.apply_mu(W.mr, mu)
    if terrain == "flat":
        prof, amp = np.zeros(HF_NCOL), 0.0
    elif terrain == "draw":
        prof = terrain_profile(int(terrain_seed))
        amp = float(amp_override) if amp_override is not None else ep["amp"]
    else:
        prof = terrain_profile(ep["terrain_seed"])
        amp = ep["amp"]
    scaled = np.clip(prof * (amp / AMP_MAX), 0.0, 1.0)
    W.set_terrain(np.stack([scaled, scaled]))
    m, d = W.mr, W.dr
    mujoco.mj_resetData(m, d)
    zf = terrain_height(prof, amp, X0 + WHEEL_DX)
    zr = terrain_height(prof, amp, X0 - WHEEL_DX)
    d.qpos[:] = [X0, max(zf, zr) + R_W - WHEEL_DZ + 0.005, 0.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)
    d.ctrl[:] = 0.0
    for _ in range(N_SETTLE):
        mujoco.mj_step(m, d)
    u = probe_commands(ep["probe_seed"])
    Q = np.empty((N1, 5))
    V = np.empty((N1, 5))
    for i in range(N1):
        d.ctrl[:] = u[i]
        for _ in range(NSUB):
            mujoco.mj_step(m, d)
        Q[i] = d.qpos
        V[i] = d.qvel
    out = {"q": Q, "v": V, "u": u}
    if noisy:
        rng = np.random.default_rng(ep["noise_seed"])
        cor = corrupt_override if corrupt_override is not None else ep
        ax = np.gradient(V[:, 0]) * CTRL_HZ
        az = np.gradient(V[:, 1]) * CTRL_HZ
        out["obs"] = {
            "phi_f": Q[:, 3] + rng.normal(0, NOISE["phi"], N1),
            "phi_r": Q[:, 4] + rng.normal(0, NOISE["phi"], N1),
            "pitch": Q[:, 2] + rng.normal(0, NOISE["pitch"], N1),
            "dpitch": V[:, 2] + rng.normal(0, NOISE["dpitch"], N1),
            "ax": cor["ksc"] * ax + cor["bx"] + rng.normal(0, NOISE["acc"], N1),
            "az": cor["ksc"] * az + cor["bz"] + rng.normal(0, NOISE["acc"], N1),
            "u": u,
        }
    return out


def knots_to_u(knots: np.ndarray) -> np.ndarray:
    """Expand a 14-knot program to the 135 blind-phase control ticks
    (zero-order hold; values clipped to [-1, 1])."""
    reps = int(np.ceil(N2 / len(knots)))
    return np.clip(np.repeat(knots, reps)[:N2], -1.0, 1.0)


def rollout_blind(mu: float, alpha_deg: float, knots: np.ndarray):
    """Phase-2 rollout: smooth uphill slope (tilted gravity), deterministic,
    no observations. Returns (positions, velocities) at the control rate."""
    W = worlds()
    W.apply_mu(W.ms, float(mu))
    a = np.deg2rad(alpha_deg)
    m, d = W.ms, W.ds
    m.opt.gravity[:] = [-9.81 * np.sin(a), 0.0, -9.81 * np.cos(a)]
    mujoco.mj_resetData(m, d)
    d.qpos[:] = [0.0, R_W - WHEEL_DZ + 0.001, 0.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)
    u = knots_to_u(np.asarray(knots, float))
    xs = np.empty(N2)
    vs = np.empty(N2)
    for i in range(N2):
        d.ctrl[:] = u[i]
        for _ in range(NSUB):
            mujoco.mj_step(m, d)
        xs[i] = d.qpos[0]
        vs[i] = d.qvel[0]
    return xs, vs


def blind_summary(xs: np.ndarray, vs: np.ndarray, s_star: float) -> dict:
    """Terminal state + settle time extracted from a blind-phase trajectory.
    The rover is "settled" at the end iff it is inside the settle window
    (|x - s_star| < SETTLE_X and |v| < SETTLE_V) at the final tick; the settle
    time t_s is the time it last entered that window."""
    inside = (np.abs(xs - s_star) < SETTLE_X) & (np.abs(vs) < SETTLE_V)
    if inside[-1]:
        idx_out = np.where(~inside)[0]
        t_s = (idx_out[-1] + 1) / CTRL_HZ if idx_out.size else 0.0
        settled = True
    else:
        t_s = T2
        settled = False
    return {"x_T": float(xs[-1]), "v_T": float(vs[-1]),
            "t_s": float(t_s), "settled": bool(settled)}


def t_opt(mu: float, alpha_deg: float, s_star: float) -> float:
    """Per-episode time scale used by the quality metric: the closed-form
    bang-bang (max accelerate, max brake) travel time achievable at friction
    ``mu`` on the slope. The grading metric evaluates this at the episode's
    true friction value."""
    a = np.deg2rad(alpha_deg)
    g = 9.81
    a1 = max(mu * g * np.cos(a) - g * np.sin(a), 0.2)
    a2 = mu * g * np.cos(a) + g * np.sin(a)
    return float(np.sqrt(2.0 * s_star * (1.0 / a1 + 1.0 / a2)))


def score_summary(s: dict, s_star: float, alpha: float, mu: float,
                  p0: float = P0) -> float:
    """Episode quality in [0, 1] from a blind-phase summary.

    Settled episodes:   q = W_POS*q_pos + W_VEL*q_vel + W_TIME*q_time, with
      q_pos  = max(0, 1 - |x_T - s_star| / P0)
      q_vel  = max(0, 1 - |v_T| / V0)
      q_time = clip((C_HI - t_s / t_opt(mu, alpha, s_star)) / (C_HI - C_LO), 0, 1)
    Unsettled episodes: q = MISS_CREDIT * q_pos * max(0, 1 - |v_T| / 1.5)
    (a fast flythrough past the target scores zero)."""
    q_pos = max(0.0, 1.0 - abs(s["x_T"] - s_star) / p0)
    if not s["settled"]:
        return float(MISS_CREDIT * q_pos * max(0.0, 1.0 - abs(s["v_T"]) / 1.5))
    q_vel = max(0.0, 1.0 - abs(s["v_T"]) / V0)
    q_time = float(np.clip((C_HI - s["t_s"] / t_opt(mu, alpha, s_star))
                           / (C_HI - C_LO), 0.0, 1.0))
    return float(W_POS * q_pos + W_VEL * q_vel + W_TIME * q_time)


def score_blind(xs, vs, s_star, alpha, mu, p0: float = P0) -> float:
    return score_summary(blind_summary(xs, vs, s_star), s_star, alpha, mu, p0)
