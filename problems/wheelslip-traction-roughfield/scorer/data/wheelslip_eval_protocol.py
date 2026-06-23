"""GRADER-SIDE evaluation protocol for the rover traction-programming task.
PRIVATE: mounted at /mcp_server/data and NOT visible to the agent.

Defines the held-out evaluation episodes and the deterministic grading
rollout. This module is self-contained on the private mount (it carries its
own copy of the simulation core) and does not import any agent-visible module;
the simulation code, constants, and quality metric are identical to the
public practice environment (/data/env/practice_env.py) -- only the
episode generator differs:

  - the evaluation friction band ``BAND_EVAL`` (private; disjoint from the
    practice generator's range), and
  - the fixed private seed family ``EVAL_SEEDS``.

Protocol (identical to the measured evidence run used to calibrate the
scorer):

  - N_EVAL = 80 held-out episodes, ``make_eval_episode(seed)`` for seed in
    EVAL_SEEDS (1000..1079).
  - Phase 1: the scripted randomized probe drive is rolled out on the
    episode's rough strip; the policy receives the observation dict
    (phi_f, phi_r, pitch, dpitch, ax, az, u, alpha, s_star, probe_seed) and
    must return the 14-knot blind-phase program.
  - Phase 2: the program is executed open-loop on the episode's slope at the
    episode's TRUE friction; the episode quality is computed by the disclosed
    stop-and-settle metric evaluated at the true friction.
  - The aggregate metric is the mean quality over the N_EVAL episodes.

Everything is seeded; grading is deterministic for a fixed submission.
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
AMP_MAX = 0.16
HF_SMOOTH = 2.2
G_GAIN = 36.0
B_DAMP = 0.015

# ---------- per-episode nuisance distributions ----------
A_LO, A_HI = 0.07, 0.15
ACC_BIAS_SD = 0.30
ACC_SCALE_RANGE = (0.93, 1.07)

# ---------- probe ----------
U_PROBE = 0.55
SLEW = 10.0

# ---------- observation noise (1-sigma) ----------
NOISE = {"phi": 0.005, "pitch": 0.004, "dpitch": 0.01, "acc": 0.25}

# ---------- friction bands ----------
BAND_EVAL = (0.42, 0.72)         # PRIVATE held-out evaluation friction band

# ---------- disclosed per-episode task parameters ----------
ALPHA_RANGE = (8.0, 10.5)
SSTAR_RANGE = (3.5, 4.4)

# ---------- episode quality metric (identical to the public module) ----------
SETTLE_X = 0.30
SETTLE_V = 0.30
P0 = 0.50
V0 = 1.00
C_LO, C_HI = 1.10, 2.10
W_POS, W_VEL, W_TIME = 0.25, 0.05, 0.70
MISS_CREDIT = 0.25

# ---------- held-out grading seeds (PRIVATE) ----------
# 80 episodes: the original 40 (seeds 1000..1039) plus 40 more (1040..1079)
# generated through the SAME factory below (make_eval_episode), same private
# friction band BAND_EVAL and the same nuisance distributions -- the seed
# family simply continues the original scheme.
EVAL_SEED0 = 1000
N_EVAL = 80
EVAL_SEEDS = list(range(EVAL_SEED0, EVAL_SEED0 + N_EVAL))

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


class PolicyContractError(Exception):
    """Raised when the submitted policy violates the submission contract
    (act() raising, wrong program shape, non-finite values). The scorer
    converts this into a real 0.0 (a malformed submission), distinct from a
    grading-engine fault."""


class Worlds:
    def __init__(self):
        self.mr = mujoco.MjModel.from_xml_string(XML_ROUGH)
        self.ms = mujoco.MjModel.from_xml_string(XML_SLOPE)
        self.dr = mujoco.MjData(self.mr)
        self.ds = mujoco.MjData(self.ms)
        self.hf_adr = self.mr.hfield_adr[0]
        self.hf_n = self.mr.hfield_nrow[0] * self.mr.hfield_ncol[0]

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


# ---------- episode factory (PRIVATE band + seed family) ----------

def make_eval_episode(seed: int) -> dict:
    """Held-out evaluation episode. Draw order matches the public episode
    builder exactly; only the friction band differs."""
    rng = np.random.default_rng(seed)
    return {
        "seed": seed,
        "mu": float(rng.uniform(*BAND_EVAL)),
        "terrain_seed": int(rng.integers(0, 2**31 - 1)),
        "amp": float(rng.uniform(A_LO, A_HI)),
        "probe_seed": int(rng.integers(0, 2**31 - 1)),
        "noise_seed": int(rng.integers(0, 2**31 - 1)),
        "bx": float(rng.normal(0, ACC_BIAS_SD)),
        "bz": float(rng.normal(0, ACC_BIAS_SD)),
        "ksc": float(rng.uniform(*ACC_SCALE_RANGE)),
        "alpha": float(rng.uniform(*ALPHA_RANGE)),
        "s_star": float(rng.uniform(*SSTAR_RANGE)),
    }


def terrain_profile(seed: int) -> np.ndarray:
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


# ---------- probe ----------

def probe_commands(probe_seed: int) -> np.ndarray:
    rng = np.random.default_rng(probe_seed)
    t = np.arange(N1) / CTRL_HZ
    u = np.zeros(N1)
    seg = lambda a, b: (t >= a) & (t < b)
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
    e0 += d_coast
    m_ramp = seg(e0, e0 + d_ramp)
    u[m_ramp] = np.linspace(0.0, a_ramp, m_ramp.sum())
    e0 += d_ramp
    u[seg(e0, e0 + d_rev)] = -a_rev
    e0 += d_rev
    t_tail = 6.0 - 1.0 - rng.uniform(0.0, 0.25)
    m1 = seg(e0, t_tail)
    fr = np.array([0.7, 1.6, 3.1]) * rng.uniform(0.85, 1.2)
    ph = rng.uniform(0, 2 * np.pi, fr.size)
    u[m1] = 0.45 * U_PROBE + 0.5 * U_PROBE * np.sum(
        np.sin(2 * np.pi * fr[None] * t[m1, None] + ph[None]), 1) / fr.size * 1.6
    d_step = rng.uniform(0.4, 0.55)
    u[seg(t_tail, t_tail + d_step)] = U_PROBE
    u[seg(t_tail + d_step, min(t_tail + d_step + 0.5, 5.55))] = -U_PROBE
    u[seg(5.55, 6.0)] = 0.0
    out = np.empty_like(u)
    cur = 0.0
    dmax = SLEW / CTRL_HZ
    for i, ui in enumerate(u):
        cur += np.clip(ui - cur, -dmax, dmax)
        out[i] = cur
    return np.clip(out, -1.0, 1.0)


# ---------- rollouts ----------

def rollout_window(ep: dict) -> dict:
    """Phase-1 grading rollout on the episode's own rough strip; returns the
    observation dict the policy receives."""
    W = worlds()
    W.apply_mu(W.mr, ep["mu"])
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
    rng = np.random.default_rng(ep["noise_seed"])
    ax = np.gradient(V[:, 0]) * CTRL_HZ
    az = np.gradient(V[:, 1]) * CTRL_HZ
    return {
        "phi_f": Q[:, 3] + rng.normal(0, NOISE["phi"], N1),
        "phi_r": Q[:, 4] + rng.normal(0, NOISE["phi"], N1),
        "pitch": Q[:, 2] + rng.normal(0, NOISE["pitch"], N1),
        "dpitch": V[:, 2] + rng.normal(0, NOISE["dpitch"], N1),
        "ax": ep["ksc"] * ax + ep["bx"] + rng.normal(0, NOISE["acc"], N1),
        "az": ep["ksc"] * az + ep["bz"] + rng.normal(0, NOISE["acc"], N1),
        "u": u,
    }


def knots_to_u(knots: np.ndarray) -> np.ndarray:
    reps = int(np.ceil(N2 / len(knots)))
    return np.clip(np.repeat(knots, reps)[:N2], -1.0, 1.0)


def rollout_blind(mu: float, alpha_deg: float, knots: np.ndarray):
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


# ---------- quality metric ----------

def blind_summary(xs: np.ndarray, vs: np.ndarray, s_star: float) -> dict:
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
    a = np.deg2rad(alpha_deg)
    g = 9.81
    a1 = max(mu * g * np.cos(a) - g * np.sin(a), 0.2)
    a2 = mu * g * np.cos(a) + g * np.sin(a)
    return float(np.sqrt(2.0 * s_star * (1.0 / a1 + 1.0 / a2)))


def score_summary(s: dict, s_star: float, alpha: float, mu: float) -> float:
    q_pos = max(0.0, 1.0 - abs(s["x_T"] - s_star) / P0)
    if not s["settled"]:
        return float(MISS_CREDIT * q_pos * max(0.0, 1.0 - abs(s["v_T"]) / 1.5))
    q_vel = max(0.0, 1.0 - abs(s["v_T"]) / V0)
    q_time = float(np.clip((C_HI - s["t_s"] / t_opt(mu, alpha, s_star))
                           / (C_HI - C_LO), 0.0, 1.0))
    return float(W_POS * q_pos + W_VEL * q_vel + W_TIME * q_time)


# ---------- grading episode ----------

def run_episode(policy, seed: int) -> dict:
    """Deterministic single-episode grading: probe rollout -> policy.act ->
    blind rollout at the TRUE friction -> quality.

    Raises ``PolicyContractError`` on any submission-contract violation (the
    scorer maps that to 0.0).
    """
    ep = make_eval_episode(seed)
    obs = rollout_window(ep)
    observation = dict(obs)
    observation["alpha"] = ep["alpha"]
    observation["s_star"] = ep["s_star"]
    observation["probe_seed"] = ep["probe_seed"]

    try:
        raw_program = policy.act(observation)
        program = np.asarray(raw_program, dtype=np.float64).ravel()
    except Exception as e:  # surfaced to the scorer as a contract failure
        raise PolicyContractError(
            f"act() or action conversion failed at seed={seed}: "
            f"{type(e).__name__}: {e}"
        ) from e
    if program.shape != (N_KNOTS,):
        raise PolicyContractError(
            f"program shape {program.shape}, expected ({N_KNOTS},) at seed={seed}"
        )
    if not np.all(np.isfinite(program)):
        raise PolicyContractError(f"non-finite program values at seed={seed}")

    xs, vs = rollout_blind(ep["mu"], ep["alpha"], program)
    summary = blind_summary(xs, vs, ep["s_star"])
    quality = score_summary(summary, ep["s_star"], ep["alpha"], ep["mu"])
    return {
        "seed": int(seed),
        "quality": float(quality),
        "settled": bool(summary["settled"]),
        "x_T": summary["x_T"],
        "v_T": summary["v_T"],
        "t_s": summary["t_s"],
    }
