"""Weak baseline: fixed-friction planning with no system identification.

The slope simulator, quality objective, heuristic initialization, and restarted
CEM search match the reference policy. The only friction estimate is the
constant 0.90, so the probe observations are ignored.
"""
from __future__ import annotations

import os

# MuJoCo/NumPy execute on the host CPU. Keep native math libraries
# single-threaded for deterministic grading and to avoid oversubscription.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np
import mujoco

# ---------- timing / interface constants (match the practice environment) ----------
DT = 0.002
CTRL_HZ = 50
NSUB = int(round(1.0 / (CTRL_HZ * DT)))
T1 = 6.0
N1 = int(T1 * CTRL_HZ)
T2 = 2.7
N2 = int(T2 * CTRL_HZ)
N_KNOTS = 14

R_W = 0.12
WHEEL_DX = 0.28
WHEEL_DZ = -0.10
G_GAIN = 36.0
B_DAMP = 0.015
U_PROBE = 0.55

MU_PHYSICAL = (0.30, 1.50)   # supported friction range (predictions clipped here)

# ---------- quality metric constants (disclosed; used as the planning objective) ----------
SETTLE_X = 0.30
SETTLE_V = 0.30
P0 = 0.50
V0 = 1.00
C_LO, C_HI = 1.10, 2.10
W_POS, W_VEL, W_TIME = 0.25, 0.05, 0.70
MISS_CREDIT = 0.25

XML_SLOPE = f"""
<mujoco model="ws_slope">
  <option timestep="{DT}" gravity="0 0 -9.81" integrator="implicitfast"/>
  <worldbody>
    <geom name="ground" type="plane" size="80 2 0.1" pos="0 0 0" friction="1.0 0.005 0.0001"/>
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
  </worldbody>
  <contact>
    <pair name="pf" geom1="gwf" geom2="ground" friction="1.0 1.0 0.005 0.0001 0.0001"/>
    <pair name="pr" geom1="gwr" geom2="ground" friction="1.0 1.0 0.005 0.0001 0.0001"/>
  </contact>
  <actuator>
    <general name="mf" joint="jwf" gaintype="fixed" gainprm="{G_GAIN} 0 0" ctrlrange="-1 1"/>
    <general name="mr" joint="jwr" gaintype="fixed" gainprm="{G_GAIN} 0 0" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


class _SlopeWorld:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_string(XML_SLOPE)
        self.d = mujoco.MjData(self.m)
        wfb = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "wf")
        self.I_W = float(self.m.body_inertia[wfb][1])
        self.M_TOT = float(self.m.body_mass.sum())


_SW = None


def _slope_world() -> _SlopeWorld:
    global _SW
    if _SW is None:
        _SW = _SlopeWorld()
    return _SW


# ---------- probe-window feature extraction ----------

def _om(phi):
    return np.gradient(phi) * CTRL_HZ


def force_observer(obs: dict):
    """Per-wheel transmitted tangential force estimate from the encoders and
    the disclosed constants (G_GAIN, B_DAMP, wheel inertia):
    f_t = (G u - I_w domega - b omega) / R."""
    I_W = _slope_world().I_W
    om_f, om_r = _om(obs["phi_f"]), _om(obs["phi_r"])
    dom_f, dom_r = _om(om_f), _om(om_r)
    u = obs["u"]
    ff = (G_GAIN * u - I_W * dom_f - B_DAMP * om_f) / R_W
    fr = (G_GAIN * u - I_W * dom_r - B_DAMP * om_r) / R_W
    return ff, fr


def window_features(obs: dict) -> np.ndarray:
    """Engineered probe-window features (102-dim)."""
    u = obs["u"]
    om_f, om_r = _om(obs["phi_f"]), _om(obs["phi_r"])
    ax, az = obs["ax"], obs["az"]
    pit, dpit = obs["pitch"], obs["dpitch"]
    ff, fr = force_observer(obs)
    t = np.arange(N1) / CTRL_HZ
    f = []

    def stats(s):
        f.extend([s.mean(), s.std(), *np.quantile(s, [0.1, 0.5, 0.9])])

    for s in (om_f, om_r, ax, az, pit, dpit, ff, fr):
        stats(s)
    # force-observer plateau statistics under hard drive/brake
    hard_p = u > 0.5 * U_PROBE
    hard_n = u < -0.5 * U_PROBE
    for w in (ff, fr):
        for msk in (hard_p, hard_n):
            if msk.sum() > 8:
                f.extend([*np.quantile(np.abs(w[msk]), [0.5, 0.75, 0.9]),
                          float(np.abs(w[msk]).mean())])
            else:
                f.extend([0.0, 0.0, 0.0, 0.0])
    # free-spin / unloading fraction proxies at multiple force thresholds
    for thr in (10.0, 25.0, 45.0, 70.0, 100.0):
        f.append(float((np.abs(ff[hard_p | hard_n]) < thr).mean()
                       if (hard_p | hard_n).sum() else 0.0))
        f.append(float((np.abs(fr[hard_p | hard_n]) < thr).mean()
                       if (hard_p | hard_n).sum() else 0.0))
    # encoder spin extremity
    for w in (om_f, om_r):
        f.extend([float(np.abs(w).max()), *np.quantile(w, [0.02, 0.98])])
    # command-conditioned accelerometer statistics
    du = np.gradient(u) * CTRL_HZ
    for msk in (u > 0.6 * U_PROBE, u < -0.6 * U_PROBE,
                np.abs(u) < 0.08, (du > 0.15) & (u > 0.1),
                (u > 0.3 * U_PROBE) & (u <= 0.6 * U_PROBE),
                (u < -0.3 * U_PROBE) & (u >= -0.6 * U_PROBE)):
        if msk.sum() > 5:
            f.extend([float(ax[msk].mean()), float(ax[msk].std())])
        else:
            f.extend([0.0, 0.0])
    # terrain-intensity context: spectral band powers of az and pitch rate
    F = np.abs(np.fft.rfft(az - az.mean())) ** 2
    frq = np.fft.rfftfreq(N1, 1 / CTRL_HZ)
    for lo, hi in ((0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 25.0)):
        f.append(float(np.log1p(F[(frq >= lo) & (frq < hi)].sum())))
    Fp = np.abs(np.fft.rfft(dpit - dpit.mean())) ** 2
    for lo, hi in ((0.0, 3.0), (3.0, 10.0), (10.0, 25.0)):
        f.append(float(np.log1p(Fp[(frq >= lo) & (frq < hi)].sum())))
    Ff = np.abs(np.fft.rfft(ff - ff.mean())) ** 2
    for lo, hi in ((0.0, 3.0), (3.0, 10.0), (10.0, 25.0)):
        f.append(float(np.log1p(Ff[(frq >= lo) & (frq < hi)].sum())))
    # coast-tail decay slopes
    msk = (t >= 5.6) & (t < 6.0)
    for w in (om_f, om_r):
        ww = w[msk]
        f.append(float(np.polyfit(t[msk], ww, 1)[0]) if ww.size > 5 else 0.0)
    # lagged command->response correlations
    for lag in (0, 1, 2):
        f.append(float(np.corrcoef(u[:N1 - lag], om_f[lag:])[0, 1]))
        f.append(float(np.corrcoef(u[:N1 - lag], ff[lag:])[0, 1]))
    return np.nan_to_num(np.asarray(f, dtype=np.float64))


# ---------- blind-phase planning (simulator + objective + CEM) ----------

def knots_to_u(knots: np.ndarray) -> np.ndarray:
    reps = int(np.ceil(N2 / len(knots)))
    return np.clip(np.repeat(knots, reps)[:N2], -1.0, 1.0)


def rollout_blind(mu: float, alpha_deg: float, knots: np.ndarray):
    W = _slope_world()
    m, d = W.m, W.d
    m.pair_friction[:, 0] = float(mu)
    m.pair_friction[:, 1] = float(mu)
    a = np.deg2rad(alpha_deg)
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


def t_opt(mu: float, alpha_deg: float, s_star: float) -> float:
    a = np.deg2rad(alpha_deg)
    g = 9.81
    a1 = max(mu * g * np.cos(a) - g * np.sin(a), 0.2)
    a2 = mu * g * np.cos(a) + g * np.sin(a)
    return float(np.sqrt(2.0 * s_star * (1.0 / a1 + 1.0 / a2)))


def score_blind(xs, vs, s_star, alpha, mu) -> float:
    """The disclosed stop-and-settle quality metric (planning objective)."""
    inside = (np.abs(xs - s_star) < SETTLE_X) & (np.abs(vs) < SETTLE_V)
    if inside[-1]:
        idx_out = np.where(~inside)[0]
        t_s = (idx_out[-1] + 1) / CTRL_HZ if idx_out.size else 0.0
        settled = True
    else:
        t_s = T2
        settled = False
    x_T, v_T = float(xs[-1]), float(vs[-1])
    q_pos = max(0.0, 1.0 - abs(x_T - s_star) / P0)
    if not settled:
        return float(MISS_CREDIT * q_pos * max(0.0, 1.0 - abs(v_T) / 1.5))
    q_vel = max(0.0, 1.0 - abs(v_T) / V0)
    q_time = float(np.clip((C_HI - t_s / t_opt(mu, alpha, s_star))
                           / (C_HI - C_LO), 0.0, 1.0))
    return float(W_POS * q_pos + W_VEL * q_vel + W_TIME * q_time)


def heuristic_knots(mu_hat: float, alpha: float, s_star: float) -> np.ndarray:
    """Physics-informed initial program: bang-bang at the mu_hat traction
    limit with a spin-avoiding torque cap, then a gravity-holding tail."""
    g = 9.81
    a = np.deg2rad(alpha)
    W = _slope_world()
    m_tot = W.M_TOT
    n_half = 0.5 * m_tot * g * np.cos(a)
    u_cap = float(np.clip(mu_hat * n_half * R_W / G_GAIN * 1.05, 0.05, 1.0))
    a1 = max(mu_hat * g * np.cos(a) - g * np.sin(a), 0.2)
    a2 = mu_hat * g * np.cos(a) + g * np.sin(a)
    v_pk = np.sqrt(2.0 * s_star * a1 * a2 / (a1 + a2))
    t1, t2 = v_pk / a1, v_pk / a2
    u_hold = float(m_tot * g * np.sin(a) * R_W / (2 * G_GAIN))
    tt = (np.arange(N_KNOTS) + 0.5) * (T2 / N_KNOTS)
    k = np.where(tt < t1, u_cap, np.where(tt < t1 + t2, -0.9 * u_cap, u_hold))
    return k.astype(float)


def plan_cem(mu_hat, alpha, s_star, seed: int, pop=40, iters=8, elite=8,
             restarts=6) -> np.ndarray:
    """Heuristic-initialized restarted CEM over the N_KNOTS open-loop knots,
    simulated and scored at mu_hat."""
    rng = np.random.default_rng(seed)
    best_k, best_q = None, -1.0
    inits = [
        heuristic_knots(mu_hat, alpha, s_star),
        np.concatenate([np.full(5, 0.6), np.full(5, 0.1), np.full(4, -0.2)]),
        np.concatenate([np.full(7, 0.9), np.full(4, -0.5), np.full(3, 0.05)]),
    ]
    for r in range(restarts):
        mu0 = inits[r % len(inits)].copy()
        sd = np.full(N_KNOTS, 0.30 if r == 0 else 0.45)
        for _ in range(iters):
            cand = np.clip(rng.normal(mu0, sd, size=(pop, N_KNOTS)), -1, 1)
            if best_k is not None:
                cand[0] = np.clip(best_k, -1, 1)
            cand[1] = np.clip(inits[r % len(inits)], -1, 1)
            quals = np.empty(pop)
            for c in range(pop):
                xs, vs = rollout_blind(mu_hat, alpha, cand[c])
                quals[c] = score_blind(xs, vs, s_star, alpha, mu_hat)
            order = np.argsort(-quals)
            if quals[order[0]] > best_q:
                best_q, best_k = quals[order[0]], cand[order[0]].copy()
            el = cand[order[:elite]]
            mu0, sd = el.mean(0), el.std(0) + 0.02
    return best_k


# ---------- the policy ----------

MU_CONSTANT = 0.90


class ConstantMuPolicy:
    """Skip system identification and plan at one fixed friction estimate."""

    def reset(self):
        pass

    def act(self, observation: dict) -> np.ndarray:
        seed = int(observation["probe_seed"]) % (2**31 - 1)
        return plan_cem(
            MU_CONSTANT,
            float(observation["alpha"]),
            float(observation["s_star"]),
            seed=seed,
        )


def load_policy() -> ConstantMuPolicy:
    return ConstantMuPolicy()
