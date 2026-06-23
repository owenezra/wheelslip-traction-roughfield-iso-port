# Program a blind uphill stop from a short traction probe

You are given a simulated planar two-wheel rover with torque-driven wheels, chassis pitch dynamics, and a body-mounted accelerometer. The wheel-ground friction coefficient changes from episode to episode and is never observed directly.

Each episode has two phases.

## Probe phase

The rover executes a scripted 6-second drive at 50 Hz over rough ground. The script contains launch, braking, coast, ramp, multisine, and step-reverse segments, with its schedule randomized from a disclosed per-episode seed.

Your policy receives:

- `phi_f`, `phi_r`, `pitch`, `dpitch`, `ax`, `az`, and `u`: NumPy arrays of shape `(300,)` containing the two wheel encoder angles, chassis pitch and pitch rate, accelerometer channels, and executed probe command;
- `alpha`: the blind-phase slope angle in degrees;
- `s_star`: the target stopping distance in metres;
- `probe_seed`: the integer seed used to generate the probe schedule.

The accelerometer has an unknown per-episode bias and scale factor plus noise. There is no position or velocity odometry.

## Blind phase

After the probe, the rover is placed at the bottom of a smooth uphill slope. No observations are available during this 2.7-second phase. Your policy must return one open-loop torque program containing exactly 14 finite values. Values are clipped to `[-1, 1]`; each knot is held for approximately 0.193 seconds.

Episode quality is in `[0, 1]` and rewards an accurate, settled, and fast stop. It combines final position accuracy, final speed, and time to settle relative to the physically achievable stop time at the episode's true friction. An unsettled fly-through receives little or no credit. The exact episode metric is implemented by the public simulator and is used unchanged for grading. The aggregate metric is mean episode quality over deterministic held-out episodes.

## Public simulator

The task provides:

- `/data/env/practice_env.py`

Read its module documentation and source. It contains the full MuJoCo model, all disclosed vehicle constants, the probe generator, public episode builders, probe and blind rollouts, and the exact quality metric. `make_train_episode(seed)` samples the public practice distribution. You may also generate episodes over explicit parameter ranges with `make_episode(...)`. There are no pre-generated training rows.

Evaluation uses the same observation/action contract and simulator dynamics with held-out episode parameters and fixed seeds. The held-out friction distribution is not disclosed and need not match the public practice generator.

## Submission

Write:

- `/tmp/output/policy.py`

The module must define:

```python
def load_policy():
    ...
```

`load_policy()` must return an object with:

```python
def act(self, observation):
    ...
```

`act(observation)` is called once per episode and must return an array-like object of shape `(14,)` containing finite numeric values. You may also define `reset(self)`; when present, the grader calls it before every episode. The loaded policy object is otherwise reused across episodes, so behavior must be deterministic.

The policy must be self-contained at inference. It may read auxiliary files placed beside `policy.py` in `/tmp/output`, but it must not read `/data`, grader files, network resources, or files outside its own output directory.
