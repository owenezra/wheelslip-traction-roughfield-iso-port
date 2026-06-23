# Public task data

`env/practice_env.py` is the complete public MuJoCo training simulator. It generates probe observations and blind-phase outcomes at runtime; no static dataset is required.

The public practice generator intentionally covers only one operating regime. Authors and agents may use the explicit episode factory to domain-randomize over the simulator's disclosed physical support.
