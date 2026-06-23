# Baselines

Each subdirectory commits the exact `policy.py` artifact scored by the learnability gate.

- `do_nothing`: all-zero torque, expected at the raw floor.
- `full_throttle`: saturated positive torque, expected at the raw floor because it does not settle.
- `constant_mu`: uses the same physics planner as the reference but assumes one friction value and performs no system identification. Legacy evidence measured mean quality about 0.179, well below the public-only reference.
