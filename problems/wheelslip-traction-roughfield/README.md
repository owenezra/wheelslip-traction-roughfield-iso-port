# Wheelslip traction on rough field - ISO ML task

This is the ISO `task_type = "ml"` port of legacy task #914. It preserves the planar-rover MuJoCo simulator, the friction system-identification problem, the 14-knot open-loop policy contract, and a deterministic held-out evaluator.

The agent trains or tunes against `/data/env/practice_env.py` and submits `/tmp/output/policy.py`. Grading uses the root-only private protocol and executes submitted code through `grading.PolicyWorker`.
