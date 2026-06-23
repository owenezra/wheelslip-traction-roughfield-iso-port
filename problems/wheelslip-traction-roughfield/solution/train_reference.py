"""Train the public-only reference friction identifier and stage the policy.

The labeled library is generated exclusively with the public simulator across
its disclosed physical friction support. MuJoCo and scikit-learn execute on the
host CPU even when the task is allocated a TPU; no fp64 operation is sent to
the TPU.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from policy import MU_PHYSICAL, window_features  # noqa: E402

PUBLIC_ENV_CANDIDATES = (
    Path(os.environ.get("TASK_DATA_DIR", "/data")) / "env",
    HERE.parent / "data" / "env",
)
for candidate in PUBLIC_ENV_CANDIDATES:
    if (candidate / "practice_env.py").is_file():
        sys.path.insert(0, str(candidate))
        break
else:
    raise FileNotFoundError("could not locate public /data/env/practice_env.py")

import practice_env  # noqa: E402

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/tmp/output"))
LIBRARY_SEEDS = range(40000, 41800)


def main() -> None:
    started = time.monotonic()
    features: list[np.ndarray] = []
    labels: list[float] = []

    for seed in LIBRARY_SEEDS:
        episode = practice_env.make_episode(seed, MU_PHYSICAL)
        observation = practice_env.rollout_window(episode)["obs"]
        features.append(window_features(observation))
        labels.append(float(episode["mu"]))

    x_train = np.nan_to_num(np.stack(features))
    y_train = np.asarray(labels)
    model = HistGradientBoostingRegressor(
        max_iter=400,
        random_state=0,
    ).fit(x_train, y_train)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "members": [model],
        "selected": "hgb_public_broadband",
        "n_train": len(y_train),
        "friction_support": tuple(float(x) for x in MU_PHYSICAL),
    }
    joblib.dump(payload, OUTPUT_DIR / "policy.pt")
    shutil.copy2(HERE / "policy.py", OUTPUT_DIR / "policy.py")

    summary = {
        "recipe": "public_broadband_hgb",
        "n_train": len(y_train),
        "feature_dim": int(x_train.shape[1]),
        "random_state": 0,
        "wall_s": round(time.monotonic() - started, 3),
    }
    (OUTPUT_DIR / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"staged {OUTPUT_DIR / 'policy.py'} and policy.pt "
        f"from {len(y_train)} public episodes"
    )


if __name__ == "__main__":
    main()
