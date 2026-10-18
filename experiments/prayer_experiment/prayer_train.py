"""Training entry for the prayer-ENGAGEMENT probe — ISOLATED.

Importing prayer_engage applies (in this process only): the richer 20-d context
patch (via context_train), the engaged floor-plan loader (prayer gate on the
female-ward west door), and the prayer_call task. This entry then wraps make_env
so the patches also hold inside each env-factory thunk, and delegates to the
unmodified train.train_sac.train. Writes only under prayer_experiment/runs.

Run with n_envs=1 so the in-process monkeypatches hold (DummyVecEnv).

Usage (one agent+seed):
  python prayer_experiment/prayer_train.py --agent cwn --seed 0 \
      --timesteps 400000 --n_envs 1 --no_camera --no_check_env \
      --log_dir prayer_experiment/runs
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prayer_experiment"))

import prayer_engage as pe   # noqa: E402  (applies ALL patches on import)
import train.train_sac as ts  # noqa: E402


if __name__ == "__main__":
    ts.make_env = pe._prayer_make_env
    if "--log_dir" not in sys.argv:
        sys.argv += ["--log_dir", str(ROOT / "prayer_experiment" / "runs")]
    ts.train(ts.parse_args())
