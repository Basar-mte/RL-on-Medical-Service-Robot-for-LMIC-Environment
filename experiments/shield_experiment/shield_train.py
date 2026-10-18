"""Shield-in-loop training — SELF-CONTAINED BONUS EXPERIMENT.

Isolation contract: this file (and everything it writes) lives entirely under
shield_experiment/. It does NOT modify any file in train/, envs/, rewards/,
or baselines/ — it reuses them read-only and injects the safety shield by
monkeypatching train.train_sac.make_env AT RUNTIME, in this process only.
Outputs go to shield_experiment/runs/. Deleting this folder fully removes the
experiment; the main paper pipeline is untouched.

The shield is applied by wrapping each env factory's product with a gym
wrapper that filters the commanded action through baselines.safety_shield.
shield_action before stepping. It wraps OUTSIDE the SB3 Monitor (Monitor
passes observations through unchanged), so the shield still sees raw LiDAR;
VecNormalize is applied later at the vec level, above the shield. The shield
is thus part of the transition dynamics — a well-defined MDP with no
replay-buffer action mismatch.

Usage (one agent+seed):
  python shield_experiment/shield_train.py --agent attention --seed 0 --timesteps 1000000 --no_camera
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from baselines.safety_shield import shield_action
import train.train_sac as ts


class _ShieldWrapper(gym.Wrapper):
    """Filter the commanded action through the reactive shield each step."""

    def __init__(self, env):
        super().__init__(env)
        self._last = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last = obs
        return obs, info

    def step(self, action):
        safe = shield_action(np.asarray(action, dtype=np.float32), self._last)
        obs, r, term, trunc, info = self.env.step(safe)
        self._last = obs
        return obs, r, term, trunc, info


_orig_make_env = ts.make_env


def _shielded_make_env(*args, **kwargs):
    factory = _orig_make_env(*args, **kwargs)   # returns a thunk building Monitor(env)

    def _thunk():
        return _ShieldWrapper(factory())        # shield outside Monitor: raw LiDAR, raw action

    return _thunk


if __name__ == "__main__":
    # Force all outputs under shield_experiment/runs/ unless caller overrode it.
    if "--log_dir" not in sys.argv:
        sys.argv += ["--log_dir", str(ROOT / "shield_experiment" / "runs")]
    ts.make_env = _shielded_make_env            # in-process monkeypatch; main files untouched
    ts.train(ts.parse_args())
