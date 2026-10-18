"""Richer-context CWN — SELF-CONTAINED, ISOLATED EXPERIMENT.

Hypothesis: CWN does not win because its 14-d context is too coarse to demand
context-varying weights (the probe found no time-of-day / prayer adaptation).
Give the weighter DIRECT cultural-risk knowledge and it should produce sharper,
more situation-specific weights and win on CVR (compliance). Either outcome
strengthens the paper: a win is a new result; no win proves the weighting
ceiling is intrinsic, not a feature-poverty artifact.

Isolation contract: this file (and everything it writes) lives entirely under
context_experiment/. It edits NO file in envs/, train/, rewards/, or baselines/.
At runtime, in this process only, it monkeypatches:
  (1) LMICHospitalNavEnv._context_vec  -> appends 6 cultural/safety features
      (14-d -> 20-d), all computed from existing env state;
  (2) LMICHospitalNavEnv.__init__      -> widens the "context" obs Box to (20,);
  (3) train.train_sac.CWN / .AttentionWeightNet -> build with context_dim=20.
Fixed-weight baselines are unaffected by the weighting change but see the same
richer observation, so the comparison stays controlled. Outputs go to
context_experiment/runs/. Run with n_envs=1 (DummyVecEnv) so the in-process
patches hold.

Added features (all in [0,1]):
  prayer_active   direct prayer-window intensity (0.3->0, 1.0->1)  [temporal]
  d_prayer        proximity to prayer room (0 near .. 1 far)
  attendant_prox  nearest-attendant proximity (1 near .. 0 far)
  in_cultural     robot currently inside a prayer/gender zone
  goal_in_cult    the task goal is inside a cultural zone (necessary-violation)
  ped_prox        nearest-pedestrian proximity (safety signal)

Usage (one agent+seed):
  python context_experiment/context_train.py --agent cwn --seed 0 \
      --timesteps 1000000 --n_envs 1 --no_camera --no_check_env \
      --log_dir context_experiment/runs
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from gymnasium import spaces

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import envs.lmic_hospital_env as em          # noqa: E402
import train.train_sac as ts                  # noqa: E402
import rewards.cwn as cwn_mod                  # noqa: E402
import baselines.attention_weighting as attn_mod  # noqa: E402

RICH_DIM = 20  # 14 base + 6 added
_CULT_ZONES = ("prayer", "male_ward", "female_ward")

_orig_context_vec = em.LMICHospitalNavEnv._context_vec
_orig_init = em.LMICHospitalNavEnv.__init__


def _rich_context_vec(self) -> np.ndarray:
    base = _orig_context_vec(self)  # 14-d
    x, y = float(self._robot_pos[0]), float(self._robot_pos[1])
    # 1. direct prayer-window intensity (the temporal signal the probe never learned)
    prayer_active = (float(self.fp.prayer_intensity(self._tod_hour)) - 0.3) / 0.7
    # 2. proximity to the prayer room (0 near .. 1 far)
    pc = self.fp.nearest_room_of_type(x, y, "prayer")
    d_prayer = (min(float(np.linalg.norm(self._robot_pos - pc)) / 10.0, 1.0)
                if pc is not None else 1.0)
    # 3. nearest-attendant proximity (1 near .. 0 far)
    fam = float(self._crowd.nearest_family_distance(self._robot_pos))
    attendant_prox = 1.0 - min(fam / 3.0, 1.0)
    # 4. currently inside a cultural zone
    in_cultural = 1.0 if self.fp.get_zone(x, y) in _CULT_ZONES else 0.0
    # 5. task goal inside a cultural zone (necessary-violation / SR-CVR trade-off)
    gz = self.fp.get_zone(float(self._task_goal[0]), float(self._task_goal[1]))
    goal_in_cult = 1.0 if gz in _CULT_ZONES else 0.0
    # 6. nearest-pedestrian proximity (safety)
    if self._crowd.agents:
        mind = min(float(np.linalg.norm(a.pos - self._robot_pos)) for a in self._crowd.agents)
    else:
        mind = 10.0
    ped_prox = 1.0 - min(mind / 3.0, 1.0)
    extra = np.array([prayer_active, d_prayer, attendant_prox,
                      in_cultural, goal_in_cult, ped_prox], dtype=np.float32)
    return np.clip(np.concatenate([base, extra]).astype(np.float32), -1.0, 1.0)


def _rich_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    self.observation_space.spaces["context"] = spaces.Box(-1.0, 1.0, (RICH_DIM,), np.float32)


def apply_patches():
    if getattr(em.LMICHospitalNavEnv, "_rich_ctx_patched", False):
        return
    em.LMICHospitalNavEnv._context_vec = _rich_context_vec
    em.LMICHospitalNavEnv.__init__ = _rich_init
    em.LMICHospitalNavEnv._rich_ctx_patched = True

    # Subclasses (not wrapper functions) so `type(instance)()` — which
    # train_sac uses to build the eval replica — also defaults to RICH_DIM.
    class _CWN20(cwn_mod.CWN):
        def __init__(self, *a, **k):
            k.setdefault("context_dim", RICH_DIM)
            super().__init__(*a, **k)

    class _Attn20(attn_mod.AttentionWeightNet):
        def __init__(self, *a, **k):
            k.setdefault("context_dim", RICH_DIM)
            super().__init__(*a, **k)

    ts.CWN = _CWN20
    ts.AttentionWeightNet = _Attn20


apply_patches()

_orig_make_env = ts.make_env


def _rich_make_env(*args, **kwargs):
    factory = _orig_make_env(*args, **kwargs)

    def _thunk():
        apply_patches()
        return factory()

    return _thunk


if __name__ == "__main__":
    ts.make_env = _rich_make_env
    if "--log_dir" not in sys.argv:
        sys.argv += ["--log_dir", str(ROOT / "experiments" / "context_experiment" / "runs")]
    ts.train(ts.parse_args())
