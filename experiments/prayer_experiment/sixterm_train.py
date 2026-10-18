"""6-TERM weighter probe - the last CWN experiment (definitive closure).

Instead of weighting the fixed cultural BUNDLE (nav/safe/cult/task), the weighter
outputs SIX weights and controls each cultural constraint independently:
    [nav, safe, prayer, gender, attendant, task].
This removes the one real limitation we found (the 4-term weighter can only raise
the whole cultural term, so it cannot up-weight prayer during a prayer window
without also over-penalizing gender-ward entry). Run on the REDESIGN env, where
prayer is ACTIVE and the rich context exposes prayer_active - the weighter's best
possible shot. If it still does not beat the 4-term CWN / fixed_equal here, finer
weighting definitively does not help: the ceiling is about the weighting AXIS, not
its granularity.

Isolated. Reuses redesign_train (redesign env config + rich 20-d context + tasks),
then in this process only:
  * N_WEIGHTS 4 -> 6 in rewards.cwn and rewards.reinforce_mixin (so CWN builds a
    6-way softmax and the REINFORCE warmup/sampling use 6);
  * env._default_reward applies the 6 weights to the three cultural sub-rewards
    (prayer / gender / attendant) separately.
The REINFORCE credit (success - lambda*CVR meta-reward) is dim-agnostic, so it
needs no change. Run n_envs=1.

Usage: python prayer_experiment/sixterm_train.py --agent cwn --seed 0 \
         --timesteps 500000 --n_envs 1 --no_camera --no_check_env
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "prayer_experiment"))

import redesign_train as rd          # noqa: E402  (redesign env + rich context + tasks on import)
import rewards.cwn as cwn_mod         # noqa: E402
import rewards.reinforce_mixin as rmix  # noqa: E402
import envs.lmic_hospital_env as em   # noqa: E402
import train.train_sac as ts          # noqa: E402


def apply_sixterm():
    cwn_mod.N_WEIGHTS = 6
    rmix.N_WEIGHTS = 6
    em.LMICHospitalNavEnv._default_reward = _reward6


_orig_setup = em.LMICHospitalNavEnv._setup_episode


def _reward6(self, collision, task_complete, action, info):
    from rewards.components import navigation_reward, safety_reward, task_reward
    from rewards.cultural import (prayer_avoidance_reward, gender_zone_reward,
                                  attendant_proxemics_reward)
    dist = info["dist_to_goal"]
    r_nav = navigation_reward(self._prev_dist, dist, task_complete, action)
    r_safe = safety_reward(collision, info["min_pedestrian_distance"])
    fam = self._crowd.nearest_family_distance(self._robot_pos)
    r_p, _ = prayer_avoidance_reward(self._robot_pos, self.fp, self._tod_hour)
    r_g, _ = gender_zone_reward(self._robot_pos, self.fp)
    r_a, _ = attendant_proxemics_reward(fam)
    r_task = task_reward(task_complete, self._urgency, self._time, self._deadline)
    ctx = self._context_vec()
    if self._weight_source is not None:
        w = self._weight_source(ctx)
    else:
        w = np.full(6, 1.0 / 6.0, dtype=np.float32)
    total = (w[0] * r_nav + w[1] * r_safe + w[2] * r_p
             + w[3] * r_g + w[4] * r_a + w[5] * r_task)
    return float(np.clip(total, -5.0, 5.0))


apply_sixterm()

_orig_make_env = ts.make_env


def _make_env(*args, **kwargs):
    factory = _orig_make_env(*args, **kwargs)

    def _thunk():
        rd._ctx.apply_patches(); rd.patch_tasks(); apply_sixterm()
        return factory()

    return _thunk


if __name__ == "__main__":
    ts.make_env = _make_env
    if "--config" not in sys.argv:
        sys.argv += ["--config", "prayer_experiment/dmch_redesign.yaml"]
    if "--log_dir" not in sys.argv:
        sys.argv += ["--log_dir", str(ROOT / "experiments" / "prayer_experiment" / "sixterm_runs")]
    ts.train(ts.parse_args())
