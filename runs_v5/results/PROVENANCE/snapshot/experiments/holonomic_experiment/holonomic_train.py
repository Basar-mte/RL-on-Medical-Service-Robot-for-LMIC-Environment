"""Holonomic-kinematics ablation — SELF-CONTAINED, ISOLATED EXPERIMENT.

Question it answers: is the Ackermann (non-holonomic) chassis the reason the
reward-weighting schemes (CWN included) do not win, or does the weighting
ceiling hold regardless of kinematics? To test it honestly we retrain the same
agents under a HOLONOMIC (unicycle / differential-drive) motion model and
recompute the comparison and the adaptivity ceiling. If CWN still does not beat
the attention weighter and the ceiling stays small, Ackermann is not the cause.

Isolation contract: this file (and everything it writes) lives entirely under
holonomic_experiment/. It does NOT edit any file in envs/, train/, rewards/,
baselines/, or configs/. It changes ONLY the robot kinematics, at runtime, in
this process, by monkeypatching `LMICHospitalNavEnv.step`: the Ackermann heading
rate `(v/L)tan(delta)` (zero at v=0 -> cannot pivot) is replaced by a rate that
uses a fixed reference speed instead of the actual v, so heading change is
DECOUPLED from speed (the robot can pivot in place). Same steering law, same max
turn rate as Ackermann-at-full-speed; only the non-holonomic v-coupling is
removed. Everything else (collision, crowd, reward, obs, termination) is the
unmodified env logic. Outputs go to holonomic_experiment/runs/.

Run all agents with n_envs=1 so the in-process monkeypatch holds (DummyVecEnv);
no SubprocVecEnv workers that would re-import the env unpatched.

Usage (one agent+seed):
  python holonomic_experiment/holonomic_train.py --agent cwn --seed 0 \
      --timesteps 1000000 --n_envs 1 --no_camera --no_check_env \
      --log_dir holonomic_experiment/runs
"""
from __future__ import annotations
import sys
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import envs.lmic_hospital_env as em   # noqa: E402
import train.train_sac as ts          # noqa: E402


def apply_holonomic_patch():
    """Replace Ackermann step() with a unicycle (pivot-in-place) version.

    Idempotent (guarded), so re-applying in an env-factory thunk is harmless.
    """
    if getattr(em.LMICHospitalNavEnv, "_holonomic_patched", False):
        return
    orig_step = em.LMICHospitalNavEnv.step
    V_REF = float(em.MAX_LINEAR)   # reference speed decoupling heading from v
    L = float(em.WHEELBASE)
    DT = float(em.DT)

    def holonomic_step(self, action):
        a = np.clip(action, self.action_space.low, self.action_space.high)
        v, delta = float(a[0]), float(a[1])
        # Heading rate uses V_REF, not v -> decoupled from speed (can pivot at v=0).
        heading_rate = (V_REF / L) * math.tan(delta)
        self._robot_heading = (self._robot_heading + heading_rate * DT
                               + math.pi) % (2 * math.pi) - math.pi
        # Delegate the rest (move along the new heading, collisions, crowd,
        # reward, obs) to the original step with ZERO steering so it applies no
        # additional Ackermann rotation on top of the heading we just set.
        return orig_step(self, np.array([v, 0.0], dtype=np.float32))

    em.LMICHospitalNavEnv.step = holonomic_step
    em.LMICHospitalNavEnv._holonomic_patched = True


apply_holonomic_patch()

# Belt-and-suspenders: also re-apply inside each env-factory thunk.
_orig_make_env = ts.make_env


def _holo_make_env(*args, **kwargs):
    factory = _orig_make_env(*args, **kwargs)

    def _thunk():
        apply_holonomic_patch()
        return factory()

    return _thunk


if __name__ == "__main__":
    ts.make_env = _holo_make_env
    if "--log_dir" not in sys.argv:
        sys.argv += ["--log_dir", str(ROOT / "holonomic_experiment" / "runs")]
    ts.train(ts.parse_args())
