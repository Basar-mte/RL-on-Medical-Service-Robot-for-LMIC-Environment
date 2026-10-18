"""Reduced-density retraining — SELF-CONTAINED EXPLORATORY EXPERIMENT.

Isolation contract: this file (and everything it writes) lives entirely under
density_experiment/. It does NOT modify any file in envs/, rewards/, train/,
baselines/, or configs/ — it reuses them read-only and overrides the pedestrian
density levels AT RUNTIME, in this process only, by mutating the module-global
`envs.lmic_hospital_env.DENSITY_LEVELS` (and wrapping `train.train_sac.make_env`
so the override also applies inside any worker env factory). Outputs go to
density_experiment/runs/. Deleting this folder fully removes the experiment; the
main benchmark (medium=45, high=74) is untouched.

Why this experiment exists: to test *honestly* whether lowering the benchmark
density (medium 45->30, high 74->50) changes which weighter wins. The only valid
test is a SAME-DENSITY comparison — CWN vs the attention weighter vs vanilla,
all retrained at the reduced density — which run_density.ps1 sets up. Comparing
a density-30 CWN against the density-45 baselines would be meaningless.

Usage (one agent+seed):
  python density_experiment/density_train.py --agent cwn --seed 0 \
      --timesteps 1000000 --n_envs 1 --no_camera
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import envs.lmic_hospital_env as env_mod   # noqa: E402
import train.train_sac as ts               # noqa: E402

# --- Reduced density levels (this process only) --------------------------------
# medium 45 -> 30, high 74 -> 50, low 23 unchanged. Per-class split keeps the
# original staff/patient/attendant/wheelchair balance.
MEDIUM = dict(medical_staff=5, patient=11, family_attendant=11, wheelchair=3)  # 30
HIGH   = dict(medical_staff=9, patient=18, family_attendant=18, wheelchair=5)  # 50


def _apply_density_override():
    env_mod.DENSITY_LEVELS["medium"] = dict(MEDIUM)
    env_mod.DENSITY_LEVELS["high"] = dict(HIGH)


# Apply in the main process (covers DummyVecEnv / in-process eval env).
_apply_density_override()

# Also apply inside each env factory's thunk (covers SubprocVecEnv workers, which
# re-import the module fresh on spawn and would otherwise use the original 45/74).
_orig_make_env = ts.make_env


def _density_make_env(*args, **kwargs):
    factory = _orig_make_env(*args, **kwargs)

    def _thunk():
        import envs.lmic_hospital_env as em
        em.DENSITY_LEVELS["medium"] = dict(MEDIUM)
        em.DENSITY_LEVELS["high"] = dict(HIGH)
        return factory()

    return _thunk


if __name__ == "__main__":
    ts.make_env = _density_make_env
    if "--log_dir" not in sys.argv:
        sys.argv += ["--log_dir", str(ROOT / "experiments" / "density_experiment" / "runs")]
    ts.train(ts.parse_args())
