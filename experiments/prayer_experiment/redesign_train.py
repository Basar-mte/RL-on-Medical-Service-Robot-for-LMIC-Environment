"""Training entry for the DMCH-REDESIGN env — ISOLATED.

Uses the new navigable floor plan (prayer_experiment/dmch_redesign.yaml) via
train_sac's own --config (no geometry monkeypatch needed). In this process only:
  * imports context_train -> richer 20-d context (adds prayer_active, d_prayer,
    attendant_prox, in_cultural, goal_in_cult, ped_prox) + CWN/Attn(context_dim=20);
  * patches env TASKS / TASK_IDS to the redesign's six tasks (STAT / male & female
    ward calls / ICU / pharmacy / prayer-room service) so sampling + urgency/
    deadline match the YAML's target rooms.
Run with n_envs=1 (DummyVecEnv) so the in-process patches hold.

Usage (one agent+seed):
  python prayer_experiment/redesign_train.py --agent cwn --seed 0 \
      --timesteps 500000 --n_envs 1 --no_camera --no_check_env
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "context_experiment"))

import context_train as _ctx          # noqa: E402  (rich 20-d context on import)
import envs.lmic_hospital_env as em    # noqa: E402
import train.train_sac as ts           # noqa: E402

# env-side task catalogue (urgency + deadline); target rooms live in the YAML.
NEW_TASKS = {
    "stat_med":       dict(urgency=1.0, deadline_s=45,  label="STAT Medicine"),
    "ward_call_f":    dict(urgency=0.9, deadline_s=60,  label="Female Ward Call"),
    "ward_call_m":    dict(urgency=0.9, deadline_s=60,  label="Male Ward Call"),
    "icu_run":        dict(urgency=0.8, deadline_s=70,  label="ICU Transfer"),
    "pharmacy_run":   dict(urgency=0.6, deadline_s=90,  label="Pharmacy Run"),
    "prayer_service": dict(urgency=0.3, deadline_s=200, label="Prayer-room Service"),
}


def patch_tasks():
    if em.TASK_IDS[:1] != ["stat_med"]:
        em.TASKS.clear(); em.TASKS.update(NEW_TASKS)
        em.TASK_IDS[:] = list(NEW_TASKS.keys())


patch_tasks()

_orig_make_env = ts.make_env


def _make_env(*args, **kwargs):
    factory = _orig_make_env(*args, **kwargs)

    def _thunk():
        _ctx.apply_patches(); patch_tasks()
        return factory()

    return _thunk


if __name__ == "__main__":
    ts.make_env = _make_env
    if "--config" not in sys.argv:
        sys.argv += ["--config", "prayer_experiment/dmch_redesign.yaml"]
    if "--log_dir" not in sys.argv:
        sys.argv += ["--log_dir", str(ROOT / "experiments" / "prayer_experiment" / "redesign_runs")]
    ts.train(ts.parse_args())
