"""Simplified Tier-2 experiment — runtime overrides, this process only.

Isolation contract (same as the other experiments/): nothing in envs/, rewards/,
train/, eval/, baselines/ or configs/ is modified. This module mutates the
already-imported env module in memory:

  * low    -> 10 pedestrians (fixed)
  * medium -> 10..20 pedestrians, drawn uniformly PER EPISODE from the env's own
              seeded RNG (so evaluation stays reproducible)
  * high   -> 20 pedestrians (fixed)

Class mix keeps the benchmark's staff/patient/attendant/wheelchair proportions
(8/16/16/5 of 45). The simplified floor plan itself is a separate YAML
(hospital_simple.yaml, built by make_config.py).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
CONFIG_REL = "experiments/simple_experiment/hospital_simple.yaml"
RUNS_NAME = "experiments/simple_experiment/runs"
N_MIN, N_MAX = 10, 20

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def split_counts(n: int) -> dict:
    staff = max(1, round(n * 8 / 45))
    wheel = max(1, round(n * 5 / 45))
    rest = n - staff - wheel
    patient = rest // 2
    return dict(medical_staff=staff, patient=patient,
                family_attendant=rest - patient, wheelchair=wheel)


def apply() -> None:
    import envs.lmic_hospital_env as em
    if getattr(em, "_SIMPLE_PATCHED", False):
        return
    em.DENSITY_LEVELS["low"] = split_counts(N_MIN)
    em.DENSITY_LEVELS["high"] = split_counts(N_MAX)
    em.DENSITY_LEVELS["medium"] = split_counts((N_MIN + N_MAX) // 2)

    orig = em.LMICHospitalNavEnv._setup_episode

    def _setup_episode(self, task_id, density_key):
        if density_key == "medium":
            n = int(self._rng.integers(N_MIN, N_MAX + 1))
            em.DENSITY_LEVELS["medium"] = split_counts(n)
        return orig(self, task_id, density_key)

    em.LMICHospitalNavEnv._setup_episode = _setup_episode
    em._SIMPLE_PATCHED = True
    print(f"[simple_experiment] patch applied (pid {os.getpid()}): "
          f"peds {N_MIN}-{N_MAX}", file=sys.stderr, flush=True)
