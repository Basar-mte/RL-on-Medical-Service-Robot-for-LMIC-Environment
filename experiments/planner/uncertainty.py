"""Unpredicted-uncertainty patches (evaluation-only, in-process, env-var driven).

  UNC_PED_NOYIELD=1   pedestrians IGNORE the robot (no robot-repulsion force): the crowd
                      no longer cooperates — worst-case behavioural uncertainty.
  UNC_LIDAR=s         multiplicative Gaussian range noise, sigma = s (e.g. 0.05 = 5 %).
  UNC_LIDAR_DROP=p    each ray independently drops out (reads max range) with prob p.
                      Noise applies to EVERY consumer of the scan (policy, shields,
                      classical wall forces) — one imperfect sensor, as on the robot.
  UNC_ACT=s           actuation noise: executed action = commanded + N(0, s*range) per
                      axis, applied INSIDE env.step AFTER any shield — motors are
                      imperfect and no software layer can see the perturbation.

All randomness uses the env's episode-seeded RNG, so every driver faces identical
noise realisations on identical episodes. apply_from_env() before creating the env.
"""
from __future__ import annotations
import os

import numpy as np

import envs.lmic_hospital_env as em
import envs.pedestrians as peds


def apply_from_env() -> dict:
    cfg = dict(noyield=os.environ.get("UNC_PED_NOYIELD") == "1",
               lidar=float(os.environ.get("UNC_LIDAR", "0") or 0),
               drop=float(os.environ.get("UNC_LIDAR_DROP", "0") or 0),
               act=float(os.environ.get("UNC_ACT", "0") or 0))
    E = em.LMICHospitalNavEnv
    if getattr(E, "_uncertainty_cfg", None) is not None:
        return E._uncertainty_cfg

    if cfg["noyield"]:
        _orig_crowd_step = peds.PedestrianCrowd.step

        def crowd_step(self, dt=0.1, robot_pos=None):
            return _orig_crowd_step(self, dt, robot_pos=None)   # the crowd cannot see the robot
        peds.PedestrianCrowd.step = crowd_step

    if cfg["lidar"] > 0 or cfg["drop"] > 0:
        _orig_cast = E._cast_lidar

        def _cast_lidar(self):
            hits = _orig_cast(self)
            rng = self._rng
            if cfg["lidar"] > 0:
                hits = hits * (1.0 + rng.normal(0.0, cfg["lidar"], hits.shape).astype(np.float32))
            if cfg["drop"] > 0:
                hits = np.where(rng.random(hits.shape) < cfg["drop"], 1.0, hits)
            return np.clip(hits, 0.0, 1.0).astype(np.float32)
        E._cast_lidar = _cast_lidar

    if cfg["act"] > 0:
        _orig_step = E.step
        v_rng = float(em.MAX_LINEAR - em.MIN_LINEAR)
        d_rng = float(2 * em.MAX_STEER)

        def step(self, action):
            a = np.asarray(action, dtype=np.float32).copy()
            a[0] += self._rng.normal(0.0, cfg["act"] * v_rng)
            a[1] += self._rng.normal(0.0, cfg["act"] * d_rng)
            return _orig_step(self, a)
        E.step = step

    E._uncertainty_cfg = cfg
    if any(v for v in cfg.values()):
        print(f"[uncertainty] patch applied: {cfg}", flush=True)
    return cfg
