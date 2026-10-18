"""Residual RL over the social-force prior (in-process env patch, no main-code change).

The executed action is  a = clip(a_prior + scale * a_policy)  where a_prior is the tuned
Ackermann social-force action toward the route planner's waypoint (sf_prior.AckermannSF)
and a_policy is the SAC output in the env's own action box ([v, steering]). A zero policy
output reproduces the classical controller exactly, so learning starts from a safe, 95 %-
success behaviour and can only add speed and judgement (up to +-scale*0.8 m/s, +-scale*30
deg). The prior's action is appended to the observation under the key "prior" so the
policy knows what it is correcting.

Env vars: PILOT_AGENT=residual_sf enables it (patch_sac.py); PILOT_RES_SCALE (default 0.5).
Applies to training AND evaluation (the class is patched, evaluate.py builds the same env).
"""
from __future__ import annotations
import os

import numpy as np
from gymnasium import spaces

import envs.lmic_hospital_env as em
from planner.sf_prior import AckermannSF


def apply_residual_patch(scale: float = 0.5) -> None:
    E = em.LMICHospitalNavEnv
    if getattr(E, "_residual_patched", False):
        return
    orig_init, orig_get_obs, orig_step = E.__init__, E._get_obs, E.step
    lo = np.array([em.MIN_LINEAR, -em.MAX_STEER], dtype=np.float32)
    hi = np.array([em.MAX_LINEAR, em.MAX_STEER], dtype=np.float32)

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        self.observation_space.spaces["prior"] = spaces.Box(lo, hi, dtype=np.float32)
        self._prior_ctrl = AckermannSF.tuned()
        self._prior_action = np.zeros(2, dtype=np.float32)
        self._prev_obs["prior"] = np.zeros(2, dtype=np.float32)

    def _prior(self, obs):
        pl = self.__dict__.get("_route_planner")
        target = self._task_goal
        if pl is not None:                                 # hierarchical: steer to the waypoint
            cfg = type(self)._planner_cfg
            wp = pl.field(self._task_goal, self._tod_hour).waypoint(self._robot_pos, cfg["L"], cfg["los"])
            if wp is not None:
                target = wp
        self._prior_ctrl.set_goal(np.asarray(target, dtype=float))
        a = self._prior_ctrl.act(self._robot_pos, self._robot_heading,
                                 obs["pedestrians"].reshape(em.K_NEIGHBORS, 4), obs["lidar"])
        return np.clip(a, lo, hi).astype(np.float32)

    def _get_obs(self):
        # Mirrors LMICHospitalNavEnv._get_obs (the original clips every key of the
        # observation space, so it cannot be called once "prior" is in the space).
        obs = {"lidar": self._cast_lidar(), "robot_state": self._robot_state_vec(),
               "pedestrians": self._ped_obs_vec(), "context": self._context_vec(), "goal": self._goal_vec()}
        if self.use_camera:
            obs["camera"] = self._render_camera()
        self._prior_action = _prior(self, obs)
        obs["prior"] = self._prior_action
        for key, subspace in self.observation_space.spaces.items():
            if key != "camera":
                obs[key] = np.clip(obs[key], subspace.low, subspace.high)
        return obs

    def step(self, action):
        a = np.clip(self._prior_action + scale * np.asarray(action, dtype=np.float32), lo, hi)
        # Evaluators cannot shield the policy output (it is only a correction): they set
        # env._env_shield = "v1" | "v2" and the shield is applied here to the EXECUTED action.
        sh = getattr(self, "_env_shield", None)
        if sh == "v1":
            from baselines.safety_shield import shield_action
            a = shield_action(a, {"lidar": self._cast_lidar()})
        elif sh == "v2":
            from planner.arc_shield import shield_action as shield_v2
            a = shield_v2(a, {"lidar": self._cast_lidar(), "pedestrians": self._ped_obs_vec(),
                              "robot_state": self._robot_state_vec()})
        self._executed_action = a
        return orig_step(self, a)

    E.__init__, E._get_obs, E.step = __init__, _get_obs, step
    E._residual_patched = True
    E._residual_scale = scale
    print(f"[residual] patch applied: a = clip(prior + {scale} * policy), prior = tuned social force", flush=True)


def apply_from_env() -> bool:
    if os.environ.get("PILOT_AGENT") != "residual_sf":
        return False
    apply_residual_patch(float(os.environ.get("PILOT_RES_SCALE", "0.5")))
    return True
