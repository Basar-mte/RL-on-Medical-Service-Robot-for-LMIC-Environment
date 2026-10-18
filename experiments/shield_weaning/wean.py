"""Shield weaning: distil a reactive safety shield into the SAC policy and
anneal the policy off it, so the final policy needs no shield at deployment.

Isolation contract (as in every experiments/ folder): nothing in envs/,
rewards/, train/, eval/, baselines/ or configs/ is modified. Everything is
injected in-process via run.py (SAC subclass, replay-buffer subclass, env
wrapper, callback).

Why: on trained Tier-2 policies the inference shield alters 59-71% of all
actions (measured 2026-10-04). The policies never learned to slow down near
obstacles; the shield does it for them. Naive shield-in-loop training made
this worse (policy leans on the filter; -15..-23 SR vs inference shielding).

Method, four pieces:
  1. Executed-action relabeling. The shield runs inside the env; the replay
     buffer stores the action that was EXECUTED, so the critic learns the
     value of what actually happened, never that an unsafe proposal was fine.
  2. Corrective distillation. On intervened transitions an auxiliary loss
     pulls the actor's mean action toward the stored (shield) action.
  3. Annealed intervention budget. The distillation weight lambda is set by
     dual ascent toward a target intervention rate that decays linearly to
     zero over the first ANNEAL_FRAC of training, weaning the policy off the
     shield; where the policy still relies on it, lambda rises.
  4. Shielded exploration throughout, so early training sees fewer
     collision terminations and longer episodes.

Modes (ablations):
  wean   full method
  fixed  as wean but the budget target stays at RHO0 (no anneal)
  naive  shield in loop, proposed action stored, no distillation
         (the earlier failed design, re-run on Tier 2 for a matched ablation)
"""
from __future__ import annotations
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym
import torch as th
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import SAC                                   # noqa: E402
from stable_baselines3.common.buffers import DictReplayBuffer        # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback          # noqa: E402
from stable_baselines3.common.type_aliases import DictReplayBufferSamples  # noqa: E402
from stable_baselines3.common.utils import polyak_update             # noqa: E402
from baselines.safety_shield import shield_action                    # noqa: E402
from envs.lmic_hospital_env import MIN_LINEAR, MAX_LINEAR, MAX_STEER  # noqa: E402

MODE = os.environ.get("WEAN_MODE", "wean")          # wean | fixed | naive
CONFIG_REL = "configs/hospital_v4.yaml"
RUNS_NAME = f"experiments/shield_weaning/runs/{MODE}"

# Intervention = normalised action deviation above this (same threshold as
# the 2026-10-04 measurement: 59-71% of steps on trained Tier-2 policies).
_SPAN = np.array([MAX_LINEAR - MIN_LINEAR, 2 * MAX_STEER], dtype=np.float32)
INTERV_EPS = 0.02
RHO0 = 0.60          # initial budget = measured reliance of unweaned policies
ANNEAL_FRAC = 0.80   # budget reaches 0 at 80% of training; last 20% is at 0
LAMBDA_INIT = 1.0
LAMBDA_MAX = 50.0
ETA = 0.05           # dual-ascent step per update
UPDATE_EVERY = 1_000  # env steps between lambda updates
WINDOW = 5_000       # steps over which the intervention rate is measured


class ShieldWrapper(gym.Wrapper):
    """Execute the shielded action; report what was executed via info."""

    def __init__(self, env):
        super().__init__(env)
        self._last_obs = None

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)
        safe = shield_action(a, self._last_obs)
        obs, r, term, trunc, info = self.env.step(safe)
        self._last_obs = obs
        info["a_exec"] = safe
        info["interv_mag"] = float(np.linalg.norm((safe - a) / _SPAN))
        return obs, r, term, trunc, info


class WeanReplayBuffer(DictReplayBuffer):
    """DictReplayBuffer + per-transition intervention magnitude."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.interv = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.last_interv: th.Tensor | None = None

    def add(self, obs, next_obs, action, reward, done, infos):
        self.interv[self.pos] = np.array([float(i.get("interv_mag", 0.0)) for i in infos])
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds, env=None) -> DictReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        obs_ = self._normalize_obs({k: o[batch_inds, env_indices, :]
                                    for k, o in self.observations.items()}, env)
        next_obs_ = self._normalize_obs({k: o[batch_inds, env_indices, :]
                                         for k, o in self.next_observations.items()}, env)
        self.last_interv = self.to_torch(self.interv[batch_inds, env_indices])
        return DictReplayBufferSamples(
            observations={k: self.to_torch(o) for k, o in obs_.items()},
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations={k: self.to_torch(o) for k, o in next_obs_.items()},
            dones=self.to_torch(self.dones[batch_inds, env_indices]
                                * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            rewards=self.to_torch(self._normalize_reward(
                self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
        )


class WeanSAC(SAC):
    """SAC with executed-action relabeling and lambda-weighted corrective
    distillation toward the shield's action on intervened transitions."""

    wean_lambda: float = LAMBDA_INIT   # saved/loaded with the model

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
        if MODE != "naive":
            buffer_action = buffer_action.copy()
            for i, info in enumerate(infos):
                if "a_exec" in info:
                    buffer_action[i] = self.policy.scale_action(np.asarray(info["a_exec"]))
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs, actor_losses, critic_losses = [], [], [], []
        bc_losses, interv_fracs = [], []
        distil = MODE != "naive"

        for gradient_step in range(gradient_steps):
            rd = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = rd.discounts if rd.discounts is not None else self.gamma
            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(rd.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(rd.next_observations)
                next_q = th.cat(self.critic_target(rd.next_observations, next_actions), dim=1)
                next_q, _ = th.min(next_q, dim=1, keepdim=True)
                next_q = next_q - ent_coef * next_log_prob.reshape(-1, 1)
                target_q = rd.rewards + (1 - rd.dones) * discounts * next_q

            current_q = self.critic(rd.observations, rd.actions)
            critic_loss = 0.5 * sum(F.mse_loss(q, target_q) for q in current_q)
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_pi = th.cat(self.critic(rd.observations, actions_pi), dim=1)
            min_q_pi, _ = th.min(q_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_q_pi).mean()

            if distil:
                # Corrective distillation on intervened transitions: pull the
                # actor's mean (squashed, [-1,1]) toward the executed action.
                mask = (self.replay_buffer.last_interv > INTERV_EPS).float()
                mu = self.actor(rd.observations, deterministic=True)
                per = ((mu - rd.actions) ** 2).sum(dim=1)
                bc = (mask * per).sum() / mask.sum().clamp(min=1.0)
                actor_loss = actor_loss + self.wean_lambda * bc
                bc_losses.append(bc.item())
                interv_fracs.append(mask.mean().item())
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        if bc_losses:
            self.logger.record("wean/bc_loss", np.mean(bc_losses))
            self.logger.record("wean/lambda", self.wean_lambda)
            self.logger.record("wean/interv_frac_batch", np.mean(interv_fracs))


class WeanCallback(BaseCallback):
    """Dual ascent on lambda toward the (annealed) intervention budget."""

    def __init__(self, total_timesteps: int):
        super().__init__()
        self.total = total_timesteps
        self.window: deque[float] = deque(maxlen=WINDOW)
        self.next_update = 0

    def target(self, t: int) -> float:
        if MODE == "fixed":
            return RHO0
        return RHO0 * max(0.0, 1.0 - t / (ANNEAL_FRAC * self.total))

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "interv_mag" in info:
                self.window.append(1.0 if info["interv_mag"] > INTERV_EPS else 0.0)
        t = self.num_timesteps
        if t >= self.next_update and len(self.window) >= 100:
            self.next_update = t + UPDATE_EVERY
            rho = float(np.mean(self.window))
            tgt = self.target(t)
            if MODE != "naive":
                lam = float(np.clip(self.model.wean_lambda + ETA * (rho - tgt), 0.0, LAMBDA_MAX))
                self.model.wean_lambda = lam
                self.logger.record("wean/lambda", lam)
            self.logger.record("wean/interv_rate", rho)
            self.logger.record("wean/budget_target", tgt)
        return True
