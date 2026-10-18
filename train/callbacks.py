"""
Custom SB3 callbacks for cultural metrics logging and CWN persistence.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rewards.reinforce_mixin import ROLLOUTS_PER_UPDATE


class CulturalMetricsCallback(BaseCallback):
    """
    Logs cultural metrics to TensorBoard and a JSON file:
      - Cultural Violation Rate (CVR) per episode
      - Social Violation Rate
      - Path Efficiency (actual / optimal)
      - Mean min-pedestrian distance
    """

    def __init__(self, log_dir: Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self._ep_cvr: list = []
        self._ep_svr: list = []
        self._ep_pe:  list = []
        self._ep_mpd: list = []
        self._ep_rewards: list = []
        self._records: list = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for i, (info, done) in enumerate(zip(infos, dones)):
            if done:
                steps = max(info.get("steps", 1), 1)
                cvr = info.get("cultural_violations", 0) / steps
                svr = info.get("social_violations", 0) / steps
                opt = info.get("path_length_optimal", 1.0)
                act = info.get("path_length_actual", opt)
                pe  = opt / max(act, 1e-3)
                mpd = info.get("min_pedestrian_distance", 0.0)

                self._ep_cvr.append(cvr)
                self._ep_svr.append(svr)
                self._ep_pe.append(pe)
                self._ep_mpd.append(mpd)
                self._records.append({
                    "step": self.num_timesteps,
                    "cvr": cvr, "svr": svr, "pe": pe, "mpd": mpd,
                    "task_id": info.get("task_id", ""),
                    "density": info.get("density_level", ""),
                })

                self.logger.record("cultural/cvr", cvr)
                self.logger.record("cultural/svr", svr)
                self.logger.record("cultural/path_efficiency", pe)
                self.logger.record("cultural/mean_ped_dist", mpd)
        return True

    def _on_training_end(self) -> None:
        out = self.log_dir / "cultural_metrics.json"
        with open(out, "w") as f:
            json.dump(self._records, f, indent=2)
        if self.verbose:
            print(f"[CulturalMetricsCallback] Saved {len(self._records)} episodes to {out}")


class CWNCallback(BaseCallback):
    """
    Drives the REINFORCE-style training update for a context-conditioned
    weight network (CWN or AttentionWeightNet — see
    rewards/reinforce_mixin.py::AdaptiveWeightReinforceMixin). Also keeps a
    second, deterministic (eval-mode) replica in sync for EvalCallback's
    eval_env, logs the softmax temperature (CWN only — a no-op for
    AttentionWeightNet, which has no log_tau), and periodically checkpoints
    the network + its optimizer state.

    Named "CWNCallback" for historical/minimal-diff reasons; it works
    identically for both agents with a trainable weight network.
    """

    def __init__(self, cwn_module, log_dir: Path, eval_weight_module=None,
                 save_freq: int = 50_000,
                 rollouts_per_update: int = ROLLOUTS_PER_UPDATE,
                 prefix: str = "cwn", verbose: int = 0) -> None:
        super().__init__(verbose)
        self.cwn = cwn_module
        self.eval_weight_module = eval_weight_module
        self.log_dir = Path(log_dir)
        self.save_freq = save_freq
        self.rollouts_per_update = rollouts_per_update
        # Derived from the actual --agent (e.g. "attention"), not hardcoded
        # to "cwn" — otherwise an attention_seed0_.../ run directory would
        # confusingly contain files literally named cwn_final.pt etc.
        self.prefix = prefix
        self._last_save = 0

    def _on_step(self) -> bool:
        import torch

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            # Per-step half of the v3 meta-reward: this step's cultural-
            # violation flag (length-guarded inside the mixin so warmup and
            # eval steps are ignored). Must run every step, before the
            # episode-end flush below.
            self.cwn.record_step_outcome(bool(info.get("cultural_violation", False)))
            if done and "episode" in info:
                # v3 meta-objective: the episode's SUCCESS flag — combined
                # with the recorded per-step violations, the network is
                # credited with success - lambda*CVR, the benchmark's own
                # vital metrics, rather than any CARS return it could game.
                self.cwn.end_episode(bool(info.get("is_success", False)))
                if self.eval_weight_module is not None:
                    self.eval_weight_module.load_state_dict(self.cwn.state_dict())

        if self.cwn.pending_episode_count() >= self.rollouts_per_update:
            metrics = self.cwn.reinforce_update()
            for k, v in metrics.items():
                self.logger.record(f"cwn/{k}", v)

        try:
            tau = float(torch.exp(self.cwn.log_tau).item())
            self.logger.record("cwn/temperature", tau)
        except AttributeError:
            pass

        if self.num_timesteps - self._last_save >= self.save_freq:
            path = self.log_dir / f"{self.prefix}_{self.num_timesteps}.pt"
            self.cwn.save(path)
            self._last_save = self.num_timesteps
        return True


class SaveVecNormalizeOnNewBest(BaseCallback):
    """
    Saves the training VecNormalize statistics alongside EvalCallback's
    best_model.zip. SB3's EvalCallback saves only the policy weights: without
    this snapshot, later evaluation pairs a best model from (say) step 400k
    with the END-of-training vec_normalize.pkl — observation statistics the
    checkpoint never trained under — silently degrading every evaluated
    agent by a normalization mismatch.
    """

    def __init__(self, vec_env, save_path: Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.vec_env = vec_env  # the VecNormalize wrapper used for training
        self.save_path = Path(save_path)

    def _on_step(self) -> bool:
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.vec_env.save(str(self.save_path / "vec_normalize_best.pkl"))
        return True


class SaveWeightNetOnNewBest(BaseCallback):
    """
    Saves the weight network alongside EvalCallback's best_model.zip,
    triggered via EvalCallback(..., callback_on_new_best=...) so the two
    checkpoints stay aligned to the same timestep — without this, only
    cwn_final.pt (end of training, possibly a worse timestep than the best
    eval) would exist as a candidate to pair with best_model.zip.
    """

    def __init__(self, weight_module, save_path: Path, prefix: str = "cwn",
                 verbose: int = 0) -> None:
        super().__init__(verbose)
        self.weight_module = weight_module
        self.save_path = Path(save_path)
        self.prefix = prefix

    def _on_step(self) -> bool:
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.weight_module.save(self.save_path / f"{self.prefix}_best.pt")
        return True
