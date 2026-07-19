"""
Custom SB3 callbacks for cultural metrics logging and CWN persistence.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


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
    Saves CWN weights at fixed intervals and logs softmax temperature.
    """

    def __init__(self, cwn_module, log_dir: Path, save_freq: int = 50_000,
                 verbose: int = 0) -> None:
        super().__init__(verbose)
        self.cwn = cwn_module
        self.log_dir = Path(log_dir)
        self.save_freq = save_freq
        self._last_save = 0

    def _on_step(self) -> bool:
        import torch
        try:
            tau = float(torch.exp(self.cwn.log_tau).item())
            self.logger.record("cwn/temperature", tau)
        except AttributeError:
            pass

        if self.num_timesteps - self._last_save >= self.save_freq:
            path = self.log_dir / f"cwn_{self.num_timesteps}.pt"
            self.cwn.save(path)
            self._last_save = self.num_timesteps
        return True
