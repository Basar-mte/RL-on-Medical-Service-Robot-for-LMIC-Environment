"""
Contextual Weight Network (CWN) — C3 contribution.

A lightweight MLP that maps a 12-dim context vector to 4 non-negative reward
weights [w_nav, w_safety, w_cultural, w_task] via a temperature-scaled softmax.

Architecture
------------
context (12) → Linear(12, 32) → ReLU → Linear(32, 16) → ReLU → Linear(16, 4)
→ Softmax(τ) → (4,) weights

The temperature τ is a learnable parameter (initialised to 1.0) that controls
the peakedness of the weight distribution. Low τ → near-uniform weights;
high τ → winner-takes-all.

Context vector (12 dims, matches lmic_hospital_env.py _context_vec)
--------------------------------------------------------------------
[0]   crowd_density      ∈ [0, 1]
[1]   task_urgency       ∈ [0, 1]
[2-9] zone_onehot        ∈ {0,1}^8
[10]  sin(2π·tod/24)     ∈ [-1, 1]
[11]  cos(2π·tod/24)     ∈ [-1, 1]

Usage
-----
cwn = CWN()
context = env_info["context_vector"]          # (12,) numpy array
weights = cwn.get_weights(context)            # (4,) numpy, sums to 1
# Then: R = sum(weights * [r_nav, r_safety, r_cult, r_task])

During SAC training, CWN is optimised jointly via the total CARS reward
(gradients flow through the weight computation into the combined reward).
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CONTEXT_DIM = 12
N_WEIGHTS    = 4        # [nav, safety, cultural, task]
HIDDEN_DIM   = 32


class CWN(nn.Module):
    """
    Contextual Weight Network.

    Parameters
    ----------
    context_dim : int, default 12
    hidden_dim  : int, default 32
    tau_init    : float, initial softmax temperature (learnable)
    """

    def __init__(
        self,
        context_dim: int = CONTEXT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        tau_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, N_WEIGHTS),
        )
        # log-softmax temperature (always positive via exp)
        self.log_tau = nn.Parameter(torch.tensor(float(np.log(tau_init))))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        context : Tensor (..., context_dim)

        Returns
        -------
        weights : Tensor (..., 4) — sums to 1 along last dim.
        """
        logits = self.net(context)
        tau = torch.exp(self.log_tau).clamp(min=0.1, max=10.0)
        return F.softmax(logits / tau, dim=-1)

    @torch.no_grad()
    def get_weights(self, context_np: np.ndarray) -> np.ndarray:
        """
        Convenience method for environment rollouts.

        Parameters
        ----------
        context_np : (12,) or (N, 12) numpy array

        Returns
        -------
        weights : (4,) or (N, 4) numpy array, sums to 1.
        """
        single = context_np.ndim == 1
        x = torch.tensor(context_np, dtype=torch.float32)
        if single:
            x = x.unsqueeze(0)
        w = self.forward(x)
        w_np = w.cpu().numpy()
        return w_np[0] if single else w_np

    def save(self, path: str | Path) -> None:
        torch.save({"state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "CWN":
        data = torch.load(path, map_location="cpu", weights_only=True)
        net = cls(**kwargs)
        net.load_state_dict(data["state_dict"])
        net.eval()
        return net


# ---------------------------------------------------------------------------
# Fixed-weight baseline (equal weights ablation)
# ---------------------------------------------------------------------------

class FixedWeightSource:
    """
    Drop-in replacement for CWN that returns constant fixed weights.
    Used for ablation: B1 (equal) and B2 (hand-tuned).
    """

    PRESETS = {
        "equal":     np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
        "nav_heavy": np.array([0.55, 0.20, 0.15, 0.10], dtype=np.float32),
        "safe_heavy":np.array([0.20, 0.45, 0.25, 0.10], dtype=np.float32),
        "cult_heavy":np.array([0.20, 0.20, 0.45, 0.15], dtype=np.float32),
        "task_heavy":np.array([0.20, 0.15, 0.15, 0.50], dtype=np.float32),
    }

    def __init__(self, preset: str = "equal") -> None:
        assert preset in self.PRESETS, f"Unknown preset '{preset}'"
        self._w = self.PRESETS[preset].copy()

    def get_weights(self, context_np: np.ndarray) -> np.ndarray:
        if context_np.ndim == 1:
            return self._w.copy()
        return np.tile(self._w, (context_np.shape[0], 1))


# ---------------------------------------------------------------------------
# CWN-backed reward shaper (wraps CWN + CARS components)
# ---------------------------------------------------------------------------

class CARSRewardShaper:
    """
    Combines CWN weight prediction with CARS reward components.

    This class is the "plug" inserted into the Gymnasium environment via the
    `reward_fn` and `weight_source` hooks (lmic_hospital_env.py).

    Usage
    -----
    shaper = CARSRewardShaper(cwn=CWN())
    # In env.__init__: env = LMICHospitalNavEnv(reward_fn=shaper, weight_source=shaper.cwn.get_weights)
    shaper(obs, next_obs, action, info)  → float reward
    """

    def __init__(
        self,
        cwn: Optional[CWN] = None,
        fixed_preset: Optional[str] = None,
        fp=None,
    ) -> None:
        if cwn is not None:
            self._weight_src = cwn
        elif fixed_preset is not None:
            self._weight_src = FixedWeightSource(fixed_preset)
        else:
            self._weight_src = FixedWeightSource("equal")
        self._fp = fp

    def __call__(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        info: dict,
    ) -> float:
        from rewards.components import navigation_reward, safety_reward, task_reward
        from rewards.cultural import cultural_reward

        ctx = info["context_vector"]
        weights = self._weight_src.get_weights(ctx)

        # Component rewards
        r_nav = navigation_reward(
            info.get("prev_dist", info["dist_to_goal"] + 0.01),
            info["dist_to_goal"],
            info["task_complete"],
            action,
        )
        r_safe = safety_reward(info["collision"], info["min_pedestrian_distance"])
        fp = self._fp or info.get("floorplan")
        if fp is not None:
            r_cult, _ = cultural_reward(
                np.array(next_obs[48:50]),
                fp,
                info.get("tod_hour", 12.0),
                info["min_pedestrian_distance"],
            )
        else:
            r_cult = 0.0  # fp unavailable — cultural term skipped
        r_task = task_reward(
            info["task_complete"],
            info["urgency"],
            info.get("elapsed_s", info["steps"] * 0.1),
            info.get("deadline_s", 90.0),
        )

        total = (weights[0] * r_nav
                 + weights[1] * r_safe
                 + weights[2] * r_cult
                 + weights[3] * r_task)
        return float(np.clip(total, -5.0, 5.0))
