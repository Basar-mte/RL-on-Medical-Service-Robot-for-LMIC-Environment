"""
Baseline B5 — MODSRL Attention-based reward weighting.

Simplified reimplementation of the attention weighting mechanism from:
  Modular Deep RL (MODSRL) — attention over reward modules based on
  current state features.

The attention scores are computed over the same 4 reward heads as CWN,
but via a simple dot-product attention over the context vector projected
by a fixed key matrix — no temperature parameter, no end-to-end learning
of the temperature.

This isolates the benefit of CWN's learnable temperature τ: both networks
are trained via the identical REINFORCE mechanism in
`rewards.reinforce_mixin.AdaptiveWeightReinforceMixin` (see that module and
`train.callbacks.CWNCallback`) — the only difference between B5 and CWN is
architectural (no learnable `log_tau` here), which is what actually isolates
the temperature mechanism's contribution. (Note: this deliberately differs
from a "trained through the standard actor loss" framing that appears in
some paper drafts — that would require routing gradients through
Stable-Baselines3's internal actor-loss computation, is unspecified beyond
a one-line aside, and would confound the tau-ablation with an unrelated
algorithmic difference between the two baselines being compared.)
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rewards.reinforce_mixin import AdaptiveWeightReinforceMixin, REINFORCE_LR


CONTEXT_DIM = 14  # matches envs.lmic_hospital_env._context_vec
N_HEADS      = 4   # nav, safety, cultural, task
KEY_DIM      = 16


class AttentionWeightNet(nn.Module, AdaptiveWeightReinforceMixin):
    """
    Attention-based reward weight predictor (B5 baseline).

    Computes dot-product attention between a query derived from the context
    and a fixed set of key vectors (one per reward head).
    """

    def __init__(
        self,
        context_dim: int = CONTEXT_DIM,
        n_heads: int = N_HEADS,
        key_dim: int = KEY_DIM,
        lr: float = REINFORCE_LR,
    ) -> None:
        super().__init__()
        self.query_proj = nn.Linear(context_dim, key_dim)
        self.key_mat = nn.Parameter(torch.randn(n_heads, key_dim) * 0.1)
        self._init()
        self._init_reinforce_state(lr=lr)

    def _init(self) -> None:
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        context : (..., context_dim)

        Returns
        -------
        weights : (..., N_HEADS) — via softmax attention
        """
        q = self.query_proj(context)          # (..., key_dim)
        # scores: (..., N_HEADS) via einsum
        scores = torch.einsum("...d,hd->...h", q, self.key_mat)
        scores = scores / (KEY_DIM ** 0.5)
        return F.softmax(scores, dim=-1)

    # get_weights() is provided by AdaptiveWeightReinforceMixin: stochastic
    # (Dirichlet-sampled, recorded for REINFORCE) in training mode, plain
    # deterministic forward() in eval mode.

    def save(self, path) -> None:
        payload = {"state_dict": self.state_dict()}
        payload.update(self._save_extra_state())
        torch.save(payload, path)

    @classmethod
    def load(cls, path, load_optimizer: bool = False, **kwargs) -> "AttentionWeightNet":
        data = torch.load(path, map_location="cpu", weights_only=True)
        net = cls(**kwargs)
        net.load_state_dict(data["state_dict"])
        net._load_extra_state(data, load_optimizer)
        net.eval()
        return net
