"""
Baseline B5 — MODSRL Attention-based reward weighting.

Simplified reimplementation of the attention weighting mechanism from:
  Modular Deep RL (MODSRL) — attention over reward modules based on
  current state features.

The attention scores are computed over the same 4 reward heads as CWN,
but via a simple dot-product attention over the context vector projected
by a fixed key matrix — no temperature parameter, no end-to-end learning
of the temperature.

This isolates the benefit of CWN's learnable temperature τ.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CONTEXT_DIM = 12
N_HEADS      = 4   # nav, safety, cultural, task
KEY_DIM      = 16


class AttentionWeightNet(nn.Module):
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
    ) -> None:
        super().__init__()
        self.query_proj = nn.Linear(context_dim, key_dim)
        self.key_mat = nn.Parameter(torch.randn(n_heads, key_dim) * 0.1)
        self._init()

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

    @torch.no_grad()
    def get_weights(self, context_np: np.ndarray) -> np.ndarray:
        single = context_np.ndim == 1
        x = torch.tensor(context_np, dtype=torch.float32)
        if single:
            x = x.unsqueeze(0)
        w = self.forward(x).cpu().numpy()
        return w[0] if single else w

    def save(self, path) -> None:
        torch.save({"state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, **kwargs) -> "AttentionWeightNet":
        data = torch.load(path, map_location="cpu", weights_only=True)
        net = cls(**kwargs)
        net.load_state_dict(data["state_dict"])
        net.eval()
        return net
