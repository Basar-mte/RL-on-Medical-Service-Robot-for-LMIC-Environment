"""
Shared REINFORCE-style training mechanism for context-conditioned reward-
weight networks (CWN, AttentionWeightNet).

Why this exists: neither network was ever actually trained before this was
added. `get_weights()` was `@torch.no_grad()` with no optimizer, buffer, or
update loop anywhere in the codebase — both networks sat frozen at random
initialization for the entire duration of any training run, at any step
count. This mixin implements the intended mechanism: a Dirichlet-sampled
stochastic weight vector each step (used both for the actual reward mixing
and for the REINFORCE log-probability), credited with the completed
episode's total return, and updated in a clean on-policy batch every
`ROLLOUTS_PER_UPDATE` episodes.

Revision history: the first version of this mixin (LR=1e-4, an update every
10 episodes, and a 2,000-step *rolling* buffer reused across many updates)
was itself diagnosed as non-functional — checkpoints 250,000 steps apart had
near-identical weights (softmax temperature pinned at 1.000 the entire run).
Root causes, all fixed below: (1) LR was 10x below SAC's own 3e-4 while
firing ~30x less often, so the ~100 updates in a 300k-step run barely moved
the network; (2) the per-step-replicated return array made the baseline
length-weighted rather than a clean per-episode average, biasing the
advantage; (3) the rolling buffer let a single update's gradient use
log-probs recomputed under *current* parameters for actions sampled up to
several updates (and several thousand steps) ago under *older* parameters —
an off-policy mismatch with no importance-sampling correction. The batch
design here (collect exactly ROLLOUTS_PER_UPDATE episodes, take one gradient
step, discard) removes (3) by construction and fixes (2) directly.

Host class contract: an `nn.Module` subclass providing a differentiable
`forward(context: Tensor) -> Tensor` returning a `(..., N_WEIGHTS)` simplex
vector (post-softmax), plus a call to `self._init_reinforce_state(lr=...)`
at the end of its own `__init__` (after all parameter-holding submodules
are assigned, since this constructs the optimizer via `self.parameters()`).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

CONTEXT_DIM = 14  # matches envs.lmic_hospital_env._context_vec
N_WEIGHTS = 4

# Matches the training-mechanism design: 50k-step hard warm start at equal
# weights, then one on-policy REINFORCE batch update every
# ROLLOUTS_PER_UPDATE completed episodes (collect, update, discard --
# no rolling buffer spanning multiple updates; see revision history above).
WARMUP_STEPS = 50_000
# 8 episodes per update (was 2): the v3 per-step reward-to-go credit gives
# each update ~2000 informative gradient terms instead of 2 antisymmetric
# episode scalars, so fewer/steadier updates win over frequent/noisy ones.
ROLLOUTS_PER_UPDATE = 8

# Dirichlet concentration scale: alpha = softmax_output * kappa + ALPHA_FLOOR.
# KAPPA_START=20 gives roughly +-10% sampling std per component -- enough
# exploration signal for the REINFORCE gradient early on. The concentration
# ANNEALS to KAPPA_END over KAPPA_ANNEAL_STEPS: late in training the sampled
# weights tighten around the network's mean output, removing the reward
# noise the sampling injects into SAC's replayed targets (a measured
# handicap: both learned-weighting agents trailed every fixed-weight
# baseline under constant KAPPA=20). ALPHA_FLOOR keeps every concentration
# parameter strictly positive near-degenerate softmax outputs.
KAPPA_START = 20.0
KAPPA_END = 150.0
KAPPA_ANNEAL_STEPS = 800_000
ALPHA_FLOOR = 1e-2

# v3 meta-objective: the network is credited with the benchmark's OWN vital
# metrics, not the (self-weighted or equal-weighted) CARS return it can game
# or that dilutes the signal: per-step meta-reward
#     m_t = -LAMBDA_VIOL * 1[cultural violation at t]
#           + SUCCESS_BONUS * 1[t is terminal and episode succeeded]
# credited per step as undiscounted reward-to-go (a weight choice is
# credited only with what happened FROM that step on -- temporal credit the
# previous whole-episode scalar destroyed).
SUCCESS_BONUS = 10.0
# Calibration matters: two ward tasks REQUIRE ~100-130 violation-steps of
# gender-zone transit to complete at all. At the original 0.2/step, a
# completed nurse_call netted 10 - ~23 = NEGATIVE meta-return, so the weight
# network learned to steer SAC away from ward tasks entirely (measured on
# the seed-0 vanguard duel: ward-task SR 5% vs 25% for fixed weights, with
# overall SR 15 points lower). At 0.02/step, required transit costs ~2.3
# against the +10 success bonus (completion decisively positive) while
# LOITERING 500+ steps in a ward still cancels a success — the penalty now
# prices dwell-time, not task feasibility.
LAMBDA_VIOL = 0.02

# Above SAC's 3e-4 (updates are ~500x less frequent), below the old 3e-3
# (the v3 gradient is far lower-variance, so it does not need the outsized
# step the 2-episode antisymmetric credit did).
REINFORCE_LR = 1.5e-3
GRAD_CLIP_NORM = 1.0
ENTROPY_COEF = 1e-3


class AdaptiveWeightReinforceMixin:
    """Mix into an nn.Module implementing forward(context) -> simplex weights."""

    def _init_reinforce_state(self, lr: float = REINFORCE_LR,
                               warmup_steps: int = WARMUP_STEPS) -> None:
        self._step_count = 0
        self._warmup_steps = warmup_steps
        # Per-episode-in-progress accumulation, flushed by end_episode().
        self._ep_ctx: list = []
        self._ep_actions: list = []
        self._ep_viol: list = []   # per recorded step: cultural-violation flag
        # Completed episodes awaiting a reinforce_update() call. Consumed
        # and discarded whole by each update (see reinforce_update) -- no
        # rolling buffer persists across updates, so every gradient step
        # only ever uses actions sampled under the current parameters.
        self._episode_queue: list = []
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def _kappa(self) -> float:
        """Annealed Dirichlet concentration (see KAPPA_* constants)."""
        frac = min(max(self._step_count - self._warmup_steps, 0)
                   / max(KAPPA_ANNEAL_STEPS - self._warmup_steps, 1), 1.0)
        return KAPPA_START + (KAPPA_END - KAPPA_START) * frac

    @torch.no_grad()
    def get_weights(self, context_np: np.ndarray) -> np.ndarray:
        """
        Convenience method for environment rollouts. In training mode
        (self.training, the nn.Module default), returns a Dirichlet-sampled
        stochastic weight vector and records (context, action) for the
        REINFORCE update once the episode completes (see end_episode()). In
        eval mode (after .eval()), returns the plain deterministic softmax
        output -- no sampling, no bookkeeping, matching the previous
        (unintentionally-always-on) behavior.

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

        if self.training:
            w = self._sample_for_training(x, context_np if single else None)
        else:
            w = self.forward(x)

        w_np = w.cpu().numpy()
        return w_np[0] if single else w_np

    def _sample_for_training(self, x: torch.Tensor,
                              context_np_single: Optional[np.ndarray]) -> torch.Tensor:
        self._step_count += 1
        if self._step_count <= self._warmup_steps:
            # Hard override to equal weights, byte-identical to
            # FixedWeightSource("equal") -- no forward() call, no recording:
            # there is nothing meaningful to credit yet.
            return torch.full((x.shape[0], N_WEIGHTS), 1.0 / N_WEIGHTS, dtype=torch.float32)

        p = self.forward(x)
        alpha = p * self._kappa() + ALPHA_FLOOR
        w = torch.distributions.Dirichlet(alpha).sample()

        if context_np_single is not None:
            self._ep_ctx.append(context_np_single.copy())
            self._ep_actions.append(w.squeeze(0).cpu().numpy())

        return w

    def record_step_outcome(self, cultural_violation: bool) -> None:
        """
        Called once per environment step (by train.callbacks.CWNCallback,
        AFTER the step's get_weights recording) with that step's cultural-
        violation flag — the per-step half of the meta-reward. Ignored while
        nothing is recorded (warmup / eval), and length-guarded so outcomes
        can never misalign with the recorded (context, action) stream.
        """
        if len(self._ep_viol) < len(self._ep_ctx):
            self._ep_viol.append(bool(cultural_violation))

    def end_episode(self, success: bool) -> None:
        """
        Call once per completed episode with its success flag. Builds the
        per-step meta-rewards m_t = -LAMBDA_VIOL*viol_t (+ SUCCESS_BONUS on
        the terminal step if the episode succeeded) and their undiscounted
        reward-to-go, then flushes this episode's (context, action, R_t)
        triples into the update queue; a no-op during warm-start.
        """
        if self._ep_ctx:
            n = len(self._ep_ctx)
            viol = np.array(self._ep_viol[:n] + [False] * (n - len(self._ep_viol)),
                            dtype=np.float32)
            m = -LAMBDA_VIOL * viol
            if success:
                m[-1] += SUCCESS_BONUS
            rtg = np.cumsum(m[::-1])[::-1].astype(np.float32)  # reward-to-go
            self._episode_queue.append(dict(
                contexts=np.stack(self._ep_ctx),
                actions=np.stack(self._ep_actions),
                rtg=rtg,
            ))
        self._ep_ctx, self._ep_actions, self._ep_viol = [], [], []

    def pending_episode_count(self) -> int:
        return len(self._episode_queue)

    def reinforce_update(self) -> dict:
        """
        Run one on-policy REINFORCE gradient step over exactly the episodes
        accumulated since the last update, then discard them. Returns a dict
        of scalars for TensorBoard logging.

        The baseline is the mean of per-episode returns (one scalar per
        episode), not the mean over per-step-replicated returns -- averaging
        over steps instead of episodes silently reweights the baseline
        toward longer episodes (a fixed-length-1000 episode contributes
        1000x the weight of a 1-step episode to a step-mean, but they should
        each count once). Using exactly this update's episodes (no rolling
        buffer from prior updates) keeps every (context, action) pair's
        log-prob computed under the same parameters that generated it --
        the previous rolling-buffer design mixed in actions from several
        updates and thousands of steps ago, recomputing their log-prob under
        newer parameters with no importance-sampling correction.
        """
        if not self._episode_queue:
            return {}

        ctx_chunks = [ep["contexts"] for ep in self._episode_queue]
        act_chunks = [ep["actions"] for ep in self._episode_queue]
        rtg_chunks = [ep["rtg"] for ep in self._episode_queue]
        n_episodes = len(self._episode_queue)
        self._episode_queue.clear()

        ctx_np = np.concatenate(ctx_chunks)
        rtg_np = np.concatenate(rtg_chunks)
        # STATE-DEPENDENT baseline, fit fresh each update by closed-form
        # least squares of reward-to-go on the context. A plain batch-mean
        # baseline is provably wrong here: contexts differ systematically in
        # achievable return (a cultural-zone step carries violation risk no
        # corridor step has), so under a constant baseline every action
        # sampled in the poorer context gets negative advantage — REINFORCE
        # then suppresses the CONTEXT rather than preferring better ACTIONS
        # within it. (Caught empirically: the synthetic-rule unit test
        # learned the exact inverse mapping under a batch-mean baseline.)
        X = np.concatenate([ctx_np, np.ones((ctx_np.shape[0], 1), dtype=np.float32)],
                           axis=1)
        beta, *_ = np.linalg.lstsq(X, rtg_np, rcond=None)
        baseline_np = X @ beta

        ctx_t = torch.tensor(ctx_np, dtype=torch.float32)
        act_t = torch.tensor(np.concatenate(act_chunks), dtype=torch.float32)
        # Insurance: an exact-zero component in a stored Dirichlet sample
        # would make log_prob +inf and one optimizer step would NaN-poison
        # the network (and thus every subsequent reward) permanently. Never
        # observed in 2M probed samples, but the cost of ruling it out is
        # one clamp + renormalize.
        act_t = act_t.clamp_min(1e-6)
        act_t = act_t / act_t.sum(dim=-1, keepdim=True)
        rtg_t = torch.tensor(rtg_np, dtype=torch.float32)
        # Per-step advantages give temporal credit (a weight choice is only
        # credited with what followed it), state-conditioned by the baseline.
        advantage = rtg_t - torch.tensor(baseline_np, dtype=torch.float32)

        p = self.forward(ctx_t)
        alpha = p * self._kappa() + ALPHA_FLOOR
        dist = torch.distributions.Dirichlet(alpha)
        log_prob = dist.log_prob(act_t)
        entropy = dist.entropy().mean()

        loss = -(log_prob * advantage).mean() - ENTROPY_COEF * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), GRAD_CLIP_NORM)
        self.optimizer.step()

        return dict(
            loss=float(loss.item()),
            baseline=float(rtg_t.mean().item()),
            n_episodes=n_episodes,
            mean_step_rtg=float(rtg_t.mean().item()),
            kappa=self._kappa(),
        )

    def _save_extra_state(self) -> dict:
        return {"optimizer_state_dict": self.optimizer.state_dict()}

    def _load_extra_state(self, data: dict, load_optimizer: bool) -> None:
        if load_optimizer and "optimizer_state_dict" in data:
            self.optimizer.load_state_dict(data["optimizer_state_dict"])
