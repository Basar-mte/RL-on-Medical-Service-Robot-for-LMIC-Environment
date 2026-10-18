"""
Phase 6.1 — Frozen evaluation protocol.

Evaluates a saved SAC model on LMICHospitalNav-v1 across all task types
and density levels. Computes:
  • SR   — Success Rate
  • CVR  — Cultural Violation Rate (novel metric, C2)
  • SPL  — Success-weighted Path Length (Anderson 2018)
  • MPD  — Mean Minimum Pedestrian Distance
  • SVR  — Social Violation Rate
  • CT   — Completion Time (seconds)

Usage:
  python eval/evaluate.py --model_dir runs/cwn_seed0_XXX/best_model \\
         --n_episodes 100 --output eval_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

from envs.lmic_hospital_env import LMICHospitalNavEnv, TASK_IDS, DENSITY_LEVELS


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_spl(success: bool, optimal_len: float, actual_len: float) -> float:
    if not success or actual_len < 1e-6:
        return 0.0
    return optimal_len / max(actual_len, optimal_len)


def aggregate_metrics(records: List[Dict]) -> Dict:
    sr   = np.mean([r["success"] for r in records])
    cvr  = np.mean([r["cvr"] for r in records])
    spl  = np.mean([r["spl"] for r in records])
    mpd  = np.mean([r["mpd"] for r in records])
    svr  = np.mean([r["svr"] for r in records])
    ct_s = np.mean([r["ct_s"] for r in records if r["success"]] or [0])
    return dict(
        sr=float(sr),
        cvr=float(cvr),
        spl=float(spl),
        mpd=float(mpd),
        svr=float(svr),
        ct_s=float(ct_s),
        n_episodes=len(records),
    )


# ---------------------------------------------------------------------------
# VecNormalize helper
# ---------------------------------------------------------------------------

def _find_vecnormalize(model_dir: Path) -> Optional[Path]:
    """
    Search for VecNormalize stats alongside the saved model.

    Search order:
      1. model_dir / "vec_normalize_best.pkl"   (snapshot saved at the same
         timestep as best_model.zip by SaveVecNormalizeOnNewBest — the ONLY
         stats the best checkpoint actually trained under; the end-of-training
         pkl can be hundreds of thousands of steps newer)
      2. run_dir / "vec_normalize.pkl"          (written at end of training)
      3. run_dir / "checkpoints/*vecnormalize*steps.pkl"  (latest checkpoint)
    where run_dir is model_dir or its parent.
    """
    for base in [model_dir, model_dir.parent]:
        best = base / "vec_normalize_best.pkl"
        if best.exists():
            return best

    candidates = []
    for base in [model_dir, model_dir.parent]:
        vn = base / "vec_normalize.pkl"
        if vn.exists():
            candidates.append(vn)
        # SB3's CheckpointCallback(name_prefix="sac", save_vecnormalize=True)
        # writes "sac_vecnormalize_<N>_steps.pkl" — a glob of "*_vecnormalize.pkl"
        # (matching only filenames ENDING in that suffix) never matches this
        # and was silently dead code; it only appeared to "work" because a
        # top-level vec_normalize.pkl also always existed alongside it.
        for pkl in sorted(base.glob("checkpoints/*vecnormalize*steps.pkl")):
            candidates.append(pkl)

    if not candidates:
        return None
    # Return the most recently modified one
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def _batch(obs):
    """Add a leading batch dim; works for a flat ndarray OR a Dict observation."""
    if isinstance(obs, dict):
        return {k: v[np.newaxis] for k, v in obs.items()}
    return obs[np.newaxis]


def _unbatch(obs):
    """Remove a leading batch dim; works for a flat ndarray OR a Dict observation."""
    if isinstance(obs, dict):
        return {k: v[0] for k, v in obs.items()}
    return obs[0]


def _infer_agent(model_dir: Path) -> Optional[str]:
    """Recover the --agent name (e.g. "cwn") from a runs/<agent>_seed<N>_<ts>/
    style path — checks both the dir itself and its parent, since model_dir
    may be .../best_model."""
    for part in (model_dir.name, model_dir.parent.name):
        if "_seed" in part:
            return part.split("_seed")[0]
    return None


def _load_weight_source(model_dir: Path, agent: Optional[str]):
    """
    For agent in ("cwn", "attention"), load the trained weight-network
    checkpoint saved alongside the SAC model (see train.callbacks.CWNCallback
    / SaveWeightNetOnNewBest) and return its deterministic (.eval()-mode)
    get_weights as this eval run's weight_source. Returns None for every
    other agent, or if no checkpoint is found (falls back to the env's
    built-in equal-weight default, with a loud warning).
    """
    if agent not in ("cwn", "attention"):
        return None

    from rewards.cwn import CWN
    from baselines.attention_weighting import AttentionWeightNet
    net_cls = CWN if agent == "cwn" else AttentionWeightNet

    run_dir = model_dir.parent if model_dir.name == "best_model" else model_dir

    candidates = []
    if model_dir.name == "best_model":
        best_path = model_dir / f"{agent}_best.pt"
        if best_path.exists():
            candidates = [best_path]
    if not candidates:
        candidates = sorted(run_dir.glob(f"{agent}_*.pt"), key=lambda p: p.stat().st_mtime)

    if not candidates:
        print(f"  [eval] WARNING: agent='{agent}' but no saved weight-net checkpoint "
              f"found under {run_dir} — evaluating with fallback equal weights")
        return None

    net = net_cls.load(candidates[-1])  # net_cls.load() already calls .eval()
    print(f"  [eval] Loaded trained {agent} weights from {candidates[-1].name}")
    return net.get_weights


def _load_model_and_env(model_dir: str, config_path: str, agent: Optional[str] = None):
    """
    Locate the saved model + VecNormalize stats, load the model (CPU), and
    build the matching environment. Shared by evaluate_model() and
    eval/log_trajectories.py so both inherit the same Dict-obs-safe loading
    logic instead of duplicating it.

    `agent` (e.g. "cwn", "attention") is auto-inferred from model_dir's path
    if not given; used to load a trained weight-network checkpoint (see
    _load_weight_source) so the eval env's reward mixing matches what the
    policy actually saw during training, rather than silently falling back
    to equal weights for these two agents.

    Returns (model, inner_env, eval_env, use_normalization). `eval_env` is
    ALWAYS a VecEnv — Monitor+DummyVecEnv, VecTransposeImage-wrapped if the
    obs space has an image key (matching train_sac.py's wrapping order
    exactly: transpose BEFORE normalize, since VecNormalize.load() requires
    its saved observation_space shape to match), plus VecNormalize on top if
    stats were found — so every reset()/step() goes through the exact same
    transform the model was trained under, regardless of whether
    VecNormalize stats exist (previously, the no-VecNormalize fallback
    stepped the raw, untransposed env directly, silently feeding
    channel-last camera frames to a model trained on channel-first ones).
    `inner_env` is the unwrapped LMICHospitalNavEnv, for reading internal
    state (e.g. _robot_pos) directly — drive episodes via `eval_env`, not
    `inner_env`, directly (see evaluate_model() for the seed/set_options
    pattern).
    """
    model_dir = Path(model_dir)
    agent = agent or _infer_agent(model_dir)

    # Find the .zip model file
    zip_files = list(model_dir.glob("*.zip"))
    if not zip_files:
        # EvalCallback saves best_model.zip directly inside the dir
        zip_files = list(model_dir.parent.glob("best_model.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No .zip model found in {model_dir}")
    model_path = sorted(zip_files, key=lambda p: p.stat().st_mtime)[-1]

    # Find VecNormalize
    vn_path = _find_vecnormalize(model_dir)
    use_normalization = vn_path is not None
    if use_normalization:
        print(f"  [eval] Using VecNormalize from: {vn_path.name}")
    else:
        print("  [eval] No VecNormalize found — evaluating without obs normalization")

    # Load model (on CPU for evaluation)
    model = SAC.load(str(model_path), device="cpu")

    weight_source = _load_weight_source(model_dir, agent)

    # Build environment. The camera key is auto-detected from the loaded
    # model's own observation space, so no-camera checkpoints evaluate in a
    # matching no-camera env and older camera checkpoints keep working.
    model_obs = model.observation_space
    use_camera = isinstance(model_obs, spaces.Dict) and "camera" in model_obs.spaces
    cfg = str(Path(__file__).parent.parent / config_path)
    env_fn = lambda: Monitor(LMICHospitalNavEnv(floorplan_path=cfg,
                                                weight_source=weight_source,
                                                use_camera=use_camera))
    dummy_env = DummyVecEnv([env_fn])
    inner_env = dummy_env.envs[0].env  # unwrap Monitor → LMICHospitalNavEnv

    eval_env = dummy_env
    if isinstance(eval_env.observation_space, spaces.Dict) and use_camera:
        eval_env = VecTransposeImage(eval_env)
    if use_normalization:
        eval_env = VecNormalize.load(str(vn_path), eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

    return model, inner_env, eval_env, use_normalization


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_model(
    model_dir: str,
    config_path: str,
    n_episodes: int = 100,
    density: str = "medium",
    deterministic: bool = True,
) -> Dict:
    """
    Run `n_episodes` rollouts with the saved model and return metrics dict.
    """
    model, inner_env, eval_env, use_normalization = _load_model_and_env(model_dir, config_path)

    records = []
    for ep in range(n_episodes):
        task_id = TASK_IDS[ep % len(TASK_IDS)]

        # Seed/options must be set on the VecEnv (not by calling
        # inner_env.reset() directly and separately "syncing" eval_env with
        # a second, unseeded reset() — that second reset would silently
        # re-randomize the episode via the env's own RNG, discarding the
        # requested task/density/seed entirely). VecEnvWrapper.seed()/
        # set_options() forward down through VecTransposeImage/VecNormalize
        # to the underlying DummyVecEnv, which passes them to the real
        # LMICHospitalNavEnv.reset(seed=..., options=...) exactly once.
        eval_env.seed(1000 + ep)
        eval_env.set_options({"task": task_id, "density": density})
        obs = _unbatch(eval_env.reset())

        done = False
        info_last: Dict = {}

        while not done:
            action, _ = model.predict(_batch(obs), deterministic=deterministic)
            action = action[0]

            # DummyVecEnv.step_wait() returns the standard 4-tuple
            # (obs, rewards, dones, infos) — NOT a 5-tuple of
            # (obs, rewards, terminated, truncated, infos); "done" already
            # folds in both terminated and truncated.
            raw_obs, _, dones, infos = eval_env.step(action[np.newaxis])
            obs = _unbatch(raw_obs)
            done = bool(dones[0])
            info_last = infos[0] if infos else {}

        steps    = info_last.get("steps", 1)
        success  = bool(info_last.get("task_complete", False))
        cult_viol = info_last.get("cultural_violations", 0)
        soc_viol  = info_last.get("social_violations", 0)
        opt_len   = info_last.get("path_length_optimal", 1.0)
        act_len   = info_last.get("path_length_actual", opt_len)
        mpd_val   = info_last.get("min_pedestrian_distance", 0.0)

        records.append(dict(
            ep=ep,
            task_id=task_id,
            success=success,
            # End-reason for post-hoc diagnosis: without it, a broken
            # benchmark (e.g. policy-independent collisions) is
            # indistinguishable from weak policies in the saved results.
            end=("success" if success
                 else "collision" if info_last.get("collision") else "timeout"),
            steps=int(steps),
            dist_to_goal=float(info_last.get("dist_to_goal", -1.0)),
            cvr=cult_viol / max(steps, 1),
            svr=soc_viol  / max(steps, 1),
            spl=compute_spl(success, opt_len, act_len),
            mpd=float(mpd_val),
            ct_s=steps * 0.1,
        ))

        if (ep + 1) % 10 == 0:
            partial = aggregate_metrics(records)
            print(f"  ep {ep+1:3d}/{n_episodes}  SR={partial['sr']:.3f}  "
                  f"CVR={partial['cvr']:.4f}  SPL={partial['spl']:.3f}  "
                  f"MPD={partial['mpd']:.3f}")

    eval_env.close()  # cascades to inner_env.close() via the wrapper chain

    metrics = aggregate_metrics(records)
    metrics["records"] = records  # per-episode detail for post-hoc diagnosis
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True,
                   help="Path to model directory (containing .zip)")
    p.add_argument("--config", default="configs/dmch_hospital.yaml")
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--density", default="medium")
    p.add_argument("--output", default="eval_results.json",
                   help="Output JSON filename (saved next to model_dir)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = evaluate_model(
        model_dir=args.model_dir,
        config_path=args.config,
        n_episodes=args.n_episodes,
        density=args.density,
    )
    out_path = Path(args.model_dir).parent / args.output
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[evaluate] Saved to {out_path}")
    print(json.dumps(results, indent=2))
