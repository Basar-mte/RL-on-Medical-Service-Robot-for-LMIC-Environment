"""PPO baseline: the CARS equal-weight reward, the same environment, wrappers,
monitors, callbacks and (through patch_sac.py) the same route-planner wiring as
the SAC agents — only the algorithm differs. Stable-Baselines3 PPO with its
default hyper-parameters (n_steps 2048, batch 64, 10 epochs, lr 3e-4, GAE 0.95,
clip 0.2) and the same [256, 256] MultiInputPolicy; observation AND reward
normalisation on (SB3's recommendation for on-policy PPO — the SAC recipe's
reward-normalisation-off fix addressed an off-policy critic pathology).

Agent id "ppo_equal" (run dirs ppo_equal_seed<s>_<ts>/, checkpoints ppo_<N>_steps.zip).
Launch through the campaign: experiments/stability_pilot/patch_sac.py experiments/ppo/train_ppo.py ...
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import train.train_sac as ts  # noqa: E402  (make_env / callbacks / NORM_OBS_KEYS — possibly planner-wrapped)
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

AGENT = "ppo_equal"
REWARD_AGENT = "fixed_equal"      # CARS, equal weights
PPO_KWARGS = dict(policy="MultiInputPolicy",
                  policy_kwargs=dict(net_arch=[256, 256], activation_fn=torch.nn.ReLU),
                  n_steps=2048, batch_size=64, n_epochs=10, learning_rate=3e-4,
                  gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, verbose=1)


def parse_args():
    p = argparse.ArgumentParser(description="PPO baseline on LMICHospitalNav-v1")
    p.add_argument("--config", default="configs/dmch_hospital.yaml")
    p.add_argument("--agent", default=AGENT, choices=[AGENT])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--n_envs", type=int, default=1)
    p.add_argument("--eval_freq", type=int, default=50_000)
    p.add_argument("--checkpoint_freq", type=int, default=100_000)
    p.add_argument("--log_dir", default="runs")
    p.add_argument("--device", default="cpu")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--density", default="medium")
    p.add_argument("--no_camera", action="store_true")
    p.add_argument("--no_check_env", action="store_true")
    p.add_argument("--resume_from", default=None)
    return p.parse_args()


def train(args):
    assert args.n_envs == 1, "campaign parity: one environment per run"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(2)
    resume_dir = Path(args.resume_from) if args.resume_from else None
    resume_steps, resume_ckpt = 0, None
    if resume_dir is not None:
        log_dir, run_name = resume_dir, resume_dir.name
        ckpts = sorted(log_dir.glob("checkpoints/ppo_[0-9]*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]))
        if not ckpts:
            raise SystemExit(f"--resume_from {log_dir}: no checkpoints/ppo_*_steps.zip found")
        resume_ckpt = ckpts[-1]; resume_steps = int(resume_ckpt.stem.split("_")[1])
        print(f"[train_ppo] RESUMING {run_name} from {resume_steps:,} steps", flush=True)
    else:
        run_name = f"{AGENT}_seed{args.seed}_{int(time.time())}"
        log_dir = Path(args.log_dir) / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train_ppo] Run: {run_name} | seed {args.seed} | log dir {log_dir}", flush=True)

    config_path = str(ROOT / args.config)
    use_camera = not args.no_camera
    monitor_csv = str(log_dir / ("monitor" if resume_dir is None else f"monitor_resume{resume_steps}"))
    env = DummyVecEnv([ts.make_env(config_path, REWARD_AGENT, args.seed, args.density,
                                   monitor_path=monitor_csv, use_camera=use_camera)])
    eval_env = DummyVecEnv([ts.make_env(config_path, REWARD_AGENT, args.seed + 100, args.density,
                                        use_camera=use_camera)])
    if resume_dir is None:
        env = ts.VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=args.gamma,
                              norm_obs_keys=ts.NORM_OBS_KEYS)
    else:
        vn_pkl = log_dir / "checkpoints" / f"ppo_vecnormalize_{resume_steps}_steps.pkl"
        if not vn_pkl.exists():
            raise SystemExit(f"--resume_from: missing {vn_pkl}")
        env = ts.VecNormalize.load(str(vn_pkl), env); env.training = True; env.norm_reward = True
    eval_env = ts.VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False,
                               norm_obs_keys=ts.NORM_OBS_KEYS)

    if resume_dir is None:
        model = PPO(env=env, gamma=args.gamma, seed=args.seed, device=args.device,
                    tensorboard_log=str(log_dir / "tb"), **PPO_KWARGS)
    else:
        model = PPO.load(str(resume_ckpt), env=env, device=args.device, tensorboard_log=str(log_dir / "tb"))
    print(f"[train_ppo] Model params: {sum(p.numel() for p in model.policy.parameters()):,}", flush=True)

    callbacks = [
        CheckpointCallback(save_freq=args.checkpoint_freq, save_path=str(log_dir / "checkpoints"),
                           name_prefix="ppo", save_vecnormalize=True),
        EvalCallback(eval_env, best_model_save_path=str(log_dir / "best_model"), log_path=str(log_dir / "eval"),
                     eval_freq=args.eval_freq, n_eval_episodes=10, deterministic=True, verbose=1,
                     callback_on_new_best=CallbackList([ts.SaveVecNormalizeOnNewBest(env, log_dir / "best_model")])),
        ts.CulturalMetricsCallback(log_dir=log_dir),
    ]
    remaining = args.timesteps - resume_steps
    if remaining <= 0:
        raise SystemExit(f"[train_ppo] nothing to resume: {resume_steps:,} >= {args.timesteps:,}")
    print(f"[train_ppo] Training for {remaining:,} steps" + (f" (resumed at {resume_steps:,})" if resume_dir else "")
          + "...", flush=True)
    model.learn(total_timesteps=remaining, callback=CallbackList(callbacks), tb_log_name=run_name,
                reset_num_timesteps=(resume_dir is None))
    model.save(str(log_dir / "final_model"))
    env.save(str(log_dir / "vec_normalize.pkl"))
    print(f"[train_ppo] Done. Saved to {log_dir}", flush=True)


if __name__ == "__main__":
    train(parse_args())
