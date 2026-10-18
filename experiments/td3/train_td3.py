"""TD3 baseline: the CARS equal-weight reward, the same environment, planner wiring,
wrappers and callbacks as the SAC agents — only the algorithm differs. TD3 is SAC's
nearest off-policy neighbour (twin critics, replay buffer, deterministic actor +
exploration noise), so it tests whether the campaign's findings are SAC-specific.

Recipe matched to the SAC stable recipe wherever the knob exists: buffer 1M,
LR 1e-4, batch 256, net [256, 256], train_freq 4 / gradient_steps 4, reward
normalisation OFF (same critic-divergence rationale), obs normalisation ON.
TD3-specific: Gaussian exploration noise sigma = 0.1 x action half-range per
dimension; default target smoothing and policy delay.

Agent id "td3_equal" (run dirs td3_equal_seed<s>_<ts>/, checkpoints td3_<N>_steps.zip).
Launch through the campaign: experiments/stability_pilot/patch_sac.py experiments/td3/train_td3.py ...
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import train.train_sac as ts  # noqa: E402  (make_env / callbacks / NORM_OBS_KEYS — possibly planner-wrapped)
from stable_baselines3 import TD3  # noqa: E402
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback  # noqa: E402
from stable_baselines3.common.noise import NormalActionNoise  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

AGENT = "td3_equal"
REWARD_AGENT = "fixed_equal"      # CARS, equal weights
TD3_KWARGS = dict(policy="MultiInputPolicy",
                  policy_kwargs=dict(net_arch=[256, 256], activation_fn=torch.nn.ReLU),
                  train_freq=4, gradient_steps=4, batch_size=256, tau=0.005, verbose=1)


def parse_args():
    p = argparse.ArgumentParser(description="TD3 baseline on LMICHospitalNav-v1")
    p.add_argument("--config", default="configs/dmch_hospital.yaml")
    p.add_argument("--agent", default=AGENT, choices=[AGENT])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--n_envs", type=int, default=1)
    p.add_argument("--eval_freq", type=int, default=50_000)
    p.add_argument("--checkpoint_freq", type=int, default=100_000)
    p.add_argument("--log_dir", default="runs")
    p.add_argument("--device", default="cpu")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--buffer_size", type=int, default=1_000_000)
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
        ckpts = sorted(log_dir.glob("checkpoints/td3_[0-9]*_steps.zip"), key=lambda p: int(p.stem.split("_")[1]))
        if not ckpts:
            raise SystemExit(f"--resume_from {log_dir}: no checkpoints/td3_*_steps.zip found")
        resume_ckpt = ckpts[-1]; resume_steps = int(resume_ckpt.stem.split("_")[1])
        print(f"[train_td3] RESUMING {run_name} from {resume_steps:,} steps", flush=True)
    else:
        run_name = f"{AGENT}_seed{args.seed}_{int(time.time())}"
        log_dir = Path(args.log_dir) / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train_td3] Run: {run_name} | seed {args.seed} | log dir {log_dir}", flush=True)

    config_path = str(ROOT / args.config)
    use_camera = not args.no_camera
    monitor_csv = str(log_dir / ("monitor" if resume_dir is None else f"monitor_resume{resume_steps}"))
    env = DummyVecEnv([ts.make_env(config_path, REWARD_AGENT, args.seed, args.density,
                                   monitor_path=monitor_csv, use_camera=use_camera)])
    eval_env = DummyVecEnv([ts.make_env(config_path, REWARD_AGENT, args.seed + 100, args.density,
                                        use_camera=use_camera)])
    # Reward normalisation OFF from the start (the SAC stable recipe; same critic rationale).
    if resume_dir is None:
        env = ts.VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=args.gamma,
                              norm_obs_keys=ts.NORM_OBS_KEYS)
    else:
        vn_pkl = log_dir / "checkpoints" / f"td3_vecnormalize_{resume_steps}_steps.pkl"
        if not vn_pkl.exists():
            raise SystemExit(f"--resume_from: missing {vn_pkl}")
        env = ts.VecNormalize.load(str(vn_pkl), env)
        env.training = True
        env.norm_reward = False
    eval_env = ts.VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False,
                               norm_obs_keys=ts.NORM_OBS_KEYS)

    n_act = env.action_space.shape[0]
    half_range = (env.action_space.high - env.action_space.low) / 2.0
    noise = NormalActionNoise(mean=np.zeros(n_act), sigma=0.1 * half_range)
    if resume_dir is None:
        model = TD3(env=env, learning_rate=args.learning_rate, buffer_size=args.buffer_size,
                    gamma=args.gamma, action_noise=noise, seed=args.seed, device=args.device,
                    tensorboard_log=str(log_dir / "tb"), **TD3_KWARGS)
    else:
        model = TD3.load(str(resume_ckpt), env=env, device=args.device, tensorboard_log=str(log_dir / "tb"))
        model.action_noise = noise
        buf = log_dir / "checkpoints" / f"td3_replay_buffer_{resume_steps}_steps.pkl"
        if buf.exists():
            model.load_replay_buffer(str(buf))
            print(f"[train_td3] Replay buffer restored ({model.replay_buffer.size():,} transitions)", flush=True)
    print(f"[train_td3] Model params: {sum(p.numel() for p in model.policy.parameters()):,}", flush=True)

    callbacks = [
        CheckpointCallback(save_freq=args.checkpoint_freq, save_path=str(log_dir / "checkpoints"),
                           name_prefix="td3", save_vecnormalize=True, save_replay_buffer=True),
        EvalCallback(eval_env, best_model_save_path=str(log_dir / "best_model"), log_path=str(log_dir / "eval"),
                     eval_freq=args.eval_freq, n_eval_episodes=10, deterministic=True, verbose=1,
                     callback_on_new_best=CallbackList([ts.SaveVecNormalizeOnNewBest(env, log_dir / "best_model")])),
        ts.CulturalMetricsCallback(log_dir=log_dir),
    ]
    remaining = args.timesteps - resume_steps
    if remaining <= 0:
        raise SystemExit(f"[train_td3] nothing to resume: {resume_steps:,} >= {args.timesteps:,}")
    print(f"[train_td3] Training for {remaining:,} steps"
          + (f" (resumed at {resume_steps:,})" if resume_dir else "") + "...", flush=True)
    model.learn(total_timesteps=remaining, callback=CallbackList(callbacks), tb_log_name=run_name,
                reset_num_timesteps=(resume_dir is None))
    model.save(str(log_dir / "final_model"))
    env.save(str(log_dir / "vec_normalize.pkl"))
    print(f"[train_td3] Done. Saved to {log_dir}", flush=True)


if __name__ == "__main__":
    train(parse_args())
