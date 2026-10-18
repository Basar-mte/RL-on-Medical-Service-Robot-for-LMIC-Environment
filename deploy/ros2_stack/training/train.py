"""Train a SAC policy on the fast HospitalNav lite env, then save best_model.zip
+ vec_normalize.pkl in the exact format controller_node.py's Policy class loads
(SAC.load + VecNormalize.load), so the result deploys straight into Gazebo.
"""
import argparse
import os

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv

from hospitalnav_lite_env import HospitalNavLiteEnv

OUT_DIR = os.path.join(os.path.dirname(__file__), "models")


def evaluate(model, vec_env, n_episodes=30):
    vec_env.training = False
    reached = collided = timeout = 0
    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            done = bool(dones[0])
        info = infos[0]
        if info.get("collided"):
            collided += 1
        elif info.get("reached"):
            reached += 1
        else:
            timeout += 1
    return {"reached": reached, "collided": collided, "timeout": timeout, "n": n_episodes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=50_000)
    ap.add_argument("--n-envs", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    env = make_vec_env(HospitalNavLiteEnv, n_envs=args.n_envs,
                        vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = SAC("MultiInputPolicy", env, verbose=1,
                policy_kwargs={"net_arch": [128, 128]},
                learning_rate=3e-4, buffer_size=200_000,
                batch_size=256, train_freq=1, gradient_steps=1,
                learning_starts=2000)

    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    model.save(os.path.join(OUT_DIR, "best_model.zip"))
    env.save(os.path.join(OUT_DIR, "vec_normalize.pkl"))
    print("saved models to", OUT_DIR)

    eval_vec = make_vec_env(HospitalNavLiteEnv, n_envs=1)
    eval_vec = VecNormalize.load(os.path.join(OUT_DIR, "vec_normalize.pkl"), eval_vec)
    stats = evaluate(model, eval_vec, n_episodes=30)
    print("eval over", stats["n"], "episodes:", stats)


if __name__ == "__main__":
    main()
