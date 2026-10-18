"""
Interactive keyboard control script for LMICHospitalNav-v1.

Arrow keys: control robot (Ackermann steering — heading only changes while
moving, matching the real chassis; no in-place pivot)
  ↑  —  move forward (v = 0.6)
  ←  —  steer left   (δ = +MAX_STEER)
  →  —  steer right  (δ = -MAX_STEER)
  ↓  —  stop (v = 0, δ = 0)
  q / ESC — quit

Usage:
  python scripts/play.py [--task TASK] [--density medium]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.lmic_hospital_env import LMICHospitalNavEnv, MAX_STEER


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="STAT_medicine")
    p.add_argument("--density", default="medium")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import pygame
    except ImportError:
        print("pygame required for play.py.  pip install pygame")
        sys.exit(1)

    env = LMICHospitalNavEnv(render_mode="human", seed=0)
    obs, info = env.reset(seed=0, options={"task": args.task, "density": args.density})

    print(f"[play] Task: {args.task}  |  Density: {args.density}")
    print("[play] Arrow keys to drive. Q/ESC to quit.")

    total_reward = 0.0
    steps = 0
    running = True

    while running:
        # Handle events
        v, steering_angle = 0.0, 0.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            v = 0.6
        if keys[pygame.K_DOWN]:
            v = 0.0
        if keys[pygame.K_LEFT]:
            steering_angle = +MAX_STEER
        if keys[pygame.K_RIGHT]:
            steering_angle = -MAX_STEER

        action = np.array([v, steering_angle], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            status = "SUCCESS" if info.get("task_complete") else "FAILED"
            print(f"\n[play] Episode {status} — steps={steps}  "
                  f"reward={total_reward:.2f}  CVR={info['cultural_violations']/max(steps,1):.3f}")
            obs, info = env.reset(seed=steps, options={"task": args.task, "density": args.density})
            total_reward = 0.0
            steps = 0

    env.close()


if __name__ == "__main__":
    main()
