"""
Phase 5 — Multi-seed training harness.

Launches all 6 agent types × 3 seeds in sequence (or parallel via subprocess).

Usage:
  python train/run_all.py --timesteps 1000000 --parallel
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

AGENTS = ["cwn", "fixed_equal", "nav_heavy", "vanilla", "he2025", "attention"]
SEEDS  = [0, 1, 2]
DENSITIES = ["medium"]

# Must stay in sync with train_sac.ADAPTIVE_WEIGHT_AGENTS: these agents hold
# the trainable weight network in-process and train_sac.py hard-exits on
# --n_envs > 1 for them, so a mixed-agent invocation with --n_envs 4 would
# otherwise silently fail exactly the two headline agents.
ADAPTIVE_WEIGHT_AGENTS = ("cwn", "attention")

SCRIPT = Path(__file__).parent / "train_sac.py"


def build_cmd(agent: str, seed: int, density: str, args: argparse.Namespace) -> list:
    n_envs = 1 if agent in ADAPTIVE_WEIGHT_AGENTS else args.n_envs
    cmd = [
        sys.executable, str(SCRIPT),
        "--agent", agent,
        "--seed", str(seed),
        "--density", density,
        "--timesteps", str(args.timesteps),
        "--log_dir", args.log_dir,
        "--device", args.device,
        "--n_envs", str(n_envs),
        "--buffer_size", str(args.buffer_size),
    ]
    if args.no_check_env:
        cmd.append("--no_check_env")
    if args.no_camera:
        cmd.append("--no_camera")
    return cmd


def run_all(args: argparse.Namespace) -> None:
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS
    combos = list(product(args.agents.split(",") if args.agents else AGENTS, seeds, DENSITIES))
    print(f"[run_all] {len(combos)} training runs scheduled.")

    if args.parallel:
        # Each child's stdout goes to its own log file: the previous
        # stdout=PIPE + sequential communicate() could deadlock once any
        # not-yet-reaped child filled its 64KB pipe buffer with SB3's
        # progress tables. --max_concurrent throttles to a wave size the
        # machine's RAM/GPU can absorb (0 = launch everything at once).
        pending = list(combos)
        running: list = []
        log_root = Path(args.log_dir)
        log_root.mkdir(parents=True, exist_ok=True)
        while pending or running:
            while pending and (args.max_concurrent <= 0
                               or len(running) < args.max_concurrent):
                agent, seed, density = pending.pop(0)
                cmd = build_cmd(agent, seed, density, args)
                log_path = log_root / f"{agent}_seed{seed}_train.log"
                fh = open(log_path, "w")
                print(f"  [launch] {agent} seed={seed} -> {log_path}")
                p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
                running.append((agent, seed, p, fh))
            time.sleep(10)
            still = []
            for agent, seed, p, fh in running:
                rc = p.poll()
                if rc is None:
                    still.append((agent, seed, p, fh))
                else:
                    fh.close()
                    status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                    print(f"  [{status}] {agent} seed={seed}", flush=True)
            running = still
    else:
        for idx, (agent, seed, density) in enumerate(combos, 1):
            cmd = build_cmd(agent, seed, density, args)
            print(f"\n[{idx}/{len(combos)}] {agent} seed={seed} density={density}")
            t0 = time.time()
            result = subprocess.run(cmd)
            elapsed = time.time() - t0
            status = "OK" if result.returncode == 0 else "FAILED"
            print(f"  [{status}] elapsed {elapsed/60:.1f} min")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--log_dir", default="runs")
    p.add_argument("--device", default="auto")
    p.add_argument("--parallel", action="store_true",
                    help="Run all combos concurrently. NOTE: each combo's "
                         "replay buffer holds full-resolution camera frames "
                         "(no optimize_memory_usage support for Dict obs) — "
                         "running several agents in parallel multiplies RAM "
                         "use accordingly; on memory-constrained machines, "
                         "prefer sequential (default) with a higher --n_envs "
                         "per run instead.")
    p.add_argument("--no_check_env", action="store_true")
    p.add_argument("--no_camera", action="store_true",
                    help="Passed through to train_sac.py --no_camera "
                         "(train vector-only, no synthetic camera).")
    p.add_argument("--max_concurrent", type=int, default=0,
                    help="With --parallel: cap on simultaneously running "
                         "training processes (0 = no cap). Size waves to "
                         "RAM: each run's replay buffer holds ~4.2 GB of "
                         "camera frames at --buffer_size 100000.")
    p.add_argument("--n_envs", type=int, default=1,
                    help="Passed through to train_sac.py --n_envs (SubprocVecEnv "
                         "parallelism within a single run).")
    p.add_argument("--buffer_size", type=int, default=100_000,
                    help="Passed through to train_sac.py --buffer_size.")
    p.add_argument("--agents", default=None,
                    help="Comma-separated subset of agents (default: all 6).")
    p.add_argument("--seeds", default=None,
                    help="Comma-separated subset of seeds (default: 0,1,2).")
    args = p.parse_args()
    run_all(args)
