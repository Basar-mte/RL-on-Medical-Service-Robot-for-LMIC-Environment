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

SCRIPT = Path(__file__).parent / "train_sac.py"


def build_cmd(agent: str, seed: int, density: str, args: argparse.Namespace) -> list:
    cmd = [
        sys.executable, str(SCRIPT),
        "--agent", agent,
        "--seed", str(seed),
        "--density", density,
        "--timesteps", str(args.timesteps),
        "--log_dir", args.log_dir,
        "--device", args.device,
    ]
    if args.no_check_env:
        cmd.append("--no_check_env")
    return cmd


def run_all(args: argparse.Namespace) -> None:
    combos = list(product(AGENTS, SEEDS, DENSITIES))
    print(f"[run_all] {len(combos)} training runs scheduled.")

    if args.parallel:
        procs = []
        for agent, seed, density in combos:
            cmd = build_cmd(agent, seed, density, args)
            print(f"  [launch] {agent} seed={seed}")
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            procs.append((agent, seed, p))
        for agent, seed, p in procs:
            stdout, _ = p.communicate()
            rc = p.returncode
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            print(f"  [{status}] {agent} seed={seed}")
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
    p.add_argument("--parallel", action="store_true")
    p.add_argument("--no_check_env", action="store_true")
    args = p.parse_args()
    run_all(args)
