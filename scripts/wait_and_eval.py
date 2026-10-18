"""Wait until all 18 runs have final_model.zip, then run the evaluation chain
once. Launcher-agnostic: it only watches the filesystem, so it works no
matter which process (or a since-dead orchestrator) trained the runs.
Idempotent — safe to re-run; skips straight to eval if already 18/18.

    .venv\\Scripts\\python.exe scripts\\wait_and_eval.py
"""
from __future__ import annotations
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
RUNS = ROOT / "runs"
AGENTS = ["cwn", "fixed_equal", "nav_heavy", "vanilla", "he2025", "attention"]
SEEDS = [0, 1, 2]


def log(m):
    print(f"[wait_eval] {time.strftime('%H:%M:%S')} {m}", flush=True)


def complete_count():
    n = 0
    for a in AGENTS:
        for s in SEEDS:
            hits = [d for d in RUNS.glob(f"{a}_seed{s}_*")
                    if d.is_dir() and (d / "final_model.zip").exists()]
            if hits:
                n += 1
    return n


def main():
    while True:
        n = complete_count()
        log(f"{n}/18 complete")
        if n >= 18:
            break
        time.sleep(120)

    log("all 18 complete — starting evaluation chain")
    for cmd in (
        [PY, "eval/post_training.py", "--runs_dir", "runs", "--n_episodes", "100"],
        [PY, "eval/run_ablations.py", "--runs_dir", "runs", "--n_episodes", "50"],
        [PY, "eval/log_trajectories.py", "--runs_dir", "runs"],
    ):
        log("running: " + " ".join(cmd[1:]))
        subprocess.run(cmd, cwd=str(ROOT))
    log("ALL DONE — results in runs/results/table.json")


if __name__ == "__main__":
    main()
