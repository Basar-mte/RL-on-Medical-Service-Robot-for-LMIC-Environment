"""Campaign recovery orchestrator (crash/power-out safe, idempotent).

Resumes every incomplete run in runs/ from its latest checkpoint via
train_sac.py --resume_from, launches never-started attention seeds fresh,
waits for all trainings, then runs the evaluation chain. Uses the same
subprocess.Popen pattern as train/run_all.py (which the campaign already
validated) — no shell quoting involved.

    .venv\\Scripts\\python.exe scripts\\resume_runs.py
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
RUNS = ROOT / "runs"
ADAPTIVE = ("cwn", "attention")
TIMESTEPS = "1000000"


def log(msg: str) -> None:
    line = f"[resume] {time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)


def main() -> None:
    procs = []  # (name, Popen, filehandle)

    def launch(name: str, cmd: list, logname: str) -> None:
        fh = open(RUNS / logname, "a", encoding="utf-8")
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
        procs.append((name, p, fh))
        log(f"launched {name} (pid {p.pid})")
        time.sleep(2)  # stagger CUDA init

    # --- Resume incomplete runs that have checkpoints ---
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"^(.+)_seed(\d+)_\d+$", d.name)
        if not m or (d / "final_model.zip").exists():
            continue
        agent, seed = m.group(1), m.group(2)
        ckpt_dir = d / "checkpoints"
        ckpts = list(ckpt_dir.glob("sac_[0-9]*_steps.zip")) if ckpt_dir.exists() else []
        if not ckpts:
            log(f"{d.name}: no checkpoint, skipping (needs fresh relaunch)")
            continue
        n_envs = "1" if agent in ADAPTIVE else "2"
        launch(f"resume {d.name}", [
            PY, "train/train_sac.py", "--agent", agent, "--seed", seed,
            "--timesteps", TIMESTEPS, "--n_envs", n_envs, "--no_camera",
            "--no_check_env", "--resume_from", str(d),
        ], f"{agent}_seed{seed}_train.log")

    # --- Fresh attention seeds that never started ---
    for seed in ("1", "2"):
        if not any(RUNS.glob(f"attention_seed{seed}_*/")):
            launch(f"fresh attention seed{seed}", [
                PY, "train/train_sac.py", "--agent", "attention", "--seed", seed,
                "--timesteps", TIMESTEPS, "--n_envs", "1", "--no_camera",
                "--no_check_env", "--log_dir", "runs",
            ], f"attention_seed{seed}_train.log")

    log(f"{len(procs)} trainings running; waiting...")
    failed = []
    for name, p, fh in procs:
        rc = p.wait()
        fh.close()
        log(f"{name}: {'OK' if rc == 0 else f'FAILED rc={rc}'}")
        if rc != 0:
            failed.append(name)

    if failed:
        log(f"WARNING: {len(failed)} runs failed: {failed} — eval proceeds with what exists")

    log("training phase done — starting evaluation chain")
    for cmd in (
        [PY, "eval/post_training.py", "--runs_dir", "runs", "--n_episodes", "100"],
        [PY, "eval/run_ablations.py", "--runs_dir", "runs", "--n_episodes", "50"],
        [PY, "eval/log_trajectories.py", "--runs_dir", "runs"],
    ):
        log("running: " + " ".join(cmd[1:]))
        subprocess.run(cmd, cwd=str(ROOT))
    log("ALL DONE")


if __name__ == "__main__":
    main()
