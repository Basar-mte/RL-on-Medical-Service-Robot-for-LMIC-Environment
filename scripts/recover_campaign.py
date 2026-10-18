"""Full idempotent campaign recovery orchestrator (power-outage safe).

Enumerates the complete 18-run matrix (6 agents x 3 seeds). For each:
  - final_model.zip present            -> skip (already complete)
  - a run dir with a checkpoint exists -> resume via --resume_from
  - no run dir                         -> launch fresh
Runs with a wave-scheduler concurrency cap, then fires the evaluation chain
exactly once when all 18 report final_model.zip. Safe to re-run at any time.

    .venv\\Scripts\\python.exe scripts\\recover_campaign.py
"""
from __future__ import annotations
import re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
RUNS = ROOT / "runs"
ADAPTIVE = ("cwn", "attention")
AGENTS = ["cwn", "fixed_equal", "nav_heavy", "vanilla", "he2025", "attention"]
SEEDS = [0, 1, 2]
TIMESTEPS = "1000000"
TOTAL_STEPS = 1_000_000
# GLOBAL live cap: counts runs already training from a PRIOR invocation
# (detected by a fresh _train.log), so this is restart-safe — re-running
# never double-launches a live run and never exceeds the cap in total.
# 12 measured comfortable (pre-outage peaked ~11 at 258 fps with GPU
# headroom to spare; 9 left the GPU at 77%).
MAX_CONCURRENT = 12
FRESH_SEC = 150   # a live SB3 run writes its log at least this often


def log(msg):
    print(f"[recover] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def find_dir(agent, seed):
    hits = sorted(RUNS.glob(f"{agent}_seed{seed}_*"))
    hits = [h for h in hits if h.is_dir()]
    return hits[-1] if hits else None


def latest_ckpt_steps(run_dir):
    cks = run_dir / "checkpoints"
    if not cks.exists():
        return None
    steps = [int(m.group(1)) for p in cks.glob("sac_*_steps.zip")
             if (m := re.match(r"sac_(\d+)_steps", p.name))]
    return max(steps) if steps else None


def is_complete(agent, seed):
    d = find_dir(agent, seed)
    return bool(d and (d / "final_model.zip").exists())


def remaining_steps(agent, seed):
    d = find_dir(agent, seed)
    ck = latest_ckpt_steps(d) if d else None
    return TOTAL_STEPS - (ck or 0)


def log_fresh(agent, seed):
    """True if this run's log was written within FRESH_SEC — i.e. a process
    (from this or a prior invocation) is actively training it right now."""
    logf = RUNS / f"{agent}_seed{seed}_train.log"
    return logf.exists() and (time.time() - logf.stat().st_mtime) < FRESH_SEC


def build_cmd(agent, seed):
    n_envs = "1" if agent in ADAPTIVE else "2"
    base = [PY, "train/train_sac.py", "--agent", agent, "--seed", str(seed),
            "--timesteps", TIMESTEPS, "--n_envs", n_envs, "--no_camera",
            "--no_check_env"]
    d = find_dir(agent, seed)
    if d and latest_ckpt_steps(d) is not None:
        return f"resume {d.name}@{latest_ckpt_steps(d)}", base + ["--resume_from", str(d)]
    return f"fresh {agent}_seed{seed}", base + ["--log_dir", "runs"]


def main():
    all_keys = [(a, s) for a in AGENTS for s in SEEDS]
    # Launch queue: incomplete runs NOT already training, longest-remaining
    # first so the fresh 1M runs (the makespan-determining long pole) start
    # earliest. Already-live runs are intentionally NOT enqueued — they are
    # only COUNTED toward the global cap, never relaunched (no duplicates).
    pending = sorted(
        [(a, s) for (a, s) in all_keys if not is_complete(a, s) and not log_fresh(a, s)],
        key=lambda k: -remaining_steps(*k))
    log(f"{sum(is_complete(a, s) for a, s in all_keys)}/18 complete; "
        f"{sum(log_fresh(a, s) and not is_complete(a, s) for a, s in all_keys)} already training; "
        f"{len(pending)} to launch (cap {MAX_CONCURRENT})")

    launched = {}   # (a,s) -> (Popen, fh)

    def live_count():
        mine = sum(1 for (p, _) in launched.values() if p.poll() is None)
        # orphans: live-log runs I did NOT launch and that aren't complete
        orphan = sum(1 for (a, s) in all_keys
                     if (a, s) not in launched and not is_complete(a, s) and log_fresh(a, s))
        return mine + orphan

    while pending or any(p.poll() is None for p, _ in launched.values()):
        while pending and live_count() < MAX_CONCURRENT:
            a, s = pending.pop(0)
            label, cmd = build_cmd(a, s)
            fh = open(RUNS / f"{a}_seed{s}_train.log", "a", encoding="utf-8")
            p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
            launched[(a, s)] = (p, fh)
            log(f"launched {label} (pid {p.pid}); live={live_count()}")
            time.sleep(5)  # let it register before recounting
        time.sleep(30)
        for (a, s), (p, fh) in list(launched.items()):
            if p.poll() is not None:
                fh.close()
                log(f"{a}_seed{s}: {'OK' if p.returncode == 0 else f'FAILED rc={p.returncode}'}")
                del launched[(a, s)]
        # Re-enqueue any orphan that died without completing (rare)
        for (a, s) in all_keys:
            if (not is_complete(a, s) and not log_fresh(a, s)
                    and (a, s) not in launched and (a, s) not in pending):
                log(f"re-enqueue stalled {a}_seed{s}")
                pending.append((a, s))
                pending.sort(key=lambda k: -remaining_steps(*k))

    done = sum(1 for a in AGENTS for s in SEEDS
               if (d := find_dir(a, s)) and (d / "final_model.zip").exists())
    log(f"training phase complete: {done}/18 have final_model.zip")
    if done < 18:
        log("WARNING: <18 complete — re-run this script to resume the rest; "
            "skipping eval")
        return

    log("starting evaluation chain")
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
