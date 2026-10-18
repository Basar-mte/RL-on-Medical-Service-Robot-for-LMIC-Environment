"""Full simplified-experiment campaign: 6 agents x 3 seeds x 1M steps, then eval.

Clone of scripts/campaign_v4.py (same wave scheduler, cap 12, idempotent:
re-running skips finished runs, resumes from checkpoints, never double-launches
live runs). Differences: every command goes through run.py (pedestrian patch),
the simplified config, outputs under experiments/simple_experiment/runs/, and
summarize.py instead of post_training.py (which would re-render paper figures).

Eval chain after 18/18:
  1. evaluate.py 500 eps at medium (10-20 peds)     -> eval_results.json
  2. evaluate.py 200 eps at low (10) and high (20)  -> eval_low/high.json
  3. summarize.py                                    -> runs/results/*.json
  4. eval_shielded_simple.py (500 eps)               -> runs/results/shielded.json
  5. log_trajectories.py                             -> runs/results/trajectories/

    .venv\\Scripts\\python.exe experiments\\simple_experiment\\campaign.py
"""
from __future__ import annotations
import os, re, shutil, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
RUN = str(HERE / "run.py")
RUNS_NAME = "experiments/simple_experiment/runs"
RUNS = ROOT / RUNS_NAME
CONFIG_REL = "experiments/simple_experiment/hospital_simple.yaml"
ADAPTIVE = ("cwn", "attention")
AGENTS = ["cwn", "fixed_equal", "nav_heavy", "vanilla", "he2025", "attention"]
SEEDS = [0, 1, 2]
TIMESTEPS = "1000000"
TOTAL_STEPS = 1_000_000
MAX_CONCURRENT = 12   # measured ceiling on this machine — do NOT raise
FRESH_SEC = 300       # a live run touches its train log or monitor CSV at least this often
CKPT_FREQ = "50000"   # model + replay buffer + vecnorm; caps a crash's loss at 50k steps
DONE_MARKER = RUNS / "ALL_DONE"   # watchdog.ps1 stops relaunching once this exists
ENV = {**os.environ, "PYTHONUTF8": "1", "MPLBACKEND": "Agg"}


def log(msg):
    print(f"[simple] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def find_dir(agent, seed):
    hits = sorted(h for h in RUNS.glob(f"{agent}_seed{seed}_*") if h.is_dir())
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
    files = [RUNS / f"{agent}_seed{seed}_train.log"]
    d = find_dir(agent, seed)
    if d:
        files += list(d.glob("monitor*.monitor.csv"))
    mt = max((f.stat().st_mtime for f in files if f.exists()), default=0)
    return time.time() - mt < FRESH_SEC


def kill_strays(agent, seed):
    """Kill any leftover trainer for this run (hung, or orphaned by a dead
    orchestrator) before relaunching, so two processes never share a run dir."""
    flt = (f"$_.CommandLine -like '*simple_experiment*train_sac.py*' -and "
           f"$_.CommandLine -like '*--agent {agent} *' -and "
           f"$_.CommandLine -like '*--seed {seed} *'")
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                    f"Where-Object {{ {flt} }} | "
                    "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"],
                   capture_output=True)


def build_cmd(agent, seed):
    n_envs = "1" if agent in ADAPTIVE else "2"
    base = [PY, RUN, "train/train_sac.py", "--config", CONFIG_REL,
            "--agent", agent, "--seed", str(seed),
            "--timesteps", TIMESTEPS, "--n_envs", n_envs, "--no_camera",
            "--no_check_env", "--checkpoint_freq", CKPT_FREQ]
    d = find_dir(agent, seed)
    if d and latest_ckpt_steps(d) is not None:
        return f"resume {d.name}@{latest_ckpt_steps(d)}", base + ["--resume_from", str(d)]
    return f"fresh {agent}_seed{seed}", base + ["--log_dir", RUNS_NAME]


def evaluate(model_dir, density, n_eps, output):
    subprocess.run([PY, RUN, "eval/evaluate.py", "--model_dir", str(model_dir),
                    "--config", CONFIG_REL, "--n_episodes", str(n_eps),
                    "--density", density, "--output", output],
                   cwd=str(ROOT), env=ENV)


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    if not (RUNS / "config_used.yaml").exists():
        shutil.copy(ROOT / CONFIG_REL, RUNS / "config_used.yaml")

    all_keys = [(a, s) for a in AGENTS for s in SEEDS]
    pending = sorted(
        [(a, s) for (a, s) in all_keys if not is_complete(a, s) and not log_fresh(a, s)],
        key=lambda k: -remaining_steps(*k))
    log(f"{sum(is_complete(a, s) for a, s in all_keys)}/18 complete; "
        f"{len(pending)} to launch (cap {MAX_CONCURRENT})")

    launched = {}

    def live_count():
        mine = sum(1 for (p, _) in launched.values() if p.poll() is None)
        orphan = sum(1 for (a, s) in all_keys
                     if (a, s) not in launched and not is_complete(a, s) and log_fresh(a, s))
        return mine + orphan

    # live_count() includes orphans left by a previous orchestrator, so a
    # restarted orchestrator keeps supervising them instead of exiting early.
    while pending or live_count() > 0:
        while pending and live_count() < MAX_CONCURRENT:
            a, s = pending.pop(0)
            kill_strays(a, s)
            label, cmd = build_cmd(a, s)
            fh = open(RUNS / f"{a}_seed{s}_train.log", "a", encoding="utf-8")
            p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh,
                                 stderr=subprocess.STDOUT, env=ENV)
            launched[(a, s)] = (p, fh)
            log(f"launched {label} (pid {p.pid}); live={live_count()}")
            time.sleep(5)
        time.sleep(30)
        for (a, s), (p, fh) in list(launched.items()):
            if p.poll() is not None:
                fh.close()
                log(f"{a}_seed{s}: {'OK' if p.returncode == 0 else f'FAILED rc={p.returncode}'}")
                del launched[(a, s)]
        for (a, s) in all_keys:
            if (not is_complete(a, s) and not log_fresh(a, s)
                    and (a, s) not in launched and (a, s) not in pending):
                log(f"re-enqueue stalled {a}_seed{s}")
                pending.append((a, s))
                pending.sort(key=lambda k: -remaining_steps(*k))

    done = sum(is_complete(a, s) for a, s in all_keys)
    log(f"training phase complete: {done}/18 have final_model.zip")
    if done < 18:
        log("WARNING: <18 complete — re-run this script to resume the rest; skipping eval")
        return

    log("evaluating: 500 eps medium + 200 eps low/high per run")
    for a in AGENTS:
        for s in SEEDS:
            d = find_dir(a, s)
            model_dir = d / "best_model" if (d / "best_model").exists() else d
            for density, n, out in (("medium", 500, "eval_results.json"),
                                    ("low", 200, "eval_low.json"),
                                    ("high", 200, "eval_high.json")):
                if not (d / out).exists():
                    log(f"  eval {d.name} {density}")
                    evaluate(model_dir, density, n, out)

    for cmd in (
        [PY, str(HERE / "summarize.py")],
        [PY, str(HERE / "eval_shielded_simple.py")],
        [PY, RUN, "eval/log_trajectories.py", "--runs_dir", RUNS_NAME,
         "--config", CONFIG_REL, "--output_dir", f"{RUNS_NAME}/results/trajectories"],
    ):
        log("running: " + " ".join(Path(c).name if i < 2 else c for i, c in enumerate(cmd)))
        subprocess.run(cmd, cwd=str(ROOT), env=ENV)
    DONE_MARKER.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    log("ALL DONE")


if __name__ == "__main__":
    main()
