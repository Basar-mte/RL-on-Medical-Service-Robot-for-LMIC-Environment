"""Shield-weaning campaign on Tier 2 (configs/hospital_v4.yaml, 45 peds, 1M).

15 runs, one wave (cap 12 + 3):
  wean   x {fixed_equal, attention, vanilla} x seeds 0-2   (the method)
  fixed  x fixed_equal x seeds 0-2                         (no anneal ablation)
  naive  x fixed_equal x seeds 0-2                         (old in-loop design)
Baselines are the existing 1M Tier-2 runs (runs_v4/results_1M/): no retraining.

Waits for experiments/simple_experiment/runs/ALL_DONE before launching (the
machine is saturated by that campaign). Same idempotent scheduler as the other
campaigns: skips finished runs, resumes from checkpoints, never double-launches
a live run, kills strays before relaunch. Then the eval chain:
  evaluate.py 500 eps (unshielded)  -> <mode>/<run>/eval_results.json
  summarize.py                      -> results/ (per-mode table.json + comparison)
  eval_shielded (500 eps, shield at inference) per mode -> <mode>/results/shielded.json

    .venv\\Scripts\\python.exe experiments\\shield_weaning\\campaign.py
"""
from __future__ import annotations
import os, re, shutil, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
RUN = str(HERE / "run.py")
RUNS = HERE / "runs"
CONFIG_REL = "configs/hospital_v4.yaml"
WAIT_FOR = ROOT / "experiments" / "simple_experiment" / "runs" / "ALL_DONE"
MATRIX = {  # mode -> agents
    "wean": ["fixed_equal", "attention", "vanilla"],
    "fixed": ["fixed_equal"],
    "naive": ["fixed_equal"],
}
SEEDS = [0, 1, 2]
ADAPTIVE = ("cwn", "attention")
TIMESTEPS = "1000000"
TOTAL_STEPS = 1_000_000
MAX_CONCURRENT = 12
FRESH_SEC = 300
CKPT_FREQ = "50000"
DONE_MARKER = RUNS / "ALL_DONE"
BASE_ENV = {**os.environ, "PYTHONUTF8": "1", "MPLBACKEND": "Agg"}
KEYS = [(m, a, s) for m, ags in MATRIX.items() for a in ags for s in SEEDS]


def log(msg):
    print(f"[wean] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def mode_dir(mode):
    return RUNS / mode


def find_dir(mode, agent, seed):
    hits = sorted(h for h in mode_dir(mode).glob(f"{agent}_seed{seed}_*") if h.is_dir())
    return hits[-1] if hits else None


def latest_ckpt_steps(run_dir):
    cks = run_dir / "checkpoints"
    if not cks.exists():
        return None
    steps = [int(m.group(1)) for p in cks.glob("sac_*_steps.zip")
             if (m := re.match(r"sac_(\d+)_steps", p.name))]
    return max(steps) if steps else None


def is_complete(k):
    d = find_dir(*k)
    return bool(d and (d / "final_model.zip").exists())


def remaining_steps(k):
    d = find_dir(*k)
    ck = latest_ckpt_steps(d) if d else None
    return TOTAL_STEPS - (ck or 0)


def log_fresh(k):
    mode, agent, seed = k
    files = [mode_dir(mode) / f"{agent}_seed{seed}_train.log"]
    d = find_dir(*k)
    if d:
        files += list(d.glob("monitor*.monitor.csv"))
    mt = max((f.stat().st_mtime for f in files if f.exists()), default=0)
    return time.time() - mt < FRESH_SEC


def kill_strays(k):
    mode, agent, seed = k
    flt = (f"$_.CommandLine -like '*shield_weaning*train_sac.py*' -and "
           f"$_.CommandLine -like '*--log_dir *runs/{mode}*' -and "
           f"$_.CommandLine -like '*--agent {agent} *' -and "
           f"$_.CommandLine -like '*--seed {seed} *'")
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                    f"Where-Object {{ {flt} }} | "
                    "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"],
                   capture_output=True)


def build_cmd(k):
    mode, agent, seed = k
    n_envs = "1" if agent in ADAPTIVE else "2"
    log_dir = f"experiments/shield_weaning/runs/{mode}"
    base = [PY, RUN, "train/train_sac.py", "--config", CONFIG_REL,
            "--agent", agent, "--seed", str(seed),
            "--timesteps", TIMESTEPS, "--n_envs", n_envs, "--no_camera",
            "--no_check_env", "--checkpoint_freq", CKPT_FREQ, "--log_dir", log_dir]
    d = find_dir(*k)
    if d and latest_ckpt_steps(d) is not None:
        return f"resume {mode}/{d.name}@{latest_ckpt_steps(d)}", base + ["--resume_from", str(d)]
    return f"fresh {mode}/{agent}_seed{seed}", base


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    for m in MATRIX:
        mode_dir(m).mkdir(exist_ok=True)
    if not (RUNS / "config_used.yaml").exists():
        shutil.copy(ROOT / CONFIG_REL, RUNS / "config_used.yaml")

    if not WAIT_FOR.exists() and not any(is_complete(k) or log_fresh(k) for k in KEYS):
        log(f"waiting for {WAIT_FOR.relative_to(ROOT)} (simple experiment) before launching")
        while not WAIT_FOR.exists():
            time.sleep(120)
        log("simple experiment finished; launching")

    pending = sorted([k for k in KEYS if not is_complete(k) and not log_fresh(k)],
                     key=lambda k: -remaining_steps(k))
    log(f"{sum(is_complete(k) for k in KEYS)}/{len(KEYS)} complete; "
        f"{len(pending)} to launch (cap {MAX_CONCURRENT})")
    launched = {}

    def live_count():
        mine = sum(1 for (p, _) in launched.values() if p.poll() is None)
        orphan = sum(1 for k in KEYS if k not in launched and not is_complete(k) and log_fresh(k))
        return mine + orphan

    while pending or live_count() > 0:
        while pending and live_count() < MAX_CONCURRENT:
            k = pending.pop(0)
            kill_strays(k)
            label, cmd = build_cmd(k)
            mode, agent, seed = k
            fh = open(mode_dir(mode) / f"{agent}_seed{seed}_train.log", "a", encoding="utf-8")
            p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                                 env={**BASE_ENV, "WEAN_MODE": mode})
            launched[k] = (p, fh)
            log(f"launched {label} (pid {p.pid}); live={live_count()}")
            time.sleep(5)
        time.sleep(30)
        for k, (p, fh) in list(launched.items()):
            if p.poll() is not None:
                fh.close()
                log(f"{'/'.join(map(str, k))}: {'OK' if p.returncode == 0 else f'FAILED rc={p.returncode}'}")
                del launched[k]
        for k in KEYS:
            if not is_complete(k) and not log_fresh(k) and k not in launched and k not in pending:
                log(f"re-enqueue stalled {'/'.join(map(str, k))}")
                pending.append(k)
                pending.sort(key=lambda k: -remaining_steps(k))

    done = sum(is_complete(k) for k in KEYS)
    log(f"training phase complete: {done}/{len(KEYS)} have final_model.zip")
    if done < len(KEYS):
        log("WARNING: incomplete — re-run this script to resume the rest; skipping eval")
        return

    log("evaluating: 500 eps unshielded per run (through run.py)")
    for k in KEYS:
        d = find_dir(*k)
        if (d / "eval_results.json").exists():
            continue
        model_dir = d / "best_model" if (d / "best_model").exists() else d
        log(f"  eval {k[0]}/{d.name}")
        subprocess.run([PY, RUN, "eval/evaluate.py", "--model_dir", str(model_dir),
                        "--config", CONFIG_REL, "--n_episodes", "500",
                        "--output", "eval_results.json"],
                       cwd=str(ROOT), env={**BASE_ENV, "WEAN_MODE": k[0]})

    log("running: summarize.py")
    subprocess.run([PY, str(HERE / "summarize.py")], cwd=str(ROOT), env=BASE_ENV)
    for mode, agents in MATRIX.items():
        log(f"running: shielded eval ({mode}: {', '.join(agents)})")
        subprocess.run([PY, str(HERE / "eval_shielded_wean.py"), *agents],
                       cwd=str(ROOT), env={**BASE_ENV, "WEAN_MODE": mode})
    log("running: summarize.py (with shield)")
    subprocess.run([PY, str(HERE / "summarize.py")], cwd=str(ROOT), env=BASE_ENV)
    DONE_MARKER.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    log("ALL DONE")


if __name__ == "__main__":
    main()
