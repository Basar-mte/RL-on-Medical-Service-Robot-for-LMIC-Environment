"""Stability pilot 2 — after fixed alpha FAILED to stop the collapse (pilot 1),
test the remaining suspects on the simplified setup (plain SAC code path,
reliable collapse by 400-600k with defaults):

  buf1M            replay buffer 1,000,000 instead of 100,000 (CLI flag)
  noNormRew        VecNormalize reward normalization OFF (in-process patch)
  lr1e-4           learning rate 1e-4 instead of 3e-4 (CLI flag)
  buf1M_noNormRew  both

All: fixed_equal, seed 0, 800k steps, auto alpha (as in the paper), n_envs 2.
Outputs under experiments/stability_pilot/runs2/<name>/. No main-code change:
the patch below is applied in-process via sitecustomize-style env var read by
patch_sac.py, which then hands off to experiments/simple_experiment/run.py.

    python experiments/stability_pilot/launch2.py            # launch
    python experiments/stability_pilot/launch2.py --status   # progress
"""
from __future__ import annotations
import csv, os, subprocess, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
RUNS = HERE / "runs2"
CFG = "experiments/simple_experiment/hospital_simple.yaml"
JOBS = {  # name -> (extra CLI args, env)
    "buf1M":           (["--buffer_size", "1000000"], {}),
    "noNormRew":       ([], {"PILOT_NORM_REWARD": "0"}),
    "lr1e-4":          (["--learning_rate", "1e-4"], {}),
    "buf1M_noNormRew": (["--buffer_size", "1000000"], {"PILOT_NORM_REWARD": "0"}),
}
# Each run gets its OWN hidden console (CREATE_NEW_CONSOLE + SW_HIDE), like
# PowerShell's Start-Process -WindowStyle Hidden. The first pilot-2 launch used
# DETACHED_PROCESS and all four runs (and their SubprocVecEnv workers) died
# together with "forrtl: error (200): window-CLOSE event" when the launching
# session's console closed.
NEW_CONSOLE = 0x00000010 | 0x00000200  # CREATE_NEW_CONSOLE | CREATE_NEW_PROCESS_GROUP


def _hidden():
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0  # SW_HIDE
    return si


def _run_dir(name):
    hits = sorted(p for p in (RUNS / name).glob("fixed_equal_seed0_*") if p.is_dir())
    return hits[-1] if hits else None


def launch(only=None, resume=False):
    RUNS.mkdir(parents=True, exist_ok=True)
    for name, (extra, envx) in JOBS.items():
        if only and name not in only:
            continue
        d = RUNS / name
        d.mkdir(exist_ok=True)
        env = {k: v for k, v in os.environ.items() if k not in ("WEAN_MODE", "PILOT_NORM_REWARD")}
        env.update(PYTHONUTF8="1", MPLBACKEND="Agg", **envx)
        cmd = [PY, str(HERE / "patch_sac.py"), "experiments/simple_experiment/run.py", "train/train_sac.py",
               "--config", CFG, "--agent", "fixed_equal", "--seed", "0", "--timesteps", "800000",
               "--n_envs", "2", "--no_camera", "--no_check_env", *extra]
        rd = _run_dir(name)
        if resume and rd is not None:
            cmd += ["--resume_from", str(rd)]
        else:
            cmd += ["--log_dir", f"experiments/stability_pilot/runs2/{name}"]
        p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, creationflags=NEW_CONSOLE, startupinfo=_hidden(),
                             stdout=open(d / "train.log", "a", encoding="utf-8"), stderr=subprocess.STDOUT)
        (d / "pid").write_text(str(p.pid))
        print(f"{'resumed' if resume and rd else 'launched'} {name} pid {p.pid}")


def status():
    def flag(v): return 1.0 if str(v).strip().lower() in ("true", "1", "1.0") else 0.0
    for name in JOBS:
        d = RUNS / name
        rd = [p for p in d.glob("fixed_equal_seed0_*") if p.is_dir()]
        if not rd:
            print(f"{name:<16} not started"); continue
        rows = []
        for mf in rd[-1].glob("monitor*.monitor.csv"):
            body = [l for l in open(mf, encoding="utf-8", errors="ignore").read().splitlines() if l and not l.startswith("#")]
            for r in csv.DictReader(body):
                try: rows.append((float(r["t"]), int(float(r["l"])), flag(r.get("task_complete")), flag(r.get("collision"))))
                except Exception: pass
        rows.sort(); cum = np.cumsum([r[1] for r in rows]) if rows else np.array([])
        sr = []
        for b in range(8):
            idx = [i for i, c in enumerate(cum) if b * 1e5 <= c < (b + 1) * 1e5]
            sr.append(f"{100*np.mean([rows[i][2] for i in idx]):3.0f}" if idx else "  -")
        log = (d / "train.log").read_text(encoding="utf-8", errors="ignore").splitlines()
        alpha = [l.split("|")[2].strip() for l in log if "ent_coef " in l and "loss" not in l]
        crit = [l.split("|")[2].strip() for l in log if "critic_loss" in l]
        steps = int(cum[-1]) if len(cum) else 0
        age = time.time() - (d / "train.log").stat().st_mtime
        print(f"{name:<16} {steps:>7,} steps  SR%/100k: {' '.join(sr)}  alpha={alpha[-1] if alpha else '-'}  critic={crit[-1] if crit else '-'}  log age {age:.0f}s")


if __name__ == "__main__":
    if "--status" in sys.argv:
        status()
    else:
        only = None
        if "--only" in sys.argv:
            only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
        launch(only, resume="--resume" in sys.argv)
