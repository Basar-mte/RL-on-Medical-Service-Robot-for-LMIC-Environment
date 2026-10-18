"""Stability pilot: does a FIXED SAC entropy coefficient prevent the late-training
collapse (alpha 0.001 -> 0.07..2, critic loss -> 1e3..1e4, success -> 0) seen in
16/18 simplified-experiment runs, 8/15 shield-weaning runs and the Tier-2 2M
extension? Healthy runs sit at alpha ~ 0.001 throughout.

Four detached runs, equal-weight agent, seed 0, 600k steps (collapse normally
shows by 300-500k), CLI flags only (no main-code change):
  wean_ec0.001 / wean_ec0.01     shield-weaning setup (WEAN_MODE=wean), Tier-2 map
  simple_ec0.001 / simple_ec0.01 simplified-experiment setup (10-20 peds, no beds)

    python experiments/stability_pilot/launch.py          # launch all four
    python experiments/stability_pilot/launch.py --status # per-100k training SR + alpha
"""
from __future__ import annotations
import csv, os, subprocess, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
RUNS = HERE / "runs"
JOBS = {
    "wean_ec0.001":   ("experiments/shield_weaning/run.py",    "configs/hospital_v4.yaml", "0.001", "wean"),
    "wean_ec0.01":    ("experiments/shield_weaning/run.py",    "configs/hospital_v4.yaml", "0.01",  "wean"),
    "simple_ec0.001": ("experiments/simple_experiment/run.py", "experiments/simple_experiment/hospital_simple.yaml", "0.001", None),
    "simple_ec0.01":  ("experiments/simple_experiment/run.py", "experiments/simple_experiment/hospital_simple.yaml", "0.01",  None),
}
DETACHED = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP


def launch():
    RUNS.mkdir(parents=True, exist_ok=True)
    for name, (wrapper, cfg, ec, mode) in JOBS.items():
        d = RUNS / name
        d.mkdir(exist_ok=True)
        env = {k: v for k, v in os.environ.items() if k != "WEAN_MODE"}
        env.update(PYTHONUTF8="1", MPLBACKEND="Agg")
        if mode:
            env["WEAN_MODE"] = mode
        cmd = [PY, wrapper, "train/train_sac.py", "--config", cfg, "--agent", "fixed_equal",
               "--seed", "0", "--timesteps", "600000", "--n_envs", "2", "--no_camera",
               "--no_check_env", "--ent_coef", ec, "--log_dir", f"experiments/stability_pilot/runs/{name}"]
        p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, creationflags=DETACHED,
                             stdout=open(d / "train.log", "a", encoding="utf-8"),
                             stderr=subprocess.STDOUT)
        (d / "pid").write_text(str(p.pid))
        print(f"launched {name} pid {p.pid}")


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
        for b in range(6):
            idx = [i for i, c in enumerate(cum) if b * 1e5 <= c < (b + 1) * 1e5]
            sr.append(f"{100*np.mean([rows[i][2] for i in idx]):3.0f}" if idx else "  -")
        log = (d / "train.log").read_text(encoding="utf-8", errors="ignore").splitlines()
        alpha = [l.split("|")[2].strip() for l in log if "ent_coef " in l and "loss" not in l]
        crit = [l.split("|")[2].strip() for l in log if "critic_loss" in l]
        steps = int(cum[-1]) if len(cum) else 0
        age = time.time() - (d / "train.log").stat().st_mtime
        print(f"{name:<16} {steps:>7,} steps  SR%/100k: {' '.join(sr)}  alpha(last)={alpha[-1] if alpha else '-'}  critic(last)={crit[-1] if crit else '-'}  log age {age:.0f}s")


if __name__ == "__main__":
    status() if "--status" in sys.argv else launch()
