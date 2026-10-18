"""Stability pilot 3 — confirm the fix on the PAPER's Tier-2 map.

Pilot 2 (simplified map, equal-weight, seed 0) showed that turning VecNormalize
reward normalization OFF (optionally with a 1M replay buffer) removes the
late-training critic divergence: deterministic SR 1.1% (defaults) -> 41-58%.
Here: configs/hospital_v4.yaml (45 peds, walled), the paper's protocol except
  --buffer_size 1000000  and  reward normalization OFF (patch_sac.py),
for equal-weight, culture-blind vanilla and the attention weighter, seeds 0-1,
1M steps (paper budget). Key questions: does SR rise over the paper's Tier-2
1M numbers (28.4 / 26.8 / 24.9) and does the cultural gap on the
prayer-shortcut task (attention 18%/CVR .089 vs vanilla 6%/.210) survive?

    python experiments/stability_pilot/launch3.py            # launch 6 runs
    python experiments/stability_pilot/launch3.py --status   # progress
    python experiments/stability_pilot/launch3.py --resume   # after a crash
"""
from __future__ import annotations
import csv, json, os, re, subprocess, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
# Pilot 4 reuses this launcher: PILOT_RUNS=runs4 PILOT_EXTRA="--learning_rate 1e-4"
RUNS = HERE / os.environ.get("PILOT_RUNS", "runs3")
EXTRA = os.environ.get("PILOT_EXTRA", "").split()
CFG = "configs/hospital_v4.yaml"
AGENTS = {"fixed_equal": "2", "vanilla": "2", "attention": "1"}  # agent -> n_envs (paper protocol)
SEEDS = [0, 1]
NEW_CONSOLE = 0x00000010 | 0x00000200  # own hidden console: survives the launcher's console closing


def _hidden():
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0
    return si


def _name(a, s):
    return f"{a}_s{s}"


def _run_dir(a, s):
    hits = sorted(p for p in (RUNS / _name(a, s)).glob(f"{a}_seed{s}_*") if p.is_dir())
    return hits[-1] if hits else None


def _latest_ckpt(rd):
    steps = [int(m.group(1)) for p in (rd / "checkpoints").glob("sac_*_steps.zip")
             if (m := re.match(r"sac_(\d+)_steps", p.name))] if rd and (rd / "checkpoints").exists() else []
    return max(steps) if steps else None


def launch(resume=False):
    RUNS.mkdir(parents=True, exist_ok=True)
    for a, n_envs in AGENTS.items():
        for s in SEEDS:
            d = RUNS / _name(a, s)
            d.mkdir(exist_ok=True)
            rd = _run_dir(a, s)
            if rd is not None and (rd / "final_model.zip").exists():
                print(f"skip {d.name}: complete"); continue
            env = {k: v for k, v in os.environ.items() if k not in ("WEAN_MODE",)}
            env.update(PYTHONUTF8="1", MPLBACKEND="Agg", PILOT_NORM_REWARD="0")
            cmd = [PY, str(HERE / "patch_sac.py"), "train/train_sac.py",
                   "--config", CFG, "--agent", a, "--seed", str(s), "--timesteps", "1000000",
                   "--n_envs", n_envs, "--no_camera", "--no_check_env", "--buffer_size", "1000000", *EXTRA]
            if resume and rd is not None and _latest_ckpt(rd) is not None:
                cmd += ["--resume_from", str(rd)]
            else:
                cmd += ["--log_dir", f"experiments/stability_pilot/{RUNS.name}/{d.name}"]
            p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, creationflags=NEW_CONSOLE, startupinfo=_hidden(),
                                 stdout=open(d / "train.log", "a", encoding="utf-8"), stderr=subprocess.STDOUT)
            (d / "pid").write_text(str(p.pid))
            print(f"{'resumed' if '--resume_from' in cmd else 'launched'} {d.name} pid {p.pid}")
            time.sleep(3)


def _flag(v):
    return 1.0 if str(v).strip().lower() in ("true", "1", "1.0") else 0.0


def _read(files):
    rows = []
    for mf in files:
        lines = open(mf, encoding="utf-8", errors="ignore").read().splitlines()
        t0 = json.loads(lines[0][1:])["t_start"]
        for r in csv.DictReader(lines[1:]):
            try:
                rows.append((t0 + float(r["t"]), int(float(r["l"])), _flag(r.get("task_complete")), _flag(r.get("collision"))))
            except Exception:
                pass
    return sorted(rows)


def curve(rd):
    """(cum_steps, success, collision) on the true step axis, across resumes."""
    groups = {"": sorted(rd.glob("monitor.monitor.csv")) + sorted(rd.glob("monitor_[0-9].monitor.csv"))}
    for f in rd.glob("monitor_resume*_*.monitor.csv"):
        groups.setdefault(re.search(r"resume(\d+)_", f.name).group(1), []).append(f)
    resumes = sorted(int(k) for k in groups if k)
    pts, seg_starts = [], [0] + resumes
    for i, key in enumerate([""] + [str(r) for r in resumes]):
        start, stop = seg_starts[i], (resumes[i] if i < len(resumes) else None)
        c = start
        for t, l, sc, k in _read(groups[key]):
            c += l
            if stop is not None and c > stop:
                break
            pts.append((c, sc, k))
    return pts


def status():
    for a in AGENTS:
        for s in SEEDS:
            d = RUNS / _name(a, s)
            rd = _run_dir(a, s)
            if rd is None:
                print(f"{d.name:<15} not started"); continue
            pts = curve(rd)
            bins = []
            for b in range(10):
                v = [(sc, k) for c, sc, k in pts if b * 1e5 <= c < (b + 1) * 1e5]
                bins.append(f"{100*np.mean([x for x, _ in v]):3.0f}" if v else "  -")
            log = (d / "train.log").read_text(encoding="utf-8", errors="ignore").splitlines()
            crit = [l.split("|")[2].strip() for l in log if "critic_loss" in l]
            steps = pts[-1][0] if pts else 0
            age = time.time() - (d / "train.log").stat().st_mtime
            done = "DONE" if (rd / "final_model.zip").exists() else ("LIVE" if age < 300 else "STALLED")
            print(f"{d.name:<15} {steps:>9,}  SR%/100k: {' '.join(bins)}  critic={crit[-1] if crit else '-'}  {done}")


if __name__ == "__main__":
    if "--status" in sys.argv:
        status()
    else:
        launch(resume="--resume" in sys.argv)
