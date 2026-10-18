"""Live dashboard for the simplified Tier-2 experiment — read-only, safe anytime.

Aggregates, per run: env steps done, throughput, ETA, rolling success /
collision rates (last 100 training episodes), best & latest eval SR,
newest checkpoint, and LIVE/STALLED/QUEUED/DONE status; plus campaign
totals, GPU load, and the orchestrator's latest message.

    .venv\\Scripts\\python.exe experiments\\simple_experiment\\dashboard.py           # one snapshot
    .venv\\Scripts\\python.exe experiments\\simple_experiment\\dashboard.py --watch   # refresh every 30 s
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = Path(__file__).resolve().parent / "runs"
AGENTS = ["cwn", "fixed_equal", "nav_heavy", "vanilla", "he2025", "attention"]
SEEDS = [0, 1, 2]
TOTAL = 1_000_000
FRESH_SEC = 300
ROLL_N = 100


def find_dir(agent, seed):
    hits = [h for h in sorted(RUNS.glob(f"{agent}_seed{seed}_*")) if h.is_dir()]
    return hits[-1] if hits else None


def _flag(v):
    """Monitor CSV stores info_keywords bools as 'True'/'False' strings."""
    return 1.0 if str(v).strip().lower() in ("true", "1", "1.0") else 0.0


def read_monitors(run_dir):
    """Return (total_steps, wall_seconds, episode rows [(t, success, collision)])."""
    steps, wall, rows = 0, 0.0, []
    for mf in run_dir.glob("monitor*.monitor.csv"):
        try:
            with open(mf, encoding="utf-8", errors="ignore") as f:
                lines = f.read().splitlines()
        except OSError:
            continue
        body = [ln for ln in lines if ln and not ln.startswith("#")]
        if len(body) < 2:
            continue
        rdr = csv.DictReader(body)
        f_steps, f_wall = 0, 0.0
        for row in rdr:
            try:
                l = int(float(row["l"])); t = float(row["t"])
                f_steps += l; f_wall = max(f_wall, t)
                rows.append((t, _flag(row.get("task_complete")),
                             _flag(row.get("collision"))))
            except (KeyError, ValueError, TypeError):
                continue
        steps += f_steps
        wall = max(wall, f_wall)
    rows.sort(key=lambda r: r[0])
    return steps, wall, rows


def eval_sr(run_dir):
    """(best eval SR, latest eval SR) from SB3 EvalCallback's evaluations.npz."""
    f = run_dir / "eval" / "evaluations.npz"
    if not f.exists():
        return None, None
    try:
        z = np.load(f, allow_pickle=True)
        if "successes" not in z.files or not len(z["successes"]):
            return None, None
        sr = z["successes"].mean(axis=1)
        return float(sr.max()), float(sr[-1])
    except Exception:
        return None, None


def latest_ckpt(run_dir):
    cks = [int(m.group(1)) for p in (run_dir / "checkpoints").glob("sac_*_steps.zip")
           if (m := re.match(r"sac_(\d+)_steps", p.name))] if (run_dir / "checkpoints").exists() else []
    return max(cks) if cks else 0


def gpu_line():
    """GPU health: load, VRAM, temperature, power, throttle state."""
    q = ("utilization.gpu,memory.used,memory.total,temperature.gpu,"
         "power.draw,power.limit,clocks_throttle_reasons.active")
    for query in (q, q.rsplit(",", 1)[0]):  # retry without throttle field
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            if out.returncode != 0:
                continue
            v = [s.strip() for s in out.stdout.strip().split(",")]
            u, m, mt, t, pd, pl = v[:6]
            temp = int(float(t))
            warn = "  ** GPU HOT **" if temp >= 83 else ""
            thr = ""
            if len(v) > 6:
                thr = " | no throttle" if int(v[6], 16) == 0 else " | ** THROTTLING **"
            return (f"GPU {u}% | VRAM {float(m)/1024:.1f}/{float(mt)/1024:.1f} GB | "
                    f"{temp}C | {float(pd):.0f}/{float(pl):.0f} W{thr}{warn}")
        except Exception:
            continue
    return "GPU n/a"


def cpu_line():
    """CPU load + RAM via CIM (no psutil dependency) + disk free."""
    parts = []
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "$c=(Get-CimInstance Win32_Processor|Measure-Object LoadPercentage -Average).Average;"
             "$o=Get-CimInstance Win32_OperatingSystem;"
             "'{0}|{1:N1}|{2:N1}' -f $c,(($o.TotalVisibleMemorySize-$o.FreePhysicalMemory)/1MB),"
             "($o.TotalVisibleMemorySize/1MB)"],
            capture_output=True, text=True, timeout=20)
        c, used, total = out.stdout.strip().split("|")
        ram_warn = "  ** RAM LOW **" if float(used) / float(total) > 0.92 else ""
        parts.append(f"CPU {float(c):.0f}% | RAM {used}/{total} GB{ram_warn}")
    except Exception:
        parts.append("CPU n/a")
    try:
        import shutil as _sh
        free_gb = _sh.disk_usage(str(ROOT)).free / 1e9
        parts.append(f"disk {free_gb:.0f} GB free" + ("  ** DISK LOW **" if free_gb < 30 else ""))
    except Exception:
        pass
    return " | ".join(parts)


def orchestrator_alive():
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object "
             "{ $_.CommandLine -like '*simple_experiment*campaign.py*' } | Measure-Object).Count"],
            capture_output=True, text=True, timeout=20)
        return int(out.stdout.strip() or 0) > 0
    except Exception:
        return False


def snapshot():
    now = time.time()
    header = (f"{'run':<22}{'steps':>9}{'%':>5}{'fps':>6}{'ETA_h':>7}"
              f"{'rollSR':>8}{'coll%':>7}{'evalSR(best/last)':>19}{'ckpt':>7}  status")
    lines = [header, "-" * len(header)]
    tot_steps = tot_fps = done_n = live_n = 0
    for agent in AGENTS:
        for seed in SEEDS:
            name = f"{agent}_s{seed}"
            d = find_dir(agent, seed)
            if d is None:
                lines.append(f"{name:<22}{'-':>9}{'-':>5}{'-':>6}{'-':>7}"
                             f"{'-':>8}{'-':>7}{'-':>19}{'-':>7}  QUEUED")
                continue
            steps, wall, rows = read_monitors(d)
            fps = steps / wall if wall > 1 else 0.0
            tot_steps += steps; tot_fps += fps
            tail = rows[-ROLL_N:]
            roll_sr = 100 * np.mean([r[1] for r in tail]) if tail else float("nan")
            roll_co = 100 * np.mean([r[2] for r in tail]) if tail else float("nan")
            best, last = eval_sr(d)
            ck = latest_ckpt(d)
            files = [RUNS / f"{agent}_seed{seed}_train.log", *d.glob("monitor*.monitor.csv")]
            fresh = now - max((f.stat().st_mtime for f in files if f.exists()), default=0) < FRESH_SEC
            if (d / "final_model.zip").exists():
                status, done_n = "DONE", done_n + 1
            elif fresh:
                status = "LIVE"; live_n += 1
            else:
                status = "STALLED"
            eta = (TOTAL - steps) / fps / 3600 if fps > 0 and status == "LIVE" else float("nan")
            ev = (f"{best:.2f}/{last:.2f}" if best is not None else "-")
            lines.append(
                f"{name:<22}{steps:>9,}{100*steps/TOTAL:>5.0f}{fps:>6.0f}"
                f"{eta:>7.1f}{roll_sr:>7.0f}%{roll_co:>6.0f}%{ev:>19}"
                f"{ck//1000:>6}k  {status}")
    remaining = 18 * TOTAL - tot_steps
    eta_all = remaining / tot_fps / 3600 if tot_fps > 0 else float("nan")
    lines.append("-" * len(header))
    lines.append(f"campaign: {done_n}/18 done | {live_n} live | "
                 f"{100*tot_steps/(18*TOTAL):.1f}% of 18M steps | "
                 f"aggregate {tot_fps:.0f} fps | ETA ~{eta_all:.1f} h (train only)")
    lines.append(f"health:   {gpu_line()}")
    lines.append(f"          {cpu_line()}")
    ologs = sorted(RUNS.glob("campaign*.log"), key=lambda p: p.stat().st_mtime)
    alive = orchestrator_alive()
    lines.append(f"orchestrator: {'RUNNING' if alive else 'NOT RUNNING (watchdog relaunches within 5 min)'}"
                 f" | restarts: {max(len(ologs) - 1, 0)}")
    if ologs:
        try:
            last_line = ologs[-1].read_text(encoding="utf-8", errors="ignore").strip().splitlines()[-1]
            lines.append(f"  last [{ologs[-1].name}]: {last_line}")
        except Exception:
            pass
    wlog = RUNS / "watchdog.log"
    if wlog.exists():
        lines.append(f"  watchdog: {wlog.read_text(encoding='utf-8', errors='ignore').strip().splitlines()[-1]}")
    n_eval = sum(1 for a in AGENTS for sd in SEEDS if (d := find_dir(a, sd))
                 for f in ("eval_results.json", "eval_low.json", "eval_high.json") if (d / f).exists())
    if n_eval:
        lines.append(f"evaluation: {n_eval}/54 eval files written")
    if (RUNS / "ALL_DONE").exists():
        lines.append("*** ALL DONE — results in experiments/simple_experiment/runs/results/ ***")
    res = RUNS / "results" / "table.json"
    if res.exists():
        lines.append(f"RESULTS READY: {res}")
    return "\n".join(lines)


def main():
    watch = "--watch" in sys.argv
    interval = 30
    while True:
        out = snapshot()
        if watch:
            os.system("cls" if os.name == "nt" else "clear")
        print(time.strftime("Simplified Tier-2 experiment (no beds, 10-20 peds) — %Y-%m-%d %H:%M:%S"))
        print(out)
        if not watch:
            break
        time.sleep(interval)


if __name__ == "__main__":
    main()
