"""Live dashboard for the shield-weaning campaign — read-only.

    .venv\\Scripts\\python.exe experiments\\shield_weaning\\dashboard.py           # snapshot
    .venv\\Scripts\\python.exe experiments\\shield_weaning\\dashboard.py --watch   # every 30 s

Reuses the per-run readers of experiments/simple_experiment/dashboard.py and
adds the weaning signals from TensorBoard: current intervention rate, budget
target and lambda.
"""
from __future__ import annotations
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "simple_experiment"))
import dashboard as base  # noqa: E402  (simple_experiment/dashboard.py)

RUNS = HERE / "runs"
MATRIX = {"wean": ["fixed_equal", "attention", "vanilla"],
          "fixed": ["fixed_equal"], "naive": ["fixed_equal"]}
SEEDS = [0, 1, 2]
TOTAL = 1_000_000
FRESH_SEC = 300
N = sum(len(a) for a in MATRIX.values()) * len(SEEDS)


def find_dir(mode, agent, seed):
    hits = sorted(h for h in (RUNS / mode).glob(f"{agent}_seed{seed}_*") if h.is_dir())
    return hits[-1] if hits else None


def wean_signals(run_dir):
    """(interv_rate, budget_target, lambda) from the newest TB event file."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        evs = sorted((run_dir / "tb").glob("**/events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
        if not evs:
            return None
        acc = EventAccumulator(str(evs[-1]), size_guidance={"scalars": 2})
        acc.Reload()
        out = []
        for tag in ("wean/interv_rate", "wean/budget_target", "wean/lambda"):
            vals = acc.Scalars(tag) if tag in acc.Tags().get("scalars", []) else []
            out.append(vals[-1].value if vals else None)
        return out
    except Exception:
        return None


def orchestrator_alive():
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object "
             "{ $_.CommandLine -like '*shield_weaning*campaign.py*' } | Measure-Object).Count"],
            capture_output=True, text=True, timeout=20)
        return int(out.stdout.strip() or 0) > 0
    except Exception:
        return False


def snapshot():
    now = time.time()
    hdr = (f"{'run':<26}{'steps':>9}{'%':>5}{'fps':>5}{'ETA_h':>6}{'rollSR':>7}{'coll%':>6}"
           f"{'eval(best/last)':>16}{'interv':>7}{'budget':>7}{'lambda':>7}  status")
    lines = [hdr, "-" * len(hdr)]
    tot_steps = tot_fps = done_n = live_n = 0
    for mode, agents in MATRIX.items():
        for agent in agents:
            for seed in SEEDS:
                name = f"{mode}/{agent}_s{seed}"
                d = find_dir(mode, agent, seed)
                if d is None:
                    lines.append(f"{name:<26}{'-':>9}{'-':>5}{'-':>5}{'-':>6}{'-':>7}{'-':>6}"
                                 f"{'-':>16}{'-':>7}{'-':>7}{'-':>7}  QUEUED")
                    continue
                steps, wall, rows = base.read_monitors(d)
                fps = steps / wall if wall > 1 else 0.0
                tot_steps += steps; tot_fps += fps
                tail = rows[-base.ROLL_N:]
                rsr = 100 * np.mean([r[1] for r in tail]) if tail else float("nan")
                rco = 100 * np.mean([r[2] for r in tail]) if tail else float("nan")
                best, last = base.eval_sr(d)
                files = [RUNS / mode / f"{agent}_seed{seed}_train.log", *d.glob("monitor*.monitor.csv")]
                fresh = now - max((f.stat().st_mtime for f in files if f.exists()), default=0) < FRESH_SEC
                if (d / "final_model.zip").exists():
                    status, done_n = "DONE", done_n + 1
                elif fresh:
                    status = "LIVE"; live_n += 1
                else:
                    status = "STALLED"
                eta = (TOTAL - steps) / fps / 3600 if fps > 0 and status == "LIVE" else float("nan")
                ev = f"{best:.2f}/{last:.2f}" if best is not None else "-"
                ws = wean_signals(d) or [None, None, None]
                fmt = lambda v: "-" if v is None else f"{v:.2f}"
                lines.append(f"{name:<26}{steps:>9,}{100*steps/TOTAL:>5.0f}{fps:>5.0f}{eta:>6.1f}"
                             f"{rsr:>6.0f}%{rco:>5.0f}%{ev:>16}{fmt(ws[0]):>7}{fmt(ws[1]):>7}"
                             f"{fmt(ws[2]):>7}  {status}")
    eta_all = (N * TOTAL - tot_steps) / tot_fps / 3600 if tot_fps > 0 else float("nan")
    lines.append("-" * len(hdr))
    lines.append(f"campaign: {done_n}/{N} done | {live_n} live | "
                 f"{100*tot_steps/(N*TOTAL):.1f}% of {N}M steps | aggregate {tot_fps:.0f} fps | "
                 f"ETA ~{eta_all:.1f} h (train only)")
    lines.append(f"health:   {base.gpu_line()}")
    lines.append(f"          {base.cpu_line()}")
    ologs = sorted(RUNS.glob("campaign*.log"), key=lambda p: p.stat().st_mtime)
    lines.append(f"orchestrator: {'RUNNING' if orchestrator_alive() else 'NOT RUNNING (watchdog relaunches within 5 min)'}"
                 f" | restarts: {max(len(ologs) - 1, 0)}")
    if ologs:
        try:
            lines.append(f"  last [{ologs[-1].name}]: "
                         f"{ologs[-1].read_text(encoding='utf-8', errors='ignore').strip().splitlines()[-1]}")
        except Exception:
            pass
    n_eval = sum(1 for m, ags in MATRIX.items() for a in ags for s in SEEDS
                 if (d := find_dir(m, a, s)) and (d / "eval_results.json").exists())
    if n_eval:
        lines.append(f"evaluation: {n_eval}/{N} unshielded eval files written")
    if (RUNS / "ALL_DONE").exists():
        lines.append("*** ALL DONE — results in experiments/shield_weaning/results/comparison.md ***")
    return "\n".join(lines)


def main():
    watch = "--watch" in sys.argv
    while True:
        out = snapshot()
        if watch:
            os.system("cls" if os.name == "nt" else "clear")
        print(time.strftime("Shield weaning (Tier 2, 45 peds, 1M) — %Y-%m-%d %H:%M:%S"))
        print(out)
        if not watch:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
