"""Live dashboard for the FINAL stable-recipe campaign (runs_v5/) — read-only.

    .venv\\Scripts\\python.exe scripts\\dashboard_v5.py           # one snapshot
    .venv\\Scripts\\python.exe scripts\\dashboard_v5.py --watch   # refresh every 60 s
"""
from __future__ import annotations
import os, subprocess, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "simple_experiment"))
import campaign_v5_stable as C  # noqa: E402  (GROUPS / RUNS / run_dir)
import dashboard as base        # noqa: E402  (read_monitors / eval_sr / latest_ckpt / gpu_line / cpu_line)


def orchestrator_alive():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object "
                              "{ $_.CommandLine -like '*campaign_v5_stable.py*' } | Measure-Object).Count"],
                             capture_output=True, text=True, timeout=20)
        return int(out.stdout.strip() or 0) > 0
    except Exception:
        return False


def snapshot():
    now = time.time()
    W = 46   # run-name column: "<group>/<agent>_s<seed> [planner]"
    hdr = f"{'run':<{W}}{'steps':>10}{'%':>5}{'fps':>5}{'ETA_h':>6}{'rollSR':>7}{'coll%':>6}{'eval(best/last)':>16}{'ckpt':>7}  status"
    lines = [hdr, "-" * len(hdr)]
    tot = done = live = 0; agg_fps = 0.0; steps_left = 0
    for g, spec in C.GROUPS.items():
        total = spec["steps"]
        for a in spec["agents"]:
            for s in spec["seeds"]:
                tot += 1
                mode = C.planner_mode(g, a)
                name = f"{g}/{C.display(a)}_s{s}" + (f" [{mode}]" if mode else "")
                rd = C.run_dir((g, a, s))
                if rd is None:
                    lines.append(f"{name:<{W}}{'-':>10}{'-':>5}{'-':>5}{'-':>6}{'-':>7}{'-':>6}{'-':>16}{'-':>7}  QUEUED")
                    steps_left += total
                    continue
                steps, wall, rows = base.read_monitors(rd)
                fps = steps / wall if wall > 1 else 0.0
                tail = rows[-base.ROLL_N:]
                rsr = 100 * np.mean([r[1] for r in tail]) if tail else float("nan")
                rco = 100 * np.mean([r[2] for r in tail]) if tail else float("nan")
                best, last = base.eval_sr(rd)
                logf = rd.parent / "train.log"
                age = now - max([logf.stat().st_mtime if logf.exists() else 0]
                                + [f.stat().st_mtime for f in rd.glob("monitor*.monitor.csv")])
                steps = min(steps, total)
                if (rd / "final_model.zip").exists():
                    status, steps = "DONE", total; done += 1
                elif age < 300:
                    status = "LIVE"; live += 1; agg_fps += fps
                else:
                    status = "STALLED"
                steps_left += total - steps
                eta = (total - steps) / fps / 3600 if fps > 0 and status == "LIVE" else float("nan")
                ev = f"{best:.2f}/{last:.2f}" if best is not None else "-"
                lines.append(f"{name:<{W}}{steps:>10,.0f}{100*steps/total:>5.0f}{fps:>5.0f}{eta:>6.1f}"
                             f"{rsr:>6.0f}%{rco:>5.0f}%{ev:>16}{base.latest_ckpt(rd)//1000:>6}k  {status}")
    lines.append("-" * len(hdr))
    eta_all = steps_left / agg_fps / 3600 if agg_fps > 0 else float("nan")
    lines.append(f"campaign: {done}/{tot} trained | {live} live | aggregate {agg_fps:.0f} steps/s | "
                 f"{steps_left/1e6:.1f}M steps left | training ETA ~{eta_all:.0f} h (+ evaluation)")
    lines.append(f"health:   {base.gpu_line()}")
    lines.append(f"          {base.cpu_line()}")
    try:
        import json
        th = json.load(open(C.RUNS / "thermal_status.json"))
        f = lambda v, u: "n/a" if v is None else f"{v:.0f}{u}"  # noqa: E731
        flag = {"ok": "OK", "WARNING": "** WARNING **", "CRITICAL": "!! CRITICAL !!"}.get(th.get("level"), th.get("level"))
        lines.append(f"thermal:  {flag} | CPU clock {f(th.get('perf'), '%')} of nominal (<100% under load = throttling)"
                     f" | throttle flag {th.get('throttle', 0):.0f} | board {f(th.get('zone_c'), 'C')}"
                     f" | GPU {f(th.get('gpu_c'), 'C')} | runs shed for heat: {th.get('shed', 0)} | at {th.get('time', '?')[11:]}")
    except Exception:
        lines.append("thermal:  (no sample yet)")
    logs = sorted(C.RUNS.glob("campaign*.log"), key=lambda p: p.stat().st_mtime)
    lines.append(f"orchestrator: {'RUNNING' if orchestrator_alive() else 'NOT RUNNING (watchdog relaunches within 5 min)'}"
                 f" | restarts: {max(len(logs) - 1, 0)}")
    if logs:
        try:
            lines.append(f"  last: {logs[-1].read_text(encoding='utf-8', errors='ignore').strip().splitlines()[-1]}")
        except Exception:
            pass
    for g in C.GROUPS:
        m = C.RUNS / "results" / g
        state = "DONE" if (m / "EVAL_DONE").exists() else ("FAILED" if (m / "EVAL_FAILED").exists() else "pending")
        lines.append(f"{g} evaluation: {state}")
    if (C.RUNS / "ALL_DONE").exists():
        lines.append("*** ALL DONE — results in runs_v5/results/ ***")
    return "\n".join(lines)


def main():
    watch = "--watch" in sys.argv
    while True:
        out = snapshot()
        if watch:
            os.system("cls" if os.name == "nt" else "clear")
        print(time.strftime("FINAL campaign, stable recipe (runs_v5) — %Y-%m-%d %H:%M:%S"))
        print(out)
        if not watch:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
