"""Run the planner evaluation matrix (pilot-4 stable policies x planner variants) in
parallel hidden processes at BELOW_NORMAL priority. Idempotent: finished result
files are skipped, so it can be re-run after an interruption.

    python experiments/planner/run_matrix.py [--phase core|abl|all] [--procs 4] [--n_eps 200]

Results: experiments/planner/results/<model>__<variant>.json  (logs in results/logs/)
Summary: python experiments/planner/summarize.py
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
RES = HERE / "results"
RUNS4 = ROOT / "experiments" / "stability_pilot" / "runs4"

MODELS = {d.name: sorted(p for p in d.glob(f"{d.name.split('_s')[0]}_seed*") if p.is_dir())[-1]
          for d in sorted(RUNS4.glob("*_s[0-9]")) if d.is_dir()}

VARIANTS = {
    "noplan_shield":       ["--mode", "off", "--shield"],
    "geo_shield":          ["--mode", "geo", "--shield"],
    "cult_shield":         ["--mode", "cult", "--shield"],
    "cult":                ["--mode", "cult"],
    # ablations (subset of models)
    "cult3_shield":        ["--mode", "cult", "--ratio", "3", "--shield"],
    "cult_L1.5_shield":    ["--mode", "cult", "--L", "1.5", "--shield"],
    "cult_L4_shield":      ["--mode", "cult", "--L", "4", "--shield"],
    "cult_geodist_shield": ["--mode", "cult", "--dist", "geo", "--shield"],
    "cult_nolos_shield":   ["--mode", "cult", "--no_los", "--shield"],
}
CORE = ["noplan_shield", "geo_shield", "cult_shield", "cult"]
ABL = [v for v in VARIANTS if v not in CORE]
ABL_MODELS = ["fixed_equal_s1", "vanilla_s1"]


def jobs(phase):
    for name, rd in MODELS.items():
        for v, extra in VARIANTS.items():
            if v in CORE and phase in ("core", "all") or v in ABL and phase in ("abl", "all") and name in ABL_MODELS:
                yield name, v, rd, extra


def campaign_running() -> bool:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object "
                          "{ $_.CommandLine -like '*campaign_v5_stable.py*' -or $_.CommandLine -like '*train_sac.py*' }"
                          " | Measure-Object).Count"], capture_output=True, text=True, timeout=60)
    return int(out.stdout.strip() or 0) > 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="core", choices=["core", "abl", "all"])
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--n_eps", type=int, default=200)
    p.add_argument("--config", default="configs/hospital_v4.yaml")
    p.add_argument("--wait_for_campaign", action="store_true",
                   help="start only once no runs_v5 training/orchestrator process is alive (CPU <= 90 %% rule)")
    a = p.parse_args()
    if a.wait_for_campaign:
        while campaign_running():
            print(time.strftime("%H:%M:%S"), "campaign still running - waiting", flush=True)
            time.sleep(30)
        print(time.strftime("%H:%M:%S"), "campaign stopped - starting the matrix", flush=True)
    RES.mkdir(exist_ok=True); (RES / "logs").mkdir(exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    si = subprocess.STARTUPINFO(); si.dwFlags |= subprocess.STARTF_USESHOWWINDOW; si.wShowWindow = 0
    flags = subprocess.CREATE_NEW_CONSOLE | 0x00004000   # hidden console, BELOW_NORMAL priority
    queue = [(n, v, rd, ex) for n, v, rd, ex in jobs(a.phase) if not (RES / f"{n}__{v}.json").exists()]
    print(f"{len(queue)} jobs to run ({a.phase}), {a.procs} at a time, {a.n_eps} eps each", flush=True)
    live = []
    while queue or live:
        live = [(pr, n, v) for pr, n, v in live if pr.poll() is None]
        while queue and len(live) < a.procs:
            n, v, rd, ex = queue.pop(0)
            out = f"experiments/planner/results/{n}__{v}.json"
            cmd = [PY, str(HERE / "eval_planner.py"), "--run_dir", str(rd), "--config", a.config,
                   "--n_eps", str(a.n_eps), "--out", out, *ex]
            log = open(RES / "logs" / f"{n}__{v}.log", "a", encoding="utf-8")
            pr = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT,
                                  creationflags=flags, startupinfo=si)
            live.append((pr, n, v))
            print(time.strftime("%H:%M:%S"), f"launched {n} {v}", flush=True)
        time.sleep(10)
    print("matrix done", flush=True)


if __name__ == "__main__":
    main()
