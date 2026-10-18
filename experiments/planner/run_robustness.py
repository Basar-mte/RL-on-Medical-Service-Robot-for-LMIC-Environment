"""Unpredicted-uncertainty robustness suite (evaluation-only, saved models, protocol seeds).

Conditions:
  unmapped   world = unmapped_cfgs/<base>_L<i>.yaml (boxes the planner cannot see;
             PLANNER_MAP keeps the planner on the clean map), 50 eps x N layouts
  noyield    pedestrians ignore the robot (UNC_PED_NOYIELD=1), 500 eps
  lidar      5 % range noise + 2 % ray dropout (UNC_LIDAR/UNC_LIDAR_DROP), 500 eps
  act        10 %-of-range actuation noise after the shield (UNC_ACT=0.1), 500 eps
Drivers: learned fixed_equal_s1 [cult], learned vanilla_s0 [geo], classical sf2 (tuned),
each without shield and with the v2 shield. 2 jobs in parallel on borrowed slots
(runs_v5/CAP=14, removed at the end). Results: runs_v5/results/tier2/robustness/.
Merge/summary: summarize_robustness.py
"""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
PATCH = str(ROOT / "experiments" / "stability_pilot" / "patch_sac.py")
RES = ROOT / "runs_v5" / "results" / "tier2" / "robustness"
LOG = open(ROOT / "runs_v5" / "robustness.log", "a", encoding="utf-8")
BELOW_NORMAL = 0x00004000
BASE_CFG = "configs/hospital_v4.yaml"
LAYOUTS = sorted((HERE / "unmapped_cfgs").glob("hospital_v4_L*.yaml"))
EPS_FULL, EPS_LAYOUT = 500, 50

CONDS = {"noyield": {"UNC_PED_NOYIELD": "1"},
         "lidar": {"UNC_LIDAR": "0.05", "UNC_LIDAR_DROP": "0.02"},
         "act": {"UNC_ACT": "0.1"}}


def log(m):
    LOG.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}\n"); LOG.flush()


def learned_cmd(agent, seed, mode, shield2, config, out):
    rd = sorted(p for p in (ROOT / "runs_v5" / "tier2" / f"{agent}_s{seed}").glob(f"{agent}_seed{seed}_*") if p.is_dir())[-1]
    cmd = [PY, str(HERE / "eval_planner.py"), "--run_dir", str(rd), "--mode", mode, "--ratio", "3",
           "--dist", "geo", "--config", config, "--n_eps", str(EPS_LAYOUT if "unmapped" in out else EPS_FULL),
           "--out", f"runs_v5/results/tier2/robustness/{out}.json"]
    return cmd + (["--shield_v2"] if shield2 else [])


def classical_cmd(shield2, config, base):
    n = EPS_LAYOUT if "unmapped" in base else EPS_FULL
    cmd = [PY, PATCH, "scripts/eval_classical_v5.py", "tier2", "planner_sf2", "--n_eps", str(n),
           "--suffix", f"__{base}"]     # eval_classical appends _shield2 itself
    return cmd + (["--shield_v2"] if shield2 else [])


DRIVERS = [("fe", lambda sh, cfg, out: learned_cmd("fixed_equal", 1, "cult", sh, cfg, out)),
           ("van", lambda sh, cfg, out: learned_cmd("vanilla", 0, "geo", sh, cfg, out)),
           ("sf2", None)]
CDIR = ROOT / "runs_v5" / "results" / "tier2" / "classical"


def _one(cname, env_extra, cfg, i=None):
    tag = cname if i is None else f"{cname}_L{i}"
    for dname, mk in DRIVERS:
        for sh in (False, True):
            base = f"{tag}__{dname}"
            out = base + ("_shield2" if sh else "")
            if dname == "sf2":
                cmd = classical_cmd(sh, cfg, base)
                marker = CDIR / f"planner_sf2__{base}{'_shield2' if sh else ''}.json"
            else:
                cmd = mk(sh, cfg, out)
                marker = RES / f"{out}.json"
            if not marker.exists():
                yield cmd, dict(env_extra), out


def jobs():
    for cname, env_extra in CONDS.items():
        yield from _one(cname, env_extra, BASE_CFG)
    for i, lay in enumerate(LAYOUTS):
        cfg = str(lay.relative_to(ROOT)).replace("\\", "/")
        yield from _one("unmapped", {"PLANNER_MAP": BASE_CFG, "EVAL_CONFIG": cfg}, cfg, i=i)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    (ROOT / "runs_v5" / "CAP").write_text("14")
    base_env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUTF8": "1",
                "PILOT_NORM_REWARD": "0"}
    for v in ("UNC_PED_NOYIELD", "UNC_LIDAR", "UNC_LIDAR_DROP", "UNC_ACT", "PLANNER_MAP", "EVAL_CONFIG"):
        base_env.pop(v, None)
    queue, live = list(jobs()), []
    log(f"robustness suite: {len(queue)} jobs, 2 parallel, layouts={len(LAYOUTS)}")
    while queue or live:
        live = [p for p in live if p.poll() is None]
        while queue and len(live) < 2:
            cmd, extra, out = queue.pop(0)
            fh = open(RES / f"{out}.log", "a", encoding="utf-8")
            live.append(subprocess.Popen(cmd, cwd=str(ROOT), env={**base_env, **extra},
                                         stdout=fh, stderr=subprocess.STDOUT, creationflags=BELOW_NORMAL))
            log(f"launched {out}")
        time.sleep(10)
    try:
        (ROOT / "runs_v5" / "CAP").unlink()
    except OSError:
        pass
    log("robustness suite done; CAP removed")


if __name__ == "__main__":
    main()
