"""FINAL paper campaign with the STABLE training recipe (stability pilots 2-4,
Oct 2026): VecNormalize reward normalization OFF, replay buffer 1M, learning
rate 1e-4. Design decisions for review-proofing:
  * 5 seeds per agent (seed-level statistics)
  * n_envs = 1 for EVERY agent (removes the n_envs confound reviewers flagged)
  * 2M steps for the main tiers; the 1M checkpoint is evaluated too
    (paper-protocol budget + convergence evidence)
  * kinematics ablation (pivot-capable unicycle) re-trained with the SAME recipe
Throughput (benchmarked 2026-10-08): CPU, 1 torch/BLAS thread per run, 22
concurrent runs = 709 env-steps/s aggregate vs 180 on the GPU.

HIERARCHICAL agents (Oct 8 route-planner fix, experiments/planner/): a global
route planner feeds the policy's goal observation (waypoint 2.5 m along the
route) and the progress reward is geodesic; CARS agents get the cultural-cost
planner ("cult"), the culture-blind vanilla/he2025 the shortest-path planner
("geo"). planner="auto" applies that rule; None = original mapless agent.

Groups, in priority order (cap MAX_CONCURRENT, longest-remaining first in group):
  tier2                   hospital_v4   6 agents x 5 seeds x 2M, planner auto
  tier2_mapless           hospital_v4   fixed_equal/vanilla x 5 x 2M, NO planner (baseline)
  tier2_planner_ablation  hospital_v4   fixed_equal->geo, vanilla->cult x 3 x 2M (2x2 completion)
  tier1                   dmch          6 agents x 5 seeds x 2M, planner auto
  tier1_mapless           dmch          fixed_equal x 3 x 2M, NO planner
  tier2_pivot             hospital_v4   fixed_equal x 3 x 1M, pivot kinematics + planner
  tier2_ppo               hospital_v4   PPO (CARS equal-weight reward, SB3 defaults) x 5 x 2M, planner
When a group's runs are all complete its evaluation (scripts/eval_v5.py, same
per-run planner settings) starts in the background while other groups train;
groups flagged classical=True also evaluate the non-learning baselines
(scripts/eval_classical_v5.py: planner + social force / pure pursuit, A*, social
force) under the same protocol. Display names: he2025 -> risk_field (NAMES).

Restart-safe/idempotent: skips complete runs, resumes from the newest
checkpoint, adopts runs still training, kills strays before relaunch, prunes
old replay-buffer pickles. Smoke test: set V5_SMOKE=1 (tiny budgets, runs_v5_smoke/).

    .venv\\Scripts\\python.exe scripts\\campaign_v5_stable.py
"""
from __future__ import annotations
import os, re, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from thermal_guard import ThermalGuard  # noqa: E402
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
PATCH = str(ROOT / "experiments" / "stability_pilot" / "patch_sac.py")
SMOKE = os.environ.get("V5_SMOKE") == "1"
RUNS = ROOT / ("runs_v5_smoke" if SMOKE else "runs_v5")
ALL6 = ["cwn", "fixed_equal", "nav_heavy", "vanilla", "he2025", "attention"]

# "mid": an intermediate checkpoint that is also evaluated (1M = the paper's budget)
T2, T1 = "configs/hospital_v4.yaml", "configs/dmch_hospital.yaml"
S5, S3 = [0, 1, 2, 3, 4], [0, 1, 2]
PPO = "experiments/ppo/train_ppo.py"       # second RL algorithm (agent id ppo_equal: CARS equal-weight reward)
# Presentation names (JSON keys keep the internal ids; dashboards/tables show these).
NAMES = {"he2025": "risk_field", "ppo_equal": "ppo_equal", "cwn": "cwn", "fixed_equal": "fixed_equal",
         "nav_heavy": "nav_heavy", "vanilla": "vanilla", "attention": "attention",
         "safe_heavy": "safe_heavy", "residual_sf": "residual_sf"}
LONG_NAMES = {"he2025": "Risk-field SAC (exponential-decay risk-field reward, He et al. 2025)",
              "ppo_equal": "PPO with the CARS equal-weight reward (SB3 defaults)",
              "safe_heavy": "SAC + CARS with safety-heavy fixed weights [0.20, 0.45, 0.25, 0.10]",
              "residual_sf": "SAC residual over the tuned social-force prior (CARS equal-weight reward)"}
if SMOKE:
    GROUPS = {
        "tier2": dict(config=T2, agents=["attention", "vanilla"], seeds=[0], steps=6000, mid=3000, planner="auto",
                      classical=True),
        "tier2_mapless": dict(config=T2, agents=["fixed_equal"], seeds=[0], steps=3000, planner=None),
        "tier2_planner_ablation": dict(config=T2, agents=["vanilla"], seeds=[0], steps=3000,
                                       planner={"vanilla": "cult"}),
        "tier2_pivot": dict(config=T2, agents=["fixed_equal"], seeds=[0], steps=3000, pivot=True, planner="auto"),
        "tier2_ppo": dict(config=T2, agents=["ppo_equal"], seeds=[0], steps=6000, mid=3000, planner="auto",
                          trainer=PPO),
        "tier2_safe": dict(config=T2, agents=["safe_heavy"], seeds=[0], steps=3000, planner="auto",
                           env={"PILOT_AGENT": "safe_heavy"}),
        "tier2_residual": dict(config=T2, agents=["residual_sf"], seeds=[0], steps=3000, planner="auto",
                               env={"PILOT_AGENT": "residual_sf"}),
    }
    CKPT_FREQ, EVAL_FREQ, MAX_CONCURRENT = 1500, 3000, 6   # 4 training slots remain while 2 evaluate
else:
    GROUPS = {
        "tier2": dict(config=T2, agents=ALL6, seeds=S5, steps=2_000_000, mid=1_000_000, planner="auto",
                      classical=True),
        "tier2_mapless": dict(config=T2, agents=["fixed_equal", "vanilla"], seeds=S5, steps=2_000_000,
                              mid=1_000_000, planner=None),
        "tier2_planner_ablation": dict(config=T2, agents=["fixed_equal", "vanilla"], seeds=S3, steps=2_000_000,
                                       mid=1_000_000, planner={"fixed_equal": "geo", "vanilla": "cult"}),
        "tier1": dict(config=T1, agents=ALL6, seeds=S5, steps=2_000_000, mid=1_000_000, planner="auto",
                      classical=True),
        "tier1_mapless": dict(config=T1, agents=["fixed_equal"], seeds=S3, steps=2_000_000, mid=1_000_000,
                              planner=None),
        "tier2_pivot": dict(config=T2, agents=["fixed_equal"], seeds=S3, steps=1_000_000, pivot=True, planner="auto"),
        "tier2_ppo": dict(config=T2, agents=["ppo_equal"], seeds=S5, steps=2_000_000, mid=1_000_000, planner="auto",
                          trainer=PPO),
        # "make RL better than the classical driver" (user, 2026-10-11): residual learning over the
        # tuned social-force prior, and the safety-weighted reward control
        "tier2_residual": dict(config=T2, agents=["residual_sf"], seeds=S3, steps=2_000_000, mid=1_000_000,
                               planner="auto", env={"PILOT_AGENT": "residual_sf"}),
        "tier2_safe": dict(config=T2, agents=["safe_heavy"], seeds=S3, steps=2_000_000, mid=1_000_000,
                           planner="auto", env={"PILOT_AGENT": "safe_heavy"}),
    }
    # User limit: total CPU <= 90%. Measured: one "single-thread" run uses ~1.2
    # cores (torch + Python/logging overhead) and other apps (e.g. AutoCAD) ~1.5
    # cores. 17 planner runs measured 87.7-91.0% (thermal.csv 2026-10-08 16:23-16:28),
    # so the cap is 16 (~85%). While an evaluation runs (2 processes) training is
    # capped at 14. Runtime override without a restart: write an integer to runs_v5/CAP.
    CKPT_FREQ, EVAL_FREQ, MAX_CONCURRENT = 100_000, 50_000, 16
EVAL_PROCS = 2
BELOW_NORMAL = 0x00004000   # Windows priority class: Windows and the desktop always come first

FRESH_SEC = 300
RECIPE = ["--buffer_size", "1000000", "--learning_rate", "1e-4", "--n_envs", "1", "--device", "cpu",
          "--checkpoint_freq", str(CKPT_FREQ), "--eval_freq", str(EVAL_FREQ), "--no_camera", "--no_check_env"]
KEYS = [(g, a, s) for g, spec in GROUPS.items() for a in spec["agents"] for s in spec["seeds"]]
BASE_ENV = {**{k: v for k, v in os.environ.items()
               if k not in ("WEAN_MODE", "PILOT_PIVOT", "PILOT_AGENT", "PILOT_ALGO") and not k.startswith("PLANNER_")},
            "PYTHONUTF8": "1", "MPLBACKEND": "Agg", "PILOT_NORM_REWARD": "0",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1"}
CULTURE_BLIND = ("vanilla", "he2025")        # no cultural reward term -> shortest-path planner
# Settings fixed by the evaluation-only matrix on the pilot-4 policies (experiments/planner/
# results, 2026-10-08): look-ahead 2.5 m > 1.5/4 m, line-of-sight pull-back +7.5 pp, geodesic
# distance channel harmless, cultural cost 3x the reward's penalty > 1x on every policy
# (+2.5..+4 pp success, prayer-room steps 0.000 vs 0.013-0.033).
PLANNER_DEFAULTS = {"PLANNER_RATIO": "3.0", "PLANNER_L": "2.5", "PLANNER_DIST": "geo",
                    "PLANNER_PROGRESS": "1", "PLANNER_LOS": "1"}


def planner_mode(g, a):
    """None | 'geo' | 'cult' for run (g, a): the group's planner setting resolved per agent."""
    p = GROUPS[g].get("planner")
    if p is None:
        return None
    if isinstance(p, dict):
        return p.get(a)
    if p == "auto":
        return "geo" if a in CULTURE_BLIND else "cult"
    return p


def env_for(k, evaluation=False):
    """Process environment for run key (g, a, s) or, with a bare group name, the group-level env."""
    g, a = (k, None) if isinstance(k, str) else (k[0], k[1])
    env = {**BASE_ENV, **({"PILOT_PIVOT": "1"} if GROUPS[g].get("pivot") else {})}
    mode = planner_mode(g, a) if a is not None else None
    if mode:
        env.update(PLANNER_DEFAULTS, PLANNER_MODE=mode)
        if evaluation:
            env["PLANNER_PROGRESS"] = "0"   # same observations; keeps info["dist_to_goal"] Euclidean in records
    if GROUPS[g].get("trainer") == PPO:
        env["PILOT_ALGO"] = "ppo"           # evaluation loads PPO checkpoints
        env["PILOT_NORM_REWARD"] = "1"      # on-policy PPO keeps SB3's reward normalisation
    env.update(GROUPS[g].get("env", {}))    # group-specific patches (agent aliases)
    return env


def display(a):
    return NAMES.get(a, a)


def log(msg):
    print(f"[v5] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def slot(k):
    g, a, s = k
    return RUNS / g / f"{a}_s{s}"


def run_dir(k):
    g, a, s = k
    hits = sorted(p for p in slot(k).glob(f"{a}_seed{s}_*") if p.is_dir())
    return hits[-1] if hits else None


def ckpt(rd):
    if rd is None or not (rd / "checkpoints").exists():
        return None
    st = [int(m.group(1)) for p in (rd / "checkpoints").glob("*_[0-9]*_steps.zip")
          if (m := re.match(r"(?:sac|ppo)_(\d+)_steps", p.name))]
    return max(st) if st else None


ADAPTIVE = ("cwn", "attention")


def complete(k):
    """Final model AND its normalizer (AND the weight net for adaptive agents)
    written, and untouched for 60 s — train_sac saves them one after another,
    so final_model.zip alone can exist before vec_normalize.pkl does."""
    rd = run_dir(k)
    if rd is None:
        return False
    need = [rd / "final_model.zip", rd / "vec_normalize.pkl"]
    if k[1] in ADAPTIVE:
        need.append(rd / f"{k[1]}_final.pt")
    return all(f.exists() and time.time() - f.stat().st_mtime > 60 for f in need)


def remaining(k):
    return GROUPS[k[0]]["steps"] - (ckpt(run_dir(k)) or 0)


def final_files_exist(k):
    """complete() without the 60-s quiet window."""
    rd = run_dir(k)
    if rd is None:
        return False
    need = [rd / "final_model.zip", rd / "vec_normalize.pkl"]
    if k[1] in ADAPTIVE:
        need.append(rd / f"{k[1]}_final.pt")
    return all(f.exists() for f in need)


def finalize_from_ckpt(k):
    """A run stopped after its last checkpoint reached the full budget but before
    train_sac wrote final_model.zip cannot be resumed (train_sac exits with
    'Nothing to resume'; seen looping in the 2026-10-11 smoke). The checkpoint at
    `steps` IS the final model state: promote it to the final artifacts."""
    g, a, s = k
    rd = run_dir(k)
    st = ckpt(rd)
    if rd is None or st is None or st < GROUPS[g]["steps"] or final_files_exist(k):
        return False
    prefix = "ppo" if GROUPS[g].get("trainer") == PPO else "sac"
    z = rd / "checkpoints" / f"{prefix}_{st}_steps.zip"
    v = rd / "checkpoints" / f"{prefix}_vecnormalize_{st}_steps.pkl"
    if not (z.exists() and v.exists()):
        return False
    if a in ADAPTIVE and not (rd / f"{a}_final.pt").exists():
        wpts = sorted((int(p.stem.split("_")[-1]), p) for p in rd.glob(f"{a}_[0-9]*.pt")
                      if p.stem.split("_")[-1].isdigit())
        if not wpts:
            return False            # no weight-net snapshot to promote; must retrain the tail
        shutil.copy(wpts[-1][1], rd / f"{a}_final.pt")
    if not (rd / "final_model.zip").exists():
        shutil.copy(z, rd / "final_model.zip")
    if not (rd / "vec_normalize.pkl").exists():
        shutil.copy(v, rd / "vec_normalize.pkl")
    log(f"{g}/{a}_s{s}: promoted checkpoint @{st:,} to final_model (stopped after its last checkpoint)")
    return True


def fresh(k):
    files = [slot(k) / "train.log"]
    rd = run_dir(k)
    if rd:
        files += list(rd.glob("monitor*.monitor.csv"))
    mt = max((f.stat().st_mtime for f in files if f.exists()), default=0)
    return time.time() - mt < FRESH_SEC


def _ps(command):
    return subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True)


def kill_strays(k):
    g, a, s = k
    # '?' = exactly one path separator, so 'tier2' never matches 'tier2_pivot'; the
    # slot path appears in both fresh (--log_dir) and resume (--resume_from) commands.
    flt = f"$_.CommandLine -like '*train_*.py*' -and $_.CommandLine -like '*{RUNS.name}?{g}?{a}_s{s}*'"
    _ps("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        f"Where-Object {{ {flt} }} | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}")


PPO_RECIPE = ["--n_envs", "1", "--device", "cpu", "--checkpoint_freq", str(CKPT_FREQ),
              "--eval_freq", str(EVAL_FREQ), "--no_camera", "--no_check_env"]   # SB3-default PPO hyper-parameters


def build_cmd(k):
    g, a, s = k
    spec = GROUPS[g]
    trainer = spec.get("trainer", "train/train_sac.py")
    cmd = [PY, PATCH, trainer, "--config", spec["config"], "--agent", a, "--seed", str(s),
           "--timesteps", str(spec["steps"]), *(PPO_RECIPE if trainer == PPO else RECIPE)]
    rd = run_dir(k)
    if ckpt(rd) is not None:
        return f"resume {g}/{rd.name}@{ckpt(rd)}", cmd + ["--resume_from", str(rd)]
    return f"fresh {g}/{a}_s{s}", cmd + ["--log_dir", f"{RUNS.name}/{g}/{a}_s{s}"]


def prune_buffers():
    """Keep only the newest replay-buffer pickle per run (each up to ~0.75 GB)."""
    for k in KEYS:
        rd = run_dir(k)
        if rd is None or not (rd / "checkpoints").exists():
            continue
        bufs = sorted((int(m.group(1)), p) for p in (rd / "checkpoints").glob("sac_replay_buffer_*_steps.pkl")
                      if (m := re.match(r"sac_replay_buffer_(\d+)_steps", p.name)))
        for _, p in bufs[:-1]:
            try:
                if time.time() - p.stat().st_mtime > 120:
                    p.unlink()
            except OSError:
                pass


def marker(g, name):
    return RUNS / "results" / g / name


def eval_alive_elsewhere(g):
    if SMOKE:          # the smoke orchestrator is alone in runs_v5_smoke/; the real campaign's
        return False   # evaluations share the group names and must not be mistaken for ours
    flt = f"$_.CommandLine -like '*eval_v5.py* {g}' -and $_.CommandLine -notlike '*--shield*'"
    out = _ps("(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
              f"Where-Object {{ {flt} }} | Measure-Object).Count")
    try:
        return int(out.stdout.strip() or 0) > 0
    except ValueError:
        return False


def pin_affinity():
    """Hard CPU cap (user limit <= 90%): restrict this orchestrator to logical CPUs
    1..n-4, leaving CPU 0 and the last three free for Windows and the user's apps.
    Training and evaluation processes inherit the mask from their parent.
    Measured 2026-10-08: 20 of 24 cores -> total CPU 85.6-89.2% (avg 87.9%)."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes
    n = os.cpu_count() or 24
    mask = sum(1 << i for i in range(1, n - 3))
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE          # pseudo-handle must stay pointer-sized
    k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    k32.SetProcessAffinityMask.restype = wintypes.BOOL
    if k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask):
        log(f"CPU affinity: {bin(mask).count('1')} of {n} logical CPUs (CPU 0 and last 3 kept free)")
    else:
        log("WARNING: could not set CPU affinity")


def main():
    RUNS.mkdir(exist_ok=True)
    pin_affinity()
    order = list(GROUPS)
    sort_key = lambda k: (order.index(k[0]), -remaining(k))  # noqa: E731
    pending = sorted([k for k in KEYS if not complete(k) and not final_files_exist(k)
                      and not finalize_from_ckpt(k) and not fresh(k)], key=sort_key)
    log(f"{sum(complete(k) for k in KEYS)}/{len(KEYS)} complete; {len(pending)} to launch (cap {MAX_CONCURRENT})"
        + (" [SMOKE]" if SMOKE else ""))
    launched, evals = {}, {}
    shed_at = {}                      # key -> time it was stopped to reduce load
    guard = ThermalGuard(RUNS, log)

    def recently_shed(k):
        return time.time() - shed_at.get(k, 0) < 2 * FRESH_SEC

    def live_keys():
        mine = [k for k, (p, _) in launched.items() if p.poll() is None]
        orphan = [k for k in KEYS if k not in launched and not complete(k) and fresh(k) and not recently_shed(k)]
        return mine + orphan

    def live():
        return len(live_keys())

    def shed_excess():
        """Stop runs above the current cap, least work lost first (newest checkpoint);
        they are re-enqueued and resume from that checkpoint when capacity returns."""
        excess = live() - cap()
        if excess <= 0:
            return

        def unsaved(k):
            """Seconds of training that stopping k would lose: since its newest
            checkpoint, or since this process segment started (fresh or resume)."""
            rd = run_dir(k)
            if rd is None:
                return 0.0
            marks = [c.stat().st_mtime for c in (rd / "checkpoints").glob("*_steps.zip")] \
                if (rd / "checkpoints").exists() else []
            marks += [m.stat().st_ctime for m in rd.glob("monitor*.monitor.csv")]   # segment start
            return time.time() - max(marks, default=time.time())
        # Never shed a run that is writing its final artifacts (remaining <= 0): it cannot
        # be resumed, only promoted — and unsaved() ~ 0 made these the FIRST picks before.
        for k in sorted([x for x in live_keys() if remaining(x) > 0], key=unsaved)[:excess]:
            lost = unsaved(k)
            if k in launched:
                launched[k][0].kill()
            kill_strays(k)
            shed_at[k] = time.time()
            log(f"shed {k[0]}/{k[1]}_s{k[2]} to respect the CPU/thermal cap "
                f"(~{lost/60:.0f} min of unsaved training; resumes later)")

    def need_eval(g):
        return (all(complete(k) for k in KEYS if k[0] == g)
                and not marker(g, "EVAL_DONE").exists() and not marker(g, "EVAL_FAILED").exists())

    def start_evals():
        for g in GROUPS:
            if g in evals or not need_eval(g) or eval_alive_elsewhere(g):
                continue
            fh = open(RUNS / f"eval_{g}.log", "a", encoding="utf-8")
            evals[g] = subprocess.Popen([PY, "scripts/eval_v5.py", g], cwd=str(ROOT), env=env_for(g),
                                        stdout=fh, stderr=subprocess.STDOUT, creationflags=BELOW_NORMAL)
            log(f"{g}: all runs complete -> evaluation started (pid {evals[g].pid})")

    def work_left():
        # Any incomplete run counts (also runs paused by a shed and runs inside the 60-s
        # completion quiet window): exiting early left a paused run unresumed and a
        # finished group unevaluated in the 2026-10-11 smoke test.
        return (pending or live() > 0 or any(p.poll() is None for p in evals.values())
                or any(need_eval(g) for g in GROUPS) or any(not complete(k) for k in KEYS))

    def cap():
        evaluating = any(p.poll() is None for p in evals.values()) or any(eval_alive_elsewhere(g) for g in GROUPS)
        base = MAX_CONCURRENT
        try:                                   # runtime override (e.g. CPU-limit tuning) without a restart
            base = int((RUNS / "CAP").read_text().strip())
        except (OSError, ValueError):
            pass
        return max(2, base - (EVAL_PROCS if evaluating else 0) - guard.shed)

    start_evals()
    while work_left():
        guard.tick()
        shed_excess()
        while pending and live() < cap():
            k = pending.pop(0)
            kill_strays(k)
            label, cmd = build_cmd(k)
            slot(k).mkdir(parents=True, exist_ok=True)
            fh = open(slot(k) / "train.log", "a", encoding="utf-8")
            p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT, env=env_for(k),
                                 creationflags=BELOW_NORMAL)
            launched[k] = (p, fh)
            log(f"launched {label} [planner {planner_mode(k[0], k[1]) or 'none'}"
                f"{', PPO' if GROUPS[k[0]].get('trainer') == PPO else ''}] (pid {p.pid}); live={live()}")
            time.sleep(3)
        time.sleep(10 if SMOKE else 30)
        for k, (p, fh) in list(launched.items()):
            if p.poll() is not None:
                fh.close()
                if p.returncode == 0:
                    log(f"{k[0]}/{k[1]}_s{k[2]}: OK")
                elif recently_shed(k):
                    log(f"{k[0]}/{k[1]}_s{k[2]}: stopped (shed; resumes from its checkpoint)")
                else:
                    log(f"{k[0]}/{k[1]}_s{k[2]}: FAILED rc={p.returncode}")
                del launched[k]
        for k in KEYS:
            if complete(k) or k in launched or k in pending or fresh(k):
                continue
            if final_files_exist(k) or finalize_from_ckpt(k):
                continue   # finished or promoted; complete() turns True after the quiet window
            log(f"re-enqueue stalled {k[0]}/{k[1]}_s{k[2]}")
            pending.append(k)
            pending.sort(key=sort_key)
        prune_buffers()
        start_evals()
        for g, p in list(evals.items()):
            if p.poll() is not None:
                log(f"{g}: evaluation finished rc={p.returncode}")
                del evals[g]
                if p.returncode != 0:
                    log(f"WARNING {g} evaluation failed; fix, delete EVAL_FAILED, re-run")
                    marker(g, "EVAL_FAILED").parent.mkdir(parents=True, exist_ok=True)
                    marker(g, "EVAL_FAILED").write_text(time.strftime("%c"))
        if (not pending and live() == 0 and not evals
                and any(need_eval(g) for g in GROUPS) and all(eval_alive_elsewhere(g) for g in GROUPS if need_eval(g))):
            time.sleep(60)   # an earlier orchestrator's evaluation is still running

    if all(marker(g, "EVAL_DONE").exists() for g in GROUPS):
        (RUNS / "ALL_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
        log("ALL DONE")
    else:
        log(f"exiting: {sum(complete(k) for k in KEYS)}/{len(KEYS)} complete; re-run to continue")


if __name__ == "__main__":
    main()
