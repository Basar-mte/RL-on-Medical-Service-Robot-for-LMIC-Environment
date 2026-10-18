"""V4 campaign 2M-step extension — resumes all 18 completed 1M runs to 2M
(user-approved 2026-08-05: eval curves still climbing at 1M, attention +8.7
SR pts in the final quarter).

Identical to scripts/campaign_v4.py except: TIMESTEPS=2M, and a one-time
prep step that ARCHIVES each run's 1M artifacts (final_model.zip ->
final_model_1M.zip, <agent>_final.pt -> ..._1M.pt, eval_results.json ->
eval_results_1M.json; runs_v4/results -> runs_v4/results_1M) so the
completion check re-arms, training resumes from the sac_1000000 checkpoint
into the same directory, and the eval chain re-runs on the 2M models.
Idempotent and outage-safe like the original.

    .venv\\Scripts\\python.exe scripts\\campaign_v4_2m.py
"""
from __future__ import annotations
import os, re, shutil, subprocess, sys, time
from pathlib import Path

# log_trajectories.py prints a '→' — force UTF-8 in all child processes so
# redirected stdout can't die on charmap (bit us once on the 1M chain).
os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
RUNS_NAME = "runs_v4"
RUNS = ROOT / RUNS_NAME
CONFIG_REL = "configs/hospital_v4.yaml"
ADAPTIVE = ("cwn", "attention")
AGENTS = ["cwn", "fixed_equal", "nav_heavy", "vanilla", "he2025", "attention"]
SEEDS = [0, 1, 2]
TIMESTEPS = "2000000"
TOTAL_STEPS = 2_000_000
MAX_CONCURRENT = 12
FRESH_SEC = 150


def log(msg):
    print(f"[campaign_v4_2m] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def find_dir(agent, seed):
    hits = [h for h in sorted(RUNS.glob(f"{agent}_seed{seed}_*")) if h.is_dir()]
    return hits[-1] if hits else None


def latest_ckpt_steps(run_dir):
    cks = run_dir / "checkpoints"
    if not cks.exists():
        return None
    steps = [int(m.group(1)) for p in cks.glob("sac_*_steps.zip")
             if (m := re.match(r"sac_(\d+)_steps", p.name))]
    return max(steps) if steps else None


def prep_extension():
    """Archive 1M artifacts so completion checks re-arm (idempotent)."""
    res = RUNS / "results"
    res_1m = RUNS / "results_1M"
    if res.exists() and not res_1m.exists():
        res.rename(res_1m)
        log("archived results/ -> results_1M/")
    for agent in AGENTS:
        for seed in SEEDS:
            d = find_dir(agent, seed)
            if d is None:
                continue
            ck = latest_ckpt_steps(d) or 0
            fm = d / "final_model.zip"
            if fm.exists() and ck < TOTAL_STEPS:
                fm.rename(d / "final_model_1M.zip")
                for extra in (d / f"{agent}_final.pt", d / "eval_results.json"):
                    if extra.exists():
                        extra.rename(extra.with_name(
                            extra.stem + "_1M" + extra.suffix))
                log(f"archived 1M artifacts in {d.name}")


def is_complete(agent, seed):
    d = find_dir(agent, seed)
    return bool(d and (d / "final_model.zip").exists())


def remaining_steps(agent, seed):
    d = find_dir(agent, seed)
    ck = latest_ckpt_steps(d) if d else None
    return TOTAL_STEPS - (ck or 0)


def log_fresh(agent, seed):
    logf = RUNS / f"{agent}_seed{seed}_train.log"
    return logf.exists() and (time.time() - logf.stat().st_mtime) < FRESH_SEC


def build_cmd(agent, seed):
    n_envs = "1" if agent in ADAPTIVE else "2"
    base = [PY, "train/train_sac.py", "--config", CONFIG_REL,
            "--agent", agent, "--seed", str(seed),
            "--timesteps", TIMESTEPS, "--n_envs", n_envs, "--no_camera",
            "--no_check_env"]
    d = find_dir(agent, seed)
    if d and latest_ckpt_steps(d) is not None:
        return f"resume {d.name}@{latest_ckpt_steps(d)}", base + ["--resume_from", str(d)]
    return f"fresh {agent}_seed{seed}", base + ["--log_dir", RUNS_NAME]


def main():
    prep_extension()
    all_keys = [(a, s) for a in AGENTS for s in SEEDS]
    pending = sorted(
        [(a, s) for (a, s) in all_keys if not is_complete(a, s) and not log_fresh(a, s)],
        key=lambda k: -remaining_steps(*k))
    log(f"{sum(is_complete(a, s) for a, s in all_keys)}/18 complete; "
        f"{sum(log_fresh(a, s) and not is_complete(a, s) for a, s in all_keys)} already training; "
        f"{len(pending)} to launch (cap {MAX_CONCURRENT})")

    launched = {}

    def live_count():
        mine = sum(1 for (p, _) in launched.values() if p.poll() is None)
        orphan = sum(1 for (a, s) in all_keys
                     if (a, s) not in launched and not is_complete(a, s) and log_fresh(a, s))
        return mine + orphan

    while pending or any(p.poll() is None for p, _ in launched.values()):
        while pending and live_count() < MAX_CONCURRENT:
            a, s = pending.pop(0)
            label, cmd = build_cmd(a, s)
            fh = open(RUNS / f"{a}_seed{s}_train.log", "a", encoding="utf-8")
            p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
            launched[(a, s)] = (p, fh)
            log(f"launched {label} (pid {p.pid}); live={live_count()}")
            time.sleep(5)
        time.sleep(30)
        for (a, s), (p, fh) in list(launched.items()):
            if p.poll() is not None:
                fh.close()
                log(f"{a}_seed{s}: {'OK' if p.returncode == 0 else f'FAILED rc={p.returncode}'}")
                del launched[(a, s)]
        for (a, s) in all_keys:
            if (not is_complete(a, s) and not log_fresh(a, s)
                    and (a, s) not in launched and (a, s) not in pending):
                log(f"re-enqueue stalled {a}_seed{s}")
                pending.append((a, s))
                pending.sort(key=lambda k: -remaining_steps(*k))

    done = sum(1 for a in AGENTS for s in SEEDS
               if (d := find_dir(a, s)) and (d / "final_model.zip").exists())
    log(f"training phase complete: {done}/18 have 2M final_model.zip")
    if done < 18:
        log("WARNING: <18 complete — re-run this script to resume the rest; "
            "skipping eval")
        return

    log("evaluating each run in the v4 map (evaluate.py --config, 500 eps)")
    for a in AGENTS:
        for s in SEEDS:
            d = find_dir(a, s)
            if not d or (d / "eval_results.json").exists():
                continue
            best = d / "best_model"
            model_dir = best if best.exists() else d
            log(f"  {d.name}")
            subprocess.run([PY, "eval/evaluate.py", "--model_dir", str(model_dir),
                            "--config", CONFIG_REL, "--n_episodes", "500",
                            "--output", "eval_results.json"], cwd=str(ROOT))

    for cmd in (
        [PY, "eval/post_training.py", "--runs_dir", RUNS_NAME, "--n_episodes", "500"],
        [PY, "eval/log_trajectories.py", "--runs_dir", RUNS_NAME,
         "--config", CONFIG_REL, "--output_dir", f"{RUNS_NAME}/results/trajectories"],
        [PY, "scripts/eval_shielded_v4.py", "cwn", "attention", "vanilla"],
    ):
        log("running: " + " ".join(cmd[1:]))
        subprocess.run(cmd, cwd=str(ROOT))
    log("ALL DONE")


if __name__ == "__main__":
    main()
