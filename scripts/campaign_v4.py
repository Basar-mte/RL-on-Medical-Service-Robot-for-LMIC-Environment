"""scripts/recover_campaign.py retargeted at the NEW layout — same wave
scheduler, same 18-run matrix, three real differences (all forced by the
no-main-code-changes rule):

  1. trains into runs_v4/ with --config configs/hospital_v4.yaml
  2. eval runs evaluate.py per run itself: post_training.py doesn't forward
     --config and run_ablations.py hardcodes the old YAML, so calling them
     for evaluation would silently score v4 models in the old map
  3. post_training.py is then only the aggregator (it prefers each run's
     cached eval_results.json and never re-evaluates)

    .venv\\Scripts\\python.exe scripts\\campaign_v4.py
"""
from __future__ import annotations
import re, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
RUNS_NAME = "runs_v4"
RUNS = ROOT / RUNS_NAME
CONFIG_REL = "configs/hospital_v4.yaml"
ADAPTIVE = ("cwn", "attention")
AGENTS = ["cwn", "fixed_equal", "nav_heavy", "vanilla", "he2025", "attention"]
SEEDS = [0, 1, 2]
TIMESTEPS = "1000000"
TOTAL_STEPS = 1_000_000
MAX_CONCURRENT = 12   # measured ceiling on this machine — do NOT raise
FRESH_SEC = 150       # a live SB3 run writes its log at least this often


def log(msg):
    print(f"[campaign_v4] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def find_dir(agent, seed):
    hits = sorted(RUNS.glob(f"{agent}_seed{seed}_*"))
    hits = [h for h in hits if h.is_dir()]
    return hits[-1] if hits else None


def latest_ckpt_steps(run_dir):
    cks = run_dir / "checkpoints"
    if not cks.exists():
        return None
    steps = [int(m.group(1)) for p in cks.glob("sac_*_steps.zip")
             if (m := re.match(r"sac_(\d+)_steps", p.name))]
    return max(steps) if steps else None


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
    RUNS.mkdir(exist_ok=True)
    if not (RUNS / "config_used.yaml").exists():
        shutil.copy(ROOT / CONFIG_REL, RUNS / "config_used.yaml")

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
    log(f"training phase complete: {done}/18 have final_model.zip")
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
        [PY, "eval/post_training.py", "--runs_dir", RUNS_NAME, "--n_episodes", "100"],
        [PY, "eval/log_trajectories.py", "--runs_dir", RUNS_NAME,
         "--config", CONFIG_REL, "--output_dir", f"{RUNS_NAME}/results/trajectories"],
        [PY, "scripts/eval_shielded_v4.py", "cwn", "attention", "vanilla"],
    ):
        log("running: " + " ".join(cmd[1:]))
        subprocess.run(cmd, cwd=str(ROOT))
    log("ALL DONE")


if __name__ == "__main__":
    main()
