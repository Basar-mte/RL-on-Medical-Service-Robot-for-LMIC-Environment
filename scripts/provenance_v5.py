"""Record the provenance of the final campaign: git HEAD + working-tree status,
sha256 of every behaviour-defining file (env, rewards, baselines, training,
evaluation, configs, in-process patches, planner, orchestrator), library
versions and the full pip freeze.

    .venv\\Scripts\\python.exe scripts\\provenance_v5.py [runs_dir]   (default runs_v5)
"""
from __future__ import annotations
import hashlib, platform, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "runs_v5")
OUT = RUNS / "results" / "PROVENANCE"
OUT.mkdir(parents=True, exist_ok=True)

GLOBS = ["envs/*.py", "rewards/*.py", "baselines/*.py", "train/*.py", "eval/*.py", "configs/*.yaml",
         "experiments/stability_pilot/patch_sac.py", "experiments/planner/*.py",
         "experiments/holonomic_experiment/holonomic_train.py",
         "scripts/campaign_v5_stable.py", "scripts/eval_v5.py", "scripts/thermal_guard.py",
         "scripts/provenance_v5.py"]


def sh(cmd):
    try:
        return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=120).stdout
    except Exception as e:  # noqa: BLE001
        return f"<{e}>"


files = sorted({p for g in GLOBS for p in ROOT.glob(g) if p.is_file()})
with open(OUT / "sha256.txt", "w", encoding="utf-8") as f:
    for p in files:
        f.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}\n")

(OUT / "git.txt").write_text(
    f"recorded {time.strftime('%Y-%m-%d %H:%M:%S')}\nHEAD {sh(['git', 'rev-parse', 'HEAD']).strip()}\n"
    f"branch {sh(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip()}\n\n--- git status --short ---\n"
    + sh(["git", "status", "--short"]) + "\n--- git diff --stat ---\n" + sh(["git", "diff", "--stat"]),
    encoding="utf-8")

import numpy, torch, gymnasium, stable_baselines3  # noqa: E402
(OUT / "versions.txt").write_text(
    f"python {platform.python_version()} ({platform.platform()})\nnumpy {numpy.__version__}\n"
    f"torch {torch.__version__}\ngymnasium {gymnasium.__version__}\nstable_baselines3 {stable_baselines3.__version__}\n"
    f"cpu_count {__import__('os').cpu_count()}\n", encoding="utf-8")
(OUT / "pip_freeze.txt").write_text(sh([sys.executable, "-m", "pip", "freeze"]), encoding="utf-8")

# Verbatim copies of the files that are not (yet) committed: orchestrator, evaluation,
# in-process patches and the planner — so the record is self-contained.
import shutil  # noqa: E402
snap = OUT / "snapshot"
snap.mkdir(exist_ok=True)
for p in files:
    rel = p.relative_to(ROOT)
    if rel.parts[0] in ("scripts", "experiments"):
        dst = snap / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
print(f"provenance written to {OUT}: {len(files)} files hashed, scripts/experiments snapshotted")
