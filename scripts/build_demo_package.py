"""Build gazebo_demo_package/ — self-contained folder with the trained
policies and everything needed to simulate/present the robot in Gazebo.

Picks each agent's best seed by final 500-episode eval SR, copies
best_model + MATCHED VecNormalize stats, the ROS 2 deploy package, Gazebo
world/zones/URDF, final results tables, and writes a provenance manifest.

    .venv\\Scripts\\python.exe scripts\\build_demo_package.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs_v4"
PKG = ROOT / "deploy" / "gazebo_demo_package"
DEPLOY = ROOT / "deploy" / "ros2_stack"
AGENTS = ["fixed_equal", "attention", "cwn", "vanilla"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def best_seed(agent):
    best = (None, -1.0, None)
    for seed in (0, 1, 2):
        dirs = [d for d in sorted(RUNS.glob(f"{agent}_seed{seed}_*")) if d.is_dir()]
        if not dirs:
            continue
        d = dirs[-1]
        f = d / "eval_results.json"
        if not f.exists():
            continue
        sr = json.load(open(f)).get("sr", -1)
        if sr > best[1]:
            best = (seed, sr, d)
    return best


def main():
    if PKG.exists():
        shutil.rmtree(PKG)
    (PKG / "models").mkdir(parents=True)
    manifest = {"built": time.strftime("%Y-%m-%d %H:%M"),
                "campaign": "runs_v4 walled tier, 2M steps, 500-ep eval",
                "config_sha256_16": sha(ROOT / "configs" / "hospital_v4.yaml"),
                "models": {}}

    for agent in AGENTS:
        seed, sr, d = best_seed(agent)
        if d is None:
            print(f"!! no eval for {agent}")
            continue
        zip_src = d / "best_model" / "best_model.zip"
        vn_src = d / "best_model" / "vec_normalize_best.pkl"
        if not vn_src.exists():
            vn_src = d / "vec_normalize.pkl"
        zd = PKG / "models" / f"{agent}_seed{seed}_best.zip"
        vd = PKG / "models" / f"{agent}_seed{seed}_vecnorm.pkl"
        shutil.copy(zip_src, zd)
        shutil.copy(vn_src, vd)
        manifest["models"][agent] = {
            "seed": seed, "eval_sr_500ep": round(sr, 3),
            "source_run": d.name, "model_sha": sha(zd), "vecnorm_sha": sha(vd)}
        print(f"{agent}: seed{seed} (SR {sr:.3f}) <- {d.name}")

    # ROS 2 deploy package (code + launch + config), gazebo assets
    shutil.copytree(DEPLOY / "hospitalnav_deploy", PKG / "hospitalnav_deploy",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    gz = PKG / "gazebo_assets"
    gz.mkdir()
    for src in [DEPLOY / "hospitalnav_deploy" / "worlds" / "hospital.world",
                DEPLOY / "hospitalnav_deploy" / "worlds" / "hospital_zones.png",
                DEPLOY / "gazebo" / "urdf" / "hospitalbot.urdf"]:
        if src.exists():
            shutil.copy(src, gz / src.name)

    # results for presentation
    res = PKG / "results"
    res.mkdir()
    for src, name in [
            (RUNS / "results" / "table.json", "table_tier2_walled_2M.json"),
            (RUNS / "results_1M" / "table.json", "table_tier2_walled_1M.json"),
            (ROOT / "runs" / "results" / "table.json", "table_tier1_open_1M.json"),
            (RUNS / "results" / "shielded.json", "shielded_tier2_2M.json"),
            (RUNS / "results" / "shield_ladder_best.json", "shield_ladder_tier2.json"),
            (RUNS / "layout_preview.png", "layout_preview.png")]:
        if src.exists():
            shutil.copy(src, res / name)
        else:
            print(f"   (skip, not present yet: {src.name} -> {name})")

    (PKG / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\npackage at {PKG}")
    total = sum(f.stat().st_size for f in PKG.rglob("*") if f.is_file()) / 1e6
    print(f"total size: {total:.1f} MB")


if __name__ == "__main__":
    main()
