"""Build Sim_New_Env/ — the portable, self-contained folder to copy to a
fresh Ubuntu machine for ROS 2 Gazebo/RViz/Nav2 simulation of the NEW
environment. Contains everything training produced; no training needed there.

    .venv\\Scripts\\python.exe scripts\\build_sim_new_env.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "deploy" / "Sim_New_Env"
DEPLOY = ROOT / "deploy" / "ros2_stack"
DEMO = ROOT / "deploy" / "gazebo_demo_package"
IGN = shutil.ignore_patterns("__pycache__", "*.pyc", ".venv", "sim_runs")


def main():
    if PKG.exists():
        shutil.rmtree(PKG)
    PKG.mkdir()

    # 1. the complete ROS 2 package (world/zones/urdf/launch/config inside)
    shutil.copytree(DEPLOY / "hospitalnav_deploy",
                    PKG / "ros2_ws_src" / "hospitalnav_deploy", ignore=IGN)

    # 2. extra gazebo assets kept outside the package (RViz config, alt launches)
    shutil.copytree(DEPLOY / "gazebo", PKG / "extra_gazebo", ignore=IGN)

    # 3. trained policies (best seed per agent, matched vecnorm pairs)
    shutil.copytree(DEMO / "models", PKG / "models")

    # 3b. EVERYTHING trained: all 6 agents x 3 seeds, best + final-2M
    #     checkpoints with their matched normalization stats, plus each
    #     run's evaluation record and learning curve.
    all_m = PKG / "models" / "all_agents"
    art = PKG / "training_artifacts"
    all_m.mkdir()
    art.mkdir()
    for agent in ["cwn", "fixed_equal", "nav_heavy", "vanilla", "he2025", "attention"]:
        for seed in (0, 1, 2):
            dirs = [d for d in sorted((ROOT / "runs_v4").glob(f"{agent}_seed{seed}_*"))
                    if d.is_dir() and (d / "final_model.zip").exists()]
            if not dirs:
                continue
            d = dirs[-1]
            tag = f"{agent}_seed{seed}"
            pairs = [
                (d / "best_model" / "best_model.zip", all_m / f"{tag}_best.zip"),
                (d / "best_model" / "vec_normalize_best.pkl", all_m / f"{tag}_best_vecnorm.pkl"),
                (d / "final_model.zip", all_m / f"{tag}_final2M.zip"),
                (d / "vec_normalize.pkl", all_m / f"{tag}_final2M_vecnorm.pkl"),
            ]
            for src, dst in pairs:
                if src.exists():
                    shutil.copy(src, dst)
            run_art = art / tag
            run_art.mkdir()
            for src in (d / "eval_results.json", d / "eval_results_1M.json",
                        d / "eval" / "evaluations.npz"):
                if src.exists():
                    shutil.copy(src, run_art / src.name)

    # 4. environment setup for a fresh Ubuntu 22.04
    shutil.copy(DEPLOY / "setup_pi5.sh", PKG / "setup_ubuntu.sh")
    shutil.copy(DEPLOY / "requirements-pi.txt", PKG / "requirements.txt")

    # 5. training provenance: config, layout source, results, manifest
    prov = PKG / "training_provenance"
    prov.mkdir()
    shutil.copy(ROOT / "configs" / "hospital_v4.yaml", prov)
    shutil.copy(DEPLOY / "training" / "layout.py", prov)
    shutil.copy(DEMO / "manifest.json", prov)
    for f in (DEMO / "results").iterdir():
        shutil.copy(f, prov / f.name)

    # 6. docs
    docs = PKG / "docs"
    docs.mkdir()
    shutil.copy(DEPLOY / "DEPLOYMENT_GUIDE.md", docs)
    shutil.copy(DEPLOY / "docs" / "real_world_test_protocol.md", docs)

    n = sum(1 for f in PKG.rglob("*") if f.is_file())
    mb = sum(f.stat().st_size for f in PKG.rglob("*") if f.is_file()) / 1e6
    print(f"Sim_New_Env built: {n} files, {mb:.1f} MB")
    print("NOTE: rebuild wipes the folder — re-create README.md and "
          "training_provenance/TRAINING_DETAILS.md afterwards (kept outside "
          "this builder).")


if __name__ == "__main__":
    main()
