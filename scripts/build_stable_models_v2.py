"""Build deploy/stable_models_v2/ — a DROP-IN model folder for the Gazebo /
MentorPi stack, containing the stable-recipe Tier-2 policies (stability pilot
4: reward normalization off, replay buffer 1M, learning rate 1e-4).

Only training changed; the observation/action interface is identical to the
August models, so these load in controller_node.py unchanged
(model_zip / vecnorm_pkl / weight_pt / agent launch arguments).

Copy the output folder next to (or into) Sim_New_Env/models on the Ubuntu
machine. This does NOT touch Sim_New_Env/ or gazebo_demo_package/.

    .venv\\Scripts\\python.exe scripts\\build_stable_models_v2.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "experiments" / "stability_pilot" / "runs4"
OUT = ROOT / "deploy" / "stable_models_v2"
RUNS = [("fixed_equal", 0), ("fixed_equal", 1), ("vanilla", 0), ("vanilla", 1),
        ("attention", 0), ("attention", 1)]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stats(f: Path):
    rec = json.load(open(f))["records"]
    stat = [r for r in rec if r["task_id"] == "STAT_medicine"]
    end = lambda k: float(np.mean([r.get("end") == k for r in rec]))  # noqa: E731
    return dict(sr=float(np.mean([r["success"] for r in rec])), collision=end("collision"),
                timeout=end("timeout"), cvr=float(np.mean([r["cvr"] for r in rec])),
                stat_sr=float(np.mean([r["success"] for r in stat])),
                stat_cvr=float(np.mean([r["cvr"] for r in stat])), n_episodes=len(rec))


def verify(zip_path: Path, vn_path: Path):
    """Load exactly as controller_node does (stock SB3) and run one prediction
    on a real Tier-2 observation."""
    sys.path.insert(0, str(ROOT))
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs.lmic_hospital_env import LMICHospitalNavEnv
    env = DummyVecEnv([lambda: LMICHospitalNavEnv(floorplan_path=str(ROOT / "configs" / "hospital_v4.yaml"),
                                                  use_camera=False)])
    vn = VecNormalize.load(str(vn_path), env)
    vn.training = False
    model = SAC.load(str(zip_path), device="cpu")
    obs = vn.reset()
    act, _ = model.predict(obs, deterministic=True)
    low, high = env.action_space.low, env.action_space.high
    ok = bool(np.all(act[0] >= low - 1e-6) and np.all(act[0] <= high + 1e-6))
    env.close()
    return ok, [round(float(x), 3) for x in act[0]]


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    manifest = {"recipe": "SAC, VecNormalize(norm_obs=True, norm_reward=False), buffer 1,000,000, "
                          "learning_rate 1e-4, 1M steps, Tier-2 (configs/hospital_v4.yaml), FINAL models",
                "source": str(SRC.relative_to(ROOT)), "models": {}}
    for agent, seed in RUNS:
        rd = sorted(p for p in (SRC / f"{agent}_s{seed}").glob(f"{agent}_seed{seed}_*") if p.is_dir())[-1]
        tag = f"{agent}_seed{seed}"
        files = {"model_zip": (rd / "final_model.zip", OUT / f"{tag}_final.zip"),
                 "vecnorm_pkl": (rd / "vec_normalize.pkl", OUT / f"{tag}_final_vecnorm.pkl")}
        if agent == "attention":
            files["weight_pt"] = (rd / "attention_final.pt", OUT / f"{tag}_final_weights.pt")
        entry = {"run_dir": str(rd.relative_to(ROOT))}
        for key, (src, dst) in files.items():
            shutil.copy(src, dst)
            entry[key] = {"file": dst.name, "sha256": sha(dst)}
        ok, act = verify(OUT / f"{tag}_final.zip", OUT / f"{tag}_final_vecnorm.pkl")
        entry["load_check"] = {"stock_sb3_load_ok": ok, "sample_action": act}
        entry["sim_eval_no_shield"] = stats(rd.parent / "eval_final500.json")
        shield_f = rd.parent / "eval_shield_final500.json"
        if shield_f.exists():
            entry["sim_eval_with_shield"] = stats(shield_f)
        manifest["models"][tag] = entry
        print(f"{tag}: load ok={ok} sample action={act}  "
              f"SR {100*entry['sim_eval_no_shield']['sr']:.1f}% "
              f"(+shield {100*entry.get('sim_eval_with_shield', {}).get('sr', float('nan')):.1f}%)")
    json.dump(manifest, open(OUT / "MANIFEST.json", "w"), indent=2)
    shutil.copy(ROOT / "deploy" / "stable_models_v2_README.md", OUT / "README.md")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
