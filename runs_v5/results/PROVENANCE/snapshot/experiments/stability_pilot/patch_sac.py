"""In-process SAC/VecNormalize tweaks for the stability pilot, then hand off to
the next wrapper (e.g. experiments/simple_experiment/run.py). No main-code change.

Env vars:
  PILOT_NORM_REWARD=0   force VecNormalize(norm_reward=False) in train_sac

    python experiments/stability_pilot/patch_sac.py <next_script> <args...>
"""
from __future__ import annotations
import os, runpy, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import train.train_sac as ts  # noqa: E402

if os.environ.get("PILOT_NORM_REWARD", "1") == "0":
    # A factory, not a subclass: the object train_sac creates must stay a plain
    # stable_baselines3 VecNormalize so it pickles/loads anywhere (a subclass
    # defined here would live in a module runpy swaps out, and the first
    # best-model save would fail to pickle it).
    _VN = ts.VecNormalize

    def _vn_factory(venv, *a, **kw):
        kw["norm_reward"] = False
        return _VN(venv, *a, **kw)

    def _vn_load(path, venv):
        vn = _VN.load(path, venv)
        vn.norm_reward = False
        return vn

    _vn_factory.load = _vn_load
    ts.VecNormalize = _vn_factory

    # train_sac's resume path sets env.norm_reward = True right after load;
    # undo that at the last possible moment, just before learning starts.
    _orig_learn = ts.SAC.learn

    def _learn(self, *a, **kw):
        for vn in (getattr(self, "_vec_normalize_env", None), self.get_env()):
            if vn is not None and hasattr(vn, "norm_reward"):
                vn.norm_reward = False
        print("[stability_pilot] norm_reward forced False before learn()", flush=True)
        return _orig_learn(self, *a, **kw)

    ts.SAC.learn = _learn
    print("[stability_pilot] VecNormalize reward normalization OFF", flush=True)

if os.environ.get("PILOT_PIVOT") == "1":
    # Pivot-capable (unicycle) kinematics ablation: importing the module applies
    # the env-class patch; its make_env wrapper re-applies it inside workers.
    sys.path.insert(0, str(ROOT / "experiments"))
    import holonomic_experiment.holonomic_train as _holo  # noqa: E402
    ts.make_env = _holo._holo_make_env
    print("[stability_pilot] PIVOT (unicycle) kinematics patch applied", flush=True)

if os.environ.get("PLANNER_MODE", "off") != "off":
    # Hierarchical agent: route planner waypoint in obs["goal"] + geodesic
    # progress reward (experiments/planner/patch.py); re-applied in workers.
    sys.path.insert(0, str(ROOT / "experiments"))
    import planner.patch as _pp  # noqa: E402
    _pp.apply_from_env()
    _pp.wrap_make_env(ts)
    print(f"[stability_pilot] PLANNER patch applied (PLANNER_MODE={os.environ['PLANNER_MODE']})", flush=True)

_ALIAS = os.environ.get("PILOT_AGENT")
if _ALIAS in ("safe_heavy", "residual_sf"):
    # Extra CARS agents without touching train_sac's --agent choices:
    #   safe_heavy   = fixed weights [0.20, 0.45, 0.25, 0.10] (FixedWeightSource preset)
    #   residual_sf  = CARS equal weights + residual action over the social-force prior
    _orig_build = ts._build_reward

    def _build_reward(agent, weight_module=None):
        if agent == "safe_heavy":
            return None, ts.FixedWeightSource("safe_heavy").get_weights
        if agent == "residual_sf":
            return None, ts.FixedWeightSource("equal").get_weights
        return _orig_build(agent, weight_module=weight_module)
    ts._build_reward = _build_reward
    _orig_parse = ts.parse_args

    def _parse_args():
        if "--agent" in sys.argv:
            i = sys.argv.index("--agent") + 1
            if i < len(sys.argv) and sys.argv[i] == _ALIAS:
                sys.argv[i] = "fixed_equal"          # pass argparse's choices; restored below
        a = _orig_parse()
        a.agent = _ALIAS
        return a
    ts.parse_args = _parse_args
    if _ALIAS == "residual_sf":
        sys.path.insert(0, str(ROOT / "experiments"))
        import planner.residual as _res  # noqa: E402
        _res.apply_from_env()
        _orig_mk = ts.make_env

        def _mk(*a, **kw):
            f = _orig_mk(*a, **kw)

            def _thunk():
                _res.apply_from_env()
                return f()
            return _thunk
        ts.make_env = _mk
    print(f"[stability_pilot] agent alias {_ALIAS} enabled", flush=True)

if os.environ.get("PILOT_ALGO") == "ppo":
    # PPO baseline (experiments/ppo/train_ppo.py): evaluation loads PPO checkpoints
    # through eval.evaluate's module-level `SAC` name.
    import eval.evaluate as _ev  # noqa: E402
    from stable_baselines3 import PPO as _PPO  # noqa: E402
    _ev.SAC = _PPO
    print("[stability_pilot] evaluation loader set to PPO", flush=True)

if __name__ == "__main__":
    nxt = sys.argv[1]
    sys.argv = [nxt] + sys.argv[2:]
    target = Path(nxt).as_posix()
    if target.endswith("train/train_sac.py"):
        # Call the already-imported (and patched) module; run_path would
        # execute a fresh, unpatched copy of train_sac as __main__.
        ts.train(ts.parse_args())
    elif target.endswith("eval/evaluate.py"):
        # Same reason: evaluate through the imported (possibly PPO-patched) module.
        import json  # noqa: E402
        import eval.evaluate as ev  # noqa: E402
        a = ev.parse_args()
        res = ev.evaluate_model(model_dir=a.model_dir, config_path=a.config, n_episodes=a.n_episodes,
                                density=a.density)
        out = Path(a.model_dir).parent / a.output
        json.dump(res, open(out, "w"), indent=2)
        print(f"\n[evaluate] Saved to {out}", flush=True)
    else:
        runpy.run_path(str(ROOT / nxt), run_name="__main__")
