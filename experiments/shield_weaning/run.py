"""Run train_sac.py (or any main-repo script) with shield weaning injected.

    set WEAN_MODE=wean|fixed|naive        (default wean)
    python experiments/shield_weaning/run.py train/train_sac.py --agent fixed_equal ...
    python experiments/shield_weaning/run.py eval/evaluate.py --model_dir ...

Training injection, all in-process: ts.SAC -> WeanSAC (so SAC(...) and
SAC.load(...) in train_sac.py both become the subclass), the replay buffer
class, the ShieldWrapper around each env factory (outside Monitor: raw LiDAR,
raw action), and the WeanCallback appended via CallbackList.
"""
from __future__ import annotations
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wean  # noqa: E402

ROOT = wean.ROOT


def _inject():
    import train.train_sac as ts
    from stable_baselines3.common import callbacks as sb3_cb

    ts.SAC = wean.WeanSAC
    ts.SAC_KWARGS["replay_buffer_class"] = wean.WeanReplayBuffer

    orig_make_env = ts.make_env

    def make_env(*a, **kw):
        factory = orig_make_env(*a, **kw)

        here = str(wean.HERE)

        def _thunk():
            # SubprocVecEnv workers unpickle this closure in a fresh process:
            # import by name there rather than capturing the module object.
            import importlib
            if here not in sys.path:
                sys.path.insert(0, here)
            w = importlib.import_module("wean")
            return w.ShieldWrapper(factory())
        return _thunk
    ts.make_env = make_env

    # train_sac builds CallbackList(callbacks) exactly once, for model.learn.
    orig_cl = ts.CallbackList

    def CallbackList(cbs, *a, **kw):
        if isinstance(cbs, list) and any(isinstance(c, sb3_cb.EvalCallback) for c in cbs):
            cbs = cbs + [wean.WeanCallback(total_timesteps=_TOTAL[0])]
        return orig_cl(cbs, *a, **kw)
    ts.CallbackList = CallbackList


_TOTAL = [1_000_000]

if __name__ == "__main__":
    script = sys.argv[1]
    sys.argv = [script] + sys.argv[2:]
    if Path(script).as_posix().endswith("train/train_sac.py"):
        _inject()
        import train.train_sac as ts
        args = ts.parse_args()
        _TOTAL[0] = args.timesteps
        print(f"[shield_weaning] mode={wean.MODE} rho0={wean.RHO0} anneal_frac={wean.ANNEAL_FRAC}",
              flush=True)
        ts.train(args)
    else:
        runpy.run_path(str(ROOT / script), run_name="__main__")
