"""Run any main-repo script with the simplified-experiment patch applied.

    python experiments/simple_experiment/run.py train/train_sac.py --agent cwn ...
    python experiments/simple_experiment/run.py eval/evaluate.py --model_dir ...

For training, make_env is wrapped so the patch is also re-applied inside
SubprocVecEnv workers (they re-import the env module fresh on spawn).
"""
from __future__ import annotations
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patch  # noqa: E402

ROOT = patch.ROOT


def _wrap_make_env():
    import train.train_sac as ts
    orig = ts.make_env

    def make_env(*args, **kwargs):
        factory = orig(*args, **kwargs)

        def _thunk():
            sys.path.insert(0, str(patch.HERE))
            import patch as p
            p.apply()
            return factory()
        return _thunk

    ts.make_env = make_env


if __name__ == "__main__":
    script = sys.argv[1]
    sys.argv = [script] + sys.argv[2:]
    patch.apply()
    if Path(script).as_posix().endswith("train/train_sac.py"):
        _wrap_make_env()
        import train.train_sac as ts
        ts.train(ts.parse_args())
    else:
        runpy.run_path(str(ROOT / script), run_name="__main__")
