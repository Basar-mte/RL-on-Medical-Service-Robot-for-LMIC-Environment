"""Run eval/eval_shielded.py against the V4 campaign (runs_v4/ +
configs/hospital_v4.yaml) WITHOUT touching main code: import the module,
repoint its RUNS/CONFIG constants, then call its main().

Run AFTER post_training.py has written runs_v4/results/table.json (the
shield table prints before/after against it).

    .venv\\Scripts\\python.exe scripts\\eval_shielded_v4.py [agents...]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval.eval_shielded as es  # noqa: E402

es.RUNS = ROOT / "runs_v4"
es.CONFIG = "configs/hospital_v4.yaml"
es.N_EP = 500  # match the campaign's 500-episode evaluation convention

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv += ["cwn", "attention", "vanilla"]  # paper's shield trio
    es.main()
