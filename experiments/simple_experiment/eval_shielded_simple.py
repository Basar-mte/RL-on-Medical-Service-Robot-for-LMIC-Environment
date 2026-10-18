"""Inference-time safety shield on the simplified experiment's models.

Repoints eval/eval_shielded.py at this experiment's runs/ and config (it
hardcodes Tier 1) and applies the pedestrian patch. Run after summarize.py
has written runs/results/table.json.

    python experiments/simple_experiment/eval_shielded_simple.py [agents...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patch  # noqa: E402

patch.apply()
import eval.eval_shielded as es  # noqa: E402

es.RUNS = patch.ROOT / patch.RUNS_NAME
es.CONFIG = patch.CONFIG_REL
es.N_EP = 500

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv += ["cwn", "attention", "vanilla", "fixed_equal"]
    es.main()
