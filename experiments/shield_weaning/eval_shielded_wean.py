"""Inference-time shield on the weaning runs of one mode (WEAN_MODE env var).

Repoints eval/eval_shielded.py (which hardcodes Tier 1) at
runs/<mode>/ and the Tier-2 config, 500 episodes. Must run with `wean`
importable (models pickle a reference to WeanReplayBuffer), which importing
it here ensures. summarize.py must have written runs/<mode>/results/table.json
first (eval_shielded prints before/after against it).

    set WEAN_MODE=wean
    python experiments/shield_weaning/eval_shielded_wean.py fixed_equal attention vanilla
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wean  # noqa: E402  (also makes WeanReplayBuffer unpicklable)
import eval.eval_shielded as es  # noqa: E402

es.RUNS = wean.ROOT / wean.RUNS_NAME
es.CONFIG = wean.CONFIG_REL
es.N_EP = 500

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv += ["fixed_equal"]
    es.main()
