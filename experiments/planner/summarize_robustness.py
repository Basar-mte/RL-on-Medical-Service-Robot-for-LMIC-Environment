"""Summarise the unpredicted-uncertainty suite: one row per condition x driver x shield,
with the clean-world reference rows for contrast.

    python experiments/planner/summarize_robustness.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
RES = ROOT / "runs_v5" / "results" / "tier2" / "robustness"
CDIR = ROOT / "runs_v5" / "results" / "tier2" / "classical"
DRIVERS = ["fe", "van", "sf2"]
LABEL = {"fe": "learned fixed_equal[cult]", "van": "learned vanilla[geo]", "sf2": "classical sf2"}


def load(cond, drv, sh):
    """Return pooled records for condition x driver x shield (merging unmapped layouts)."""
    recs = []
    if cond == "unmapped":
        for i in range(50):
            f = (RES / f"unmapped_L{i}__{drv}{'_shield2' if sh else ''}.json") if drv != "sf2" else \
                (CDIR / f"planner_sf2__unmapped_L{i}__sf2{'_shield2' if sh else ''}.json")
            if f.exists():
                recs += json.load(open(f))["records"]
    else:
        f = (RES / f"{cond}__{drv}{'_shield2' if sh else ''}.json") if drv != "sf2" else \
            (CDIR / f"planner_sf2__{cond}__sf2{'_shield2' if sh else ''}.json")
        if f.exists():
            recs = json.load(open(f))["records"]
    return recs


def row(recs):
    n = len(recs)
    if n == 0:
        return None
    sr = np.mean([r["success"] for r in recs])
    co = np.mean([r["end"] == "collision" for r in recs])
    to = np.mean([r["end"] == "timeout" for r in recs])
    svr = np.mean([r["svr"] for r in recs])
    spl = np.mean([r["spl"] for r in recs])
    return f"SR {100*sr:5.1f}%  coll {100*co:5.1f}%  tout {100*to:5.1f}%  SVR {svr:.3f}  SPL {spl:.2f}  n={n}"


CLEAN = {("fe", False): "SR  81.0 (clean ref)", ("fe", True): "SR  94.3 (clean ref)",
         ("van", False): "SR  84.7 (clean ref)", ("van", True): "SR  95.3 (clean ref)",
         ("sf2", False): "SR  95.4 (clean ref)", ("sf2", True): "SR  97.8 (clean ref)"}

for cond in ("unmapped", "noyield", "lidar", "act"):
    print(f"\n=== {cond} ===")
    for drv in DRIVERS:
        for sh in (False, True):
            r = row(load(cond, drv, sh))
            tag = f"{LABEL[drv]}{' + v2 shield' if sh else ''}"
            print(f"  {tag:<38} {r if r else '(pending)'}   [{CLEAN[(drv, sh)]}]")
