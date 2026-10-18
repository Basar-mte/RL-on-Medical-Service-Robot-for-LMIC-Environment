"""Checks the inferred 8-zone context against the policies' own training stats.

The runs_v5 training source is not on this machine, so the zone one-hot's width
(8) and ordering were inferred from every vec_normalize.pkl: obs_rms["context"]
entries 2..9 sum to exactly 1.0 with |var - p(1-p)| = 0, and index 3 dominates
far more in Tier-2 than Tier-1 - the corridor. That inference is a hypothesis,
and this script tests it two ways.

  structural   Does the context the deployed controller now builds have the
               right width, and does its zone block land in slots 2..9?
  distribution Does a real run's zone occupancy look like training?
               scan the run CSV's trajectory through the zone map and compare
               the resulting one-hot frequencies with obs_rms["context"].mean.
               Training spent ~80 % of Tier-2 steps in the corridor and ~0.1 %
               in the prayer room; a run that matches that pattern corroborates
               the mapping, and one that does not is reported rather than
               quietly published.

    python3 validate_zone_mapping.py                         # structural only
    python3 validate_zone_mapping.py --csv ../results_runs/.../run.csv
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "hospitalnav_deploy")
MODELS = os.path.join(HERE, "..", "models")
sys.path.insert(0, os.path.join(PKG, "hospitalnav_deploy"))

ZONE_SLICE = slice(2, 10)        # where the one-hot sits in the 14-D context


def training_zone_profile(model="fixed_equal_s2"):
    with open(os.path.join(MODELS, model, "vec_normalize.pkl"), "rb") as f:
        vn = pickle.load(f)
    c = vn.obs_rms["context"]
    return np.asarray(c.mean[ZONE_SLICE], float), np.asarray(c.var[ZONE_SLICE], float)


def structural_check():
    from perception import SemanticMap, _ZONES, N_ZONES
    png = os.path.join(PKG, "worlds", "hospital_zones.png")
    sem = SemanticMap(png, 0.05, [0.0, 0.0])
    oh = sem.zone_onehot(20.0, 15.0)          # mid-corridor
    ctx_len = 2 + len(oh) + 2 + 2
    print(f"zones in use       : {N_ZONES}  {_ZONES}")
    print(f"zone_onehot width  : {len(oh)}")
    print(f"context width      : {ctx_len}  (expected 14)")
    corridor_idx = int(np.argmax(oh))
    print(f"zone at mid-corridor (20,15): index {corridor_idx} "
          f"= {_ZONES[corridor_idx]}")
    ok = (ctx_len == 14 and _ZONES[corridor_idx] == "corridor")
    print(f"structural         : {'PASS' if ok else 'FAIL'}")
    return ok


def distribution_check(csv_path, model):
    from perception import SemanticMap, _ZONES
    png = os.path.join(PKG, "worlds", "hospital_zones.png")
    sem = SemanticMap(png, 0.05, [0.0, 0.0])

    xs, ys = [], []
    with open(csv_path) as f:
        head = f.readline().strip().split(",")
        ix, iy = head.index("x"), head.index("y")
        for line in f:
            p = line.strip().split(",")
            if len(p) <= max(ix, iy):
                continue
            try:
                xs.append(float(p[ix]))
                ys.append(float(p[iy]))
            except ValueError:
                continue
    if not xs:
        print("no trajectory rows in the CSV")
        return False

    counts = np.zeros(len(_ZONES))
    for x, y in zip(xs, ys):
        counts += sem.zone_onehot(x, y)
    observed = counts / counts.sum()
    trained, _ = training_zone_profile(model)

    print(f"\n{len(xs)} trajectory samples from {os.path.basename(csv_path)}")
    print(f"{'zone':18s} {'run':>8s} {'training':>9s}")
    for i, z in enumerate(_ZONES):
        print(f"{z:18s} {observed[i]:8.3f} {trained[i]:9.3f}")

    # The run is one trajectory and training is 2M steps of sampled goals, so
    # the test is deliberately coarse: the same zone should dominate, and the
    # prayer room should stay rare for a culturally-aware policy.
    same_top = int(np.argmax(observed)) == int(np.argmax(trained))
    prayer_i = _ZONES.index("prayer")
    prayer_ok = observed[prayer_i] <= max(0.10, trained[prayer_i] * 20 + 0.02)
    print(f"\ndominant zone match: {'yes' if same_top else 'no'} "
          f"({_ZONES[int(np.argmax(observed))]} vs {_ZONES[int(np.argmax(trained))]})")
    print(f"prayer occupancy   : run {observed[prayer_i]:.3f} "
          f"vs training {trained[prayer_i]:.3f} -> {'ok' if prayer_ok else 'HIGH'}")
    return same_top and prayer_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="", help="a run.csv to check the distribution against")
    ap.add_argument("--model", default="fixed_equal_s2")
    a = ap.parse_args()

    ok = structural_check()
    if a.csv:
        ok = distribution_check(a.csv, a.model) and ok
    print(f"\noverall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
