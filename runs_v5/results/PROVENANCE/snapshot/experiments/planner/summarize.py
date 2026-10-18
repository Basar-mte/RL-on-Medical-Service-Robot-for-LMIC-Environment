"""Summarise experiments/planner/results/*.json as tables (per variant x model, and
variant means), plus per-task success for the shielded cultural-planner variant.

    python experiments/planner/summarize.py [results_dir]
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
ORDER = ["noplan_shield", "geo_shield", "cult_shield", "cult", "cult3_shield", "cult_L1.5_shield",
         "cult_L4_shield", "cult_geodist_shield", "cult_nolos_shield"]

rows = {}
for f in sorted(RES.glob("*__*.json")):
    model, variant = f.stem.split("__", 1)
    rows[(model, variant)] = json.load(open(f))["summary"]

hdr = f"{'variant':<22}{'model':<16}{'n':>4}{'SR%':>7}{'coll%':>7}{'tout%':>7}{'frozen%':>8}{'CVR':>7}{'prayCVR':>8}{'SPL':>6}"
print(hdr); print("-" * len(hdr))
by_variant = defaultdict(list)
for variant in ORDER + sorted({v for _, v in rows} - set(ORDER)):
    for (model, v), s in sorted(rows.items()):
        if v != variant:
            continue
        by_variant[v].append(s)
        print(f"{v:<22}{model:<16}{s['n']:>4}{100*s['sr']:>7.1f}{100*s['collision']:>7.1f}{100*s['timeout']:>7.1f}"
              f"{100*s['frozen']:>8.1f}{s['cvr']:>7.3f}{s['prayer_cvr']:>8.3f}{s['spl']:>6.2f}")
print("-" * len(hdr))
print("variant means over models:")
for v, ss in by_variant.items():
    m = lambda k: 100 * np.mean([s[k] for s in ss])  # noqa: E731
    print(f"{v:<22}{'(' + str(len(ss)) + ' models)':<16}{'':>4}{m('sr'):>7.1f}{m('collision'):>7.1f}{m('timeout'):>7.1f}"
          f"{m('frozen'):>8.1f}{np.mean([s['cvr'] for s in ss]):>7.3f}{np.mean([s['prayer_cvr'] for s in ss]):>8.3f}"
          f"{np.mean([s['spl'] for s in ss]):>6.2f}")

tasks = ["STAT_medicine", "nurse_call", "specimen", "restock", "idle_return"]
for variant in ("noplan_shield", "cult_shield"):
    print(f"\nper-task SR% — {variant}:")
    print(f"{'model':<16}" + "".join(f"{t[:11]:>12}" for t in tasks))
    for (model, v), s in sorted(rows.items()):
        if v == variant:
            print(f"{model:<16}" + "".join(f"{100*s['per_task'][t]:>12.0f}" for t in tasks))
