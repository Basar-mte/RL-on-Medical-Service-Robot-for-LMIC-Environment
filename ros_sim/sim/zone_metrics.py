"""Cultural-zone occupancy of a run, from its trajectory CSV.

For the culturally-aware (CARS) policies the outcome that matters besides
arrival is WHERE the robot drove: simulation seconds and metres of path spent
inside the prayer room and inside the gendered wards (the three CULTURAL zones
of layout.py), plus whether it entered them at all. In the walled tier the
nurse-desk -> emergency-room route never needs any of them; in the open tier
(no partitions) the straight line passes through the prayer zone, which is
exactly where a culture-blind planner and a cultural one should differ.

DEPTH matters as much as time. The cultural route planner prices the prayer
zone but has no buffer around it, so in the open tier its route runs along the
zone's edge (measured 2026-10-16: y = 13.05 against an edge at 13.0) and a
tracking robot dips a few decimetres in, while the culture-blind route crosses
the room's centre ~6 m deep. max_depth_m - the deepest point reached, measured
from the zone's nearest edge - separates skimming a boundary from crossing a room.

    python3 zone_metrics.py ../results_runs/open__vanilla__escape-on__peds-busy/run.csv
"""
from __future__ import annotations
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import layout as L          # noqa: E402

DT = 0.1                    # csv_logger rows are one per control tick


def zone_metrics(csv_path, start_t=None):
    """dict: per cultural zone, seconds + metres inside, from the first row with a goal."""
    rows = list(csv.DictReader(open(csv_path)))
    rects = {name: (x0, y0, x1, y1) for (x0, y0, x1, y1, name, _d) in L.ROOMS}
    out = {z: {"seconds": 0.0, "metres": 0.0, "max_depth_m": 0.0} for z in sorted(L.CULTURAL)}
    started = False
    prev = None
    for r in rows:
        if not started:
            if r.get("goal_x", "nan") == "nan":
                continue
            started = True
        x, y = float(r["x"]), float(r["y"])
        z = L.zone_at(x, y)
        if z in out:
            out[z]["seconds"] += DT
            if prev is not None:
                out[z]["metres"] += math.hypot(x - prev[0], y - prev[1])
            x0, y0, x1, y1 = rects[z]
            out[z]["max_depth_m"] = max(out[z]["max_depth_m"], min(x - x0, x1 - x, y - y0, y1 - y))
        prev = (x, y)
    for z in out:
        out[z] = {k: round(v, 2) for k, v in out[z].items()}
    out["entered_cultural_zone"] = any(out[z]["seconds"] > 0 for z in L.CULTURAL)
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, zone_metrics(p))
