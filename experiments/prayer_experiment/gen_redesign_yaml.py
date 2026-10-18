"""Generate the DMCH-redesign floor plan as a navigable YAML.

Emits prayer_experiment/dmch_redesign.yaml in the exact schema FloorPlan reads
(map / rooms / static_obstacles / robot_spawn_points / pedestrian_spawn_zones /
tasks / prayer_windows / shift_change_windows). 40 x 30 landscape, per the
approved redesign_bd3 concept:

  * base 'corridor' room makes the whole floor walkable; functional rooms overlay
    their zone; wall obstacles (with gate gaps) carve the rooms back out.
  * cultural zones: male_ward, female_ward, mosque(prayer). Neutral: general
    ward, ICU, emergency, pharmacy, intern/staff/store, wudu.
  * every TASK-TARGET room is gated to the circulation (validated separately).
  * beds/Mejhe are obstacles, kept clear of gates.

No main-code changes; this only writes a YAML under prayer_experiment/.
"""
from __future__ import annotations
from pathlib import Path
import yaml

OUT = Path(__file__).resolve().parent / "dmch_redesign.yaml"
T = 0.1  # half wall thickness


def _sub(lo, hi, gaps):
    segs, cur = [], lo
    for a, b in sorted(gaps):
        a, b = max(a, lo), min(b, hi)
        if a > cur:
            segs.append((cur, a))
        cur = max(cur, b)
    if cur < hi:
        segs.append((cur, hi))
    return segs


def rect(x0, y0, x1, y1):
    return [[round(x0, 2), round(y0, 2)], [round(x1, 2), round(y0, 2)],
            [round(x1, 2), round(y1, 2)], [round(x0, 2), round(y1, 2)]]


def wall_segs(x0, y0, x1, y1, gates):
    out = []
    for side, e in (("S", y0), ("N", y1)):
        for a, b in _sub(x0, x1, [(g[1], g[2]) for g in gates if g[0] == side]):
            out.append(rect(a, e - T, b, e + T))
    for side, e in (("W", x0), ("E", x1)):
        for a, b in _sub(y0, y1, [(g[1], g[2]) for g in gates if g[0] == side]):
            out.append(rect(e - T, a, e + T, b))
    return out


def bed_col(x, y0, y1, pitch, w, h, skips, horiz=True):
    """Return bed rects up a column, skipping any overlapping a gate range.
    horiz=True -> bed is w wide x h tall (ward bed); else vertical overflow."""
    out, y = [], y0
    while y + h <= y1 + 1e-6:
        if all(y + h <= s0 or y >= s1 for s0, s1 in skips):
            out.append(rect(x, y, x + w, y + h))
        y += pitch
    return out


# (id, type, x0,y0,x1,y1, gates, spawn, cultural, density, label)
ROOMS = [
    ("main_corridor", "corridor", 0, 0, 40, 30, [], False, False, 0.85, "Main Circulation"),
    ("male_ward", "male_ward", 0, 4, 9, 20.3, [("E", 7, 9), ("E", 15, 17)], False, True, 0.85, "Male General Ward"),
    ("female_ward", "female_ward", 31, 4, 40, 20.3, [("W", 7, 9), ("W", 15, 17)], False, True, 0.85, "Female General Ward"),
    ("general_ward", "ward", 15, 9.3, 25, 19.8,
        [("S", 18.5, 21.5), ("N", 18.5, 21.5), ("W", 13.5, 15.5), ("E", 13.5, 15.5)], False, False, 0.8, "General Ward"),
    ("nurse_station", "nurse_station", 15, 9.3, 18, 12.4, [], True, False, 0.9, "Nurse Station"),
    ("mosque", "prayer", 9, 24, 15, 30, [("S", 11, 13)], False, True, 0.4, "Mosque (Namaz Ghor)"),
    ("wudu", "walkable", 9, 22.6, 12, 24, [("N", 10, 11.5)], False, False, 0.3, "Wudu"),
    ("icu", "icu", 25, 25, 31, 30, [("S", 27, 29)], False, False, 0.5, "ICU"),
    ("emergency", "emergency", 0, 0, 9, 4, [("E", 1, 3)], False, False, 0.9, "Emergency Dept"),
    ("pharmacy", "pharmacy", 31, 0, 40, 3.8, [("W", 1, 3)], False, False, 0.7, "Pharmacy"),
    ("store_room", "ward", 26, 0.4, 31, 3.8, [("N", 27.5, 29.5), ("E", 1, 3)], False, False, 0.3, "Store Room"),
    ("male_intern", "ward", 0, 20.3, 4.5, 30, [("S", 1.5, 3)], False, False, 0.4, "Intern Doctor Room"),
    ("male_staff", "ward", 4.5, 20.3, 9, 30, [("S", 6, 7.5)], False, False, 0.4, "Staff Room"),
    ("female_intern", "ward", 31, 20.3, 35.5, 30, [("S", 32.5, 34)], False, False, 0.4, "Intern Doctor Room"),
    ("female_staff", "ward", 35.5, 20.3, 40, 30, [("S", 37, 38.5)], False, False, 0.4, "Staff Room"),
]

MG = [(7, 9), (15, 17)]   # ward gate y-ranges (keep beds clear)


def build():
    rooms_yaml, obstacles = [], []
    for rid, typ, x0, y0, x1, y1, gates, spawn, cult, dens, label in ROOMS:
        r = dict(id=rid, type=typ, polygon=rect(x0, y0, x1, y1),
                 cultural_zone=cult, urgency_level=1, default_pedestrian_density=dens, label=label)
        if spawn:
            r["robot_spawn"] = True
        rooms_yaml.append(r)
        # main_corridor is the walkable base; nurse_station is a zone overlay
        # INSIDE the general ward — neither gets its own walls.
        if rid not in ("main_corridor", "nurse_station"):
            for w in wall_segs(x0, y0, x1, y1, gates):
                obstacles.append(dict(polygon=w, label=f"{rid} wall"))

    # outer wall with entrance/exit gaps
    for w in wall_segs(0, 0, 40, 30, [("S", 3.5, 6), ("S", 17.5, 22.5), ("N", 17.5, 22.5)]):
        obstacles.append(dict(polygon=w, label="Outer wall"))

    # ---- beds (kept clear of gates) -----------------------------------------
    beds = []
    # male ward: left col (head at west wall) full; right col skips east gates
    for r in bed_col(0.4, 4.4, 20.0, 2.6, 2.1, 1.7, []):        beds.append(("Male ward bed", r))
    for r in bed_col(6.5, 4.4, 20.0, 2.6, 2.1, 1.7, MG):        beds.append(("Male ward bed", r))
    for r in bed_col(9.2, 4.5, 19.8, 2.6, 1.4, 2.0, MG):        beds.append(("Male overflow bed", r))
    # female ward mirror
    for r in bed_col(37.5, 4.4, 20.0, 2.6, 2.1, 1.7, []):       beds.append(("Female ward bed", r))
    for r in bed_col(31.4, 4.4, 20.0, 2.6, 2.1, 1.7, MG):      beds.append(("Female ward bed", r))
    for r in bed_col(29.4, 4.5, 19.8, 2.6, 1.4, 2.0, MG):      beds.append(("Female overflow bed", r))
    # general ward perimeter (clear of the 4 doors + nurse corner)
    for r in [rect(15.3, 17.9, 17.4, 19.5), rect(21.8, 17.9, 23.9, 19.5),   # north
              rect(21.8, 9.6, 23.9, 11.2),                                  # south (east of door)
              rect(22.7, 12.6, 24.7, 14.0), rect(22.7, 16.4, 24.7, 17.8),   # east (clear of E door)
              rect(15.2, 16.4, 17.2, 17.8)]:                                # west (clear of W door)
        beds.append(("General ward bed", r))
    # ICU + emergency (ICU beds flank the S door at x27-29, keeping it clear)
    for bx in (25.4, 29.1):
        beds.append(("ICU bed", rect(bx, 25.4, bx + 1.6, 27.3)))
    for k in range(2):
        beds.append(("Emergency bed", rect(0.7 + k * 3.6, 0.7, 3.4 + k * 3.6, 2.4)))
    # a few Mejhe floor mattresses in ward aisles (clear of gates)
    for y in (5.0, 10.2, 12.8, 18.0):
        beds.append(("Mejhe MW", rect(3.7, y, 4.9, y + 0.6)))
        beds.append(("Mejhe FW", rect(35.0, y, 36.2, y + 0.6)))
    for (rid, r) in beds:
        obstacles.append(dict(polygon=r, label=rid))
    # nurse desk (small)
    obstacles.append(dict(polygon=rect(15.3, 11.4, 17.6, 12.0), label="Nurse desk"))

    cfg = dict(
        map=dict(width_m=40.0, height_m=30.0, cell_size_m=0.1),
        rooms=rooms_yaml,
        static_obstacles=obstacles,
        robot_spawn_points=[[16.5, 10.4], [20.0, 7.5], [12.0, 14.5], [28.0, 14.5]],
        pedestrian_spawn_zones=dict(
            medical_staff=[[[15, 9.3], [25, 19.8]], [[9, 4], [15, 24]], [[25, 4], [31, 25]]],
            patient=[[[0, 4], [9, 20.3]], [[31, 4], [40, 20.3]], [[15, 9.3], [25, 19.8]]],
            family_attendant=[[[0, 4], [9, 20.3]], [[31, 4], [40, 20.3]], [[9, 4], [15, 24]], [[25, 4], [31, 25]]],
            wheelchair=[[[9, 4], [31, 9]], [[9, 20], [31, 24]]],
        ),
        tasks={
            "stat_med": dict(start_room_id="nurse_station", target_room_id="emergency", urgency_level=1.0, deadline_s=45, label="STAT Medicine"),
            "ward_call_f": dict(start_room_id="nurse_station", target_room_id="female_ward", urgency_level=0.9, deadline_s=60, label="Female Ward Call"),
            "ward_call_m": dict(start_room_id="nurse_station", target_room_id="male_ward", urgency_level=0.9, deadline_s=60, label="Male Ward Call"),
            "icu_run": dict(start_room_id="nurse_station", target_room_id="icu", urgency_level=0.8, deadline_s=70, label="ICU Transfer"),
            "pharmacy_run": dict(start_room_id="nurse_station", target_room_id="pharmacy", urgency_level=0.6, deadline_s=90, label="Pharmacy Run"),
            "prayer_service": dict(start_room_id="nurse_station", target_room_id="mosque", urgency_level=0.3, deadline_s=200, label="Prayer-room Service"),
        },
        prayer_windows=[
            {"start_hour": 5.0, "end_hour": 6.5, "name": "Fajr"},
            {"start_hour": 12.0, "end_hour": 13.0, "name": "Dhuhr"},
            {"start_hour": 15.0, "end_hour": 16.0, "name": "Asr"},
            {"start_hour": 17.5, "end_hour": 18.5, "name": "Maghrib"},
            {"start_hour": 19.0, "end_hour": 20.0, "name": "Isha"},
        ],
        shift_change_windows=[
            {"start_hour": 7.5, "end_hour": 8.5},
            {"start_hour": 13.5, "end_hour": 14.5},
            {"start_hour": 21.5, "end_hour": 22.5},
        ],
    )
    OUT.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None), encoding="utf-8")
    print(f"wrote {OUT}  ({len(rooms_yaml)} rooms, {len(obstacles)} obstacles, {len(beds)} beds)")


if __name__ == "__main__":
    build()
