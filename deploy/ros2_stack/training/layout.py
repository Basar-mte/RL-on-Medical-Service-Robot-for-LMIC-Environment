"""Single source of truth for the 40x30 m hospital floor plan: walls, rooms,
furniture, and zone rectangles. Consumed by generate_world_assets.py (emits
the Gazebo .world + hospital_zones.png) and hospitalnav_lite_env.py
(training). Origin is the map's bottom-left corner (0,0) -> (MAP_W, MAP_H),
matching the [0,map_w] normalization already hardcoded in controller_node.py
and the [0,0]-origin SemanticMap in perception.py.

Seven real, fully-walled rooms, each with exactly one doorway onto the
corridor/open area (no interior connecting doors between rooms — every room
is reached from the corridor, same as a real ward floor): male_ward,
general_ward_1 (the old nurse-station footprint, now a general ward),
female_ward line the north strip; ward_icu_emerg (emergency), prayer,
general_ward_3, and general_ward_2 sit south of the corridor as one
contiguous row of adjoining rooms. There is no "nurse station" room — just a
small desk (see NURSE_DESK) tucked into the corridor corner right outside
female_ward. Everything else (the corridor and the remaining open south
area) is unwalled "walkable"/"corridor" floor.
"""
import math

MAP_W = 40.0
MAP_H = 30.0
WALL_T = 0.2      # wall thickness (m)
WALL_H = 1.2      # wall height (m), for Gazebo only
# Doorway width: 1.8 m double-leaf hospital standard. Ward/ED doors must
# pass 0.9 m-wide patient beds, so 1.0 m was unrealistically narrow — and it
# measurably crippled RL training (2026-08-04: all agents ~10x below the
# old benchmark's SR at equal steps; a scripted A* oracle managed only 1/10).
DOOR_W = 1.8      # doorway gap width (m)

# Zone index table — MUST match perception.py's _ZONES exactly.
ZONES = ["walkable", "corridor", "general_ward_1", "general_ward_2",
         "ward_icu_emerg", "prayer", "male_ward", "female_ward", "general_ward_3"]
CULTURAL = {"prayer", "male_ward", "female_ward"}

# Named rooms: (x0, y0, x1, y1, zone_name, doors). doors is a list of
# (wall, center) pairs — wall in {"N","S","E","W"} names which side the
# doorway is cut into, center is the coordinate (x for N/S, y for E/W) of
# its middle. Every room has exactly one doorway, onto the corridor/open
# area — no interior doors between adjoining rooms.
ROOMS = [
    (0.0,  17.0, 13.0, 30.0, "male_ward",       [("S", 6.5)]),
    (13.0, 17.0, 27.0, 30.0, "general_ward_1",  [("S", 20.0)]),
    (27.0, 17.0, 40.0, 30.0, "female_ward",      [("S", 33.5)]),
    (1.0,  2.0,  9.0,  13.0, "ward_icu_emerg",   [("N", 5.0)]),
    # prayer has 2 gates: one onto the open south area (N), one onto
    # SOUTH_SIDE_CORRIDOR (E) — general_ward_3 is detached from prayer by
    # that corridor, not sharing a wall with it.
    (9.0,  2.0,  17.0, 13.0, "prayer",           [("N", 13.0), ("E", 7.5)]),
    (20.0, 2.0,  27.0, 13.0, "general_ward_3",   [("N", 23.5)]),
    (27.0, 2.0,  35.0, 13.0, "general_ward_2",   [("N", 31.0)]),
]
CORRIDOR = (0.0, 13.0, 40.0, 17.0)          # y in [13,17], full width
# a second, narrower corridor strip separating prayer from general_ward_3
# (south side of the floor) — same idea as the main CORRIDOR, just local to
# this one gap instead of running the full width of the map.
SOUTH_SIDE_CORRIDOR = (17.0, 2.0, 20.0, 13.0)
# everything else (south of y=13, outside the 4 south rooms/side corridor)
# is "walkable".

# static "people" obstacles (x, y, radius) — used for both Gazebo cylinders
# and lidar obstacle raycasting in training.
PEOPLE = [
    (20.0, 15.0, 0.18),   # in the corridor
    (35.5, 16.7, 0.18),   # near the nurse desk / female-ward far corner
    (17.5, 15.0, 0.18),   # in the corridor
]

# default start/goal-sampling free rectangle (avoids spawning inside walls)
FREE_MARGIN = 0.5

# Canonical spawn point: right in front of the nurse desk, in the corridor
# at female_ward's far (south-east-ish) corner (matches
# gazebo_full_bringup.launch.py's x_pose/y_pose defaults) — the robot's
# first patrol leg drives from here to the emergency room.
START_POINT = (37.0, 15.0)

# Small furniture-only desk (visual + real collision, NOT a room/zone) —
# "the nurse station" is just this desk, tucked into the corridor corner
# right outside female_ward, near x=37 along its south wall (y=17).
NURSE_DESK = (36.4, 15.95, 37.6, 16.85)

# Hospital beds — two facing rows per ward room, headboards against the two
# side walls (perpendicular to that room's entrance, "wall-facing"), foot
# ends toward a clear centre aisle. Real collision (the robot must route
# around them, same as walls/furniture in training). Not placed in the
# emergency room or prayer room.
BED_LEN = 1.9   # extends away from the wall, into the room
BED_W = 0.9     # spacing dimension along the wall


def _beds_two_rows(x0, y0, x1, y1, y_centers):
    """Two rows of beds against the room's west (x0) and east (x1) walls,
    headboard-to-wall, one bed per y_center in each row."""
    beds = []
    for yc in y_centers:
        beds.append((x0 + WALL_T / 2, yc - BED_W / 2, x0 + WALL_T / 2 + BED_LEN, yc + BED_W / 2))
        beds.append((x1 - WALL_T / 2 - BED_LEN, yc - BED_W / 2, x1 - WALL_T / 2, yc + BED_W / 2))
    return beds


BEDS = (
    _beds_two_rows(0.0, 17.0, 13.0, 30.0, [20.0, 24.0, 28.0])   # male_ward
    + _beds_two_rows(13.0, 17.0, 27.0, 30.0, [20.0, 24.0, 28.0])  # general_ward_1
    + _beds_two_rows(27.0, 17.0, 40.0, 30.0, [20.0, 24.0, 28.0])  # female_ward
    + _beds_two_rows(20.0, 2.0, 27.0, 13.0, [5.0, 8.5])           # general_ward_3
    + _beds_two_rows(27.0, 2.0, 35.0, 13.0, [5.0, 8.5])           # general_ward_2
    # corridor overflow beds — outside general_ward_1 specifically (not
    # outside emergency/prayer), pushed against the corridor's north wall
    # with their wide (long) side flush against it; irregular spacing
    # (not evenly centred on the ward's door) mirrors real ad-hoc overflow.
    + [(15.0 - BED_LEN / 2, 15.85, 15.0 + BED_LEN / 2, 16.75),
       (23.5 - BED_LEN / 2, 15.85, 23.5 + BED_LEN / 2, 16.75)]
)

# V4 crowding (2026-08-03): extra overflow beds for the fresh benchmark
# campaign — the corridor gets beds along BOTH walls and every ward gets
# denser rows. Placement keeps clear: all door lanes (+0.3 m margin), the
# x=[17,20] junction into SOUTH_SIDE_CORRIDOR, the nurse-desk corner and
# START_POINT, and a >=1.7 m corridor lane wherever both walls carry beds
# (the fatal micro-aisles of the old sim campaigns were 0.8 m).
EXTRA_BEDS = (
    # corridor, north wall only (same row as the two ad-hoc overflow beds).
    # A south-wall mirror row was tried and removed 2026-08-04: beds on BOTH
    # corridor walls left 1.7 m pinches that made even the door-free
    # idle_return task unlearnable (0/5 at 180k steps).
    [(c - BED_LEN / 2, 15.85, c + BED_LEN / 2, 16.75)
     for c in (3.5, 10.5, 27.0, 30.8)]
    # denser ward rows: two extra y-centers per north ward, one per south ward
    + _beds_two_rows(0.0, 17.0, 13.0, 30.0, [22.0, 26.0])
    + _beds_two_rows(13.0, 17.0, 27.0, 30.0, [22.0, 26.0])
    + _beds_two_rows(27.0, 17.0, 40.0, 30.0, [22.0, 26.0])
    + _beds_two_rows(20.0, 2.0, 27.0, 13.0, [11.0])
    + _beds_two_rows(27.0, 2.0, 35.0, 13.0, [11.0])
)
BEDS = BEDS + EXTRA_BEDS

# All static furniture obstacles (beds + the nurse desk), for training's
# obstacle raycasting/free-point checks — kept separate from BEDS/NURSE_DESK
# above so Gazebo can still render them with distinct colors/heights.
FURNITURE = BEDS + [NURSE_DESK]

# Visual-only floor tinting + RViz text labels — NOT consulted by zone_at() or
# any training/perception code (it's purely cosmetic, generated straight from
# ROOMS below), so it never changes the RL observation.
ROOM_COLOR = {
    "male_ward":       (0.25, 0.45, 0.90),
    "general_ward_1":  (0.30, 0.80, 0.45),
    "female_ward":     (0.90, 0.35, 0.60),
    "ward_icu_emerg":  (0.95, 0.25, 0.15),
    "prayer":          (0.85, 0.75, 0.20),
    "general_ward_3":  (0.45, 0.65, 0.85),
    "general_ward_2":  (0.55, 0.35, 0.85),
}
ROOM_LABEL = {
    "male_ward": "Male Ward",
    "general_ward_1": "General Ward 1",
    "female_ward": "Female Ward",
    "ward_icu_emerg": "Emergency Room",
    "prayer": "Prayer Room",
    "general_ward_3": "General Ward 3",
    "general_ward_2": "General Ward 2",
}


def _wall_with_gap(axis, coord, span0, span1, gap_center, gap_w=DOOR_W):
    g0, g1 = gap_center - gap_w / 2, gap_center + gap_w / 2
    segs = []
    if g0 > span0:
        segs.append((span0, g0))
    if g1 < span1:
        segs.append((g1, span1))
    out = []
    for a, b in segs:
        if axis == "H":     # horizontal wall at y=coord, x in [a,b]
            out.append((a, coord - WALL_T / 2, b, coord + WALL_T / 2))
        else:               # vertical wall at x=coord, y in [a,b]
            out.append((coord - WALL_T / 2, a, coord + WALL_T / 2, b))
    return out


def _room_walls(x0, y0, x1, y1, doors):
    """4 boundary walls for a room rectangle, with a door gap cut into each
    (wall, center) pair in doors. A wall with no matching door is solid."""
    door_map = {}
    for wall, center in doors:
        door_map.setdefault(wall, []).append(center)

    walls = []
    for side, y in (("S", y0), ("N", y1)):
        if side in door_map:
            span0 = x0
            for center in sorted(door_map[side]):
                g0 = center - DOOR_W / 2
                if g0 > span0:
                    walls.append((span0, y - WALL_T / 2, g0, y + WALL_T / 2))
                span0 = center + DOOR_W / 2
            if span0 < x1:
                walls.append((span0, y - WALL_T / 2, x1, y + WALL_T / 2))
        else:
            walls.append((x0, y - WALL_T / 2, x1, y + WALL_T / 2))
    for side, x in (("W", x0), ("E", x1)):
        if side in door_map:
            span0 = y0
            for center in sorted(door_map[side]):
                g0 = center - DOOR_W / 2
                if g0 > span0:
                    walls.append((x - WALL_T / 2, span0, x + WALL_T / 2, g0))
                span0 = center + DOOR_W / 2
            if span0 < y1:
                walls.append((x - WALL_T / 2, span0, x + WALL_T / 2, y1))
        else:
            walls.append((x - WALL_T / 2, y0, x + WALL_T / 2, y1))
    return walls


# Wall rectangles as (x0, y0, x1, y1) axis-aligned boxes — used for both
# Gazebo boxes and lidar ray/box intersection in training.
WALLS = [
    # outer perimeter
    (-WALL_T / 2, -WALL_T / 2, MAP_W + WALL_T / 2, WALL_T / 2),               # south
    (-WALL_T / 2, MAP_H - WALL_T / 2, MAP_W + WALL_T / 2, MAP_H + WALL_T / 2),  # north
    (-WALL_T / 2, -WALL_T / 2, WALL_T / 2, MAP_H + WALL_T / 2),               # west
    (MAP_W - WALL_T / 2, -WALL_T / 2, MAP_W + WALL_T / 2, MAP_H + WALL_T / 2),  # east
]
for (x0, y0, x1, y1, _name, _doors) in ROOMS:
    WALLS += _room_walls(x0, y0, x1, y1, _doors)


def zone_at(x, y):
    """Return the zone name for a point in map coordinates."""
    for (x0, y0, x1, y1, name, _doors) in ROOMS:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return name
    cx0, cy0, cx1, cy1 = CORRIDOR
    if cx0 <= x <= cx1 and cy0 <= y <= cy1:
        return "corridor"
    sx0, sy0, sx1, sy1 = SOUTH_SIDE_CORRIDOR
    if sx0 <= x <= sx1 and sy0 <= y <= sy1:
        return "corridor"
    return "walkable"


def zone_index(name):
    return ZONES.index(name)
