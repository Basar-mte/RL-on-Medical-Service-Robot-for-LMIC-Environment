"""Dynamic pedestrians for the gz-sim hospital: walking <actor>s on closed
loops, each with a physical body, driven by the hospitalnav_gz_plugins
PedestrianBodies system.

WHAT IS SIMULATED
  gz-sim's animated actors are what the LD19 actually sees: the probe test on
  2026-10-16 showed the gpu_lidar returning the walking nurse's legs at the
  scan plane, so the deployment stack's own LiDAR people tracker detects them
  exactly as it would a person on the ward. Each actor also has a 70 kg
  collision cylinder ("<actor>_body") so the robot cannot drive through a
  person and a person shoves the robot.

  Walkers keep to their own loop - they do not plan around the robot - but
  they YIELD (plugin <yield>, default on): a walker whose path the robot
  occupies slows to a stop, and after a moment steps around it on whichever
  side the occupancy map leaves room for (or waits, in a doorway). Measured
  2026-10-17: with 45 NON-yielding walkers the 2 kg robot was knocked onto its
  side in 2 of 2 runs - a person modelled as an unstoppable 70 kg piston.
  plugin(yield_=False) keeps the original non-yielding schedule.

  Because a walker can wait, its pose is no longer a function of sim time
  alone: ground truth is what the plugin simulated, published on the gz topic
  /hospitalnav/pedestrians (ped_gt republishes it as /pedestrian_poses) and
  logged per run to peds.csv, which plugin(playback=...) replays exactly for
  the video renderer. pose_at() below is the UNOBSTRUCTED schedule - exact for
  a walker that never met the robot, and the fallback when no plugin runs
  (stills rendered without bodies).

TIMING (measured 2026-10-16, and why the keyframes are dense)
  gz interpolates an actor's keyframes with a gz::math::Spline whose tension
  is the trajectory's tension attribute. tension="1" zeroes every tangent, so
  each leg is walked with ease-in/ease-out (smoothstep) timing: a lidar probe
  watching a 7 m leg fitted smoothstep at 0.18 m mean error and constant speed
  at 0.45 m, and on a 35 m corridor leg the two differ by ~3 m. tension="0" is
  Catmull-Rom, whose tangent at a keyframe is half the chord between its
  neighbours - so on a straight leg cut into EQUAL sub-segments of equal
  duration the interpolation is exactly constant-speed. Every leg is therefore
  emitted as uniform sub-segments of at most SUB_DT seconds, and linear
  interpolation between those keyframes (pose_at) matches the rendered actor
  to within a few centimetres; the only deviation is the Catmull-Rom overshoot
  at a corner keyframe, bounded by ~15 % of one sub-segment.

ROUTES
  Each route is a closed loop of (x, y) waypoints in the world frame of the
  40 x 30 m floor, walked at a constant speed with a short pause-and-turn at
  every corner. They are drawn for the walled tier (every room is entered
  through its own doorway) and are therefore valid in the open tier too.
  Lanes keep clear of the trained static obstacles (the two corridor overflow
  beds, the three standing people and the nurse desk) so a route never
  "walks through" mapped geometry: sim/route_clearance.py checks every route
  of a level against the walled map (all >= 0.30 m since 2026-10-17, when
  four designed routes were moved off a door jamb, a wall corner and a bed).

LEVELS
  none   no actors (the benchmark's static world)
  light  4 walkers: one long corridor patrol, a doctor doing rounds between
         General Ward 1 and the Emergency Room, a visitor walking from the
         Female Ward to the Prayer Room, a slow patient with a cane
  busy   all 8: adds a nurse crossing the corridor, a second visitor at the
         Male Ward, a brisk doctor on the east corridor and a patient in the ER
  crowd45  the 8 designed walkers + 37 generated ones (crowd_routes.py, made by
         sim/gen_crowd_routes.py): 20 corridor walkers on six lanes and 17
         visitors entering every room through its doorway, each loop planned
         on the walled map with 0.32 m clearance from mapped obstacles

PEDESTRIAN-PEDESTRIAN CONTACT
  The bodies carry category_bitmask 0x0002 and collide_bitmask 0xFFFD:
  gz-physics ignores a pair when ((cat1 & col2) | (cat2 & col1)) == 0, so two
  bodies pass through each other (their scripted actors do too, and a body
  shoved off its actor would put an invisible obstacle where the lidar sees
  nobody), while the robot, walls and furniture (default 0xFFFF both ways)
  still collide with every body.
"""
from __future__ import annotations
import math

ACTOR_MODELS = {
    "nurse":   ("nursefemalewalk",   "NurseFemaleWalk.dae"),
    "doctor":  ("doctorfemalewalk",  "DoctorFemaleWalk.dae"),
    "visitor": ("femalevisitorwalk", "FemaleVisitorWalk.dae"),
    "patient": ("patientwalkingcane", "PatientWalkingCane.dae"),
}
TURN_S = 0.35          # pause at a corner while the heading swings to the next leg
SUB_DT = 0.25          # longest sub-segment along a leg (s); see TIMING above
TENSION = 0.0          # Catmull-Rom; see TIMING above
PED_RADIUS = 0.25      # body radius used for proximity events (matches the shield)
BODY_RADIUS = 0.22     # physical collision cylinder that follows each actor
BODY_MASS = 70.0       # kg: a person - the robot (2 kg) cannot push it, it pushes the robot


# name -> (model, speed m/s, waypoints)
ROUTES = {
    # corridor patrol, the full length both ways
    "ped_nurse_corridor": ("nurse", 1.20, [
        (3.0, 14.0), (38.5, 14.0), (38.5, 15.55), (26.5, 15.5), (12.5, 15.5), (3.0, 15.3)]),
    # doctor's rounds: General Ward 1 door -> corridor -> Emergency Room and back
    "ped_doctor_rounds": ("doctor", 1.10, [
        (20.0, 18.2), (19.4, 16.2), (19.0, 13.6), (5.3, 13.65), (5.0, 12.4), (5.0, 10.0),
        (4.0, 8.0), (5.0, 10.0), (5.0, 12.4), (5.3, 13.65), (19.0, 13.6), (19.4, 16.2)]),
    # visitor: Female Ward -> corridor -> Prayer Room and back
    "ped_visitor_prayer": ("visitor", 1.00, [
        (33.5, 18.2), (33.5, 16.3), (32.0, 14.4), (14.5, 14.4), (13.0, 13.6), (13.0, 11.0),
        (12.0, 8.5), (13.0, 11.0), (13.0, 13.6), (14.5, 14.4), (32.0, 14.4), (33.5, 16.3)]),
    # slow patient with a cane at the General Ward 2 doorway
    "ped_patient_ward2": ("patient", 0.45, [
        (31.0, 12.4), (31.0, 13.8), (28.5, 14.3), (28.5, 13.6), (30.6, 13.6)]),
    # nurse crossing the corridor between General Ward 3 and General Ward 1
    "ped_nurse_cross": ("nurse", 1.00, [
        (23.5, 12.4), (23.5, 13.9), (21.5, 16.3), (20.2, 16.5), (20.0, 17.6), (20.2, 16.5),
        (21.5, 16.3), (23.5, 13.9)]),
    # second visitor at the Male Ward door
    "ped_visitor_male": ("visitor", 0.90, [
        (6.5, 18.5), (6.5, 16.0), (8.0, 15.2), (10.5, 15.2), (8.0, 15.2), (6.5, 16.0)]),
    # brisk doctor on the east corridor, past the nurse desk. Lanes at 15.6 and
    # 14.4 pass 0.6 m from the robot's spawn (37, 15), clear of the 0.5 m
    # contact distance, so a parked robot never logs a contact by geometry alone
    "ped_doctor_east": ("doctor", 1.30, [
        (38.3, 15.6), (33.0, 15.6), (27.0, 15.3), (27.0, 14.4), (38.3, 14.4)]),
    # patient pacing inside the Emergency Room, near its door
    "ped_patient_er": ("patient", 0.50, [
        (6.5, 12.2), (6.5, 10.5), (7.5, 10.5), (7.5, 12.2)]),
}
DESIGNED = list(ROUTES)
try:                               # generated data, 37 walkers
    from .crowd_routes import CROWD  # noqa: E402
except ImportError:                # imported as a top-level module by sim/ scripts
    from crowd_routes import CROWD   # noqa: E402
ROUTES.update(CROWD)
LEVELS = {
    "none": [],
    "light": ["ped_nurse_corridor", "ped_doctor_rounds", "ped_visitor_prayer", "ped_patient_ward2"],
    "busy": DESIGNED,
    "crowd45": DESIGNED + list(CROWD),
}


def keyframes(speed, pts):
    """[(t, x, y, yaw)] keyframes for a closed loop through pts at `speed`.
    Each leg is cut into equal sub-segments of at most SUB_DT seconds (constant
    speed under a Catmull-Rom spline), and every corner - including the loop's
    own closing corner - gets a TURN_S pause so the heading swings in place and
    the last keyframe equals the first."""
    n = len(pts)
    keys = []
    t = 0.0
    yaw0 = math.atan2(pts[1 % n][1] - pts[0][1], pts[1 % n][0] - pts[0][0])
    keys.append((t, pts[0][0], pts[0][1], yaw0))
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        yaw = math.atan2(y1 - y0, x1 - x0)
        if i > 0:                                 # turn in place at this corner
            t += TURN_S
            keys.append((t, x0, y0, yaw))
        dur = math.hypot(x1 - x0, y1 - y0) / speed
        m = max(1, math.ceil(dur / SUB_DT))
        for k in range(1, m + 1):
            f = k / m
            keys.append((t + f * dur, x0 + f * (x1 - x0), y0 + f * (y1 - y0), yaw))
        t += dur
    # closing corner: turn back to the first leg's heading at pts[0]
    t += TURN_S
    keys.append((t, pts[0][0], pts[0][1], yaw0))
    return keys


def schedule(name):
    model, speed, pts = ROUTES[name]
    return keyframes(speed, pts)


def period(name):
    return schedule(name)[-1][0]


def pose_at(name, t, keys=None):
    """(x, y, yaw) of actor `name` at simulation time t (looping)."""
    keys = keys if keys is not None else _keys(name)
    T = keys[-1][0]
    u = t % T if T > 0 else 0.0
    for (t0, x0, y0, w0), (t1, x1, y1, w1) in zip(keys, keys[1:]):
        if t0 <= u <= t1:
            f = (u - t0) / (t1 - t0) if t1 > t0 else 0.0
            dw = (w1 - w0 + math.pi) % (2 * math.pi) - math.pi
            return x0 + f * (x1 - x0), y0 + f * (y1 - y0), w0 + f * dw
    return keys[-1][1], keys[-1][2], keys[-1][3]


_KEYS: dict = {}


def _keys(name):
    if name not in _KEYS:
        _KEYS[name] = schedule(name)
    return _KEYS[name]


def actor_block(name, model, keys, mesh_uri_base="package://hospitalnav_deploy/meshes/fuel"):
    folder, dae = ACTOR_MODELS[model]
    uri = f"{mesh_uri_base}/{folder}/{dae}"
    wps = "".join(
        f"\n          <waypoint><time>{t:.4f}</time><pose>{x:.4f} {y:.4f} 0 0 0 {yaw:.4f}</pose></waypoint>"
        for (t, x, y, yaw) in keys)
    return f"""
    <actor name="{name}">
      <skin><filename>{uri}</filename><scale>1.0</scale></skin>
      <animation name="walk"><filename>{uri}</filename><scale>1.0</scale><interpolate_x>true</interpolate_x></animation>
      <script>
        <loop>true</loop><delay_start>0.0</delay_start><auto_start>true</auto_start>
        <trajectory id="0" type="walk" tension="{TENSION}">{wps}
        </trajectory>
      </script>
    </actor>"""


SPAWN = (37.0, 15.0)    # the robot's start pose: no walker may START near it
SEED_CLEAR_SPAWN = 1.2  # m
SEED_CLEAR_PED = 0.6    # m between two walkers' start points


def phases(level, seed=0):
    """{walker: start phase on its loop (s)} for pedestrian seed `seed`.
    Seed 0 starts every walker at the beginning of its loop - the original
    schedule, so all seed-0 runs reproduce exactly. Any other seed draws each
    walker's starting point uniformly along its own loop (deterministically
    from the seed), redrawing a start within SEED_CLEAR_SPAWN of the robot's
    spawn or SEED_CLEAR_PED of another walker's start: repeated runs then
    meet different crowds, not the same crowd with a little timing noise."""
    import random
    names = LEVELS.get(level, [])
    if not seed:
        return {n: 0.0 for n in names}
    rng = random.Random(f"hospitalnav-peds:{level}:{seed}")
    out, starts = {}, []
    for n in names:
        T = period(n)
        for _ in range(200):
            t0 = rng.uniform(0.0, T)
            x, y, _ = pose_at(n, t0)
            if (math.dist((x, y), SPAWN) >= SEED_CLEAR_SPAWN
                    and all(math.dist((x, y), q) >= SEED_CLEAR_PED for q in starts)):
                break
        out[n] = round(t0, 3)
        starts.append((x, y))
    return out


def body_block(name, keys, t0=0.0):
    x, y, _ = pose_at(name, t0, keys)
    r, h, m = BODY_RADIUS, 1.70, BODY_MASS
    ixx = m * (3 * r * r + h * h) / 12.0
    izz = m * r * r / 2.0
    return f"""
    <model name="{name}_body">
      <pose>{x:.3f} {y:.3f} 0 0 0 0</pose>
      <link name="link">
        <gravity>false</gravity>
        <inertial><mass>{m}</mass><inertia><ixx>{ixx:.3f}</ixx><iyy>{ixx:.3f}</iyy><izz>{izz:.3f}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="c">
          <pose>0 0 {0.03 + h / 2:.3f} 0 0 0</pose>
          <geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry>
          <surface><contact><category_bitmask>2</category_bitmask><collide_bitmask>65533</collide_bitmask></contact></surface>
        </collision>
      </link>
    </model>"""


def plugin(yield_=True, map_yaml=None, log=None, playback=None, phase=None):
    """The PedestrianBodies <plugin> block. map_yaml (a map_server yaml) tells
    sidesteps where there is room; log writes the walkers' simulated poses to a
    CSV; playback replays such a CSV instead of simulating (no robot, no
    yielding - the video renderer)."""
    import os
    opts = [f"<yield>{'true' if yield_ else 'false'}</yield>"]
    if map_yaml:
        import yaml
        meta = yaml.safe_load(open(map_yaml))
        img = meta["image"] if os.path.isabs(meta["image"]) else os.path.join(os.path.dirname(map_yaml), meta["image"])
        opts += [f"<map>{img}</map>", f"<map_origin>{meta['origin'][0]} {meta['origin'][1]}</map_origin>",
                 f"<map_resolution>{meta['resolution']}</map_resolution>"]
    if log:
        opts.append(f"<log>{log}</log>")
    if playback:
        opts.append(f"<playback>{playback}</playback>")
    for n, t0 in (phase or {}).items():
        if t0:
            opts.append(f'<phase walker="{n}">{t0}</phase>')
    body = "".join(f"\n      {o}" for o in opts)
    return f"""
    <plugin filename="hospitalnav_pedestrian_bodies" name="hospitalnav::PedestrianBodies">{body}
    </plugin>"""


def actor_sdf(level, mesh_uri_base="package://hospitalnav_deploy/meshes/fuel", bodies=True, seed=0,
              **plugin_opts):
    """SDF <actor> blocks (+ physical bodies and the plugin) for every route in
    `level`, to splice before </world>. bodies=False with no playback gives the
    plain scripted actors (gz plays the schedule itself: stills). seed picks
    every walker's starting point on its loop (phases(); 0 = loop start)."""
    names = LEVELS[level]
    ph = phases(level, seed)
    out = "".join(actor_block(name, ROUTES[name][0], _keys(name), mesh_uri_base) for name in names)
    if names and bodies:
        out += "".join(body_block(name, _keys(name), ph[name]) for name in names)
    if names and (bodies or plugin_opts.get("playback")):
        out += plugin(phase=ph if seed else None, **plugin_opts)
    return out


def inject_actors(world_sdf, level, mesh_uri_base="package://hospitalnav_deploy/meshes/fuel", bodies=True,
                  seed=0, **plugin_opts):
    if level == "none" or not LEVELS.get(level):
        return world_sdf
    return world_sdf.replace("\n  </world>", actor_sdf(level, mesh_uri_base, bodies, seed, **plugin_opts)
                             + "\n  </world>")


def load_log(path):
    """peds.csv (the plugin's <log>) -> (t array, {name: (x, y, yaw) arrays})."""
    import numpy as np
    with open(path) as f:
        head = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    names = [c[:-2] for c in head[1:] if c.endswith("_x")]
    out = {n: (data[:, 1 + 4 * k], data[:, 2 + 4 * k], data[:, 3 + 4 * k]) for k, n in enumerate(names)}
    return data[:, 0], out


def logged_pose(log, name, t):
    """(x, y, yaw) of `name` at sim time t from a load_log() result."""
    import numpy as np
    ts, d = log
    x, y, w = d[name]
    i = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1))
    f = float(np.clip((t - ts[i - 1]) / max(ts[i] - ts[i - 1], 1e-9), 0.0, 1.0))
    dw = (w[i] - w[i - 1] + math.pi) % (2 * math.pi) - math.pi
    return x[i - 1] + f * (x[i] - x[i - 1]), y[i - 1] + f * (y[i] - y[i - 1]), w[i - 1] + f * dw


if __name__ == "__main__":
    for n in ROUTES:
        print(f"{n:22s} {ROUTES[n][0]:8s} {ROUTES[n][1]:.2f} m/s  loop {period(n):6.1f} s  "
              f"{len(schedule(n))} keyframes")
