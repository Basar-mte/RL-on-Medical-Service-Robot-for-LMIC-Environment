"""Generate a Gazebo world (SDF) that matches the trained sim exactly, from
configs/dmch_hospital.yaml. Walls/beds/carts become boxes; cultural zones become
coloured floor patches. So the policy sees the same geometry it trained on -> no
retraining.

  python gen_gazebo_world.py            # -> worlds/hospital.world
Run from hardware_deploy/gazebo/. Uses the repo's config two levels up.
"""
from __future__ import annotations
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs" / "dmch_hospital.yaml"
OUT = Path(__file__).resolve().parent / "worlds" / "hospital.world"

ZONE_RGBA = {"prayer": "0.3 0.8 0.3 0.5", "male_ward": "0.3 0.5 0.9 0.4",
             "female_ward": "0.95 0.6 0.3 0.4"}


def box_model(name, cx, cy, sx, sy, h, rgba="0.6 0.6 0.6 1"):
    return f"""    <model name="{name}"><static>true</static>
      <pose>{cx:.3f} {cy:.3f} {h/2:.3f} 0 0 0</pose>
      <link name="l"><collision name="c"><geometry><box><size>{sx:.3f} {sy:.3f} {h:.3f}</size></box></geometry></collision>
      <visual name="v"><geometry><box><size>{sx:.3f} {sy:.3f} {h:.3f}</size></box></geometry>
        <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material></visual></link>
    </model>
"""


def floor_patch(name, cx, cy, sx, sy, rgba):
    return f"""    <model name="{name}"><static>true</static>
      <pose>{cx:.3f} {cy:.3f} 0.01 0 0 0</pose>
      <link name="l"><visual name="v"><geometry><box><size>{sx:.3f} {sy:.3f} 0.02</size></box></geometry>
        <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material></visual></link>
    </model>
"""


def bbox(poly):
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def main():
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    W = cfg["map"]["width_m"]; H = cfg["map"]["height_m"]
    models = []

    # cultural-zone floor patches (visual reference; context comes from the PNG)
    for r in cfg.get("rooms", []):
        if r.get("type") in ZONE_RGBA:
            x0, y0, x1, y1 = bbox(r["polygon"])
            models.append(floor_patch(f"zone_{r['id']}", (x0+x1)/2, (y0+y1)/2, x1-x0, y1-y0, ZONE_RGBA[r["type"]]))

    # obstacles: walls tall, beds/carts/mejhe low
    for i, obs in enumerate(cfg.get("static_obstacles", [])):
        x0, y0, x1, y1 = bbox(obs["polygon"])
        sx, sy = max(x1-x0, 0.05), max(y1-y0, 0.05)
        lbl = obs.get("label", "").lower()
        h = 2.0 if "wall" in lbl else 0.6
        rgba = "0.55 0.55 0.6 1" if "wall" in lbl else "0.8 0.7 0.4 1"
        models.append(box_model(f"obs_{i}", (x0+x1)/2, (y0+y1)/2, sx, sy, h, rgba))

    # outer boundary walls
    t = 0.2
    for name, cx, cy, sx, sy in [("wall_s", W/2, 0, W, t), ("wall_n", W/2, H, W, t),
                                 ("wall_w", 0, H/2, t, H), ("wall_e", W, H/2, t, H)]:
        models.append(box_model(name, cx, cy, sx, sy, 2.0, "0.5 0.5 0.55 1"))

    world = f"""<?xml version="1.0"?>
<sdf version="1.7">
  <world name="hospital">
    <!-- default step (1ms, 1000 Hz) is needlessly expensive for this simple
         scene — 10ms/100Hz matches classic Gazebo's default and cuts physics
         CPU load roughly 10x with no visible difference for this robot. -->
    <physics type="dart"><max_step_size>0.01</max_step_size><real_time_update_rate>100</real_time_update_rate></physics>
    <!-- ROS 2 Humble's official Gazebo pairing is "Fortress" (Ignition Gazebo 6),
         which uses the pre-rebrand ignition-gazebo-* plugin names, not the
         newer gz-sim-* naming from later Gazebo releases. -->
    <plugin filename="ignition-gazebo-physics-system" name="ignition::gazebo::systems::Physics"/>
    <plugin filename="ignition-gazebo-sensors-system" name="ignition::gazebo::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <plugin filename="ignition-gazebo-user-commands-system" name="ignition::gazebo::systems::UserCommands"/>
    <plugin filename="ignition-gazebo-scene-broadcaster-system" name="ignition::gazebo::systems::SceneBroadcaster"/>
    <light type="directional" name="sun"><pose>0 0 10 0 0 0</pose><diffuse>1 1 1 1</diffuse><direction>-0.3 -0.3 -1</direction></light>
    <model name="ground"><static>true</static><link name="l">
      <collision name="c"><geometry><plane><normal>0 0 1</normal><size>{W} {H}</size></plane></geometry></collision>
      <visual name="v"><geometry><plane><normal>0 0 1</normal><size>{W} {H}</size></plane></geometry>
        <material><ambient>0.9 0.9 0.9 1</ambient></material></visual></link>
      <pose>{W/2} {H/2} 0 0 0 0</pose></model>
{''.join(models)}  </world>
</sdf>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(world, encoding="utf-8")
    print(f"wrote {OUT}  ({len(models)} models: walls/beds/zones)")


if __name__ == "__main__":
    main()
