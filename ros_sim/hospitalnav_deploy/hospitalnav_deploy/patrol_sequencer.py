"""Goal sequencer for Gazebo runs: drives the trained hierarchy through a list
of destinations and records, per goal, whether the robot actually arrived.

WHY NOT room_patrol + Nav2
  room_patrol sequences goals through Nav2's NavigateToPose action, which means
  a run cannot start until the whole Nav2 lifecycle activates. Two problems on
  ROS 2 Jazzy:

    1. Jazzy's navigation_launch.py brings up a collision_monitor whose output
       IS /cmd_vel (controller_server -> cmd_vel_nav -> velocity_smoother ->
       cmd_vel_smoothed -> collision_monitor -> cmd_vel). With Nav2 active there
       would be two publishers on /cmd_vel - Nav2 and the RL policy's adapter -
       fighting for the robot. Nav2 is not part of the trained hierarchy at all:
       route_planner_node is its global level and the policy is its local one.
    2. That collision_monitor has no section in this campaign's Humble-era
       params file, so lifecycle bringup aborts and the action server never
       appears - room_patrol then waits forever.

  So this node replaces Nav2 for sequencing only. Nothing about the policy, the
  planner, the shield or the observation changes: it publishes the SAME
  /patrol_goal PoseStamped that room_patrol published, and route_planner_node
  consumes it exactly as before.

ARRIVAL
  A goal counts as reached when the robot is within goal_tol of it - the same
  GOAL_TOL (0.4 m) the training environment used to end an episode. Each goal
  also has a SIMULATION-TIME budget; exceeding it is recorded as a timeout, not
  a silent retry, so a run's outcome is always one of reached / timeout.

CLOCK
  Every duration here is simulation time (the node runs with use_sim_time and
  reads its own clock). The worlds run gz unthrottled, so wall-clock budgets
  would have shrunk or grown with machine load - the same run would time out
  on a busy machine and succeed on an idle one. Sim-time budgets make a run's
  outcome independent of how fast the simulator happened to execute.

OUTPUT
  /patrol_status (String) live, and a JSON summary at result_path listing every
  goal with its outcome, duration and path length. That file is the run record -
  per-goal binary outcomes, never a synthesised success rate.

PEDESTRIANS
  With dynamic pedestrians in the world (ped_gt publishes their ground-truth
  poses on /pedestrian_poses) the record also carries proxemic outcomes:
  ped_min_m, the closest any walker came; ped_contacts, the number of distinct
  intrusions inside PED_CONTACT_M (robot radius + body radius, the distance at
  which the physical robot would touch someone - actors have no physics body,
  so this is how a collision with a person is counted); and ped_personal_s,
  simulation seconds spent inside the 1.2 m personal zone.
"""
from __future__ import annotations
import json
import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseArray, PointStamped
from std_msgs.msg import String, Int32MultiArray
from tf2_ros import Buffer, TransformListener

PED_CONTACT_M = 0.25 + 0.25     # robot radius + pedestrian body radius (benchmark definition)
PED_TOUCH_M = 0.35              # MentorPi chassis (~0.12 m) actually touching a body (~0.2 m)
PED_PERSONAL_M = 1.2            # proxemic personal zone

# ROUTE PROTOCOLS
#   direct (default) - the benchmark task: ONE goal, the emergency room, from
#       the nurse-desk spawn. The route planner (the trained hierarchy's global
#       level) chooses the path on the prebuilt map and hands the policy its
#       look-ahead waypoint, exactly as in training. 2026-10-16: the hop list
#       below made every policy time out at the same hop, (18.5, 15), a point
#       wedged between the two standing people at (17.5, 15) and (20, 15) where
#       the pedestrian lanes also converge - it measured a chokepoint the
#       benchmark never posed, not navigation. The hops were only ever a
#       work-around for live-SLAM bootstrapping, which the prebuilt map makes
#       unnecessary.
#   hops - the older corridor-hop tour (room_patrol.WAYPOINTS), kept for
#       demonstrations of the whole floor.
WAYPOINTS_DIRECT = [
    ("emergency room", 5.0, 7.5),
]
WAYPOINTS_HOPS = [
    ("corridor (near nurse desk)", 35.0, 15.0),
    ("corridor (bootstrap 3)", 28.0, 15.0),
    ("corridor (from nurse desk)", 24.0, 15.0),
    ("corridor (mid)", 18.5, 15.0),
    ("corridor (bootstrap 2)", 14.0, 15.0),
    ("corridor (bootstrap 1)", 9.0, 15.0),
    ("corridor (near ER)", 5.0, 14.0),
    ("emergency room", 5.0, 7.5),
    ("corridor (near prayer)", 13.0, 14.0),
    ("prayer room", 13.0, 7.5),
    ("corridor (near ward 3)", 23.5, 14.0),
    ("general ward 3", 23.5, 7.5),
    ("corridor (near ward 2)", 31.0, 14.0),
    ("general ward 2", 31.0, 7.5),
    ("male ward", 7.5, 23.5),
    ("corridor (bootstrap 2b)", 14.0, 15.0),
    ("general ward 1", 20.0, 23.5),
    ("corridor (east)", 35.0, 15.0),
    ("female ward", 32.5, 23.5),
]


class PatrolSequencer(Node):
    def __init__(self):
        super().__init__("patrol_sequencer")
        gp = self.declare_parameter
        self.goal_tol = float(gp("goal_tol", 0.4).value)        # training GOAL_TOL
        self.goal_timeout = float(gp("goal_timeout", 120.0).value)
        self.stop_at = str(gp("stop_at_first_goal", "emergency room").value)
        self.result_path = str(gp("result_path", "/tmp/patrol_result.json").value)
        self.run_label = str(gp("run_label", "run").value)
        self.start_delay = float(gp("start_delay", 20.0).value)
        # Mapless policies take a straight-line goal from goal_publisher and have
        # no route planner, so there is nothing to sequence - but the run still
        # needs the same reached/timeout record. In monitor_only mode this node
        # publishes no goals at all and simply watches the single destination.
        self.monitor_only = bool(gp("monitor_only", False).value)
        mg = list(gp("monitor_goal", [5.0, 7.5]).value)
        self.monitor_goal = (float(mg[0]), float(mg[1]))
        self.route = str(gp("route", "direct").value)
        self.waypoints = WAYPOINTS_HOPS if self.route == "hops" else WAYPOINTS_DIRECT

        self.pub = self.create_publisher(PoseStamped, "/patrol_goal", 10)
        # Mapless arm: the goal goes straight to the policy (/rl_goal, z = 0 =
        # straight-line distance, as those policies were trained) - published
        # HERE, after the same start gate as the planner arms, so every arm is
        # timed, path-integrated and proxemics-counted from the moment it is
        # first given a goal. (2026-10-16: a free-running goal_publisher let the
        # mapless robot drive before the gate, so its record missed the first
        # metres of path and up to start_delay of its budget.)
        self.rl_goal_pub = self.create_publisher(PointStamped, "/rl_goal", 10)
        self.status = self.create_publisher(String, "/patrol_status", 10)
        self.create_subscription(Int32MultiArray, "/escape_count",
                                 self._escapes, 10)
        self.create_subscription(PoseArray, "/pedestrian_poses", self._peds, 10)
        self.peds = []
        self.ped_min = float("inf")
        self.ped_contacts = 0
        self.ped_in_contact = False
        self.ped_touches = 0
        self.ped_in_touch = False
        self.ped_personal_s = 0.0
        self.last_tick = None

        self.tf = Buffer()
        TransformListener(self.tf, self)

        self.idx = 0
        self.t_goal = None
        self.t0 = None                 # set on the first tick that has a sim clock
        self.wall0 = time.time()
        self.started = False
        self.done = False
        self.escapes = [0, 0]
        self.path_len = 0.0
        self.last_xy = None
        self.records: list[dict] = []
        self.create_timer(0.2, self._tick)
        self.get_logger().info(
            f"patrol_sequencer up: route={self.route} ({len(self.waypoints)} goals), "
            f"tol={self.goal_tol} m, budget={self.goal_timeout} s per goal, "
            f"stopping after '{self.stop_at}'")

    def _now(self) -> float:
        """Simulation time in seconds (use_sim_time). 0.0 until /clock arrives -
        never wall time, so no timestamp can mix the two clocks."""
        return self.get_clock().now().nanoseconds * 1e-9

    def _peds(self, m: PoseArray):
        self.peds = [(q.position.x, q.position.y) for q in m.poses]

    def _proxemics(self, x, y, now):
        """Update the pedestrian outcome counters from the ground-truth poses."""
        dt = 0.0 if self.last_tick is None else max(0.0, now - self.last_tick)
        self.last_tick = now
        if not self.peds:
            return
        d = min(math.hypot(px - x, py - y) for px, py in self.peds)
        self.ped_min = min(self.ped_min, d)
        if d < PED_CONTACT_M and not self.ped_in_contact:
            self.ped_contacts += 1
            self.ped_in_contact = True
            self.get_logger().warning(f"pedestrian contact #{self.ped_contacts}: {d:.2f} m")
        elif d > PED_CONTACT_M + 0.2:
            self.ped_in_contact = False
        if d < PED_TOUCH_M and not self.ped_in_touch:
            self.ped_touches += 1
            self.ped_in_touch = True
        elif d > PED_TOUCH_M + 0.2:
            self.ped_in_touch = False
        if d < PED_PERSONAL_M:
            self.ped_personal_s += dt

    def _escapes(self, m: Int32MultiArray):
        if len(m.data) >= 2:
            self.escapes = [int(m.data[0]), int(m.data[1])]

    def _pose(self):
        try:
            t = self.tf.lookup_transform("map", "base_link", rclpy.time.Time())
            return t.transform.translation.x, t.transform.translation.y
        except Exception:
            return None

    def _current(self):
        if self.monitor_only:
            return ("mapless goal", self.monitor_goal[0], self.monitor_goal[1])
        return self.waypoints[self.idx]

    def _publish_goal(self):
        if self.monitor_only:
            g = PointStamped()
            g.header.frame_id = "map"
            g.header.stamp = self.get_clock().now().to_msg()
            g.point.x, g.point.y, g.point.z = self.monitor_goal[0], self.monitor_goal[1], 0.0
            self.rl_goal_pub.publish(g)
            return
        name, gx, gy = self.waypoints[self.idx]
        m = PoseStamped()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.position.x = gx
        m.pose.position.y = gy
        m.pose.orientation.w = 1.0
        self.pub.publish(m)

    def _finish(self, outcome: str):
        self.done = True
        summary = {
            "run_label": self.run_label,
            "route": self.route,
            "outcome": outcome,
            "goals": self.records,
            "escapes": {"stall": self.escapes[0], "wedge": self.escapes[1]},
            "pedestrians": {
                "present": bool(self.peds),
                "min_distance_m": round(self.ped_min, 3) if self.peds else None,
                "contacts": self.ped_contacts,
                "touches": self.ped_touches,
                "personal_zone_s": round(self.ped_personal_s, 1),
            },
            "sim_time_s": round(self._now() - (self.t0 or 0.0), 1),
            "wall_time_s": round(time.time() - self.wall0, 1),
            "path_length_m": round(self.path_len, 2),
            "clock": "sim",
        }
        try:
            with open(self.result_path, "w") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            self.get_logger().error(f"could not write {self.result_path}: {e}")
        self.get_logger().info(f"PATROL {outcome.upper()} -> {self.result_path}")
        self.status.publish(String(data=f"done:{outcome}"))

    def _tick(self):
        if self.done:
            return
        now = self._now()
        if now <= 0.0:
            return                     # no simulation clock yet
        if self.t0 is None:
            self.t0 = now
        if not self.started:
            if now - self.t0 < self.start_delay:
                return
            self.started = True
            self.t_goal = now
            self.get_logger().info(f"patrol starting (sim t={now:.1f} s)")

        p = self._pose()
        if p is None:
            # no pose this tick - but the budget still runs out: decide on the
            # last known position (measured 2026-10-17: a pose outage let two
            # runs overrun the 300 s budget by 58 and 75 s before timing out)
            if (self.last_xy is None or self.t_goal is None
                    or (now - self.t_goal) <= self.goal_timeout):
                return
            p = self.last_xy
        x, y = p
        if self.last_xy is not None:
            self.path_len += math.dist((x, y), self.last_xy)
        self.last_xy = (x, y)
        self._proxemics(x, y, now)

        self._publish_goal()
        name, gx, gy = self._current()
        d = math.hypot(gx - x, gy - y)
        n_total = 1 if self.monitor_only else len(self.waypoints)
        self.status.publish(String(
            data=f"{self.idx+1}/{n_total} -> {name}: {d:.2f} m"))

        reached = d <= self.goal_tol
        timed_out = (now - self.t_goal) > self.goal_timeout
        if not (reached or timed_out):
            return

        self.records.append({
            "goal": name, "x": gx, "y": gy,
            "outcome": "reached" if reached else "timeout",
            "final_distance_m": round(d, 3),
            "seconds": round(now - self.t_goal, 1),
            "escapes_so_far": {"stall": self.escapes[0], "wedge": self.escapes[1]},
        })
        self.get_logger().info(
            f"goal '{name}': {'REACHED' if reached else 'TIMEOUT'} "
            f"(d={d:.2f} m, {now - self.t_goal:.0f} s)")

        if timed_out and not reached:
            self._finish("timeout")
            return
        if self.monitor_only:
            self._finish("reached")
            return
        if self.stop_at and name == self.stop_at:
            self._finish("reached")
            return
        self.idx += 1
        self.t_goal = now
        if self.idx >= len(self.waypoints):
            self._finish("completed_all")


def main():
    rclpy.init()
    node = PatrolSequencer()
    try:
        rclpy.spin(node)
    finally:
        if not node.done:
            node._finish("aborted")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
