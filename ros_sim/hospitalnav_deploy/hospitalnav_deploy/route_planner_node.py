"""Route planner node: the trained hierarchy's global level on the robot.

Plans on the live /map (SLAM or map_server) with the SAME cost field the policies
were trained with (route_planner.py), and publishes the policy's goal input:
  /rl_goal   geometry_msgs/PointStamped (map frame) — the look-ahead waypoint; z carries
             the REMAINING ROUTE LENGTH in metres (the policy's distance channel;
             controller_node uses z when > 0)
  /rl_route  nav_msgs/Path — the full route, for RViz
Goal input (any of): parameters goal_x/goal_y; /nav_goal (PointStamped);
/goal_pose (PoseStamped, RViz "2D Goal Pose"); /patrol_goal (PoseStamped, room_patrol).
Subscribes /map (nav_msgs/OccupancyGrid, transient-local QoS) and TF map->base_link.

Parameters: lookahead 2.5 m, los true, cultural_ratio 3.0 (0 = shortest path for
vanilla/he2025), robot_radius 0.25, safety_margin 0.05, plan_resolution 0.1 (the map
is coarsened conservatively to this), occupied_threshold 50, unknown_is_free false,
tod_hour 12.0 + prayer_windows (flat [start,end,...] hours), semantic_png/res/origin
(the zone-label image used by the controller), replan_period 5.0 s (map updates).
"""
from __future__ import annotations
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PointStamped, PoseStamped
from tf2_ros import Buffer, TransformListener

from .route_planner import RoutePlanner, downsample_blocked, dilate, in_prayer_window, DEFAULT_PRAYER_WINDOWS


class RoutePlannerNode(Node):
    def __init__(self):
        super().__init__("route_planner")
        gp = self.declare_parameter
        self.lookahead = float(gp("lookahead", 2.5).value)
        self.los = bool(gp("los", True).value)
        self.ratio = float(gp("cultural_ratio", 3.0).value)
        self.robot_radius = float(gp("robot_radius", 0.25).value)
        self.margin = float(gp("safety_margin", 0.05).value)
        self.plan_res = float(gp("plan_resolution", 0.1).value)
        self.occ_thr = int(gp("occupied_threshold", 50).value)
        self.unknown_free = bool(gp("unknown_is_free", False).value)
        self.tod_hour = float(gp("tod_hour", 12.0).value)
        pw = list(gp("prayer_windows", [h for w in DEFAULT_PRAYER_WINDOWS for h in w]).value)
        self.windows = [(float(pw[i]), float(pw[i + 1])) for i in range(0, len(pw) - 1, 2)]
        self.replan_period = float(gp("replan_period", 5.0).value)
        gx, gy = float(gp("goal_x", float("nan")).value), float(gp("goal_y", float("nan")).value)
        self.goal = None if (math.isnan(gx) or math.isnan(gy)) else (gx, gy)
        sem_png = gp("semantic_png", "").value
        self.sem = None
        if sem_png:
            from .perception import SemanticMap
            self.sem = SemanticMap(sem_png, float(gp("semantic_res", 0.05).value),
                                   gp("semantic_origin", [0.0, 0.0]).value)

        self.map_msg = None
        self.map_dirty = False
        self.planner = None
        self.planned_goal = None
        self.last_build = 0.0
        self.tf = Buffer(); TransformListener(self.tf, self)

        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(OccupancyGrid, "/map", self._map, map_qos)
        self.create_subscription(PointStamped, "/nav_goal", self._goal_point, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self._goal_pose, 10)
        self.create_subscription(PoseStamped, "/patrol_goal", self._goal_pose, 10)
        self.pub = self.create_publisher(PointStamped, "/rl_goal", 10)
        self.route_pub = self.create_publisher(Path, "/rl_route", 1)
        self.create_timer(0.1, self._tick)
        self.create_timer(1.0, self._publish_route)
        self.get_logger().info(f"route_planner up: lookahead {self.lookahead} m, cultural_ratio {self.ratio}, "
                               f"prayer window now: {in_prayer_window(self.tod_hour, self.windows)}")

    # ---- inputs ----
    def _map(self, m: OccupancyGrid):
        self.map_msg = m
        self.map_dirty = True

    def _goal_point(self, p: PointStamped):
        self._set_goal(p.point.x, p.point.y)

    def _goal_pose(self, p: PoseStamped):
        self._set_goal(p.pose.position.x, p.pose.position.y)

    def _set_goal(self, x, y):
        if self.goal is None or math.hypot(self.goal[0] - x, self.goal[1] - y) > 0.05:
            self.goal = (x, y)
            self.get_logger().info(f"goal -> ({x:.2f}, {y:.2f})")

    def _pose(self):
        try:
            t = self.tf.lookup_transform("map", "base_link", rclpy.time.Time())
            return t.transform.translation.x, t.transform.translation.y
        except Exception:
            return None

    # ---- planning ----
    def _build(self):
        m = self.map_msg
        info = m.info
        data = np.asarray(m.data, dtype=np.int16).reshape(info.height, info.width)
        blocked = data >= self.occ_thr
        if not self.unknown_free:
            blocked |= data < 0
        factor = max(1, int(round(self.plan_res / info.resolution)))
        res = info.resolution * factor
        blocked = downsample_blocked(blocked, factor)
        blocked = dilate(blocked, int(math.ceil((self.robot_radius + self.margin) / res)))
        ox, oy = info.origin.position.x, info.origin.position.y
        zone = None
        if self.sem is not None and self.ratio > 0:
            # sample the zone label at each plan-cell centre (SemanticMap: row 0 = top of the image)
            H, W = blocked.shape
            lab = self.sem.lab
            xs = ox + (np.arange(W) + 0.5) * res
            ys = oy + (np.arange(H) + 0.5) * res
            cc = np.clip(((xs - self.sem.ox) / self.sem.res).astype(int), 0, lab.shape[1] - 1)
            rr = np.clip(lab.shape[0] - 1 - ((ys - self.sem.oy) / self.sem.res).astype(int), 0, lab.shape[0] - 1)
            zone = lab[rr[:, None], cc[None, :]].astype(np.int16)
        t0 = time.time()
        self.planner = RoutePlanner(blocked, res, (ox, oy), zone, self.ratio,
                                    in_prayer_window(self.tod_hour, self.windows), self.lookahead, self.los)
        ok = self.planner.set_goal(*self.goal)
        self.planned_goal = self.goal
        self.map_dirty = False
        self.last_build = time.time()
        self.get_logger().info(f"route field built in {time.time() - t0:.1f} s ({blocked.shape[1]}x{blocked.shape[0]} "
                               f"cells @ {res:.2f} m){'' if ok else ' — GOAL UNREACHABLE'}")

    def _tick(self):
        if self.map_msg is None or self.goal is None:
            return
        if (self.planner is None or self.planned_goal != self.goal
                or (self.map_dirty and time.time() - self.last_build > self.replan_period)):
            self._build()
        p = self._pose()
        if p is None:
            return
        wp = self.planner.waypoint(*p)
        if wp is None:
            self.get_logger().warn("no route from the current pose to the goal", throttle_duration_sec=5.0)
            return
        wx, wy, remaining = wp
        msg = PointStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x, msg.point.y, msg.point.z = float(wx), float(wy), float(remaining)
        self.pub.publish(msg)

    def _publish_route(self):
        if self.planner is None or self.planner.length is None:
            return
        p = self._pose()
        if p is None:
            return
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()
        for x, y in self.planner.route_xy(*p):
            ps = PoseStamped(); ps.header = path.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.route_pub.publish(path)


def main():
    rclpy.init(); rclpy.spin(RoutePlannerNode()); rclpy.shutdown()


if __name__ == "__main__":
    main()
