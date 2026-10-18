"""Bridges the controller/shield's [v, steering-angle] to the [v, yaw-rate] Twist
gz-sim's AckermannSteering plugin consumes, and optionally provides the
stall/wedge escape the physical MentorPi's bridge performs.

WHEELBASE: WHY 0.20 AND NOT THE ROBOT'S REAL 0.137
  The policy was trained against a bicycle integrator that advances heading by
  yaw += (v / 0.20) * tan(delta) * dt  (training/hospitalnav_lite_env.py), so a
  commanded steering angle delta MEANS the curvature kappa = tan(delta) / 0.20.

  gz's AckermannSteering takes a Twist and derives its own steering angles from
  linear.x and angular.z using the URDF's <wheel_base>. The curvature the robot
  actually drives is therefore omega / v - the plugin's wheel_base cancels out
  entirely. So publishing omega = (v / L) * tan(delta) realises kappa =
  tan(delta) / L, and L here must be the TRAINING wheelbase for the robot to
  trace the arc the policy intended.

  Using the chassis's real 0.137 m instead (as an earlier port did) makes every
  commanded arc 0.20/0.137 = 1.46x tighter than trained. The real chassis can
  still execute the trained curvature: at full lock it needs atan(tan(30deg) *
  0.137 / 0.20) = 21.6 deg of its +/-30 deg range.

ESCAPE (escape_enabled, default true)
  A real Ackermann chassis has a minimum turning radius and cannot rotate in
  place, so it can deadlock in ways the holonomic box robot of the original
  benchmark never could. Two cases, both recovered the way the physical robot's
  bridge recovers:
    1. commanded-zero stall - the shield holds v=0 and the policy never
       re-commands motion. Recovery: straight reverse.
    2. wedge - the policy keeps commanding motion with saturated steering while
       the robot is physically pinned, so it never moves. Invisible to (1)
       because the command is non-zero. Recovery: reverse with COUNTER-steer,
       the middle leg of a three-point turn - the only manoeuvre that gains
       heading for a car-like robot.
  Both refuse to reverse into an occupied rear sector, and both own /cmd_vel
  outright while active.

  An escape is an INTERVENTION that the published benchmark did not have, so it
  is counted and published on /escape_count for the run log. Set
  escape_enabled:=false for the no-intervention arm of the ablation.

Subscribes: /cmd_shielded (Twist: linear.x = v m/s, angular.z = steering rad),
            /odom (actual displacement), /scan (rear clearance)
Publishes:  /cmd_vel (Twist: linear.x = v m/s, angular.z = yaw rate rad/s),
            /escape_count (Int32MultiArray: [n_stall, n_wedge])
"""
from __future__ import annotations
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32MultiArray

WHEELBASE = 0.20          # TRAINING wheelbase - see module docstring
MAX_STEER = math.radians(30.0)

STALL_S = 8.0             # commanded ~zero this long, mid-mission = stuck
WEDGE_S = 6.0             # commanding motion this long...
WEDGE_M = 0.10            # ...having moved less than this = wedged
ESCAPE_V = -0.3           # reverse speed (m/s)
ESCAPE_S = 1.6            # straight-reverse duration (~0.5 m)
WEDGE_ESCAPE_S = 2.2      # counter-steer reverse: longer, it must gain heading
ESCAPE_COOLDOWN = 12.0
REAR_MIN_M = 0.35         # refuse to reverse with less rear clearance
REAR_HALF_DEG = 25        # +/- degrees around dead astern


class GzCmdAdapter(Node):
    def __init__(self):
        super().__init__("gz_cmd_adapter")
        self.escape_enabled = bool(
            self.declare_parameter("escape_enabled", True).value)
        self.wheelbase = float(
            self.declare_parameter("wheelbase", WHEELBASE).value)

        self.create_subscription(Twist, "/cmd_shielded", self._cmd, 10)
        self.create_subscription(Odometry, "/odom", self._odom, 10)
        self.create_subscription(LaserScan, "/scan", self._scan,
                                 qos_profile_sensor_data)
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.count_pub = self.create_publisher(Int32MultiArray, "/escape_count", 10)

        self.had_motion = False
        self.zero_since: float | None = None
        self.last_delta = 0.0
        self.cmd_moving_since: float | None = None
        self.ref_xy: tuple[float, float] | None = None
        self.xy: tuple[float, float] | None = None
        self.rear_m: float | None = None

        self.escape_until = 0.0
        self.escape_steer = 0.0
        self.last_escape = 0.0
        self.n_stall = 0
        self.n_wedge = 0

        self.create_timer(1.0 / 15.0, self._tick)
        self.create_timer(1.0, self._publish_counts)
        self.get_logger().info(
            f"gz_cmd_adapter up: wheelbase={self.wheelbase} (training curvature), "
            f"escape={'ENABLED' if self.escape_enabled else 'DISABLED'}")

    # ---- plumbing ----
    def _now(self) -> float:
        """Simulation time (use_sim_time): the stall/wedge thresholds are then
        robot-seconds, not machine-load seconds, exactly as a physical robot's
        own clock would measure them. Wall time only until /clock arrives."""
        return self.get_clock().now().nanoseconds * 1e-9

    def _odom(self, m: Odometry):
        self.xy = (m.pose.pose.position.x, m.pose.pose.position.y)

    def _scan(self, m: LaserScan):
        # Derive astern from the scan's OWN metadata: the real LD19 publishes
        # angle_min=0 (index 180 is astern) but the sim lidar publishes
        # angle_min=-pi (index 180 is dead AHEAD). Hard-coding 180 reads FORWARD
        # clearance in sim and refuses every escape as "blocked behind".
        n = len(m.ranges)
        if n < 8 or m.angle_increment == 0.0:
            return
        rear_i = int(round((math.pi - m.angle_min) / m.angle_increment)) % n
        half = max(1, int(round(math.radians(REAR_HALF_DEG) / abs(m.angle_increment))))
        vals = [m.ranges[(rear_i + k) % n] for k in range(-half, half + 1)]
        vals = [v for v in vals if 0.05 < v < m.range_max]
        self.rear_m = min(vals) if vals else None

    def _rear_clear(self) -> bool:
        return self.rear_m is None or self.rear_m > REAR_MIN_M

    def _publish_counts(self):
        m = Int32MultiArray()
        m.data = [self.n_stall, self.n_wedge]
        self.count_pub.publish(m)

    # ---- the conversion itself ----
    def _cmd(self, t: Twist):
        now = self._now()
        v, delta = t.linear.x, t.angular.z
        if self.escape_enabled and now < self.escape_until:
            return                                  # escape owns /cmd_vel
        self.last_delta = delta
        if now <= 0.0:
            pass                                    # no sim clock yet: convert, but no timers
        elif abs(v) > 0.05:
            self.had_motion = True
            self.zero_since = None
            if self.cmd_moving_since is None:
                self.cmd_moving_since = now
                self.ref_xy = self.xy
        else:
            self.cmd_moving_since = None
            self.ref_xy = None
            if self.had_motion and self.zero_since is None:
                self.zero_since = now
        out = Twist()
        out.linear.x = v
        out.angular.z = (v / self.wheelbase) * math.tan(delta)
        self.pub.publish(out)

    # ---- escapes ----
    def _begin_escape(self, now: float, dur: float, steer: float):
        self.last_escape = now
        self.escape_until = now + dur
        self.escape_steer = steer
        self.zero_since = None
        self.cmd_moving_since = None
        self.ref_xy = None

    def _tick(self):
        if not self.escape_enabled:
            return
        now = self._now()
        if now <= 0.0:
            return
        if now < self.escape_until:
            out = Twist()
            out.linear.x = ESCAPE_V
            # reversing with a steering angle follows the same bicycle model;
            # negative v flips the turn direction, which is exactly what makes
            # this the middle leg of a three-point turn
            out.angular.z = (ESCAPE_V / self.wheelbase) * math.tan(self.escape_steer)
            self.pub.publish(out)
            return
        if now - self.last_escape < ESCAPE_COOLDOWN:
            return

        # 1. commanded-zero stall -> straight reverse
        if self.zero_since is not None and now - self.zero_since > STALL_S:
            if not self._rear_clear():
                self.get_logger().info(
                    f"stall escape held: rear clearance {self.rear_m:.2f} m < {REAR_MIN_M} m",
                    throttle_duration_sec=10.0)
                return
            self.n_stall += 1
            self._begin_escape(now, ESCAPE_S, 0.0)
            self._publish_counts()        # the run record must not lag the event
            self.get_logger().warning(
                f"stall escape #{self.n_stall}: commanded zero for {STALL_S:.0f}s "
                f"mid-mission - reversing straight {abs(ESCAPE_V)*ESCAPE_S:.2f} m")
            return

        # 2. wedge -> counter-steer reverse
        if (self.cmd_moving_since is not None
                and now - self.cmd_moving_since > WEDGE_S
                and self.ref_xy is not None and self.xy is not None):
            moved = math.dist(self.xy, self.ref_xy)
            if moved < WEDGE_M:
                if not self._rear_clear():
                    # pinned front AND rear: let the policy keep trying, but
                    # restart the window so we re-check instead of spinning here
                    self.cmd_moving_since = now
                    self.ref_xy = self.xy
                    return
                steer = max(-MAX_STEER, min(MAX_STEER, -self.last_delta))
                self.n_wedge += 1
                self._begin_escape(now, WEDGE_ESCAPE_S, steer)
                self._publish_counts()
                self.get_logger().warning(
                    f"wedge escape #{self.n_wedge}: commanded motion for "
                    f"{WEDGE_S:.0f}s but moved {moved:.3f} m - reversing with "
                    f"counter-steer {math.degrees(steer):+.0f} deg")
            else:
                self.cmd_moving_since = now
                self.ref_xy = self.xy


def main():
    rclpy.init()
    rclpy.spin(GzCmdAdapter())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
