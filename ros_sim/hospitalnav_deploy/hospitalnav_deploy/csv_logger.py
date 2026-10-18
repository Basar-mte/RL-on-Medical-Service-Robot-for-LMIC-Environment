"""Records the numbers from a sim run to a CSV for later inspection/plotting.

Subscribes: /cmd_shielded (Twist), /odom (Odometry), /rl_goal (PointStamped),
            TF map->base_link
Writes one row every tick (DT, simulation time) to the 'csv_path' parameter:
  t (sim s), wall (s), v_cmd, steering_cmd, x, y, yaw, goal_x, goal_y, goal_dist,
  n_people (tracked pedestrians the policy currently sees), ped_min_m (ground-
  truth distance to the nearest simulated pedestrian, nan without actors),
  sim_abs (absolute simulation time - the clock the actor schedule runs on, so
  a row can be matched to ped_routes.pose_at after the fact),
  z (m) and tilt_deg (angle between the chassis up-axis and the world vertical:
  ~0 on the floor; a robot shoved over by a pedestrian shows up here)
"""
from __future__ import annotations
import math
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PointStamped, PoseArray
from std_msgs.msg import Float32MultiArray
from tf2_ros import Buffer, TransformListener
from .scan_utils import DT


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class CsvLogger(Node):
    def __init__(self):
        super().__init__("hospitalnav_csv_logger")
        path = self.declare_parameter("csv_path", "/tmp/hospitalnav_run.csv").value
        self.f = open(path, "w")
        self.f.write("t,wall,v_cmd,steering_cmd,x,y,yaw,goal_x,goal_y,goal_dist,n_people,ped_min_m,sim_abs,z,tilt_deg\n")
        self.wall0 = time.time()
        self.t0 = None
        self.cmd = (0.0, 0.0)
        self.goal = None
        self.n_people = 0
        self.peds = []
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.create_subscription(Twist, "/cmd_shielded", self._cmd, 10)
        self.create_subscription(PointStamped, "/rl_goal", self._goal, 10)
        self.create_subscription(Float32MultiArray, "/people", self._people, 10)
        self.create_subscription(PoseArray, "/pedestrian_poses", self._peds, 10)
        self.create_timer(DT, self._tick)
        self.get_logger().info(f"logging run numbers to {path}")

    def _cmd(self, t: Twist):
        self.cmd = (t.linear.x, t.angular.z)

    def _goal(self, p: PointStamped):
        self.goal = (p.point.x, p.point.y)

    def _people(self, m: Float32MultiArray):
        d = list(m.data)
        self.n_people = sum(1 for i in range(0, len(d) - 1, 4) if abs(d[i]) + abs(d[i + 1]) > 1e-6)

    def _peds(self, m: PoseArray):
        self.peds = [(q.position.x, q.position.y) for q in m.poses]

    def _tick(self):
        try:
            tr = self.tf.lookup_transform("map", "base_link", rclpy.time.Time())
            x = tr.transform.translation.x
            y = tr.transform.translation.y
            q = tr.transform.rotation
            yaw = _yaw(q)
            z = tr.transform.translation.z
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q.x * q.x + q.y * q.y)))))
        except Exception:
            return
        gx, gy = self.goal if self.goal else (float("nan"), float("nan"))
        dist = math.hypot(gx - x, gy - y) if self.goal else float("nan")
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0
        wall = time.time() - self.wall0
        pmin = min((math.hypot(px - x, py - y) for px, py in self.peds), default=float("nan"))
        self.f.write(f"{t:.2f},{wall:.2f},{self.cmd[0]:.4f},{self.cmd[1]:.4f},{x:.3f},{y:.3f},{yaw:.3f},"
                     f"{gx:.3f},{gy:.3f},{dist:.3f},{self.n_people},{pmin:.3f},{now:.2f},{z:.3f},{tilt:.1f}\n")
        self.f.flush()

    def destroy_node(self):
        self.f.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = CsvLogger()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
