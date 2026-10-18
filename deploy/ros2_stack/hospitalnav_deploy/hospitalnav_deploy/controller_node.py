"""Main culturally-aware controller: assembles the 90-D observation, runs the
trained SAC policy, applies the safety shield, and publishes the motor command.
Also publishes the live weight vector for the interpretability overlay.

Deploy a policy trained on the ORIGINAL env (14-D context). Set the agent's
weight checkpoint to show CWN's adaptive weights.

Subscribes: /scan, /odom (velocity), /people (Float32MultiArray), /rl_goal
            (PointStamped, map), TF map->base_link
Publishes:  /cmd_shielded (Twist), /policy_weights (Float32MultiArray [nav,safe,cult,task])
"""
from __future__ import annotations
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, PointStamped
from tf2_ros import Buffer, TransformListener
from .scan_utils import scan_to_lidar48, shield, shield_v2, N_RAYS, DT, MAX_LINEAR, MIN_LINEAR, MAX_STEER
from .classical_sf import AckermannSF


def _yaw(q):
    return math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))


class Policy:
    """SB3 policy + VecNormalize inference (obs dict -> [v, steering])."""
    def __init__(self, model_zip, vn_pkl, algo="sac"):
        import gymnasium as gym
        from stable_baselines3 import SAC, TD3, PPO
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        cls = {"sac": SAC, "td3": TD3, "ppo": PPO}[str(algo).lower()]
        self.model = cls.load(model_zip, device="cpu")
        space = self.model.observation_space

        class _Shell(gym.Env):
            def __init__(s): s.observation_space = space; s.action_space = gym.spaces.Box(-1, 1, (2,))
        self.vn = VecNormalize.load(vn_pkl, DummyVecEnv([lambda: _Shell()]))
        self.vn.training = False; self.vn.norm_reward = False

    def act(self, obs):
        norm = self.vn.normalize_obs({k: v[None] for k, v in obs.items()})
        a, _ = self.model.predict(norm, deterministic=True)
        return float(a[0][0]), float(a[0][1])


class Controller(Node):
    def __init__(self):
        super().__init__("hospitalnav_controller")
        gp = self.declare_parameter
        self.model_zip = gp("model_zip", "").value
        self.vn_pkl    = gp("vecnorm_pkl", "").value
        self.weight_pt = gp("weight_pt", "").value            # optional CWN/attn .pt
        self.agent     = gp("agent", "cwn").value
        self.algo      = gp("algo", "sac").value              # sac / td3 / ppo
        self.map_w     = float(gp("map_w", 40.0).value)
        self.map_h     = float(gp("map_h", 30.0).value)
        self.urgency   = float(gp("urgency", 0.9).value)
        self.tod_hour  = float(gp("tod_hour", 12.0).value)
        self.offset    = float(gp("bearing_offset", 0.0).value)
        # 2 = kinematic swept-path shield (zero collisions in the benchmark); 1 = forward-cone v1
        self.shield_version = int(gp("shield_version", 2).value)
        # residual_sf: the policy's output is a CORRECTION added to the classical prior
        # (benchmark: 99.4 % success, 0 collisions with the v2 shield).
        # classical_sf2: no model at all — the tuned social-force driver alone (97.8 %).
        self.res_scale = float(gp("residual_scale", 0.5).value)
        sem_png = gp("semantic_png", "").value
        sem_res = float(gp("semantic_res", 0.05).value)
        sem_org = gp("semantic_origin", [0.0, 0.0]).value

        self.policy = None if self.agent == "classical_sf2" else Policy(self.model_zip, self.vn_pkl, self.algo)
        self.prior = AckermannSF.tuned() if self.agent in ("classical_sf2", "residual_sf") else None
        self.sem = None
        if sem_png:
            from .perception import SemanticMap
            self.sem = SemanticMap(sem_png, sem_res, sem_org)
        self.wnet = self._load_weightnet()

        self.lidar = np.ones(N_RAYS, np.float32)
        self.vel = 0.0
        self.people = np.zeros(20, np.float32)
        self.goal = None                                       # (x, y) in map
        self.tf = Buffer(); TransformListener(self.tf, self)

        self.create_subscription(LaserScan, "/scan", self._scan, 10)
        self.create_subscription(Odometry, "/odom", self._odom, 10)
        self.create_subscription(Float32MultiArray, "/people", self._people, 10)
        self.create_subscription(PointStamped, "/rl_goal", self._goal, 10)
        self.cmd = self.create_publisher(Twist, "/cmd_shielded", 10)
        self.wpub = self.create_publisher(Float32MultiArray, "/policy_weights", 10)
        self.create_timer(DT, self._tick)                       # 10 Hz
        self.get_logger().info(f"controller up: agent={self.agent}")

    # ---- callbacks ----
    def _scan(self, s):
        # policy input: nearest beam per bearing (as trained); shield input: sector
        # minimum, so a thin obstacle between sampled beams cannot slip past safety
        self.lidar = scan_to_lidar48(s.ranges, s.angle_min, s.angle_increment, s.range_max, self.offset)
        self.lidar_min = scan_to_lidar48(s.ranges, s.angle_min, s.angle_increment, s.range_max, self.offset,
                                         sector_min=True)
    def _odom(self, o):  self.vel = o.twist.twist.linear.x
    def _people(self, m): self.people = np.asarray(m.data, np.float32)
    def _goal(self, p):
        self.goal = (p.point.x, p.point.y)
        # route_planner publishes the look-ahead waypoint with z = remaining ROUTE
        # length: hierarchical (planner-trained) policies read that as their distance
        # channel. z == 0 (goal_publisher / older bridges) -> straight-line distance.
        self.goal_dist = p.point.z if p.point.z > 0.0 else None

    def _pose(self):
        try:
            t = self.tf.lookup_transform("map", "base_link", rclpy.time.Time())
            return t.transform.translation.x, t.transform.translation.y, _yaw(t.transform.rotation)
        except Exception:
            return None

    # ---- context (14-D, matches envs base _context_vec) ----
    # named _context_vec, not _context: rclpy.node.Node already owns a
    # `self._context` attribute (its rclpy.context.Context) that silently
    # shadows a same-named method — this collided in practice the first time
    # this code path actually ran (TypeError: 'Context' object is not callable).
    def _context_vec(self, x, y):
        dens = np.sum(np.abs(self.people[0::4]) > 0)           # crude count of tracked people
        d5 = min(dens / 30.0, 1.0); d2 = min(dens / 8.0, 1.0)
        zone = self.sem.zone_onehot(x, y) if self.sem else np.zeros(9, np.float32)
        cd = self.sem.cult_dist(x, y) if self.sem else 1.0
        tod = 2 * math.pi * self.tod_hour / 24.0
        return np.concatenate([[d5, self.urgency], zone,
                               [math.sin(tod), math.cos(tod)], [d2, cd]]).astype(np.float32)

    def _tick(self):
        p = self._pose()
        if p is None or self.goal is None:
            return
        x, y, yaw = p
        rs = np.array([x/self.map_w*2-1, y/self.map_h*2-1, math.sin(yaw), math.cos(yaw),
                       self.vel/MAX_LINEAR], np.float32)
        gx, gy = self.goal; b = math.atan2(gy-y, gx-x) - yaw
        b = (b + math.pi) % (2 * math.pi) - math.pi
        dist = self.goal_dist if getattr(self, "goal_dist", None) else math.hypot(gx-x, gy-y)
        goal = np.array([math.sin(b), math.cos(b), min(dist/25.0, 1.0)], np.float32)
        ctx = self._context_vec(x, y)
        obs = {"lidar": self.lidar, "robot_state": rs, "pedestrians": self.people,
               "context": ctx, "goal": goal}
        # --- choose the raw command by agent type -------------------------------
        if self.agent == "classical_sf2":
            cmd = self.prior.act(b, dist, self.people, self.lidar)
        elif self.agent == "residual_sf":
            p = self.prior.act(b, dist, self.people, self.lidar)
            obs["prior"] = p.astype(np.float32)          # the policy sees what it is correcting
            dv, dd = self.policy.act(obs)
            cmd = (float(np.clip(p[0] + self.res_scale * dv, MIN_LINEAR, MAX_LINEAR)),
                   float(np.clip(p[1] + self.res_scale * dd, -MAX_STEER, MAX_STEER)))
        else:
            cmd = self.policy.act(obs)
        scan_min = getattr(self, "lidar_min", self.lidar)                # sector-minimum scan for safety
        if self.shield_version == 2:
            v, delta = shield_v2(*cmd, scan_min, self.people)            # swept-path shield (people in robot frame)
        else:
            v, delta = shield(*cmd, scan_min)                            # forward-cone shield (v1)
        out = Twist(); out.linear.x = v; out.angular.z = delta
        self.cmd.publish(out)
        if self.wnet is not None:                              # interpretability overlay
            w = np.asarray(self.wnet.get_weights(ctx), np.float32)
            m = Float32MultiArray(); m.data = w.tolist(); self.wpub.publish(m)

    def _load_weightnet(self):
        if not self.weight_pt or self.agent not in ("cwn", "attention"):
            return None  # fixed-weight, residual and classical agents have no weight overlay
        try:
            import sys, torch
            sys.path.insert(0, "/path/to/lmic-hospitalnav")   # so rewards/baselines import
            from rewards.cwn import CWN
            from baselines.attention_weighting import AttentionWeightNet
            cls = CWN if self.agent == "cwn" else AttentionWeightNet
            net = cls(context_dim=14)                          # original-env policies use 14-D context
            data = torch.load(self.weight_pt, map_location="cpu", weights_only=False)
            net.load_state_dict(data["state_dict"] if isinstance(data, dict) and "state_dict" in data else data)
            net.eval(); return net
        except Exception as e:
            self.get_logger().warn(f"weight overlay disabled: {e}"); return None


def main():
    rclpy.init(); rclpy.spin(Controller()); rclpy.shutdown()


if __name__ == "__main__":
    main()
