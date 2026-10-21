#!/usr/bin/env python3
"""
crip_rl_nav.rl_controller

Runtime inference node for the CRIP RL local planner (Milestone 6).

It builds the SAME observation the training env uses (downsampled lidar +
goal vector in the robot frame), runs the trained policy, and publishes the
resulting (v, w) on /cmd_vel.

Policy source, in priority order:
  1. An ONNX policy at the `policy_path` parameter, run via onnxruntime.
  2. If that is missing or onnxruntime is unavailable, a deterministic
     heuristic controller (the same "turn toward goal, drive when aligned"
     baseline the env is validated against). This keeps the node runnable
     before any training has happened, clearly logged as a fallback.

Goal source:
  * /goal_pose (geometry_msgs/PoseStamped), e.g. published by the LLM brain.
  * Goal is transformed into the robot frame using /odom.

This node is an alternative velocity source to Nav2's controller and to the
reactive safety_pilot -- run only one /cmd_vel publisher at a time. The
production path (documented in the README) wraps this same policy as a C++
nav2_core::Controller plugin so it lives inside Nav2's costmap/recovery stack.
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped


class Policy:
    """Loads ONNX if possible; otherwise a heuristic fallback."""

    def __init__(self, path, logger):
        self.session = None
        self.logger = logger
        if path:
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(
                    path, providers=["CPUExecutionProvider"])
                self.input_name = self.session.get_inputs()[0].name
                logger.info(f"rl_controller: loaded ONNX policy {path}")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"rl_controller: could not load ONNX ({exc}); "
                               "using heuristic fallback policy")
        else:
            logger.warning("rl_controller: no policy_path set; using heuristic "
                           "fallback policy")

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self.session is not None:
            out = self.session.run(None, {self.input_name:
                                          obs[None, :].astype(np.float32)})
            return np.asarray(out[0]).reshape(-1)[:2]
        # heuristic fallback: obs[-2]=sin(bearing), obs[-1]=cos(bearing)
        sin_b, cos_b = float(obs[-2]), float(obs[-1])
        bearing = math.atan2(sin_b, cos_b)
        w = max(-1.0, min(1.0, bearing))
        v = 1.0 if abs(bearing) < 0.5 else -0.2
        return np.array([v, w], dtype=np.float32)


class RLController(Node):
    def __init__(self):
        super().__init__('rl_controller')
        self.declare_parameter('policy_path', '')
        self.declare_parameter('n_beams', 16)
        self.declare_parameter('max_range', 8.0)
        self.declare_parameter('v_max', 0.26)
        self.declare_parameter('w_max', 1.0)
        self.declare_parameter('goal_tol', 0.30)
        self.declare_parameter('rate', 10.0)

        self.n_beams = int(self.get_parameter('n_beams').value)
        self.max_range = float(self.get_parameter('max_range').value)
        self.v_max = float(self.get_parameter('v_max').value)
        self.w_max = float(self.get_parameter('w_max').value)
        self.goal_tol = float(self.get_parameter('goal_tol').value)

        self.policy = Policy(self.get_parameter('policy_path').value,
                             self.get_logger())

        self.scan = None
        self.x = self.y = self.yaw = 0.0
        self.goal = None  # (gx, gy)

        self.create_subscription(LaserScan, 'scan', self._on_scan,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, 'odom', self._on_odom,
                                 qos_profile_sensor_data)
        self.create_subscription(PoseStamped, 'goal_pose', self._on_goal, 10)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        period = 1.0 / float(self.get_parameter('rate').value)
        self.create_timer(period, self._tick)
        self.get_logger().info('rl_controller ready')

    def _on_scan(self, msg):
        self.scan = msg

    def _on_odom(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)

    def _on_goal(self, msg):
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f'new goal: {self.goal}')

    def _downsample_scan(self):
        """LaserScan -> n_beams normalized ranges (0..1), robust to inf/nan."""
        if self.scan is None:
            return np.ones(self.n_beams, dtype=np.float32)
        ranges = np.asarray(self.scan.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=self.max_range,
                               posinf=self.max_range, neginf=0.0)
        if len(ranges) == 0:
            return np.ones(self.n_beams, dtype=np.float32)
        idx = np.linspace(0, len(ranges) - 1, self.n_beams).astype(int)
        beams = np.clip(ranges[idx] / self.max_range, 0.0, 1.0)
        return beams.astype(np.float32)

    def _build_obs(self):
        beams = self._downsample_scan()
        gx, gy = self.goal
        d = math.hypot(gx - self.x, gy - self.y)
        bearing = math.atan2(gy - self.y, gx - self.x) - self.yaw
        extra = np.array([min(d / self.max_range, 1.0),
                          math.sin(bearing), math.cos(bearing)],
                         dtype=np.float32)
        return np.concatenate([beams, extra]).astype(np.float32), d

    def _tick(self):
        if self.goal is None:
            return
        obs, dist = self._build_obs()
        if dist <= self.goal_tol:
            self.cmd_pub.publish(Twist())  # stop
            self.get_logger().info('goal reached; stopping')
            self.goal = None
            return
        a = np.clip(self.policy.act(obs), -1.0, 1.0)
        cmd = Twist()
        cmd.linear.x = float((a[0] + 1.0) * 0.5 * self.v_max)
        cmd.angular.z = float(a[1] * self.w_max)
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = RLController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
