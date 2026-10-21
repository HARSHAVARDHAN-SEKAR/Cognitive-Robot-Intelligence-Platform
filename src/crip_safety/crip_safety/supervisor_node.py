#!/usr/bin/env python3
"""
crip_safety.supervisor_node

The CRIP safety supervisor (Module 10) -- the veto layer on the wire.

It sits between every velocity source and the robot:

    <planners/controllers>  --/cmd_vel_raw-->  [supervisor]  --/cmd_vel-->  robot

It subscribes to /cmd_vel_raw, builds a SafetyContext from the live /scan (and
/odom for the geofence), runs the SafetyEngine, and publishes only the vetted
command on /cmd_vel. Every non-trivial decision is logged and published on
/safety/audit. An /safety/estop (std_msgs/Bool) latches an emergency stop.

To put the supervisor in charge, sources publish to /cmd_vel_raw instead of
/cmd_vel. A watchdog publishes a stop if no command arrives within a timeout,
so a crashed upstream node cannot leave the robot driving.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy,
                       qos_profile_sensor_data)
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String

from crip_safety.rules import (SafetyEngine, SafetyContext, SafetyConfig,
                               Command, Decision)
from crip_safety.confidence import ConfidenceManager, HOLD


class SupervisorNode(Node):
    def __init__(self):
        super().__init__('safety_supervisor')
        self.declare_parameter('rate', 20.0)
        self.declare_parameter('cmd_timeout', 0.5)
        self.declare_parameter('max_v', 0.26)
        self.declare_parameter('stop_distance', 0.35)
        self.declare_parameter('arena_half', 5.0)

        cfg = SafetyConfig(
            max_v=float(self.get_parameter('max_v').value),
            stop_distance=float(self.get_parameter('stop_distance').value),
            arena_half=float(self.get_parameter('arena_half').value),
        )
        self.engine = SafetyEngine(cfg=cfg)
        self.cmd_timeout = float(self.get_parameter('cmd_timeout').value)

        self.ctx = SafetyContext()
        self._raw = Command(0.0, 0.0)
        self._last_cmd_t = 0.0
        self._prev_out = Command(0.0, 0.0)

        # BUGFIX (supervisor was BLIND: "LIDAR 0.0 Hz 0 beams [NO DATA]"):
        # 1. topics must be ABSOLUTE ('/scan' not 'scan') -- relative names
        #    resolve under the node namespace and silently match nothing.
        # 2. the ros_gz bridge publishes LaserScan/Odometry as RELIABLE, but
        #    qos_profile_sensor_data is BEST_EFFORT. A BEST_EFFORT subscriber
        #    CAN receive from a RELIABLE publisher, but not vice-versa, and the
        #    depth/durability must line up -- so we declare an explicit
        #    compatible profile instead of relying on the sensor-data default.
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10)

        self.create_subscription(Twist, '/cmd_vel_raw', self.on_raw, 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, sensor_qos)
        self.create_subscription(Odometry, '/odom', self.on_odom, sensor_qos)
        self.create_subscription(Bool, '/safety/estop', self.on_estop, 10)
        self.create_subscription(String, '/confidence/report',
                                 self.on_confidence, 10)

        # Confidence-aware gating (review improvement): sources report
        # "name:0.87"; fused confidence scales allowed speed, never raises it.
        # In sim, odom is ground-truth -> localization is seeded confident;
        # perception is seeded on first scan. On a real robot these come from
        # the actual modules.
        self.confidence = ConfidenceManager()
        self.confidence.report('localization', 0.9)
        self._conf_seeded = False

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.audit_pub = self.create_publisher(String, '/safety/audit', 10)

        period = 1.0 / float(self.get_parameter('rate').value)
        self.create_timer(period, self.tick)
        self._estop = False
        self.get_logger().info('safety_supervisor active (gating /cmd_vel_raw -> /cmd_vel)')

    def on_raw(self, msg: Twist):
        self._raw = Command(msg.linear.x, msg.angular.z)
        self._last_cmd_t = self.now()

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_scan(self, msg: LaserScan):
        n = len(msg.ranges)
        if n == 0:
            return
        if not self._conf_seeded:
            # live sensor data flowing -> perception baseline confidence
            self._conf_seeded = True
        self.confidence.report('perception', 0.85)

        def sector_min(lo, hi):
            best = float('inf')
            for i, r in enumerate(msg.ranges):
                a = msg.angle_min + i * msg.angle_increment
                if lo <= a < hi and math.isfinite(r) and r > 0.0:
                    best = min(best, r)
            return best

        self.ctx.front = sector_min(-math.pi/4, math.pi/4)
        self.ctx.left = sector_min(math.pi/4, 3*math.pi/4)
        self.ctx.right = sector_min(-3*math.pi/4, -math.pi/4)
        self.ctx.rear = min(sector_min(3*math.pi/4, math.pi),
                            sector_min(-math.pi, -3*math.pi/4))

    def on_odom(self, msg: Odometry):
        self.ctx.x = msg.pose.pose.position.x
        self.ctx.y = msg.pose.pose.position.y
        # keep localization confidence fresh; if odom stops arriving it goes
        # stale and the confidence manager forces a HOLD (fail-safe).
        self.confidence.report('localization', 0.9)

    def on_estop(self, msg: Bool):
        if msg.data and not self._estop:
            self.get_logger().warn('EMERGENCY STOP engaged')
        elif not msg.data and self._estop:
            self.get_logger().info('emergency stop cleared')
        self._estop = bool(msg.data)

    def on_confidence(self, msg: String):
        # format "source:0.87"
        try:
            source, _, val = msg.data.partition(':')
            self.confidence.report(source.strip(), float(val))
        except ValueError:
            pass

    def tick(self):
        # watchdog: stale command -> treat as stop request
        stale = (self.now() - self._last_cmd_t) > self.cmd_timeout
        raw = Command(0.0, 0.0) if stale else self._raw.copy()

        self.ctx.estop = self._estop
        self.ctx.prev_v = self._prev_out.v
        self.ctx.prev_w = self._prev_out.w

        result = self.engine.check(raw, self.ctx)
        out = result.command

        # confidence gate: scale (never raise) allowed speed by fused confidence
        fused = self.confidence.fuse()
        if fused['speed_scale'] < 1.0:
            out = Command(out.v * fused['speed_scale'],
                          out.w * fused['speed_scale'])
            if fused['policy'] == HOLD:
                self.audit_pub.publish(String(
                    data=f"confidence HOLD -- {fused['reason']}"))

        twist = Twist()
        twist.linear.x = out.v
        twist.angular.z = out.w
        self.cmd_pub.publish(twist)
        self._prev_out = out

        # audit only when something was modified/vetoed or on stale watchdog
        acted = any(d.decision in (Decision.MODIFY, Decision.VETO)
                    for d in result.decisions)
        if acted or stale:
            line = ("watchdog stop (no command) " if stale else "") + \
                   result.audit_line()
            self.audit_pub.publish(String(data=line))


def main(args=None):
    rclpy.init(args=args)
    node = SupervisorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
