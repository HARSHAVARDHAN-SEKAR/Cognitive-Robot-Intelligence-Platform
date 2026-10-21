#!/usr/bin/env python3
"""
crip_learning.logger_node

Records live experience to a JSONL file for the self-learning loop.

Sources:
  * /prediction/summary -- parsed for per-track positions (the moving actor),
    logged as {"kind":"track", ...} records that feed forecaster retraining.
  * /memory/log         -- mission record confirmations, logged as
    {"kind":"mission", ...} for dataset statistics and future loops.

The log path is the `log_path` parameter (default
~/.crip/experience.jsonl). The learning pipeline reads this file.

Parsing /prediction/summary keeps the logger decoupled from message-package
availability; if richer logging is wanted later, subscribe to the per-track
nav_msgs/Path topics directly.
"""
import os
import re
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from crip_learning.dataset import ExperienceLog


class LoggerNode(Node):
    def __init__(self):
        super().__init__('experience_logger')
        default_path = os.path.join(os.path.expanduser('~'), '.crip',
                                    'experience.jsonl')
        self.declare_parameter('log_path', default_path)
        self.log = ExperienceLog(self.get_parameter('log_path').value)

        self.create_subscription(String, 'prediction/summary',
                                 self.on_prediction, 10)
        self.create_subscription(String, 'memory/log', self.on_memory, 10)
        self._n = 0
        self.get_logger().info(
            f'experience_logger writing to {self.get_parameter("log_path").value}')

    def on_prediction(self, msg: String):
        # summary format per track: "id<ID> @(<x>,<y>) v=(...) -> (...)"
        t = self.get_clock().now().nanoseconds * 1e-9
        for m in re.finditer(r'id(\d+)\s+@\(([-\d.]+),([-\d.]+)\)', msg.data):
            tid, x, y = int(m.group(1)), float(m.group(2)), float(m.group(3))
            self.log.append({"kind": "track", "t": t, "id": tid,
                             "x": x, "y": y})
            self._n += 1
        if self._n and self._n % 50 == 0:
            self.get_logger().info(f'logged {self._n} track points')

    def on_memory(self, msg: String):
        t = self.get_clock().now().nanoseconds * 1e-9
        # "recorded #<id>: mission: <text> | outcome: <o> | tasks: <...>"
        outcome = "unknown"
        mo = re.search(r'outcome:\s*(\w+)', msg.data)
        if mo:
            outcome = mo.group(1)
        self.log.append({"kind": "mission", "t": t,
                         "mission": msg.data, "outcome": outcome})


def main(args=None):
    rclpy.init(args=args)
    node = LoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
