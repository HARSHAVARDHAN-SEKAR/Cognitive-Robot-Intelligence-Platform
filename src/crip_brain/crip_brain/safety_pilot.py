#!/usr/bin/env python3
"""
crip_brain.safety_pilot

Milestone-1 stand-in for the LLM Brain (Module 3) + AI Safety Supervisor
(Module 10). Later, the high-level command source becomes an LLM task
planner; for now it is a deterministic reactive controller so the stack
is fully runnable and testable end-to-end.

Behaviour:
  * drive forward at cruise speed
  * subscribe to /perception/front_range
  * if an obstacle is closer than stop_distance -> STOP and rotate to find
    a clear heading (the "safety supervisor" vetoes forward motion)
  * publish /cmd_vel and an explainability string on /brain/explanation
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, String


class SafetyPilot(Node):
    def __init__(self):
        super().__init__('safety_pilot')
        self.declare_parameter('cruise_speed', 0.25)
        self.declare_parameter('turn_speed', 0.6)
        self.declare_parameter('stop_distance', 0.7)
        self.cruise = self.get_parameter('cruise_speed').value
        self.turn = self.get_parameter('turn_speed').value
        self.stop_distance = self.get_parameter('stop_distance').value

        self.front_range = float('inf')
        self.create_subscription(Float32, 'perception/front_range',
                                 self.on_front, 10)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.exp_pub = self.create_publisher(String, 'brain/explanation', 10)
        self.timer = self.create_timer(0.1, self.tick)
        self.get_logger().info('safety_pilot started')

    def on_front(self, msg: Float32):
        # -1.0 means "nothing detected"
        self.front_range = float('inf') if msg.data < 0 else msg.data

    def tick(self):
        cmd = Twist()
        if self.front_range < self.stop_distance:
            cmd.linear.x = 0.0
            cmd.angular.z = self.turn
            reason = (f"obstacle at {self.front_range:.2f}m "
                      f"< stop_distance {self.stop_distance:.2f}m -> "
                      f"vetoing forward motion, rotating to find clear path")
        else:
            cmd.linear.x = self.cruise
            cmd.angular.z = 0.0
            reason = f"path clear (front={self.front_range:.2f}m) -> cruising"
        self.cmd_pub.publish(cmd)
        self.exp_pub.publish(String(data=reason))


def main(args=None):
    rclpy.init(args=args)
    node = SafetyPilot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
