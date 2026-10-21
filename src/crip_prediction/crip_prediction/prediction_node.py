#!/usr/bin/env python3
"""
crip_prediction.prediction_node

The CRIP trajectory-prediction node (Module 6 / Milestone 7).

  * Subscribes to /dynamic_pose/info (bridged Gazebo actor/model poses) and
    extracts 2D positions of moving entities (the walking person).
  * Feeds them to a multi-object Tracker (velocity estimation).
  * Runs the forecaster (ONNX LSTM if available, else constant-velocity) to
    predict each track's next H positions.
  * Publishes one nav_msgs/Path per track on /prediction/tracks/<id> and an
    aggregate MarkerArray-free summary on /prediction/summary.
  * Feeds predicted future positions into the world model as short-lived
    "predicted_obstacle" hazards via /perception/detections, so the planner
    layer can avoid where the person *will be*.

Pose source filtering: the robot itself and static models are ignored; only
entities whose names look dynamic (default: contain 'person' or 'actor') are
tracked. Configurable via the `dynamic_name_substrings` parameter.
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from crip_prediction.tracker import Tracker
from crip_prediction.forecaster import make_forecaster, HORIZON


class PredictionNode(Node):
    def __init__(self):
        super().__init__('prediction')
        self.declare_parameter('model_path', '')
        self.declare_parameter('registry_dir', '')
        self.declare_parameter('horizon', HORIZON)
        self.declare_parameter('dt', 0.5)
        self.declare_parameter('dynamic_name_substrings', ['person', 'actor'])
        self.declare_parameter('publish_hazards', True)

        self.horizon = int(self.get_parameter('horizon').value)
        self.dt = float(self.get_parameter('dt').value)
        self.substrings = [s.lower() for s in
                           self.get_parameter('dynamic_name_substrings').value]
        self.publish_hazards = bool(self.get_parameter('publish_hazards').value)

        self.tracker = Tracker(gate=1.5, max_age=2.0)
        model_path = self.get_parameter('model_path').value or None
        registry_dir = self.get_parameter('registry_dir').value
        if not model_path and registry_dir:
            # prefer the self-learning registry's current model if present
            import os
            cur = os.path.join(registry_dir, 'current', 'model.onnx')
            if os.path.exists(cur):
                model_path = cur
                self.get_logger().info(f'using registry current model: {cur}')
        self.forecaster = make_forecaster(model_path, self.get_logger())

        self.create_subscription(TFMessage, 'dynamic_pose/info',
                                 self.on_poses, 10)
        self._path_pubs = {}
        self.pub_summary = self.create_publisher(String, 'prediction/summary', 10)
        self.pub_detect = self.create_publisher(String, 'perception/detections', 10)
        self.get_logger().info('prediction node ready')

    def _is_dynamic(self, name: str) -> bool:
        n = name.lower()
        return any(s in n for s in self.substrings)

    def on_poses(self, msg: TFMessage):
        t = self.get_clock().now().nanoseconds * 1e-9
        observations = []
        for tfs in msg.transforms:
            name = tfs.child_frame_id
            if not self._is_dynamic(name):
                continue
            x = tfs.transform.translation.x
            y = tfs.transform.translation.y
            observations.append((x, y))
        if not observations:
            return

        tracks = self.tracker.update(observations, t)
        summaries = []
        for tr in tracks:
            hist = tr.positions()
            if len(hist) < 2:
                continue
            future = self.forecaster.predict(hist, self.horizon, self.dt)
            self._publish_path(tr.track_id, future)
            summaries.append(
                f"id{tr.track_id} @({tr.x:.1f},{tr.y:.1f}) "
                f"v=({tr.vx:.2f},{tr.vy:.2f}) -> "
                f"({future[-1][0]:.1f},{future[-1][1]:.1f})")
            if self.publish_hazards:
                # publish the mid-horizon predicted point as a transient hazard
                mid = future[min(len(future) - 1, self.horizon // 2)]
                # report as offset 0,0 placeholder label carrying absolute pos
                self.pub_detect.publish(String(
                    data=f"predicted_person:hazard:{mid[0]:.2f}:{mid[1]:.2f}"))

        if summaries:
            self.pub_summary.publish(String(data="; ".join(summaries)))

    def _publish_path(self, track_id, future):
        if track_id not in self._path_pubs:
            self._path_pubs[track_id] = self.create_publisher(
                Path, f'prediction/tracks/{track_id}', 10)
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for (x, y) in future:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position = Point(x=float(x), y=float(y), z=0.0)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self._path_pubs[track_id].publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = PredictionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
