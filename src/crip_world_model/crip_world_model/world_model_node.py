#!/usr/bin/env python3
"""
crip_world_model.world_model_node

The CRIP World Model (Milestone 2).

Responsibilities:
  * Track the robot's current pose from /odom.
  * Ingest semantic detections and stamp them into the map frame using the
    robot pose, persisting them via the storage backend.
  * Serve two query services so the brain (and a human) can interrogate the
    long-term memory:
        ~/query_entities  (crip_interfaces/QueryEntities)
        ~/have_i_seen     (crip_interfaces/HaveISeen)
  * Republish the full known map on /world_model/entities for visualization.

Detection input format (Milestone-2 bridge):
  Until the M1 perception node emits structured detections, this node accepts
  a simple std_msgs/String on /perception/detections of the form
      "label:category:robot_dx:robot_dy"
  where dx,dy are the object's offset from the robot in the robot frame.
  When M1 is upgraded to publish vision_msgs/Detection arrays, only the
  `on_detection` callback changes.
"""
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from geometry_msgs.msg import Point

from crip_interfaces.msg import SemanticEntity
from crip_interfaces.srv import QueryEntities, HaveISeen

from crip_world_model.storage import make_store, Entity


# Named landmarks in crip_world.sdf (BIG 16x16 arena) with true positions.
# Seeded at startup so the LLM brain can ground "go to the charging station".
# On a real robot these labels come from semantic perception (YOLO/CLIP).
KNOWN_LANDMARKS = [
    # Navigable approach pose in front of the physical station.
    ('charging_station', 'landmark', 5.2, 5.2),
    ('shelf_a',          'static',   -6.5,  3.0),
    ('shelf_b',          'static',   -6.5, -1.0),
    ('inspection_table', 'static',    5.0, -5.5),
    ('pallet',           'static',    0.0,  5.0),
    ('pillar',           'static',    2.0, -2.0),
]


class WorldModelNode(Node):
    def __init__(self):
        super().__init__('world_model')
        self.store = make_store(self.get_logger())

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.create_subscription(Odometry, 'odom', self.on_odom,
                                 qos_profile_sensor_data)
        self.create_subscription(String, 'perception/detections',
                                 self.on_detection, 10)

        self.entity_pub = self.create_publisher(
            SemanticEntity, 'world_model/entities', 10)

        self.srv_query = self.create_service(
            QueryEntities, '~/query_entities', self.handle_query)
        self.srv_seen = self.create_service(
            HaveISeen, '~/have_i_seen', self.handle_have_i_seen)

        # Seed known landmarks (disable with seed_landmarks:=false)
        self.declare_parameter('seed_landmarks', True)
        if self.get_parameter('seed_landmarks').value:
            self._seed_landmarks()

        # periodically republish the map for visualization/debug
        self.create_timer(2.0, self.publish_map)
        self.get_logger().info('world_model node ready')

    def _seed_landmarks(self):
        for label, category, x, y in KNOWN_LANDMARKS:
            e = Entity(label=label, category=category, x=x, y=y, z=0.0,
                       confidence=1.0, last_seen=time.time())
            self.store.upsert(e)
        self.get_logger().info(
            f'seeded {len(KNOWN_LANDMARKS)} known landmarks: '
            + ', '.join(l for l, _, _, _ in KNOWN_LANDMARKS))

    # ----- pose tracking -----
    def on_odom(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        # yaw from quaternion (z-axis rotation)
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        import math
        self.robot_yaw = math.atan2(siny, cosy)

    # ----- ingest -----
    def on_detection(self, msg: String):
        try:
            label, category, dx, dy = msg.data.split(':')
            dx, dy = float(dx), float(dy)
        except ValueError:
            self.get_logger().warn(f'bad detection format: {msg.data!r}')
            return
        # transform offset from robot frame into map frame
        import math
        mx = self.robot_x + dx * math.cos(self.robot_yaw) - dy * math.sin(self.robot_yaw)
        my = self.robot_y + dx * math.sin(self.robot_yaw) + dy * math.cos(self.robot_yaw)
        e = Entity(label=label, category=category, x=mx, y=my, z=0.0,
                   confidence=1.0, last_seen=time.time())
        stored = self.store.upsert(e)
        self.get_logger().debug(
            f'observed {label} at ({mx:.2f},{my:.2f}) '
            f'[id={stored.id} count={stored.observation_count}]')

    # ----- services -----
    def _to_msg(self, e: Entity) -> SemanticEntity:
        m = SemanticEntity()
        m.label = e.label
        m.category = e.category or ''
        m.position = Point(x=e.x, y=e.y, z=e.z)
        m.confidence = float(e.confidence)
        m.observation_count = int(e.observation_count)
        m.id = int(e.id)
        sec = int(e.last_seen)
        m.last_seen.sec = sec
        m.last_seen.nanosec = int((e.last_seen - sec) * 1e9)
        return m

    def handle_query(self, req, resp):
        near = (req.near.x, req.near.y) if req.max_distance > 0 else None
        results = self.store.query(
            req.label or None, req.category or None,
            near, req.max_distance, req.max_results)
        resp.entities = [self._to_msg(e) for e in results]
        resp.found = len(results) > 0
        return resp

    def handle_have_i_seen(self, req, resp):
        seen, times, closest = self.store.have_i_seen(req.label)
        resp.seen = seen
        resp.times_seen = times
        resp.closest = self._to_msg(closest) if closest else SemanticEntity()
        return resp

    # ----- viz -----
    def publish_map(self):
        for e in self.store.query(None, None, None, 0.0, 0):
            self.entity_pub.publish(self._to_msg(e))


def main(args=None):
    rclpy.init(args=args)
    node = WorldModelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
