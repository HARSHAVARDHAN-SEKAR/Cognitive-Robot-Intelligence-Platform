#!/usr/bin/env python3
"""
crip_dashboard.dashboard_node

Web dashboard bridge for the CRIP platform (Milestone 10).

Runs a FastAPI server (in a background thread) alongside a ROS node:
  * Subscribes to the stack's topics and folds them into a DashboardState.
  * Serves the single-page dashboard at /  and a WebSocket at /ws that
    streams JSON state snapshots ~5 Hz to connected browsers.
  * POST /mission {"mission": "..."} calls the LLM brain's plan_mission
    service so an operator can command the robot from the browser.

If FastAPI/uvicorn are not installed the node logs a clear message and still
runs as a ROS node (so it never hard-crashes the launch).
"""
import json
import math
import os
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from crip_dashboard.state import DashboardState

try:
    from crip_interfaces.srv import PlanMission
    _HAVE_BRAIN = True
except ImportError:
    PlanMission = None
    _HAVE_BRAIN = False


# Static obstacles matching crip_world.sdf (for the map view).
WORLD_OBSTACLES = [
    {"x": 4.0,  "y": 3.0,  "r": 0.5,  "shape": "box"},     # box_obstacle
    {"x": -4.0, "y": -4.0, "r": 0.4,  "shape": "circle"},  # cyl_obstacle
    {"x": 6.5,  "y": 6.5,  "r": 0.45, "shape": "box"},     # charging_station
    {"x": -6.5, "y": 3.0,  "r": 1.0,  "shape": "box"},     # shelf_a
    {"x": -6.5, "y": -1.0, "r": 1.0,  "shape": "box"},     # shelf_b
    {"x": 5.0,  "y": -5.5, "r": 0.6,  "shape": "box"},     # inspection_table
    {"x": 0.0,  "y": 5.0,  "r": 0.5,  "shape": "box"},     # pallet
    {"x": 2.0,  "y": -2.0, "r": 0.25, "shape": "circle"},  # pillar
]
ARENA_HALF = 8.0


class DashboardNode(Node):
    def __init__(self):
        super().__init__('dashboard')
        self.declare_parameter('port', 8088)
        self.port = int(self.get_parameter('port').value)

        self.state = DashboardState()
        self.state.set_obstacles(WORLD_OBSTACLES)

        # health monitor (review improvement): topic activity = heartbeat
        from crip_dashboard.health import HealthMonitor
        self.health = HealthMonitor(
            stale_after=5.0,
            expected={'odom': 20.0, 'prediction': 2.0, 'safety': 1.0,
                      'brain': 0.1, 'vlm': 0.2, 'learning': 0.01,
                      'fleet': 0.3},
            on_alert=lambda name, old, new: self.state.add_event(
                'health', f'{name}: {old} -> {new}'))
        self.create_timer(2.0, self._health_tick)

        self.create_subscription(Odometry, 'odom', self.on_odom,
                                 qos_profile_sensor_data)
        self.create_subscription(PoseStamped, 'goal_pose', self.on_goal, 10)
        self.create_subscription(String, 'brain/explanation', self.on_brain, 10)
        self.create_subscription(String, 'prediction/summary', self.on_pred, 10)
        self.create_subscription(String, 'vlm/assessment', self.on_vlm_str, 10)
        self.create_subscription(String, 'vlm/alert', self.on_vlm_alert, 10)
        self.create_subscription(String, 'learning/status', self.on_learning, 10)
        self.create_subscription(String, 'safety/audit', self.on_safety, 10)
        self.create_subscription(String, 'explain/why', self.on_why, 10)
        self.create_subscription(String, 'fleet/summary', self.on_fleet, 10)

        self.cli_brain = (self.create_client(PlanMission,
                                             '/llm_brain/plan_mission')
                          if _HAVE_BRAIN else None)

        self._start_web()
        self.get_logger().info(f'dashboard node ready (http://0.0.0.0:{self.port})')

    # ---- ROS callbacks ----
    def on_odom(self, msg):
        self.health.beat('odom')
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        self.state.update_robot(p.x, p.y, yaw)

    def on_goal(self, msg):
        self.state.update_goal(msg.pose.position.x, msg.pose.position.y)

    def on_brain(self, msg):
        self.health.beat('brain')
        self.state.update_brain(msg.data)

    def on_pred(self, msg):
        self.health.beat('prediction')
        # parse "id<ID> @(x,y) ... -> (fx,fy)"; show current + endpoint
        import re
        for m in re.finditer(
                r'id(\d+)\s+@\(([-\d.]+),([-\d.]+)\).*?->\s*\(([-\d.]+),([-\d.]+)\)',
                msg.data):
            tid = int(m.group(1))
            cur = (float(m.group(2)), float(m.group(3)))
            end = (float(m.group(4)), float(m.group(5)))
            # simple 2-point predicted segment (full path also on /prediction/tracks)
            self.state.update_prediction(tid, [cur, end])

    def on_vlm_str(self, msg):
        # SceneAssessment is published as a typed msg, but we also accept a
        # String fallback; here we only get risk via the alert topic.
        pass

    def on_vlm_alert(self, msg):
        self.health.beat('vlm')
        # format "[RISK] rationale -> recommendation"
        risk = "medium"
        if msg.data.startswith("[HIGH]"):
            risk = "high"
        elif msg.data.startswith("[LOW]"):
            risk = "low"
        self.state.update_vlm(risk, msg.data)

    def on_learning(self, msg):
        self.health.beat('learning')
        self.state.update_learning(msg.data)

    def on_safety(self, msg):
        self.health.beat('safety')
        self.state.add_event('safety', msg.data)

    def on_why(self, msg):
        self.state.why = msg.data

    def on_fleet(self, msg):
        self.health.beat('fleet')
        self.state.fleet = msg.data

    def _health_tick(self):
        self.state.health = self.health.summary()

    # ---- web server ----
    def _start_web(self):
        try:
            import uvicorn
            from fastapi import FastAPI, WebSocket, WebSocketDisconnect
            from fastapi.responses import HTMLResponse
            from fastapi.staticfiles import StaticFiles
        except ImportError:
            self.get_logger().warn(
                'FastAPI/uvicorn not installed; dashboard web server disabled. '
                'Install with: pip install fastapi "uvicorn[standard]"')
            return

        web_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'web')
        # also resolve installed share path
        if not os.path.exists(os.path.join(web_dir, 'index.html')):
            from ament_index_python.packages import get_package_share_directory
            try:
                web_dir = os.path.join(
                    get_package_share_directory('crip_dashboard'), 'web')
            except Exception:  # noqa: BLE001
                pass

        app = FastAPI()
        node = self

        @app.get('/', response_class=HTMLResponse)
        def index():
            path = os.path.join(web_dir, 'index.html')
            if os.path.exists(path):
                return open(path).read()
            return '<h1>CRIP dashboard</h1><p>index.html not found.</p>'

        @app.websocket('/ws')
        async def ws(sock: WebSocket):
            import asyncio
            await sock.accept()
            try:
                while True:
                    await sock.send_text(json.dumps(node.state.snapshot()))
                    await asyncio.sleep(0.2)  # ~5 Hz
            except Exception:  # noqa: BLE001
                return

        @app.post('/mission')
        async def mission(payload: dict):
            text = (payload or {}).get('mission', '').strip()
            if not text:
                return {"ok": False, "error": "empty mission"}
            if node.cli_brain is None or not node.cli_brain.wait_for_service(
                    timeout_sec=2.0):
                node.state.add_event('operator', f'mission (no brain): {text}')
                return {"ok": False, "error": "brain service unavailable"}
            req = PlanMission.Request(mission=text)
            fut = node.cli_brain.call_async(req)
            node.state.add_event('operator', f'mission: {text}')
            return {"ok": True, "submitted": text}

        def serve():
            uvicorn.run(app, host='0.0.0.0', port=self.port, log_level='warning')

        threading.Thread(target=serve, daemon=True).start()


def main(args=None):
    rclpy.init(args=args)
    node = DashboardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
