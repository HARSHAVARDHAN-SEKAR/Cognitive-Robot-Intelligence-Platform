#!/usr/bin/env python3
"""
crip_vlm.vlm_node

The CRIP Vision-Language node (Milestone 4).

  * Subscribes to /camera/image_raw and keeps the latest frame.
  * Serves ~/analyze_scene (crip_interfaces/AnalyzeScene) for on-demand
    assessment, and optionally analyses on a timer.
  * Publishes structured results on /vlm/assessment.
  * Publishes a human-readable alert on /vlm/alert when risk is medium/high.
  * Feeds observed labels into the M2 world model via /perception/detections
    so vision enriches the same semantic map the brain queries.

Image conversion is done without cv_bridge (Pillow only) to keep the runtime
light: rgb8/bgr8/mono8 sensor_msgs/Image -> PNG bytes for the VLM backend.
"""
import io
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from crip_interfaces.msg import SceneAssessment
from crip_interfaces.srv import AnalyzeScene

from crip_vlm.vlm import make_vlm, VLMError


class VLMNode(Node):
    def __init__(self):
        super().__init__('vlm')
        self.declare_parameter('auto_period', 0.0)  # 0 = on-demand only
        self.declare_parameter('detect_distance', 1.0)  # placeholder offset
        self.auto_period = self.get_parameter('auto_period').value

        self.vlm = make_vlm(self.get_logger())
        self._latest = None  # (bytes, media_type)
        self._lock = threading.Lock()

        self.create_subscription(Image, 'camera/image_raw', self.on_image,
                                 qos_profile_sensor_data)
        self.srv = self.create_service(AnalyzeScene, '~/analyze_scene',
                                       self.handle_analyze)
        self.pub_assess = self.create_publisher(SceneAssessment, 'vlm/assessment', 10)
        self.pub_alert = self.create_publisher(String, 'vlm/alert', 10)
        self.pub_detect = self.create_publisher(String, 'perception/detections', 10)

        if self.auto_period and self.auto_period > 0:
            self.create_timer(self.auto_period, self._auto_tick)
            self.get_logger().info(f'vlm: auto analysis every {self.auto_period}s')
        self.get_logger().info('vlm node ready')

    # ---- image intake ----
    def on_image(self, msg: Image):
        try:
            png = self._image_to_png(msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'image decode failed: {exc}')
            return
        with self._lock:
            self._latest = (png, 'image/png')

    @staticmethod
    def _image_to_png(msg: Image) -> bytes:
        from PIL import Image as PILImage
        data = bytes(msg.data)
        enc = msg.encoding.lower()
        if enc in ('rgb8', 'bgr8'):
            img = PILImage.frombytes('RGB', (msg.width, msg.height), data)
            if enc == 'bgr8':
                b, g, r = img.split()
                img = PILImage.merge('RGB', (r, g, b))
        elif enc in ('mono8', '8uc1'):
            img = PILImage.frombytes('L', (msg.width, msg.height), data)
        else:
            # best-effort: assume rgb8 stride
            img = PILImage.frombytes('RGB', (msg.width, msg.height), data)
        out = io.BytesIO()
        img.save(out, 'PNG')
        return out.getvalue()

    # ---- analysis ----
    def _analyze_latest(self, question=''):
        with self._lock:
            latest = self._latest
        if latest is None:
            return None, 'no camera frame received yet'
        png, media = latest
        try:
            a = self.vlm.analyze(png, media, question)
        except VLMError as exc:
            return None, f'vlm error: {exc}'
        except Exception as exc:  # noqa: BLE001
            return None, f'vlm backend failed: {exc}'
        self._publish(a)
        return a, ''

    def _publish(self, a):
        msg = SceneAssessment()
        msg.stamp = self.get_clock().now().to_msg()
        msg.caption = a.caption
        msg.labels = list(a.labels)
        msg.risk_level = a.risk_level
        msg.rationale = a.rationale
        msg.recommendation = a.recommendation
        msg.confidence = float(a.confidence)
        msg.backend = a.backend
        self.pub_assess.publish(msg)

        if a.risk_level in ('medium', 'high'):
            self.pub_alert.publish(String(
                data=f'[{a.risk_level.upper()}] {a.rationale} -> {a.recommendation}'))
            self.get_logger().warn(
                f'risk={a.risk_level}: {a.rationale} -> {a.recommendation}')

        # Feed observed labels to the world model (offset in robot frame;
        # without depth we place them a nominal distance ahead).
        d = self.get_parameter('detect_distance').value
        for lbl in a.labels:
            cat = 'hazard' if a.risk_level in ('medium', 'high') else 'static'
            self.pub_detect.publish(String(data=f'{lbl}:{cat}:{d:.2f}:0.0'))

    def _auto_tick(self):
        self._analyze_latest('')

    def handle_analyze(self, req, resp):
        a, err = self._analyze_latest(req.question)
        if a is None:
            resp.ok = False
            resp.error = err
            return resp
        resp.ok = True
        resp.assessment.stamp = self.get_clock().now().to_msg()
        resp.assessment.caption = a.caption
        resp.assessment.labels = list(a.labels)
        resp.assessment.risk_level = a.risk_level
        resp.assessment.rationale = a.rationale
        resp.assessment.recommendation = a.recommendation
        resp.assessment.confidence = float(a.confidence)
        resp.assessment.backend = a.backend
        return resp


def main(args=None):
    rclpy.init(args=args)
    node = VLMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
