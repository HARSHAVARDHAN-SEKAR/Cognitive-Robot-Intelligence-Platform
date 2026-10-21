#!/usr/bin/env python3
"""
crip_explain.explain_node

Explainable-AI node (Module 12).

Subscribes to the stack's decision signals, folds them into an
ExplanationTrace, and answers "why did you do that?" on demand:

  * service ~/explain (crip_interfaces/ExplainDecision) -> causal narrative
  * publishes the current explanation on /explain/why whenever a decisive
    event arrives (so the dashboard can show a live "why")

Decisive events (safety vetoes, high/medium VLM risk, navigation failures) are
flagged so the trace ranks them above routine status.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from crip_explain.trace import ExplanationTrace

try:
    from crip_interfaces.srv import ExplainDecision
    _HAVE_SRV = True
except ImportError:
    ExplainDecision = None
    _HAVE_SRV = False


class ExplainNode(Node):
    def __init__(self):
        super().__init__('explain')
        self.trace = ExplanationTrace()

        self.create_subscription(String, 'brain/explanation', self.on_brain, 10)
        self.create_subscription(String, 'vlm/alert', self.on_vlm, 10)
        self.create_subscription(String, 'prediction/summary', self.on_pred, 10)
        self.create_subscription(String, 'safety/audit', self.on_safety, 10)
        self.create_subscription(String, 'learning/status', self.on_learning, 10)

        self.pub_why = self.create_publisher(String, 'explain/why', 10)

        if _HAVE_SRV:
            self.srv = self.create_service(ExplainDecision, '~/explain',
                                           self.handle_explain)
        self.get_logger().info('explain node ready')

    # ---- intake ----
    def on_brain(self, msg):
        decisive = 'abort' in msg.data.lower() or 'reject' in msg.data.lower()
        self.trace.add('brain', msg.data, decisive)

    def on_vlm(self, msg):
        decisive = msg.data.startswith('[HIGH]') or msg.data.startswith('[MEDIUM]')
        self.trace.add('vlm', msg.data, decisive)
        if decisive:
            self._emit_why()

    def on_pred(self, msg):
        self.trace.add('prediction', msg.data, False)

    def on_safety(self, msg):
        decisive = 'VETO' in msg.data or 'watchdog' in msg.data
        self.trace.add('safety', msg.data, decisive)
        if decisive:
            self._emit_why()

    def on_learning(self, msg):
        self.trace.add('learning', msg.data, False)

    # ---- output ----
    def _emit_why(self):
        ans = self.trace.why()
        self.pub_why.publish(String(data=ans['summary']))

    def handle_explain(self, req, resp):
        k = req.top_k if req.top_k and req.top_k > 0 else 3
        ans = self.trace.why(top_k=k)
        resp.summary = ans['summary']
        resp.primary_cause = (ans.get('primary_cause') or {}).get('text', '')
        resp.factors = [f['text'] for f in ans['factors']]
        return resp


def main(args=None):
    rclpy.init(args=args)
    node = ExplainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
