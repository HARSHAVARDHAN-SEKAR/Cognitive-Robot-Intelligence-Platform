#!/usr/bin/env python3
"""
CRIP formal benchmark suite.

Measures throughput/latency of the platform's logic cores on the current
machine and prints a report. These are *logic* benchmarks (no ROS graph, no
Gazebo); they answer "is each component fast enough for its control-loop
budget?" -- e.g. the safety engine must comfortably exceed the 20 Hz
supervisor tick.

Run:  python3 benchmarks/run_benchmarks.py
"""
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
for pkg in ("crip_safety", "crip_world_model", "crip_llm_brain",
            "crip_prediction", "crip_memory", "crip_fleet", "crip_explain",
            "crip_dashboard"):
    sys.path.insert(0, os.path.join(SRC, pkg))


def bench(name, fn, n, unit="ops/s", budget=None):
    # warmup
    for _ in range(min(50, n // 10 or 1)):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = time.perf_counter() - t0
    rate = n / dt
    lat_us = (dt / n) * 1e6
    verdict = ""
    if budget is not None:
        verdict = "  PASS" if rate >= budget else f"  BELOW BUDGET ({budget:g})"
    print(f"{name:38s} {rate:12,.0f} {unit}   {lat_us:9.1f} us/op{verdict}")
    return rate


def main():
    print("=" * 78)
    print("CRIP LOGIC BENCHMARKS")
    print("=" * 78)

    # ---- safety engine: full 6-rule pipeline per command ----
    from crip_safety.rules import SafetyEngine, SafetyContext, Command
    eng = SafetyEngine()
    ctx = SafetyContext(front=0.7, left=2.0, right=2.0, prev_v=0.2)
    cmd = Command(0.25, 0.3)
    bench("safety engine (6-rule check)", lambda: eng.check(cmd, ctx),
          20000, budget=20)  # must beat 20 Hz supervisor tick easily

    # ---- confidence fusion ----
    from crip_safety.confidence import ConfidenceManager
    cm = ConfidenceManager()
    cm.report("perception", 0.9); cm.report("localization", 0.85)
    cm.report("llm_plan", 0.95); cm.report("prediction", 0.6)
    bench("confidence fusion", cm.fuse, 20000, budget=20)

    # ---- planner: mission -> plan (mock backend) ----
    os.environ["CRIP_LLM_BACKEND"] = "mock"
    from crip_llm_brain.planner import make_planner
    planner = make_planner()
    labels = ["charging_station", "door"]
    bench("planner (mock plan)",
          lambda: planner.plan("go to the charging station and report",
                               labels), 2000)

    # ---- knowledge graph ----
    from crip_world_model.relations import RelationStore
    kg = RelationStore()
    for i in range(200):
        kg.add(f"obj{i}", "in", f"room{i % 10}")
    for r in range(10):
        kg.add(f"room{r}", "in", "building")
    bench("knowledge graph contents_of()",
          lambda: kg.contents_of("building"), 5000)
    positions = {f"e{i}": (i * 0.3 % 10, i * 0.7 % 10) for i in range(60)}
    bench("knowledge graph infer_near(60)",
          lambda: kg.infer_near(positions), 500)

    # ---- tracker + forecaster fallback ----
    from crip_prediction.tracker import Tracker
    tr = Tracker()
    t = [0.0]
    def track_step():
        t[0] += 0.1
        tr.update([(2.0 + 0.01 * t[0], 1.0)], t[0])
    bench("tracker update (1 target)", track_step, 20000, budget=20)

    # ---- memory: hashing embed + recall over 500 records ----
    os.environ["CRIP_EMBED_BACKEND"] = "hashing"
    from crip_memory.embedder import make_embedder
    from crip_memory.store import NumpyStore
    emb = make_embedder()
    store = NumpyStore(dim=emb.dim)
    for i in range(500):
        txt = f"mission {i} inspect area {i % 7}"
        store.add(txt, emb.embed(txt), {"outcome": "success"})
    q = emb.embed("inspect area 3")
    bench("memory recall (500 records)", lambda: store.search(q, k=3), 2000)

    # ---- fleet merge ----
    from crip_fleet.sync import InMemoryBroker, FleetSync
    broker = InMemoryBroker()
    a = FleetSync("alpha", broker); b = FleetSync("beta", broker)
    i = [0]
    def share():
        i[0] += 1
        a.share_entity(f"e{i[0] % 100}", "static",
                       (i[0] % 20) * 0.5, 1.0, 0.8)
    bench("fleet share+merge (2 robots)", share, 5000)

    # ---- explanation trace ----
    from crip_explain.trace import ExplanationTrace
    trace = ExplanationTrace()
    for k in range(100):
        trace.add("brain", f"status {k}", decisive=(k % 9 == 0))
    bench("explanation why()", lambda: trace.why(), 5000)

    # ---- dashboard snapshot ----
    from crip_dashboard.state import DashboardState
    st = DashboardState()
    for k in range(150):
        st.update_robot(k * 0.1, k * 0.05, 0.1)
        st.add_event("brain", f"e{k}")
    bench("dashboard snapshot()", st.snapshot, 5000, budget=5)

    print("=" * 78)
    print("Budgets: control-path components must exceed 20 Hz (supervisor tick");
    print("rate); dashboard snapshot must exceed its 5 Hz stream rate.")
    print("NOTE: logic benchmarks only -- end-to-end latency (ROS transport,")
    print("Gazebo physics) requires the integrated run on a ROS machine.")
    print("=" * 78)


if __name__ == "__main__":
    main()
