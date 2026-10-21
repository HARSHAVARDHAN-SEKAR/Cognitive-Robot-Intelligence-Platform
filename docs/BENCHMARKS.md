# CRIP — Benchmark Results

Performance of the platform's logic cores, measured with
`benchmarks/run_benchmarks.py`. These are **logic benchmarks**: they measure
each component's pure-Python execution speed in isolation, with no ROS
transport and no Gazebo physics. They answer one question — *is each component
fast enough for its control-loop budget?* — not *how fast is the integrated
system end to end* (that requires the live run and is not yet measured).

**Machine:** results below were captured on the CI/build container (single
core, no GPU). Absolute numbers vary by machine; the **headroom ratios** are
what matter and are stable.

## Results

| Component | Throughput | Latency | Budget | Headroom |
|---|---:|---:|---:|---:|
| Safety engine (6-rule check) | 199,617 /s | 5.0 µs | ≥ 20 Hz | ~9,980× |
| Confidence fusion | 375,415 /s | 2.7 µs | ≥ 20 Hz | ~18,770× |
| Tracker update (1 target) | 553,735 /s | 1.8 µs | ≥ 20 Hz | ~27,690× |
| Dashboard snapshot | 522,560 /s | 1.9 µs | ≥ 5 Hz | ~104,510× |
| Planner (mock plan) | 438,720 /s | 2.3 µs | per-mission | ample |
| Memory recall (500 records) | 55,342 /s | 18.1 µs | per-query | ample |
| Fleet share + merge (2 robots) | 85,112 /s | 11.7 µs | per-event | ample |
| Knowledge graph `contents_of()` | 32,879 /s | 30.4 µs | per-query | ample |
| Explanation `why()` | 12,150 /s | 82.3 µs | on-demand | ample |
| Knowledge graph `infer_near(60)` | 4,219 /s | 237 µs | periodic | ample |

## Interpretation

The **control-path** components — those that must run inside the 20 Hz safety
supervisor tick — all clear their budget by four to five orders of magnitude.
The safety engine, the most critical component, evaluates a full six-rule
pipeline in ~5 µs; the supervisor's 50 ms budget is never at risk from
compute. The dashboard snapshot, which streams at 5 Hz, is ~100,000× faster
than required.

The **per-query / on-demand** components (planner, memory recall, knowledge
graph, explanation) are not on the hot loop; their sub-millisecond times mean
they add no perceptible latency to a mission.

## What this does and does not show

- **Shows:** no logic core is a compute bottleneck. The architecture's
  per-cycle budget is dominated by sensing and physics, not by the decision
  code — which is the correct place for the cost to sit.
- **Does not show:** end-to-end mission latency, ROS DDS transport overhead,
  Gazebo real-time factor, or navigation accuracy. Those require the
  integrated run on a ROS 2 machine and are the subject of the next
  measurement pass.

## Reproduce

```bash
python3 benchmarks/run_benchmarks.py
```

No ROS, GPU, or network required — the benchmark suite runs on the logic cores
directly.
