# CRIP — Reproducibility Guide

This document lets another person reproduce every verified result and attempt
the integrated run. It is written to be followed top to bottom.

## What is verified vs. what needs a machine

| Result | Reproducible where | Command |
|---|---|---|
| 113 unit tests pass | any Linux/macOS + Python 3.12 | see §1 |
| All 94 Python files compile | any machine | see §1 |
| Benchmark numbers | any machine | `python3 benchmarks/run_benchmarks.py` |
| Real PPO + LSTM training → ONNX | machine with pip (CPU ok) | see §3 |
| `colcon build` succeeds | ROS 2 Jazzy machine / CI | see §4 |
| Integrated Gazebo simulation | ROS 2 Jazzy + Gazebo machine | see §4 |

Sections 1–3 need **no ROS**. Section 4 needs a ROS 2 machine (or just push to
GitHub and let CI attempt it).

## 1. Logic tests (no ROS, ~1 minute)

```bash
python3 -m pip install pytest numpy fastapi httpx "uvicorn[standard]" pillow gymnasium

# compile check
find src -name '*.py' -exec python3 -m py_compile {} +

# unit suites
cd src/crip_safety      && PYTHONPATH=. pytest test/test_safety.py test/test_confidence.py -q; cd ../..
cd src/crip_world_model && PYTHONPATH=. pytest test/test_storage.py test/test_relations.py -q; cd ../..
cd src/crip_dashboard   && PYTHONPATH=. pytest test/test_dashboard.py test/test_health.py -q; cd ../..
# ... (all suites listed in the CI workflow .github/workflows/ci.yml)
```

Expected: all suites report `passed`, 113 tests total.

## 2. Benchmarks (no ROS, seconds)

```bash
python3 benchmarks/run_benchmarks.py
```

Expected: a table; control-path components marked `PASS`. See
`docs/BENCHMARKS.md` for reference numbers.

## 3. Train the models (no ROS, CPU ok, minutes)

```bash
python3 -m pip install torch stable-baselines3 gymnasium onnx onnxscript onnxruntime

# RL navigation policy -> ONNX
cd src/crip_rl_nav && PYTHONPATH=. python3 crip_rl_nav/train.py --timesteps 20000; cd ../..

# LSTM forecaster -> ONNX
cd src/crip_prediction && PYTHONPATH=. python3 crip_prediction/train.py --epochs 15; cd ../..
```

Expected: `.onnx` files written; the export self-check reports ONNX output
matching the trained model to ~1e-8. (These are proof-level models, not tuned
to production quality — see the report.)

## 4. Full build + simulation (needs ROS 2 Jazzy)

### Option A — Docker (recommended; one machine, isolated)

```bash
docker compose up --build          # first build needs internet
docker exec -it crip bash -c "source install/setup.bash && \
  ros2 launch crip_bringup mission.launch.py rviz:=false"
# open http://localhost:8088 and send a mission
```

### Option B — native ROS 2 Jazzy

```bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
colcon test && colcon test-result --verbose      # unit + launch tests
ros2 launch crip_bringup mission.launch.py
```

### Option C — let CI do it

Push to GitHub. `.github/workflows/ci.yml` runs `colcon build` + `colcon test`
inside the official `ros:jazzy` container on every push. The build log is the
evidence.

## Known first-run issues (be ready for these)

The integrated simulation has **not yet been run**. On the first `colcon
build` / launch, the most likely issues, in order:

1. **`rosidl` interface generation** — a custom `.msg`/`.srv` field type or a
   missing `DEPENDENCIES` entry in `crip_interfaces/CMakeLists.txt`.
2. **`ros_gz` bridge topic types** — a mismatch between a Gazebo message type
   and its ROS 2 counterpart in `crip_bringup/config/gz_bridge.yaml`.
3. **URDF spawn** — an inertia or joint issue only Gazebo flags at load.

These are expected integration steps, not design defects. Capture the first
error text; it is the concrete next task.

## Environment captured

- ROS 2 **Jazzy**, Gazebo **Harmonic**, Nav2, `ros_gz`
- Python 3.12; key libs: numpy, torch, stable-baselines3, gymnasium, onnx/
  onnxruntime, fastapi, uvicorn, paho-mqtt, pillow
- Exact versions are pinned in the Docker image (`Dockerfile`).
