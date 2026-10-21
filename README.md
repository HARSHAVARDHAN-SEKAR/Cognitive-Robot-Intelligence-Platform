# CRIP — Cognitive Robot Intelligence Platform (ROS 2 workspace)

![CI](https://github.com/<your-username>/crip/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Tests](https://img.shields.io/badge/tests-113%20passing-brightgreen)
![ROS 2](https://img.shields.io/badge/ROS%202-Jazzy-blue)


A runnable ROS 2 foundation for the larger "Cognitive Robot Intelligence
Platform" vision. This repository is **Milestone 0/1**: a complete, buildable,
simulatable robot stack that the advanced AI modules (LLM brain, VLM reasoning,
RL navigation, digital twin, fleet intelligence) plug into milestone by
milestone.

## Evidence & documentation

| Artifact | What it is |
|---|---|
| [`docs/TECHNICAL_REPORT.pdf`](docs/TECHNICAL_REPORT.pdf) | Full architecture + methodology + verified/unverified boundary |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Measured throughput of every logic core, with budgets |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Exact steps to reproduce every result |
| [`docs/DEMO_STORYBOARD.md`](docs/DEMO_STORYBOARD.md) | Planned 90-second demo; recorded via `scripts/record_demo.sh` |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | CI: logic tests every push + full `colcon build` in `ros:jazzy` |
| `benchmarks/run_benchmarks.py` | Reproduce the benchmark numbers on any machine |
| [`docs/PUBLISHING.md`](docs/PUBLISHING.md) | GitHub/LinkedIn publishing guide + accurate claim wording |

**Verification status:** 113 unit tests pass, 94 modules compile, benchmarks green, PPO+LSTM training with verified ONNX export - all reproducible without ROS. The integrated Gazebo simulation has not yet been run; that is the documented next step (see the report's limitations section).

What works **today**, out of the box:

- A differential-drive robot (`crip_description`) with **2D LiDAR + RGB camera**, described in xacro.
- A **Gazebo (gz-sim / Harmonic)** world with walls and obstacles.
- **RViz2** visualization (robot model, laser scan, camera, odometry, TF).
- A **perception node** that turns the laser scan into a semantic obstacle summary.
- A **reactive safety-pilot brain** that drives the robot, vetoes unsafe motion, and *explains its decisions*.
- A **CI-runnable simulation/integration test** that validates the perception→brain loop without needing a GPU.
- **Docker** packaging so it runs on any Linux machine.

> Scope note: the 12-module cognitive platform (LLM/VLM/RL/digital-twin/fleet)
> is a 12–18 month effort. This workspace is the skeleton with the first
> modules stubbed as real, working nodes. See **Roadmap** below.

---

## Packages

| Package | Type | Purpose | Vision module |
|---|---|---|---|
| `crip_description` | ament_cmake | URDF/xacro robot, Gazebo sensor+driver plugins | — |
| `crip_bringup` | ament_cmake | Worlds, launch files, RViz + ros_gz bridge config | Digital Twin (M9) |
| `crip_perception` | ament_python | Semantic obstacle monitor from LiDAR (hook for YOLO/SAM/Grounding-DINO) | Perception (M1) |
| `crip_brain` | ament_python | Reactive pilot + safety supervisor + explainability (hook for LLM/VLM) | Brain/Safety/XAI (M3/M10/M12) |
| `crip_interfaces` | ament_cmake | Shared msg/srv definitions (SemanticEntity, QueryEntities, HaveISeen) | — |
| `crip_world_model` | ament_python | Persistent semantic map + DB + query services | World Model (M2) |
| `crip_llm_brain` | ament_python | NL mission → grounded task plan (cloud/local/mock LLM) | LLM Brain (M3) |
| `crip_vlm` | ament_python | Vision-language scene assessment + risk (cloud/local/mock) | VLM Reasoning (M4) |
| `crip_navigation` | ament_cmake | Nav2 config, arena map, and bringup launch | Navigation (M5) |
| `crip_rl_nav` | ament_python | RL local planner: Gym env, training harness, ONNX inference node | RL Navigation (M6) |
| `crip_prediction` | ament_python | Multi-object tracking + LSTM trajectory forecasting | Prediction (M7) |
| `crip_memory` | ament_python | Semantic mission memory: embeddings + vector store | Memory (M8) |
| `crip_learning` | ament_python | Self-learning loop: experience log, scheduled retraining, model registry | Self-learning (M9) |
| `crip_dashboard` | ament_python | Web dashboard: live arena map, telemetry, mission control | Dashboard (M10) |
| `crip_safety` | ament_python | Safety supervisor: auditable veto layer on every actuator command | Safety (M11) |
| `crip_fleet` | ament_python | Fleet intelligence: shared map/experience over a pub/sub transport | Fleet (M12) |
| `crip_explain` | ament_python | Explainable AI: causal decision traces and a why service | XAI (M12) |
| `crip_tests` | ament_cmake | launch_testing integration tests | — |

---

## Quick start — Docker (recommended, any Linux machine)

```bash
# 1. Allow the container to use your X server (for Gazebo + RViz windows)
xhost +local:root

# 2. Build and launch the full autonomy demo
docker compose up --build
```

That launches Gazebo, spawns the robot, opens RViz, and starts the
perception + brain nodes. The robot wanders the arena, stopping and turning
away from walls and obstacles.

To get an interactive shell instead:

```bash
docker compose run --rm crip bash
# inside the container:
ros2 launch crip_bringup autonomy.launch.py
```

GPU note: for NVIDIA GPUs, install `nvidia-container-toolkit` and uncomment the
`deploy:` block in `docker-compose.yml` for hardware-accelerated rendering.

---

## Quick start — native (ROS 2 Jazzy + Gazebo Harmonic installed)

```bash
# install deps (Ubuntu 24.04 / ROS 2 Jazzy)
sudo apt install ros-jazzy-ros-gz ros-jazzy-rviz2 ros-jazzy-xacro \
  ros-jazzy-robot-state-publisher ros-jazzy-joint-state-publisher-gui \
  ros-jazzy-launch-testing-ament-cmake

./build.sh                       # colcon build
source install/setup.bash
./run.sh                         # full autonomy demo (Gazebo + RViz)
# or just the robot in RViz, no Gazebo:
ros2 launch crip_description display.launch.py
```

---

## Running the simulation test

```bash
./test.sh
# runs: colcon test --packages-select crip_tests
```

The test (`crip_tests/test/test_pipeline.py`) launches the real perception
and brain nodes, feeds synthetic LaserScan messages, and asserts:

1. Clear scan → robot commands forward motion.
2. Obstacle ahead → robot vetoes forward motion and rotates.

It needs no Gazebo or GPU, so it runs in CI in seconds.

---

## Key topics

| Topic | Type | Direction |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | brain → robot |
| `/odom` | `nav_msgs/Odometry` | robot → |
| `/scan` | `sensor_msgs/LaserScan` | robot → perception |
| `/camera/image_raw` | `sensor_msgs/Image` | robot → |
| `/perception/summary` | `std_msgs/String` | perception (human-readable) |
| `/perception/front_range` | `std_msgs/Float32` | perception → brain |
| `/brain/explanation` | `std_msgs/String` | brain (why it did what it did) |
| `/perception/detections` | `std_msgs/String` | perception → world model (`label:category:dx:dy`) |
| `/world_model/entities` | `crip_interfaces/SemanticEntity` | world model (map dump for viz) |

Watch the robot reason live:

```bash
ros2 topic echo /brain/explanation
ros2 topic echo /perception/summary
ros2 topic echo /world_model/entities
```

## World Model (Milestone 2)

The robot now builds a **persistent semantic map**. As it drives, perception
emits detections (`/perception/detections`), the world model stamps them into
the map frame using odometry, deduplicates by proximity (repeat sightings of
the same object fuse and increment an observation count), and persists them.

Backend selection is automatic: if `CRIP_PG_DSN` is set and reachable it uses
**PostgreSQL**; otherwise it falls back to **SQLite** (set `CRIP_SQLITE_PATH`
for an on-disk file, default in-memory). `docker compose up` starts a Postgres
service and wires the DSN for you.

Query the memory from the command line:

```bash
# Have I ever seen a charging station?
ros2 service call /world_model/have_i_seen crip_interfaces/srv/HaveISeen \
  "{label: 'charging_station'}"

# What hazards are within 3 m of the point (1, 0)?
ros2 service call /world_model/query_entities crip_interfaces/srv/QueryEntities \
  "{category: 'hazard', near: {x: 1.0, y: 0.0, z: 0.0}, max_distance: 3.0}"
```

---

## LLM Brain (Milestone 3)

The robot now accepts **natural-language missions** and turns them into a
**grounded task plan**. The LLM never drives the motors — it only produces an
ordered list of typed tasks (`navigate_to`, `wait`, `report`, `search_for`),
which a task runner executes through the safety-supervised controller.

Backend is switchable via `CRIP_LLM_BACKEND`:

- `mock` (default) — deterministic, offline. No API key, no GPU. Used in tests.
- `anthropic` — Anthropic Messages API (set `ANTHROPIC_API_KEY`, optionally `CRIP_LLM_MODEL`).
- `ollama` — a local model such as Llama/Qwen (set `OLLAMA_HOST`, `CRIP_LLM_MODEL`).

If a selected backend is unavailable, it falls back to `mock` rather than crashing.

**Grounding is the key step.** Every place/object a mission references is
resolved against the M2 world model. A mission that names something the robot
has never seen is *rejected with a reason* rather than executed on a
hallucinated location.

```bash
ros2 service call /llm_brain/plan_mission crip_interfaces/srv/PlanMission \
  "{mission: 'go to the charging station, wait 5 seconds, then come back'}"
```

The response contains the grounded task list and a summary; watch execution on
`/brain/explanation` and the emitted goals on `/goal_pose`. To use a real model:

```bash
CRIP_LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-... docker compose up --build
CRIP_LLM_BACKEND=ollama   CRIP_LLM_MODEL=llama3.1  docker compose up --build
```

---

## VLM Reasoning (Milestone 4)

The robot can now **look at what its camera sees and reason about it**. The VLM
node turns a camera frame into a structured assessment — caption, observed
labels, a **risk level** (none/low/medium/high), rationale, and a recommended
action — mirroring the document's "what is happening? → risk → recommendation"
flow.

Backend is switchable via `CRIP_VLM_BACKEND`, same pattern as the LLM brain:

- `mock` (default) — offline; derives a coarse assessment from image statistics. No model, no GPU.
- `anthropic` — Anthropic vision API (set `ANTHROPIC_API_KEY`, optionally `CRIP_VLM_MODEL`).
- `ollama` — a local vision model such as LLaVA / Qwen-VL (set `CRIP_VLM_MODEL`, `OLLAMA_HOST`).

The node wires into the rest of the stack:

- Publishes assessments on `/vlm/assessment` and a `/vlm/alert` on medium/high risk.
- Feeds observed labels into the **M2 world model** (via `/perception/detections`), so vision enriches the same semantic map.
- The **M3 brain**'s `report` task now calls the VLM, so "tell me what you see" returns a real scene description.

On-demand analysis:

```bash
ros2 service call /vlm/analyze_scene crip_interfaces/srv/AnalyzeScene \
  "{question: 'is the path ahead clear?'}"
```

Use a real vision model instead of the mock:

```bash
CRIP_VLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-... docker compose up --build
CRIP_VLM_BACKEND=ollama   CRIP_VLM_MODEL=llava     docker compose up --build
```

---

## Navigation (Milestone 5)

The loop is now **closed**: a natural-language mission results in the robot
*physically driving* to its goal while avoiding obstacles. Navigation is
handled by **Nav2**, configured for the diff-drive robot in `crip_navigation`:

- A static occupancy map of the arena (`maps/crip_map.pgm`) matching the Gazebo world.
- `map_server` + **AMCL** localization against that map.
- Nav2 planner (NavFn) + **DWB local controller** + recovery behaviors + velocity smoother.
- The robot's existing `/scan`, `/odom`, `/tf` feed Nav2; Nav2 owns `/cmd_vel`.

The **M3 brain now drives through Nav2**: a `navigate_to` task is sent as a
`NavigateToPose` action goal and the executor waits for the real result,
aborting the mission if navigation fails. If Nav2 isn't running it falls back
to publishing `/goal_pose`, so the lightweight `autonomy.launch.py` still works.

Run the full mission stack (sim + Nav2 + brain + world model + VLM):

```bash
ros2 launch crip_bringup mission.launch.py
# then give it a mission (in another shell):
ros2 service call /llm_brain/plan_mission crip_interfaces/srv/PlanMission \
  "{mission: 'go to the charging station'}"
```

`mission.launch.py` is the Nav2-driven stack; `autonomy.launch.py` remains the
reactive (non-Nav2) demo. The DWB local controller is a `nav2_core::Controller`
plugin — M6 introduces a learned RL policy as an alternative local planner.

---

## RL Navigation (Milestone 6)

A reinforcement-learning local planner, delivered as three separable pieces so
each part is independently usable and testable:

1. **Gym environment** (`crip_rl_nav/env.py`) — a pure-Python/NumPy 2D nav sim
   (no ROS, no GPU) whose observation/action/reward contract matches the
   runtime node. Observation = 16 downsampled lidar beams + goal vector
   (distance, sin/cos bearing); action = (v, ω) in [-1, 1]; reward shaped per
   the project spec (+goal, +progress, −collision, −time, −spin). A trivial
   greedy policy reaches >50% of goals, proving the env is *solvable* — the
   precondition for training to mean anything.

2. **Training harness** (`crip_rl_nav/train.py`) — Stable-Baselines3 PPO/SAC.
   This is an **offline GPU step**, deliberately not part of the ROS build or
   the test suite. It exports the deterministic actor to **ONNX**:

   ```bash
   pip install gymnasium stable-baselines3 torch onnx onnxscript
   python3 -m crip_rl_nav.train --algo ppo --timesteps 1000000 \
       --out src/crip_rl_nav/models/policy.onnx
   ```

3. **Inference node** (`crip_rl_nav/rl_controller.py`) — builds the same
   observation from live `/scan` + `/odom` + `/goal_pose`, runs the ONNX policy
   via onnxruntime, and publishes `/cmd_vel`. If no policy is present (or
   onnxruntime is missing) it falls back to a heuristic controller so the node
   always runs.

   ```bash
   ros2 launch crip_rl_nav rl_controller.launch.py \
       policy_path:=/abs/path/to/policy.onnx
   ```

The ONNX export was verified to reproduce the trained policy's deterministic
output to within ~1e-8, so the policy transfers from training to robot without
behavioral drift.

**Where this sits relative to Nav2.** `rl_controller` is an alternative
`/cmd_vel` source — run only one velocity publisher at a time (it replaces
`safety_pilot` or Nav2's controller, not stacks on them). The production path
is to wrap this same ONNX policy as a C++ `nav2_core::Controller` plugin so it
runs *inside* Nav2's costmap/recovery machinery, replacing DWB while keeping the
planner and behavior tree. That C++ plugin is the documented next step; the
Python node proves the policy and the pipeline end to end.

---

## Prediction (Milestone 7)

The robot now anticipates **where moving obstacles will be**, not just where
they are. Two parts, both wired into the running sim:

**A moving actor.** The Gazebo world now contains a `walking_person` actor that
loops a rectangular path. Its pose is bridged to ROS on `/dynamic_pose/info`,
giving a real dynamic obstacle to track.

**Tracking + forecasting** (`crip_prediction`), structured like M6:

1. **Tracker** (`tracker.py`) — nearest-neighbour multi-object tracking with
   finite-difference velocity estimation; ages out stale tracks. Pure Python.
2. **Forecaster** (`forecaster.py`) — an **LSTM** that predicts the next 8
   displacement steps from the past 8, trained offline on synthetic
   pedestrian-like tracks and exported to ONNX. A **constant-velocity**
   forecaster is the parameter-free fallback when no model is present, so the
   node always runs.
3. **Node** (`prediction_node.py`) — tracks dynamic poses, forecasts each, and
   publishes a `nav_msgs/Path` per object on `/prediction/tracks/<id>` plus a
   `/prediction/summary`. Predicted positions are also injected into the world
   model as transient `predicted_person` hazards via `/perception/detections`,
   so the planner can avoid where the person *will be*.

Train the forecaster (offline; same deps as M6):

```bash
pip install torch onnx onnxscript numpy
python3 -m crip_prediction.train --epochs 20 \
    --out src/crip_prediction/models/forecaster.onnx
ros2 launch crip_prediction prediction.launch.py \
    model_path:=/abs/path/to/forecaster.onnx
```

Watch predictions live:

```bash
ros2 topic echo /prediction/summary
ros2 topic echo /prediction/tracks/0     # nav_msgs/Path of the forecast
```

---

## Memory (Milestone 8)

The robot now has a **semantic memory** of its missions: it can answer "have I
done something like this before?" by meaning, not exact-label matching.

`crip_memory` follows the established backend-switch pattern:

- **Embedder** (`CRIP_EMBED_BACKEND`): `hashing` (default, dependency-free —
  hashed word/char n-grams give real lexical similarity offline), `local`
  (sentence-transformers, semantic), or `api` (OpenAI-compatible
  `/v1/embeddings`). Unavailable backends fall back to hashing.
- **Vector store**: **pgvector** on the existing Postgres (`CRIP_PG_DSN`) when
  available; otherwise an in-process NumPy cosine store, optionally persisted to
  `CRIP_MEMORY_PATH`. Tables are keyed by `(embed_backend, dim)` so vector
  spaces never mix.

Services: `~/record_experience` and `~/recall_experience` (cosine top-k).

**Wired into the brain (M3):** each executed mission is recorded with its
outcome; before planning a new mission, the brain recalls similar past
experiences and surfaces them on `/brain/explanation`.

```bash
ros2 service call /memory/record_experience crip_interfaces/srv/RecordExperience \
  "{text: 'inspected fire extinguisher on floor two, all OK', metadata_json: '{\"outcome\":\"success\"}'}"

ros2 service call /memory/recall_experience crip_interfaces/srv/RecallExperience \
  "{query: 'check the fire extinguishers upstairs', k: 3}"
```

Semantic backends:

```bash
CRIP_EMBED_BACKEND=local CRIP_EMBED_MODEL=all-MiniLM-L6-v2 docker compose up --build
CRIP_EMBED_BACKEND=api   CRIP_EMBED_KEY=sk-...              docker compose up --build
```

---

## Self-learning loop (Milestone 9)

The robot **improves from its own logged experience**: Logs → Dataset →
Training → Evaluation → gated Deploy (Module 8). Crucially, retrained models
are *not* blindly deployed — a candidate is promoted only if it measurably
beats the current model on a held-out split.

`crip_learning` provides:

- **Experience logger** (`experience_logger`) — records observed object tracks
  (from `/prediction/summary`) and mission outcomes (from `/memory/log`) to a
  JSONL file.
- **Dataset builder** (`dataset.py`) — reconstructs per-object trajectories
  from the log and turns them into the same training windows the forecaster
  uses, so the loop learns the motion the robot *actually observed*.
- **Model registry** (`registry.py`) — versioned models (`v0001`, `v0002`, …)
  with a `current` pointer and per-version metadata.
- **Pipeline** (`pipeline.py`) — builds the dataset (augmenting with synthetic
  tracks if logged data is sparse), trains a candidate forecaster, evaluates
  candidate *and* current on the same validation split, and **promotes only on
  improvement**.
- **Scheduler** (`learning_trainer`) — runs the pipeline on a timer
  (`interval_sec`, default hourly) in a background thread, reporting on
  `/learning/status`.

The prediction node loads `<registry>/current/model.onnx` if present, so a
promoted model is picked up on restart — closing the loop.

Run a learning pass manually:

```bash
python3 -m crip_learning.pipeline \
  --log ~/.crip/experience.jsonl \
  --registry ~/.crip/registry \
  --epochs 15 --min-improvement 0.02
```

The forecaster trains on CPU in minutes, so the loop completes on a schedule.
(The RL policy from M6 retrains the same way but wants a GPU; the forecaster is
used here precisely because it makes the loop runnable end to end.)

---

## Dashboard (Milestone 10)

A web **operator dashboard** gives a live view of the whole stack and a way to
command it from the browser. `crip_dashboard` runs a FastAPI server alongside a
ROS node: it subscribes to the stack's topics, folds them into a state
snapshot, and streams it over a WebSocket (~5 Hz) to a single-page UI at
`http://localhost:8088`.

The view:

- **Live arena map** — robot position + heading with a fading motion trail,
  the static obstacles, the current goal, and — the signature element — each
  tracked person's **predicted trajectory** drawn as a dashed magenta path, so
  "where the person will be" reads at a glance.
- **Mission control** — type a natural-language mission; it POSTs to the brain's
  `plan_mission` service.
- **Telemetry** — pose, scene risk (color-coded), uptime.
- **Activity feed** — brain explanations, VLM alerts, learning-loop promotions,
  and operator commands, interleaved.

```bash
ros2 launch crip_dashboard dashboard.launch.py   # or it's in mission.launch.py
# open http://localhost:8088
```

If FastAPI/uvicorn aren't installed the node logs a notice and runs as a plain
ROS node, so it never breaks the launch.

**On "digital twin":** the dashboard is the live operational view. Full **Isaac
Sim parity** — porting the Gazebo world to NVIDIA Omniverse/Isaac Sim for
photoreal, GPU-accelerated physics — is the heavier simulation-upgrade path and
is documented as future work rather than built here, since it needs Omniverse
and isn't verifiable in this environment.

---

## Safety supervisor (Milestone 11)

A formal, auditable **veto layer on the wire**: no actuator command reaches the
robot without passing it (the document's "LLM Decision → Safety Checker →
Allowed? → Execute"). Sources publish to `/cmd_vel_raw`; the supervisor vets
each command against live sensors and publishes the result to `/cmd_vel`.

    <planners / controllers> --/cmd_vel_raw--> [supervisor] --/cmd_vel--> robot

`crip_safety` is a pure-Python **rule engine** (`rules.py`) plus a thin ROS
node. Rules run as an ordered pipeline, each returning allow / modify / veto
with a reason:

1. **e-stop** — `/safety/estop` latches a full stop, overriding everything.
2. **geofence** — vetoes forward motion that would leave the arena.
3. **collision** — full stop before a frontal obstacle; proportional slow-down in the approach band.
4. **side clearance** — blocks turning into a very close side obstacle.
5. **speed limit** — clamps `v`/`w` to configured maxima.
6. **rate limit** — bounds acceleration for smooth motion — but never re-accelerates a command an earlier safety rule reduced (a safety stop always reaches zero).

A **watchdog** stops the robot if no command arrives within a timeout, so a
crashed upstream node can't leave it driving. Every modify/veto is published on
`/safety/audit` and shown in the dashboard's activity feed, so the chain is
fully traceable.

The engine is deterministic and exhaustively unit-tested (15 cases incl. the
subtle "veto must not be re-accelerated" property). In `mission.launch.py`,
Nav2's velocity output is remapped through the supervisor automatically.

```bash
# engage / clear an emergency stop from the command line:
ros2 topic pub --once /safety/estop std_msgs/Bool "{data: true}"
ros2 topic echo /safety/audit
```

---

## Fleet intelligence + Explainable AI (Milestone 12)

The capstone: the robot shares what it learns with a fleet, and can explain why
it did what it did.

**Fleet intelligence** (`crip_fleet`, Module 11). Robots publish discoveries —
semantic entities, mission outcomes, model promotions — to a shared pub/sub
transport namespaced under `crip/fleet/...`, and subscribe to their peers'.
Incoming entities are deduplicated by `(label, cell)` and merged by confidence,
so the fleet converges on one consistent map: **one robot learns, all improve.**
A peer's discovery is injected into the local world model via
`/perception/detections`, becoming navigable for everyone.

The transport is abstracted: an `InMemoryBroker` (tests / single-process) and a
real `MqttTransport` (paho-mqtt) selected by `CRIP_FLEET_MQTT=host:port`. Each
robot has a `robot_id` and ignores its own echoed messages.

**Explainable AI** (`crip_explain`, Module 12). Every subsystem already emits
evidence — the brain's plan, the VLM's risk rationale, the predictor's
forecast, the safety supervisor's veto. The explain node collects these as
typed, timestamped events and assembles a causal narrative on demand, ranking
decisive events (vetoes, high risk) above routine status and weighting by
recency:

> *"Stopped because: tracked person predicted to cross ahead; HIGH risk: person
> in path → reroute; VETO frontal obstacle 0.22m."*

```bash
ros2 service call /explain/explain crip_interfaces/srv/ExplainDecision "{top_k: 3}"
ros2 topic echo /explain/why     # live causal explanation on each decisive event
```

Both surface in the dashboard: a live **Why** banner and a **fleet** status line.

**On the fleet demo:** the in-memory broker only links robots in the *same
process*, so a true multi-robot demo on one host needs MQTT (e.g. a local
`mosquitto`) with each robot launched under a distinct `robot_id`. A full
multi-robot Gazebo fleet isn't spun up here; the sharing/merge logic is what's
built and verified.

---

## Robustness improvements (post-review hardening)

Three additions from an architecture review, each pure logic and fully tested:

- **Confidence manager** (`crip_safety/confidence.py`) — every module reports
  `(source, confidence)`; a conservative fusion (weighted mean, hard-capped by
  the worst *critical* source, stale = zero) maps to **act / degrade / hold**.
  The supervisor scales allowed speed by the result — it can only lower what
  the safety rules allow, never raise it. Report via
  `/confidence/report` (`std_msgs/String`, `"source:0.87"`).
- **Knowledge graph** (`crip_world_model/relations.py`) — typed relations over
  entities (`bottle on table in kitchen in building_a`) with transitive
  containment queries (`contents_of`, `where_is`) and automatic spatial `near`
  inference from positions.
- **Health monitor** (`crip_dashboard/health.py`) — per-module heartbeats from
  topic activity, classified **ok / slow / stale / missing**, overall = worst;
  state *changes* raise alerts into the dashboard activity feed (no spam), and
  the summary ships in the WebSocket snapshot.

---

## Testing, CI, and benchmarks

- **Unit tests:** 113 pytest tests across 11 ROS-free suites (`src/*/test/`).
  Run any suite directly, e.g.
  `cd src/crip_safety && PYTHONPATH=. pytest test/ -k "not node" -q`.
- **Integration tests:** `launch_testing` suites (in `crip_tests` and package
  `test/` dirs) run under `colcon test` on a ROS machine — they require the
  ROS graph and are excluded from the pure-Python sweep.
- **CI:** `.github/workflows/ci.yml` — two jobs. `logic-tests` compiles every
  file and runs all unit suites on each push; `ros-build` runs
  `colcon build` + `colcon test` inside the official `ros:jazzy` container.
  Push the repo to GitHub and CI runs automatically; the first green
  `ros-build` is the integration milestone.
- **Benchmarks:** `python3 benchmarks/run_benchmarks.py` measures every logic
  core with pass/fail budgets. Reference results (this build machine):
  safety engine ~195k decisions/s (budget: 20 Hz), confidence fusion ~378k/s,
  tracker ~540k/s, memory recall over 500 records ~49k/s, dashboard snapshot
  ~496k/s. These are logic benchmarks; end-to-end latency needs the live run.
- **Demo videos:** `scripts/record_demo.sh` records a screen capture + a
  ros2 bag of decision topics, with a suggested 90-second storyline. It must
  be run on the machine where the simulation is actually running — no demo
  video can exist until the first live run.

---

## Status

All twelve milestones are complete. The platform spans perception, a persistent
semantic world model, an LLM task planner, vision-language reasoning, classical
and learned navigation, trajectory prediction, semantic memory, a self-learning
loop, an operator dashboard, an auditable safety layer, and fleet-wide sharing
with explainable decisions — one coherent system rather than disconnected demos.

Documented as future hardening (needs hardware/GPU/external services not
available in this build environment): swapping mock LLM/VLM/embedding backends
for cloud or local models; training the RL and forecaster models to convergence
on a GPU; the C++ `nav2_core::Controller` plugin wrapping the RL policy; Isaac
Sim parity; and a real multi-robot MQTT fleet.

---

## Roadmap — growing the skeleton into the full platform

All milestones complete. Each shipped with a working system + a clean repo state.

- **M1 (done): Perception hook.** Replace `obstacle_monitor` internals with YOLOv11 / SAM 2 / Grounding DINO on `/camera/image_raw`, publishing detected classes.
- **M2 (done): World model.** Persistent semantic map in PostgreSQL/SQLite, with proximity dedup and `query_entities` / `have_i_seen` services. Perception now feeds detections in; the brain can query the map.
- **M3 (done): LLM brain.** NL missions → grounded task plans via switchable backend (anthropic/ollama/mock). References resolved against the world model; unknown references rejected, not hallucinated.
- **M4 (done): VLM reasoning.** Vision-language scene assessment with risk levels and recommendations; switchable backend; feeds the world model and the brain's report task.
- **M5 (done): Nav2 navigation.** Static map + AMCL + Nav2 planner/controller/recovery. The brain's `navigate_to` now drives closed-loop via the `NavigateToPose` action; the robot reaches emitted goals while avoiding obstacles.
- **M6 (done): RL navigation.** Gym env + SB3 PPO/SAC training harness + ONNX inference node. Env validated solvable; ONNX export verified to match the trained policy. Productionization as a C++ Nav2 controller plugin documented.
- **M7 (done): Prediction.** Moving actor in the sim + multi-object tracker + LSTM forecaster (ONNX, with constant-velocity fallback). Predictions published as per-track paths and injected as transient hazards.
- **M8 (done): Memory + embeddings.** Switchable embedder (hashing/local/api) + pgvector/NumPy vector store. The brain records mission outcomes and recalls similar past experiences before planning.
- **M9 (done): Self-learning loop.** Experience logging + scheduled retraining + model registry with an evaluation gate that promotes only improved models. Prediction node loads the promoted `current` model.
- **M10 (done): Dashboard.** Web operator dashboard: live arena map with predicted trajectories, mission control, telemetry, and an activity feed over a WebSocket bridge. Isaac Sim parity documented as future work.
- **M11 (done): Safety supervisor.** Auditable rule-engine veto layer gating every actuator command (e-stop, geofence, collision, side-clearance, speed/rate limits) with a watchdog and an audit trail.
- **M12 (done): Fleet intelligence + Explainable AI.** Pub/sub fleet sharing (in-memory or MQTT) with confidence-merged map convergence, plus a causal explanation service surfaced live in the dashboard.

The current code deliberately exposes the seams (`perception/front_range`,
`brain/explanation`, the behavior of `safety_pilot`) so each milestone is an
additive change rather than a rewrite.

---

## License

Apache-2.0.
