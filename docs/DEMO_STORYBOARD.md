# CRIP — Demo Storyboard

The demo video is the single highest-value piece of evidence, and it is the
one artifact that **cannot be produced until the simulation runs on a ROS 2
machine**. This document plans it fully so that, once the sim runs, recording
is mechanical. Recording is automated by `scripts/record_demo.sh`.

## Setup (on the ROS machine)

1. `docker compose up --build`
2. In another terminal:
   `docker exec -it crip bash -c "source install/setup.bash && ros2 launch crip_bringup mission.launch.py"`
3. Arrange three windows side by side: **Gazebo** (the world), **RViz** (the
   robot's view), **browser at http://localhost:8088** (the dashboard).
4. In a third terminal: `./scripts/record_demo.sh 90`

## The 90-second storyline

| Time | On screen | What it proves |
|---|---|---|
| 0:00–0:10 | Dashboard open, robot idle, health panel green | System is up; all nodes alive (health monitor) |
| 0:10–0:20 | Type mission: *"go to the charging station and report what you see"* | Natural-language command → structured plan (LLM brain) |
| 0:20–0:35 | Robot begins driving in Gazebo; trail appears on dashboard map | Grounding + Nav2 navigation working end to end |
| 0:35–0:50 | A person actor crosses ahead; magenta predicted path appears | Perception + trajectory prediction |
| 0:50–1:05 | Safety VETO appears in audit feed; robot stops; "Why" banner explains | Safety supervisor + explainability |
| 1:05–1:20 | Person passes; robot resumes and reaches the goal | Recovery, closed-loop mission completion |
| 1:20–1:30 | Learning/fleet lines update; VLM scene report shows | Self-learning loop + fleet + VLM |

This single take exercises 10 of the 12 subsystems in one continuous shot —
which is far stronger evidence than twelve separate clips.

## What `record_demo.sh` captures

- `demos/crip_demo_screen_*.mp4` — the screen recording (the shareable video).
- `demos/crip_demo_*.bag` — a ROS 2 bag of the decision topics
  (`/cmd_vel`, `/safety/audit`, `/explain/why`, `/prediction/summary`, …).
  The bag is *replayable evidence*: a reviewer can replay it and inspect the
  actual messages, not just watch a video.

## After recording

1. Trim to the storyline above (any editor).
2. Add it to the README with a thumbnail linking to the hosted video.
3. Commit the `.bag` (or a link) as reproducible evidence alongside the video.

Until this runs, the README states honestly that the demo is pending the first
integrated run — no staged or mocked footage.
