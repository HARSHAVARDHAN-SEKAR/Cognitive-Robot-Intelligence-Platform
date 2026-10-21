#!/usr/bin/env bash
# =============================================================================
# CRIP demo recorder
#
# Records demo videos of the running system. Run this ON THE MACHINE where the
# simulation is actually running (it cannot work anywhere else -- the sim must
# be up first):
#
#   Terminal 1:  docker compose up --build
#   Terminal 2:  docker exec -it crip bash -c \
#                  "source install/setup.bash && ros2 launch crip_bringup mission.launch.py"
#   Terminal 3:  ./scripts/record_demo.sh
#
# Produces in ./demos/:
#   crip_demo_screen.mp4   -- screen capture (Gazebo + RViz + dashboard)
#   crip_demo.bag/         -- ros2 bag of key topics (replayable evidence)
#
# Requirements on the host: ffmpeg (apt install ffmpeg), an X11 session.
# =============================================================================
set -euo pipefail

OUT_DIR="demos"
DURATION="${1:-90}"            # seconds to record (default 90)
DISPLAY_ID="${DISPLAY:-:0}"
RES="$(xdpyinfo | awk '/dimensions/{print $2}' 2>/dev/null || echo 1920x1080)"

mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "[demo] recording ${DURATION}s of screen ${DISPLAY_ID} (${RES})"
echo "[demo] tip: arrange Gazebo, RViz and http://localhost:8088 side by side,"
echo "[demo]      then send a mission from the dashboard so the video shows"
echo "[demo]      command -> plan -> drive -> safety in one take."

# 1) screen capture
ffmpeg -loglevel warning -y \
  -f x11grab -video_size "$RES" -framerate 30 -i "$DISPLAY_ID" \
  -t "$DURATION" -c:v libx264 -preset veryfast -pix_fmt yuv420p \
  "$OUT_DIR/crip_demo_screen_${STAMP}.mp4" &
FFMPEG_PID=$!

# 2) rosbag of the decision-relevant topics (inside the container if needed)
BAG_TOPICS="/odom /cmd_vel /cmd_vel_raw /safety/audit /explain/why \
/prediction/summary /vlm/alert /learning/status /fleet/summary"
if command -v ros2 >/dev/null 2>&1; then
  timeout "$DURATION" ros2 bag record -o "$OUT_DIR/crip_demo_${STAMP}.bag" \
    $BAG_TOPICS || true
else
  echo "[demo] ros2 not on host PATH -- record the bag inside the container:"
  echo "       docker exec -it crip bash -c 'source install/setup.bash && \\"
  echo "         ros2 bag record -o /ws/demos/crip_demo.bag $BAG_TOPICS'"
  sleep "$DURATION"
fi

wait "$FFMPEG_PID" || true
echo "[demo] done. Files in $OUT_DIR/:"
ls -lh "$OUT_DIR" | tail -n +2

cat <<'EOF'

[demo] Suggested demo storyline (90 seconds):
  0-10s   dashboard open, robot idle, health panel green
  10-20s  type mission: "go to the charging station and report what you see"
  20-60s  robot drives in Gazebo; dashboard shows trail + predicted person path
  60-75s  person crosses ahead -> safety VETO appears in audit + Why banner
  75-90s  robot resumes, reaches goal; learning/fleet lines update
EOF
