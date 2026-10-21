# CRIP — Cognitive Robot Intelligence Platform
# ROS 2 Jazzy + Gazebo Harmonic, runs on any Linux machine with Docker.
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# Core ROS + simulation dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-ros-gz \
    ros-jazzy-ros-gz-sim \
    ros-jazzy-ros-gz-bridge \
    ros-jazzy-navigation2 \
    ros-jazzy-nav2-bringup \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-joint-state-publisher \
    ros-jazzy-joint-state-publisher-gui \
    ros-jazzy-xacro \
    ros-jazzy-rviz2 \
    ros-jazzy-launch-testing \
    ros-jazzy-launch-testing-ament-cmake \
    ros-jazzy-launch-ros \
    python3-colcon-common-extensions \
    python3-pip \
    python3-psycopg2 \
    python3-pil \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# RL inference runtime (M6). onnxruntime is small; it lets rl_controller run a
# trained ONNX policy. Training deps (torch, stable-baselines3, gymnasium) are
# intentionally NOT installed here -- training is an offline GPU step:
#   pip install gymnasium stable-baselines3 torch onnx onnxscript
# The self-learning loop (M9) trainer node also needs torch+onnx+onnxscript to
# run a retraining pass; install them in the container if you want the
# in-container learning_trainer to actually train (forecaster trains on CPU).
RUN pip install --no-cache-dir --break-system-packages onnxruntime || true

# Web dashboard (M10): FastAPI + uvicorn serve the operator dashboard and the
# WebSocket telemetry stream on port 8088.
RUN pip install --no-cache-dir --break-system-packages fastapi "uvicorn[standard]" || true

# Fleet intelligence (M12): paho-mqtt for the real multi-robot transport.
# Without it (or a broker), the fleet node uses an in-process broker.
RUN pip install --no-cache-dir --break-system-packages paho-mqtt || true

# Build the workspace
WORKDIR /crip_ws
COPY src/ ./src/
RUN source /opt/ros/jazzy/setup.bash && \
    colcon build --symlink-install

# Source the overlay automatically in interactive shells
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo "source /crip_ws/install/setup.bash" >> /root/.bashrc

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
