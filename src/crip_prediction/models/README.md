# Trained forecaster

Train with:
    python3 -m crip_prediction.train --epochs 20 --out models/forecaster.onnx

Run the node against it:
    ros2 run crip_prediction prediction --ros-args -p model_path:=<abs>/models/forecaster.onnx

With no model present, the node uses a constant-velocity forecaster.
