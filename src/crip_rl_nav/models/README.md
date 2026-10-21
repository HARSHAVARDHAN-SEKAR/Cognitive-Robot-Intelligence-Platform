# Trained policies

Place exported policies here. Train with:

    python3 -m crip_rl_nav.train --algo ppo --timesteps 1000000 --out models/policy.onnx

Then run the controller pointing at it:

    ros2 run crip_rl_nav rl_controller --ros-args -p policy_path:=<abs path>/models/policy.onnx

If no policy is present, rl_controller uses a heuristic fallback so the node
still runs.
