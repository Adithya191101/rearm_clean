# Third-Party Sources

This repository tracks project-owned source and patches only. Bootstrap obtains
third-party code and assets at the immutable revisions in
[`dependencies.lock`](dependencies.lock).

| Component | Upstream terms |
|---|---|
| NVIDIA Isaac ROS Manipulation | Apache-2.0 |
| topic_based_ros2_control | BSD-3-Clause |
| Seeed reBotArmController_ROS2 | Package metadata declares Apache-2.0 |
| Seeed reBot-Isaacsim USD | MIT |

Each generated checkout retains its upstream license and copyright files. The
top-level Apache-2.0 license does not relicense those dependencies, NVIDIA
models, the TensorRT container, Isaac Sim, or runtime-streamed YCB/RealSense
assets.

`scripts/install_models.sh` downloads NVIDIA content only after explicit
`--accept-eula`. Review the current NVIDIA Isaac ROS, NGC, model, and Isaac Sim
terms before running it.
