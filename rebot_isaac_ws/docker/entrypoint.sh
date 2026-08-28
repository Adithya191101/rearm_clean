#!/usr/bin/env bash
# Container entrypoint: source ROS underlay, then the workspace overlay if it
# has been built. NVIDIA's docs require sourcing install/setup.bash in every
# shell, so this does it once here rather than relying on the user remembering.
set -e

source /opt/ros/"${ROS_DISTRO}"/setup.bash

if [ -f "${ISAAC_ROS_WS}/install/setup.bash" ]; then
    source "${ISAAC_ROS_WS}/install/setup.bash"
fi

# The reBot launch wrapper resolves its installed package profile directly.
# Keep this variable canonical for upstream tools that still consult it.
if REBOT_DRIVER_SHARE="$(
    ros2 pkg prefix --share isaac_ros_manipulation_rebot_driver_utils 2>/dev/null
)"; then
    export ISAAC_ROS_MANIPULATION_WORKFLOW_CONFIG_DIR="${REBOT_DRIVER_SHARE}/params"
else
    export ISAAC_ROS_MANIPULATION_WORKFLOW_CONFIG_DIR="${ISAAC_ROS_WS}/src/rebot_b601dm_isaac/isaac_ros_manipulation_rebot_driver_utils/params"
fi
export ISAAC_ROS_MANIPULATION_PICK_AND_PLACE_CONFIG_DIR="${ISAAC_ROS_WS}/config/pick_and_place"

exec "$@"
