#!/usr/bin/env bash
# Launch the reBot PICK_AND_PLACE workflow with LIVE perception:
# Grounding DINO ('soup can.') -> FoundationPose 6DOF -> cuMotion -> pick.
# CUDA_MPS_PIPE_DIRECTORY bypass is MANDATORY: without it every CUDA client in the
# container (the TensorRT DNN nodes: Grounding DINO, FoundationPose) deadlocks on
# the host MPS control socket exactly like trtexec did -- 0% GPU, hung forever.
# See MEMORY: rebot-container-cuda-mps-hang.
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WS_HOST="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WS=/workspaces/rebot_isaac_ws
HF_CACHE_HOST="${HF_CACHE_HOST:-/tmp/rebot_hf_cache}"
IMAGE="${REBOT_IMAGE:-rebot-isaac:4.5}"
REPO_ROOT="$(cd -- "$WS_HOST/.." && pwd)"
SOURCE_VERSION="${REBOT_SOURCE_VERSION:-archive}"
if git -C "$REPO_ROOT" rev-parse --verify HEAD >/dev/null 2>&1; then
  SOURCE_VERSION="$(git -C "$REPO_ROOT" rev-parse HEAD)"
  if [[ -n "$(git -C "$REPO_ROOT" status --porcelain -- rebot_isaac_ws)" ]]; then
    SOURCE_VERSION="${SOURCE_VERSION}-dirty"
  fi
fi
mkdir -p "$HF_CACHE_HOST"
docker run --rm -i \
  --name rebot-isaac-sim-wf \
  --gpus all \
  --network host \
  --ipc host \
  -e ROS_DOMAIN_ID=42 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e ISAAC_ROS_WS=$WS \
  -e ISAAC_ROS_MANIPULATION_PICK_AND_PLACE_CONFIG_DIR=$WS/config/pick_and_place \
  -e REBOT_SOURCE_VERSION="$SOURCE_VERSION" \
  -e CUDA_MPS_PIPE_DIRECTORY=/tmp/no-mps-live \
  -e ISAAC_ROS_ACCEPT_EULA=1 \
  -v "$WS_HOST:$WS:rw" \
  -v "$HF_CACHE_HOST:/root/.cache/huggingface:rw" \
  -v /etc/localtime:/etc/localtime:ro \
  -w "$WS" \
  "$IMAGE" \
  bash -lc '
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/no-mps-live
    source /workspaces/rebot_isaac_ws/install/setup.bash
    /workspaces/rebot_isaac_ws/docker/verify_overlay.sh || exit $?
    ros2 launch isaac_ros_manipulation_rebot_driver_utils rebot_workflow.launch.py
  '
