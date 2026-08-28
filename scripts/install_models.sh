#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WS_HOST="$REPO_ROOT/rebot_isaac_ws"
WS_CONTAINER=/workspaces/rebot_isaac_ws
IMAGE="${REBOT_IMAGE:-rebot-isaac:4.5}"
ACCEPT_EULA="${ISAAC_ROS_ACCEPT_EULA:-0}"

usage() {
  cat <<'EOF'
usage: install_models.sh --accept-eula

Downloads NVIDIA Grounding DINO and FoundationPose models, downloads the
FoundationPose demonstration assets, and builds TensorRT engines for this GPU.

--accept-eula acknowledges that NVIDIA model and runtime terms apply. Review:
  https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/grounding_dino
  https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/models/foundationpose
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --accept-eula) ACCEPT_EULA=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ "$ACCEPT_EULA" != "1" ]]; then
  echo "ERROR: explicit NVIDIA EULA acceptance is required." >&2
  echo "Review the URLs in '$0 --help', then pass --accept-eula." >&2
  exit 2
fi

docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "ERROR: Docker image '$IMAGE' is missing; run ./scripts/bootstrap.sh." >&2
  exit 1
}

mkdir -p "$WS_HOST/isaac_ros_assets"

docker run --rm \
  --gpus all \
  --network host \
  --ipc host \
  --entrypoint bash \
  -e ISAAC_ROS_ACCEPT_EULA=1 \
  -e MANIPULATOR_INSTALL_ASSETS=1 \
  -e CUDA_MPS_PIPE_DIRECTORY=/tmp/no-mps-model-install \
  -e "ISAAC_ROS_WS=$WS_CONTAINER" \
  -v "$WS_HOST:$WS_CONTAINER:rw" \
  -w "$WS_CONTAINER" \
  "$IMAGE" -lc '
    set -eo pipefail
    set +u
    source /opt/ros/jazzy/setup.bash
    set -u
    fp_dir="$ISAAC_ROS_WS/isaac_ros_assets/models/foundationpose"
    if [[ ! -s "$fp_dir/refine_model.onnx" ||
          ! -s "$fp_dir/score_model.onnx" ]]; then
      /opt/ros/jazzy/lib/isaac_ros_foundationpose_models_install/install_foundationpose_models.sh
    fi
    if [[ ! -s "$fp_dir/refine_trt_engine.plan" ]]; then
      echo "Generating FoundationPose refine TensorRT engine."
      /usr/src/tensorrt/bin/trtexec \
        --onnx="$fp_dir/refine_model.onnx" \
        --saveEngine="$fp_dir/refine_trt_engine.plan" \
        --minShapes=input1:1x160x160x6,input2:1x160x160x6 \
        --optShapes=input1:1x160x160x6,input2:1x160x160x6 \
        --maxShapes=input1:42x160x160x6,input2:42x160x160x6 \
        >/dev/null
    fi
    if [[ ! -s "$fp_dir/score_trt_engine.plan" ]]; then
      echo "Generating FoundationPose score TensorRT engine."
      /usr/src/tensorrt/bin/trtexec \
        --onnx="$fp_dir/score_model.onnx" \
        --saveEngine="$fp_dir/score_trt_engine.plan" \
        --minShapes=input1:1x160x160x6,input2:1x160x160x6 \
        --optShapes=input1:1x160x160x6,input2:1x160x160x6 \
        --maxShapes=input1:252x160x160x6,input2:252x160x160x6 \
        >/dev/null
    fi
    gdino_engine="$ISAAC_ROS_WS/isaac_ros_assets/models/grounding_dino/grounding_dino_model.plan"
    if [[ ! -s "$gdino_engine" ]]; then
      /opt/ros/jazzy/lib/isaac_ros_grounding_dino_models_install/install_grounding_dino_models.sh
    fi
    python3 \
      /workspaces/rebot_isaac_ws/src/isaac_ros_manipulation/isaac_ros_manipulation_asset_bringup/scripts/setup_perception_models.py \
      --workspace /workspaces/rebot_isaac_ws \
      --models foundationpose
  '

"$WS_HOST/docker/verify_models.sh"
