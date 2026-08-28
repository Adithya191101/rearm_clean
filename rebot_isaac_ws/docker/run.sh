#!/usr/bin/env bash
# Launch the reBot Isaac ROS container.
#
#   ./run.sh sim [command...]
#
# This clean repository supports the simulation profile only.
#
# Topology: Isaac Sim runs on the HOST and owns /clock; this container runs the
# ROS 2 Jazzy side. They meet over DDS, which is why --network host and a
# matching ROS_DOMAIN_ID / RMW_IMPLEMENTATION are mandatory on both sides.
set -euo pipefail

PROFILE="${1:-sim}"
shift || true

WS_HOST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$WS_HOST/.." && pwd)"
WS_CONTAINER=/workspaces/rebot_isaac_ws
IMAGE="${REBOT_IMAGE:-rebot-isaac:4.5}"
NAME="rebot-isaac-${PROFILE}"
SOURCE_VERSION="${REBOT_SOURCE_VERSION:-archive}"
if git -C "$REPO_ROOT" rev-parse --verify HEAD >/dev/null 2>&1; then
  SOURCE_VERSION="$(git -C "$REPO_ROOT" rev-parse HEAD)"
  if [[ -n "$(git -C "$REPO_ROOT" status --porcelain -- rebot_isaac_ws)" ]]; then
    SOURCE_VERSION="${SOURCE_VERSION}-dirty"
  fi
fi

ARGS=(
  --rm -i
  --name "${NAME}-$$"
  --gpus all
  --network host          # simplest reliable host<->container DDS
  --ipc host              # DDS shared-memory transport
  -e ROS_DOMAIN_ID=42
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  -e "ISAAC_ROS_WS=${WS_CONTAINER}"
  -e "REBOT_IMAGE=${IMAGE}"
  -e "REBOT_SOURCE_VERSION=${SOURCE_VERSION}"
  -v "${WS_HOST}:${WS_CONTAINER}:rw"
  -v "${REPO_ROOT}/dependencies.lock:/workspaces/dependencies.lock:ro"
  -v /etc/localtime:/etc/localtime:ro
  -w "${WS_CONTAINER}"
)

# Allocate a TTY only when stdin actually is one. Hardcoding -t makes every
# scripted invocation fail with "the input device is not a TTY", which would
# rule out running the gates unattended.
[ -t 0 ] && ARGS+=(-t)

# GUI tools inside the container when a display is available.
if [ -n "${DISPLAY:-}" ]; then
  ARGS+=(-e "DISPLAY=${DISPLAY}" -v /tmp/.X11-unix:/tmp/.X11-unix:rw)
  xhost +local:root >/dev/null 2>&1 || true
fi

[[ "$PROFILE" == "sim" ]] || {
  echo "ERROR: this repository supports only the sim profile" >&2
  echo "usage: $0 sim [command...]" >&2
  exit 2
}
ARGS+=(-e REBOT_PROFILE=sim -e REBOT_USE_SIM_TIME=true)

exec docker run "${ARGS[@]}" "${IMAGE}" "${@:-bash}"
