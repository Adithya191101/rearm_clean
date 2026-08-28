#!/usr/bin/env bash
set -Eeuo pipefail

ISAACSIM_ROOT="${ISAACSIM_ROOT:-$HOME/isaacsim-5.1}"

if [[ ! -x "$ISAACSIM_ROOT/isaac-sim.sh" ]]; then
    echo "Isaac Sim launcher not found at $ISAACSIM_ROOT/isaac-sim.sh" >&2
    exit 1
fi
if [[ ! -x "$ISAACSIM_ROOT/python.sh" ]]; then
    echo "Isaac Sim Python launcher not found at $ISAACSIM_ROOT/python.sh" >&2
    exit 1
fi

# The host-wide MPS daemon serializes servers by UID. If another UID has active
# clients, Isaac is queued during renderer startup. A private pipe path keeps
# Isaac on direct CUDA without disrupting those existing MPS clients.
runtime_dir="${XDG_RUNTIME_DIR:-/tmp}"
if [[ ! -d "$runtime_dir" || ! -w "$runtime_dir" ]]; then
    runtime_dir="/tmp"
fi
mps_pipe_dir="$(mktemp -d "$runtime_dir/isaacsim-no-mps.XXXXXX")"
trap 'rmdir "$mps_pipe_dir" 2>/dev/null || true' EXIT
export CUDA_MPS_PIPE_DIRECTORY="$mps_pipe_dir"

# rclpy defaults to $HOME/.ros/log, which may be read-only in managed runtime
# environments. Keep Isaac's logs in a writable, per-user host directory.
ros_log_dir="/tmp/rebot-isaac-ros-${UID:-$(id -u)}"
mkdir -p "$ros_log_dir"
chmod 700 "$ros_log_dir"
export ROS_LOG_DIR="$ros_log_dir"

# Remove overrides from previous troubleshooting attempts.
unset DISABLE_NGX VK_ICD_FILENAMES __NV_PRIME_RENDER_OFFLOAD
unset __GLX_VENDOR_LIBRARY_NAME

# Isaac owns its Python runtime and ROS bridge. A shell that previously sourced
# host Humble otherwise makes Python 3.11 import Humble's Python 3.10 rclpy
# before Isaac can load its bundled Jazzy module.
unset PYTHONPATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
unset ROS_PACKAGE_PATH ROS_VERSION ROS_PYTHON_VERSION
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    filtered_ld=""
    IFS=: read -r -a ld_entries <<< "$LD_LIBRARY_PATH"
    for entry in "${ld_entries[@]}"; do
        [[ "$entry" == /opt/ros/* ]] && continue
        filtered_ld+="${filtered_ld:+:}${entry}"
    done
    export LD_LIBRARY_PATH="$filtered_ld"
fi

# Stop only stale Isaac Sim 5.1 Kit processes.
pkill -TERM -f "$ISAACSIM_ROOT/kit/kit" 2>/dev/null || true
sleep 1
pkill -KILL -f "$ISAACSIM_ROOT/kit/kit" 2>/dev/null || true

if [[ "${1:-}" == "--python" ]]; then
    shift
    if [[ "$#" -eq 0 ]]; then
        echo "usage: $0 --python SCRIPT [ARGS...]" >&2
        exit 2
    fi
    "$ISAACSIM_ROOT/python.sh" "$@"
    exit
fi

cd "$ISAACSIM_ROOT"
./isaac-sim.sh --/renderer/multiGpu/enabled=false "$@"
