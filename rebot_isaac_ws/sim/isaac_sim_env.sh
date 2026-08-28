#!/usr/bin/env bash
# Host-side Isaac Sim 5.1 environment for the reBot stack.  Source, don't run:
#
#     source sim/isaac_sim_env.sh
#     $ISAACSIM_ROOT/isaac-sim.sh
#
# WHY THIS FILE EXISTS AT ALL
#
# Isaac Sim 5.1 ships ROS 2 client libraries for BOTH distros inside the bridge
# extension (exts/isaacsim.ros2.bridge/{humble,jazzy}/lib), and picks one at
# extension startup.  The default setting is
#
#     exts."isaacsim.ros2.bridge".ros_distro = "system_default"
#
# and `system_default` is resolved by ros2_common.py:get_ubuntu_version() through
#
#     SUPPORTED_ROS_DISTROS = {"22": "humble", "24": "jazzy"}
#
# keyed on the HOST's /etc/os-release VERSION_ID.  This host is Ubuntu 22.04, so
# autodetection selects **humble** — which cannot talk to the Jazzy container.
# Nothing errors: the bridge starts happily, publishes Humble-typed messages, and
# the container simply never receives anything.  Exactly the silent failure the
# connectivity gate exists to catch.
#
# extension.py:52 reads os.environ["ROS_DISTRO"] FIRST and only falls back to the
# setting when it is unset, so exporting ROS_DISTRO=jazzy overrides the
# autodetection.  That is the whole purpose of this script.
#
# WHY THIS IS SAFE ON A JAMMY HOST
#
# The bundled Jazzy libraries are built for Noble, but verified against this host:
#   max GLIBC symbol required   2.34   <= host 2.35
#   max GLIBCXX required      3.4.30   <= host 3.4.30
#   max CXXABI required       1.3.13   <= host
# and `isaacsim.ros2.bridge.check <jazzy/lib/>` exits 0 with both
# rmw_fastrtps_cpp and rmw_cyclonedds_cpp.  No host Jazzy install is needed or
# wanted; the host keeps its /opt/ros/humble untouched and unused.
#
# We deliberately do NOT source /opt/ros/humble/setup.bash here.  Doing so would
# put Humble's rclpy/rmw on the path ahead of the internal Jazzy libs and
# reintroduce the mismatch this file prevents.
#
# DDS AGREEMENT WITH THE CONTAINER
#
# CycloneDDS, because docker/run.sh sets the same on the container side.  The
# versions match exactly (bundled libddsc 0.10.5 vs container ros-jazzy-cyclonedds
# 0.10.5), so the wire protocol is identical.  ROS_DOMAIN_ID must also match or
# discovery silently fails — it is the single most common cause of "the gate sees
# no /clock".

ISAACSIM_ROOT="${ISAACSIM_ROOT:-$HOME/isaacsim-5.1}"

if [ ! -d "${ISAACSIM_ROOT}" ]; then
  echo "ERROR: Isaac Sim 5.1 not found at ${ISAACSIM_ROOT}" >&2
  echo "       Set ISAACSIM_ROOT, or keep 4.5 at ~/isaacsim and 5.1 here." >&2
  return 1 2>/dev/null || exit 1
fi

_BRIDGE_EXT="${ISAACSIM_ROOT}/exts/isaacsim.ros2.bridge"
_BRIDGE_LIB="${_BRIDGE_EXT}/jazzy/lib"

if [ ! -d "${_BRIDGE_LIB}" ]; then
  echo "ERROR: no bundled Jazzy libraries at ${_BRIDGE_LIB}" >&2
  echo "       This is Isaac Sim 4.5 (Humble-only), not 5.1." >&2
  return 1 2>/dev/null || exit 1
fi

export ISAACSIM_ROOT
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
export LD_LIBRARY_PATH="${_BRIDGE_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# CUDA MPS BYPASS -- required on this host, not an optimisation.
#
# A root-owned `nvidia-cuda-mps-control -d` daemon runs here, so every CUDA
# client is routed through nvidia-cuda-mps-server by default.  Kit then hangs
# forever during device init: it gets as far as
#
#     [ext: carb.windowing.plugins-1.0.0] startup
#     [854ms] [Info] [gpu.foundation.plugin]        <-- last line, ever
#
# and sits there with no banner, no error and no timeout, burning ~2 s of CPU
# per minute.  It looks exactly like a slow first-run shader compile.  The tell
# is /tmp/nvidia-mps/log: a FIFO whose mtime updates on each launch attempt and
# which has no reader.
#
# Pointing CUDA_MPS_PIPE_DIRECTORY at a path with no MPS control pipe makes the
# CUDA runtime fall back to talking to the driver directly.  With this set, the
# same launch reaches "Simulation App Startup Complete" in ~32 s.
#
# This is per-process and read-only with respect to MPS: it does NOT stop the
# daemon or disturb the other tenants on this GPU (there is a ~25 GB vLLM engine
# resident).  Do not "fix" this by killing the MPS daemon.
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/isaacsim-no-mps}"

# Isaac Sim must be the sole /clock owner; container nodes consume it via
# use_sim_time per launch profile.  Nothing to set here — recorded so the
# invariant is visible where the sim side is configured.

cat <<EOF
Isaac Sim 5.1 environment (host side)
  ISAACSIM_ROOT      ${ISAACSIM_ROOT}
  ROS_DISTRO         ${ROS_DISTRO}      (forced; autodetect would pick humble on 22.04)
  RMW_IMPLEMENTATION ${RMW_IMPLEMENTATION}
  ROS_DOMAIN_ID      ${ROS_DOMAIN_ID}
  bridge libs        ${_BRIDGE_LIB}
  MPS pipe dir       ${CUDA_MPS_PIPE_DIRECTORY}   (bypassed; MPS blocks Kit device init)

Launch:  \$ISAACSIM_ROOT/isaac-sim.sh
Gate:    ./docker/run.sh sim python3 docker/connectivity_gate.py
EOF

unset _BRIDGE_EXT _BRIDGE_LIB
