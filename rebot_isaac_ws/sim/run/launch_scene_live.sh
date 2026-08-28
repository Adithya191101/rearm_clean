#!/usr/bin/env bash
# Relaunch the Isaac Sim pick scene (host side) for a LONG run so the scene
# cameras keep publishing while we drive/verify the live perception loop.
#
# The scene script exits when --duration elapses (that is why the sim went down:
# the previous run hit its finite duration and cleanly shut the Simulation App).
# Give it a generous window here. The workflow container (Grounding DINO +
# FoundationPose + BT) reconnects over DDS automatically once cameras are back.
#
# The scene has one canonical configuration. scene_cam_0 remains static and sees
# the can regardless of arm motion.
# See sim/isaac_sim_env.sh for the DDS / MPS / ROS_DISTRO recipe.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
REPO="$(cd -- "$WS/.." && pwd)"
cd "$WS"
# shellcheck disable=SC1091
source sim/isaac_sim_env.sh
DURATION="${1:-900}"
if [[ "$#" -gt 0 ]]; then
  shift
fi
echo "launching pick_scene.py --duration $DURATION (headless)"
exec "$REPO/launch_isaacsim.sh" --python "$WS/sim/pick_scene.py" \
  --duration "$DURATION" "$@"
