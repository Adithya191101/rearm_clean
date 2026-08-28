#!/usr/bin/env bash
# Prove that every patched runtime package resolves from one workspace overlay.
set -euo pipefail

WS="${ISAAC_ROS_WS:-/workspaces/rebot_isaac_ws}"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
PACKAGE_FILE="$WS/docker/overlay-packages.txt"
OVERLAY_SETUP="$WS/install/setup.bash"
MANIFEST="$WS/install/rebot_overlay_manifest.txt"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -f "$OVERLAY_SETUP" ]] || fail \
  "workspace overlay is missing; run $WS/docker/build_overlay.sh"
[[ -f "$PACKAGE_FILE" ]] || fail "missing $PACKAGE_FILE"
[[ -f "$MANIFEST" ]] || fail \
  "overlay build manifest is missing; rebuild with build_overlay.sh"

set +u
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "$OVERLAY_SETUP"
set -u

packages=()
while IFS= read -r package; do
  [[ -z "$package" || "$package" == \#* ]] && continue
  packages+=("$package")
done < "$PACKAGE_FILE"

for package in "${packages[@]}"; do
  prefix="$(ros2 pkg prefix "$package")" ||
    fail "package is not discoverable: $package"
  [[ "$prefix" == "$WS/install/$package" ]] ||
    fail "$package resolved to $prefix instead of the workspace overlay"
done

bringup_share="$(ros2 pkg prefix --share isaac_ros_manipulation_bringup)"
core_launch="$bringup_share/launch/workflows/core.launch.py"
[[ -f "$core_launch" ]] || fail "installed workflow source is missing: $core_launch"

driver_share="$(
  ros2 pkg prefix --share isaac_ros_manipulation_rebot_driver_utils
)"
installed_profile="$driver_share/params/rebot_sim_launch_params.yaml"
canonical_profile="$WS/src/rebot_b601dm_isaac/isaac_ros_manipulation_rebot_driver_utils/params/rebot_sim_launch_params.yaml"
compatibility_profile="$WS/config/workflows/rebot_sim_launch_params.yaml"
cmp -s "$canonical_profile" "$installed_profile" ||
  fail "installed workflow profile differs from the canonical package profile"
[[ "$(readlink -f "$compatibility_profile")" == \
   "$(readlink -f "$canonical_profile")" ]] ||
  fail "workspace workflow profile is not an alias of the canonical profile"

python3 - "$WS" <<'PY'
from importlib import import_module
from pathlib import Path
import sys

workspace = Path(sys.argv[1]).resolve()
modules = (
    'isaac_ros_manipulation_ros_python_utils.core',
    'isaac_ros_manipulation_ros_python_utils.config',
    'isaac_ros_manipulation_ros_python_utils.constants',
    'isaac_ros_manipulation_orchestration.behaviors.motion_behaviors.read_grasp_poses',
    'isaac_ros_manipulation_orchestration.behaviors.motion_behaviors.close_gripper',
    'isaac_ros_manipulation_orchestration.behaviors.motion_behaviors.attach_object',
)

for name in modules:
    path = Path(import_module(name).__file__).resolve()
    try:
        path.relative_to(workspace)
    except ValueError as exc:
        raise SystemExit(
            f'ERROR: {name} resolved outside the workspace overlay: {path}'
        ) from exc
    print(f'overlay module {name}={path}')
PY

echo "Overlay verification passed (${#packages[@]} packages)."
