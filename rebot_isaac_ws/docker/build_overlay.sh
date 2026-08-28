#!/usr/bin/env bash
# Build the complete repository overlay inside rebot-isaac:4.5.
set -euo pipefail

WS="${ISAAC_ROS_WS:-/workspaces/rebot_isaac_ws}"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
PACKAGE_FILE="$WS/docker/overlay-packages.txt"
DEPENDENCY_LOCK="$(cd -- "$WS/.." && pwd)/dependencies.lock"

if [[ ! -f "/opt/ros/$ROS_DISTRO/setup.bash" ]]; then
  echo "ERROR: run this script inside the rebot-isaac:4.5 container." >&2
  exit 2
fi
if [[ ! -f "$PACKAGE_FILE" ]]; then
  echo "ERROR: overlay package manifest is missing: $PACKAGE_FILE" >&2
  exit 2
fi
if [[ ! -f "$DEPENDENCY_LOCK" ]]; then
  echo "ERROR: dependency lock is missing: $DEPENDENCY_LOCK" >&2
  exit 2
fi

set +u
source "/opt/ros/$ROS_DISTRO/setup.bash"
set -u
cd "$WS"

packages=()
while IFS= read -r package; do
  [[ -z "$package" || "$package" == \#* ]] && continue
  packages+=("$package")
done < "$PACKAGE_FILE"

colcon build \
  --symlink-install \
  --packages-select "${packages[@]}" \
  --cmake-args -DBUILD_TESTING=OFF

set +u
source "$WS/install/setup.bash"
set -u
manifest="$WS/install/rebot_overlay_manifest.txt"
{
  printf 'source_version=%s\n' "${REBOT_SOURCE_VERSION:-archive}"
  printf 'image=%s\n' "${REBOT_IMAGE:-rebot-isaac:4.5}"
  printf 'base_image_digest=%s\n' \
    'sha256:81c48cacaecbc586e0ad1e9c15f7cf7769a1a146b6a12f3bbaf83f9c5c16e623'
  printf 'dependencies_sha256=%s\n' \
    "$(sha256sum "$DEPENDENCY_LOCK" | cut -d' ' -f1)"
  printf 'dockerfile_sha256=%s\n' \
    "$(sha256sum "$WS/docker/Dockerfile" | cut -d' ' -f1)"
  printf 'package_manifest_sha256=%s\n' \
    "$(sha256sum "$PACKAGE_FILE" | cut -d' ' -f1)"
  for package in "${packages[@]}"; do
    printf 'package.%s=%s\n' "$package" "$(ros2 pkg prefix "$package")"
  done
} > "$manifest"

"$WS/docker/verify_overlay.sh"
