#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="$REPO_ROOT/dependencies.lock"
CHECK_ONLY=0

usage() {
  cat <<'EOF'
usage: fetch_sources.sh [--check]

Fetches immutable upstream revisions, applies the project-owned Isaac ROS
patch, and materializes Seeed visual/USD assets needed by the simulation.

  --check  Verify an existing source tree without network access or changes.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --check) CHECK_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

validate_destination() {
  local destination="$1"
  [[ -n "$destination" && "$destination" != /* ]] ||
    fail "dependency destination must be repository-relative: $destination"
  case "/$destination/" in
    */../*|*/./*) fail "unsafe dependency destination: $destination" ;;
  esac
}

checkout_dependency() {
  local name="$1"
  local url="$2"
  local revision="$3"
  local destination="$4"
  local sparse_path="$5"
  local checkout="$REPO_ROOT/$destination"
  local current_url=""
  local current_revision=""

  validate_destination "$destination"
  [[ "$revision" =~ ^[0-9a-f]{40}$ ]] ||
    fail "$name revision is not a full Git SHA: $revision"

  if [[ ! -d "$checkout/.git" ]]; then
    [[ "$CHECK_ONLY" -eq 0 ]] ||
      fail "$name is missing at $checkout; run scripts/fetch_sources.sh"
    [[ ! -e "$checkout" || -d "$checkout" ]] ||
      fail "$checkout exists but is not a directory"
    mkdir -p "$checkout"
    git -C "$checkout" init -q
    git -C "$checkout" remote add origin "$url"
  fi

  current_url="$(git -C "$checkout" remote get-url origin 2>/dev/null || true)"
  [[ "$current_url" == "$url" ]] ||
    fail "$name remote is $current_url, expected $url"

  if [[ "$CHECK_ONLY" -eq 0 ]] &&
     ! git -C "$checkout" cat-file -e "${revision}^{commit}" 2>/dev/null; then
    git -C "$checkout" fetch --filter=blob:none --depth=1 origin "$revision"
  fi

  if [[ -n "$sparse_path" && "$CHECK_ONLY" -eq 0 ]]; then
    # Some managed Git templates omit .git/info in a new, unborn repository.
    # sparse-checkout expects that directory to exist before first checkout.
    mkdir -p "$checkout/.git/info"
    git -C "$checkout" sparse-checkout init --cone
    git -C "$checkout" sparse-checkout set "$sparse_path"
  fi

  current_revision="$(git -C "$checkout" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$current_revision" != "$revision" ]]; then
    [[ "$CHECK_ONLY" -eq 0 ]] ||
      fail "$name is at ${current_revision:-no revision}, expected $revision"
    [[ -z "$(git -C "$checkout" status --porcelain)" ]] ||
      fail "$name has local changes and cannot move to $revision"
    git -C "$checkout" checkout -q --detach "$revision"
  fi

  printf 'source %-27s %s\n' "$name" "$revision"
}

while IFS='|' read -r name url revision destination sparse_path; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  [[ -n "$url" && -n "$revision" && -n "$destination" ]] ||
    fail "malformed dependency row for $name"
  checkout_dependency "$name" "$url" "$revision" "$destination" "$sparse_path"
done < "$LOCK_FILE"

ISAAC_SOURCE="$REPO_ROOT/rebot_isaac_ws/src/isaac_ros_manipulation"
ISAAC_PATCH="$REPO_ROOT/patches/isaac_ros_manipulation.patch"
if git -C "$ISAAC_SOURCE" apply --reverse --check "$ISAAC_PATCH" \
  >/dev/null 2>&1; then
  echo "patch  isaac_ros_manipulation already applied"
elif [[ "$CHECK_ONLY" -eq 1 ]]; then
  fail "Isaac ROS patch is missing or does not match the locked source"
elif git -C "$ISAAC_SOURCE" apply --check "$ISAAC_PATCH"; then
  git -C "$ISAAC_SOURCE" apply "$ISAAC_PATCH"
  echo "patch  isaac_ros_manipulation applied"
else
  fail "Isaac ROS patch does not apply to the locked source"
fi

VISUAL_SOURCE="$REPO_ROOT/rebot_isaac_ws/.upstream/reBotArmController_ROS2/src/rebotarm_bringup/description/meshes_b601_gripper"
VISUAL_DEST="$REPO_ROOT/rebot_isaac_ws/src/rebot_b601dm_description/meshes/visual"
VISUAL_FILES=(
  base_link.STL
  gripper_left.STL
  gripper_link.STL
  gripper_right.STL
  link1.STL
  link2.STL
  link3.STL
  link4.STL
  link5.STL
  link6.STL
)

if [[ "$CHECK_ONLY" -eq 0 ]]; then
  mkdir -p "$VISUAL_DEST"
fi
for mesh in "${VISUAL_FILES[@]}"; do
  [[ -f "$VISUAL_SOURCE/$mesh" ]] ||
    fail "Seeed visual mesh is missing: $VISUAL_SOURCE/$mesh"
  if [[ "$CHECK_ONLY" -eq 0 ]]; then
    install -m 0644 "$VISUAL_SOURCE/$mesh" "$VISUAL_DEST/$mesh"
  fi
  [[ -f "$VISUAL_DEST/$mesh" ]] ||
    fail "materialized visual mesh is missing: $VISUAL_DEST/$mesh"
  cmp -s "$VISUAL_SOURCE/$mesh" "$VISUAL_DEST/$mesh" ||
    fail "materialized visual mesh differs from Seeed source: $mesh"
done
echo "assets Seeed visual meshes materialized (${#VISUAL_FILES[@]} files)"

VENDOR_LINK="$REPO_ROOT/rebot_isaac_ws/usd/vendor/reBot_B601_DM"
VENDOR_TARGET="../../.upstream/reBot-Isaacsim/usd/reBot_B601_DM"
if [[ -L "$VENDOR_LINK" ]]; then
  [[ "$(readlink "$VENDOR_LINK")" == "$VENDOR_TARGET" ]] ||
    fail "$VENDOR_LINK points to an unexpected target"
elif [[ -e "$VENDOR_LINK" ]]; then
  fail "$VENDOR_LINK exists and is not the generated symlink"
elif [[ "$CHECK_ONLY" -eq 1 ]]; then
  fail "vendor USD link is missing: $VENDOR_LINK"
else
  ln -s "$VENDOR_TARGET" "$VENDOR_LINK"
fi

VENDOR_ROOT="$VENDOR_LINK/reBot_B601_DM.usda"
EXPECTED_VENDOR_SHA="6b9d39de1200732c581c91e895bee412844e101006fb0c3df54259d81ee28e84"
[[ -f "$VENDOR_ROOT" ]] || fail "Seeed vendor USD is missing: $VENDOR_ROOT"
ACTUAL_VENDOR_SHA="$(sha256sum "$VENDOR_ROOT" | awk '{print $1}')"
[[ "$ACTUAL_VENDOR_SHA" == "$EXPECTED_VENDOR_SHA" ]] ||
  fail "Seeed vendor USD checksum mismatch: $ACTUAL_VENDOR_SHA"
echo "assets Seeed Isaac Sim USD linked and verified"

echo "Upstream source preparation passed."
