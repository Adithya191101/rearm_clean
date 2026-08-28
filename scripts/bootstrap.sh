#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${REBOT_IMAGE:-rebot-isaac:4.5}"
ACCEPT_EULA="${ISAAC_ROS_ACCEPT_EULA:-0}"
REBUILD_IMAGE=0
SKIP_MODELS=0

usage() {
  cat <<'EOF'
usage: bootstrap.sh --accept-eula [--rebuild-image] [--skip-models]

Fetches pinned upstream sources, creates a local test virtualenv, builds the
pinned Docker runtime when needed, installs the model bundle, and runs the
complete preparation gate.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --accept-eula) ACCEPT_EULA=1 ;;
    --rebuild-image) REBUILD_IMAGE=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ "$ACCEPT_EULA" != "1" ]]; then
  echo "ERROR: explicit NVIDIA EULA acceptance is required." >&2
  echo "Review ./scripts/install_models.sh --help, then pass --accept-eula." >&2
  exit 2
fi

"$REPO_ROOT/scripts/fetch_sources.sh"
"$REPO_ROOT/scripts/preflight.sh" --skip-models

if [[ ! -x "$REPO_ROOT/.venv/bin/python" ]]; then
  python3 -m venv "$REPO_ROOT/.venv"
fi
"$REPO_ROOT/.venv/bin/python" -m pip install --upgrade pip
"$REPO_ROOT/.venv/bin/python" -m pip install -r "$REPO_ROOT/requirements-dev.txt"

if [[ "$REBUILD_IMAGE" -eq 1 ]] ||
   ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build \
    -t "$IMAGE" \
    -f "$REPO_ROOT/rebot_isaac_ws/docker/Dockerfile" \
    "$REPO_ROOT/rebot_isaac_ws/docker"
fi

if [[ "$SKIP_MODELS" -eq 0 ]] &&
   ! "$REPO_ROOT/rebot_isaac_ws/docker/verify_models.sh" >/dev/null 2>&1; then
  "$REPO_ROOT/scripts/install_models.sh" --accept-eula
fi

REBOT_PYTHON="$REPO_ROOT/.venv/bin/python" \
  "$REPO_ROOT/rebot_isaac_ws/docker/reproduce.sh" prepare

printf '\nBOOTSTRAP RESULT status=success\n'
