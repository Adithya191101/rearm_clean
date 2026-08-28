#!/usr/bin/env bash
# Single entry point for preparing and running the validated live workflow.
set -euo pipefail

WS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "$WS/.." && pwd)"
COMMAND="${1:-prepare}"
shift || true
PYTHON="${REBOT_PYTHON:-python3}"

case "$COMMAND" in
  prepare)
    "$WS/docker/verify_models.sh"
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
      "$PYTHON" -m pytest -q "$WS/test"
    "$WS/docker/run.sh" sim ./docker/build_overlay.sh
    "$WS/docker/run.sh" sim ./docker/verify_overlay.sh
    ;;
  preflight)
    exec "$REPO_ROOT/scripts/preflight.sh" "$@"
    ;;
  install-models)
    exec "$REPO_ROOT/scripts/install_models.sh" "$@"
    ;;
  bootstrap)
    exec "$REPO_ROOT/scripts/bootstrap.sh" "$@"
    ;;
  demo)
    exec "$REPO_ROOT/scripts/run_demo.sh" "$@"
    ;;
  verify-models)
    exec "$WS/docker/verify_models.sh" "$@"
    ;;
  build-overlay)
    exec "$WS/docker/run.sh" sim ./docker/build_overlay.sh "$@"
    ;;
  verify-overlay)
    exec "$WS/docker/run.sh" sim ./docker/verify_overlay.sh "$@"
    ;;
  scene)
    cd "$REPO_ROOT"
    exec "$WS/sim/run/launch_scene_live.sh" "$@"
    ;;
  workflow)
    cd "$REPO_ROOT"
    exec "$WS/sim/run/launch_workflow_live.sh" "$@"
    ;;
  goal)
    exec docker exec rebot-isaac-sim-wf bash -lc \
      'source /workspaces/rebot_isaac_ws/install/setup.bash &&
       exec python3 /workspaces/rebot_isaac_ws/sim/send_pick_goal.py "$@"' \
      rebot-pick-goal "$@"
    ;;
  help|-h|--help)
    cat <<'EOF'
usage: reproduce.sh COMMAND [ARGS...]

Developer setup:
  preflight       Validate the host, GPU, Docker, Isaac Sim, and local assets
  install-models  Download models and build GPU-specific TensorRT engines
  bootstrap       Build the image, install models, test, and build the overlay

Runtime:
  demo            Run one managed end-to-end demonstration
  scene           Start the host Isaac Sim scene
  workflow        Start the Docker Isaac ROS workflow
  goal            Send one goal; additional arguments go to send_pick_goal.py

Validation:
  prepare         Verify models, run host tests, build and verify the overlay
  verify-models   Verify portable assets and generated TensorRT engines
  build-overlay   Build the selected ROS overlay packages
  verify-overlay  Confirm patched packages resolve from this workspace
EOF
    ;;
  *)
    echo "ERROR: unknown command: $COMMAND" >&2
    echo "Run '$0 help' for available commands." >&2
    exit 2
    ;;
esac
