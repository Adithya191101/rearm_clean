#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPRODUCE="$REPO_ROOT/rebot_isaac_ws/docker/reproduce.sh"
ACCEPT_EULA="${ISAAC_ROS_ACCEPT_EULA:-0}"
DURATION=900
STARTUP_TIMEOUT=360
GUI=0
CAPTURE_OUTPUT=""
CAPTURE_FPS=10
CAPTURE_DURATION=300
CAPTURE_SETTLE_SECONDS=8
SCENE_RENDER_FPS=60
GOAL_ARGS=()
RUN_DIR=""
SCENE_PID=""
WORKFLOW_PID=""
CAPTURE_PID=""
CAPTURE_STOP_FILE=""

usage() {
  cat <<'EOF'
usage: run_demo.sh --accept-eula [OPTIONS] [-- GOAL_OPTIONS]

Runs one managed Isaac Sim + Isaac ROS pick-and-place demonstration and cleans
up both processes afterward.

Options:
  --accept-eula          Acknowledge the applicable NVIDIA runtime/model terms
  --duration SECONDS     Isaac Sim lifetime (default: 900)
  --startup-timeout SEC  Wait for live perception (default: 360)
  --gui                  Show the Isaac Sim viewport
  --capture FILE         Record and encode the synchronized 1920x1080 flow
  --capture-fps FPS      Capture/encode frame rate (default: 10)
  --capture-duration SEC Recorder timeout (default: 300)
  --capture-settle SEC   Seconds recorded after the goal ends (default: 8)
  -h, --help             Show this help

Arguments after -- are passed to send_pick_goal.py. For example:
  -- --drop-x 0.35 --drop-y 0.25 --require-complete
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --accept-eula) ACCEPT_EULA=1; shift ;;
    --duration)
      [[ "$#" -ge 2 ]] || { echo "ERROR: --duration needs a value" >&2; exit 2; }
      DURATION="$2"; shift 2
      ;;
    --startup-timeout)
      [[ "$#" -ge 2 ]] || {
        echo "ERROR: --startup-timeout needs a value" >&2
        exit 2
      }
      STARTUP_TIMEOUT="$2"; shift 2
      ;;
    --capture)
      [[ "$#" -ge 2 ]] || { echo "ERROR: --capture needs a path" >&2; exit 2; }
      CAPTURE_OUTPUT="$2"; shift 2
      ;;
    --capture-fps)
      [[ "$#" -ge 2 ]] || {
        echo "ERROR: --capture-fps needs a value" >&2
        exit 2
      }
      CAPTURE_FPS="$2"; shift 2
      ;;
    --capture-duration)
      [[ "$#" -ge 2 ]] || {
        echo "ERROR: --capture-duration needs a value" >&2
        exit 2
      }
      CAPTURE_DURATION="$2"; shift 2
      ;;
    --capture-settle)
      [[ "$#" -ge 2 ]] || {
        echo "ERROR: --capture-settle needs a value" >&2
        exit 2
      }
      CAPTURE_SETTLE_SECONDS="$2"; shift 2
      ;;
    --gui) GUI=1; shift ;;
    --) shift; GOAL_ARGS=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$DURATION" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: --duration must be a positive integer" >&2
  exit 2
}
[[ "$STARTUP_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: --startup-timeout must be a positive integer" >&2
  exit 2
}
[[ "$CAPTURE_FPS" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: --capture-fps must be a positive integer" >&2
  exit 2
}
[[ "$CAPTURE_DURATION" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: --capture-duration must be a positive integer" >&2
  exit 2
}
[[ "$CAPTURE_SETTLE_SECONDS" =~ ^[0-9]+$ ]] || {
  echo "ERROR: --capture-settle must be a non-negative integer" >&2
  exit 2
}
if [[ "$ACCEPT_EULA" != "1" ]]; then
  echo "ERROR: explicit NVIDIA EULA acceptance is required." >&2
  echo "Pass --accept-eula after reviewing THIRD_PARTY_LICENSES.md." >&2
  exit 2
fi
if [[ -n "$CAPTURE_OUTPUT" ]] && ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg is required with --capture" >&2
  exit 1
fi

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$CAPTURE_PID" ]] && kill -0 "$CAPTURE_PID" 2>/dev/null; then
    [[ -n "$CAPTURE_STOP_FILE" ]] && touch "$CAPTURE_STOP_FILE"
    kill -TERM "$CAPTURE_PID" 2>/dev/null || true
    wait "$CAPTURE_PID" 2>/dev/null || true
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null |
     grep -qx 'rebot-isaac-sim-wf'; then
    docker stop --timeout 20 rebot-isaac-sim-wf >/dev/null 2>&1 || true
  fi
  for pid in "$WORKFLOW_PID" "$SCENE_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  if [[ -n "$RUN_DIR" ]]; then
    printf 'run logs: %s\n' "$RUN_DIR"
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

"$REPO_ROOT/scripts/preflight.sh"
"$REPO_ROOT/rebot_isaac_ws/docker/verify_models.sh"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$REPO_ROOT/.rebot/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
SCENE_LOG="$RUN_DIR/scene.log"
WORKFLOW_LOG="$RUN_DIR/workflow.log"
GOAL_LOG="$RUN_DIR/goal.log"
CONNECTIVITY_LOG="$RUN_DIR/connectivity.log"
CAPTURE_LOG="$RUN_DIR/capture.log"

scene_args=("$DURATION")
[[ "$GUI" -eq 1 ]] && scene_args+=(--gui)
if [[ -n "$CAPTURE_OUTPUT" ]]; then
  scene_record_every=$((SCENE_RENDER_FPS / CAPTURE_FPS))
  if (( scene_record_every < 1 )); then
    scene_record_every=1
  fi
  CAPTURE_ROOT="$REPO_ROOT/rebot_isaac_ws/tmp/captures/$RUN_ID"
  CONTAINER_CAPTURE_ROOT="/workspaces/rebot_isaac_ws/tmp/captures/$RUN_ID"
  CAPTURE_STOP_FILE="$CAPTURE_ROOT/stop_capture"
  mkdir -p \
    "$CAPTURE_ROOT/sim_frames" \
    "$CAPTURE_ROOT/wide_frames" \
    "$CAPTURE_ROOT/presentation"
  scene_args+=(
    --record
    --record-dir "$CAPTURE_ROOT/sim_frames"
    --record-wide-dir "$CAPTURE_ROOT/wide_frames"
    --record-every "$scene_record_every"
    --record-state-file "$CAPTURE_ROOT/sim_state.json"
  )
fi
"$REPRODUCE" scene "${scene_args[@]}" >"$SCENE_LOG" 2>&1 &
SCENE_PID=$!

deadline=$((SECONDS + STARTUP_TIMEOUT))
until grep -q 'LOOP: entering main loop' "$SCENE_LOG" 2>/dev/null; do
  kill -0 "$SCENE_PID" 2>/dev/null || {
    echo "ERROR: Isaac Sim exited during startup; see $SCENE_LOG" >&2
    exit 1
  }
  (( SECONDS < deadline )) || {
    echo "ERROR: timed out waiting for the Isaac Sim scene; see $SCENE_LOG" >&2
    exit 1
  }
  sleep 2
done
echo "Isaac Sim scene is publishing."

HF_CACHE_HOST="$REPO_ROOT/repro_cache/huggingface" \
  "$REPRODUCE" workflow >"$WORKFLOW_LOG" 2>&1 &
WORKFLOW_PID=$!

deadline=$((SECONDS + STARTUP_TIMEOUT))
until docker exec rebot-isaac-sim-wf bash -lc \
  'source /workspaces/rebot_isaac_ws/install/setup.bash &&
   actions="$(ros2 action list 2>/dev/null)" &&
   grep -qx /multi_object_pick_and_place <<<"$actions" &&
   grep -qx /detect_objects <<<"$actions" &&
   grep -qx /estimate_pose_foundation_pose <<<"$actions"' \
  >/dev/null 2>&1; do
  kill -0 "$WORKFLOW_PID" 2>/dev/null || {
    echo "ERROR: Isaac ROS workflow exited during startup; see $WORKFLOW_LOG" >&2
    exit 1
  }
  (( SECONDS < deadline )) || {
    echo "ERROR: timed out waiting for workflow action servers; see $WORKFLOW_LOG" >&2
    exit 1
  }
  sleep 2
done

if ! docker exec rebot-isaac-sim-wf bash -lc \
  'source /workspaces/rebot_isaac_ws/install/setup.bash &&
   python3 /workspaces/rebot_isaac_ws/docker/connectivity_gate.py \
     --camera-topic /scene_cam_0/rgb/image_raw \
     --require-camera' >"$CONNECTIVITY_LOG" 2>&1; then
  echo "ERROR: live scene/workflow connectivity failed; see $CONNECTIVITY_LOG" >&2
  exit 1
fi
echo "Workflow/perception action servers and live scene camera are ready."

if [[ -n "$CAPTURE_OUTPUT" ]]; then
  docker exec \
    -e "REBOT_CAPTURE_ROOT=$CONTAINER_CAPTURE_ROOT" \
    -e "REBOT_CAPTURE_FPS=$CAPTURE_FPS" \
    -e "REBOT_CAPTURE_DURATION=$CAPTURE_DURATION" \
    rebot-isaac-sim-wf bash -lc '
      source /workspaces/rebot_isaac_ws/install/setup.bash
      exec python3 \
        /workspaces/rebot_isaac_ws/sim/capture_pipeline_diagnostics.py \
        --output-dir "$REBOT_CAPTURE_ROOT/presentation" \
        --sim-frame-dir "$REBOT_CAPTURE_ROOT/sim_frames" \
        --wide-sim-frame-dir "$REBOT_CAPTURE_ROOT/wide_frames" \
        --sim-state-file "$REBOT_CAPTURE_ROOT/sim_state.json" \
        --stop-file "$REBOT_CAPTURE_ROOT/stop_capture" \
        --duration "$REBOT_CAPTURE_DURATION" \
        --fps "$REBOT_CAPTURE_FPS"
    ' >"$CAPTURE_LOG" 2>&1 &
  CAPTURE_PID=$!

  deadline=$((SECONDS + STARTUP_TIMEOUT))
  until [[ -f "$CAPTURE_ROOT/presentation/capture_ready.json" ]]; do
    kill -0 "$CAPTURE_PID" 2>/dev/null || {
      echo "ERROR: diagnostic capture exited; see $CAPTURE_LOG" >&2
      exit 1
    }
    (( SECONDS < deadline )) || {
      echo "ERROR: timed out waiting for diagnostic capture" >&2
      exit 1
    }
    sleep 1
  done
  echo "Synchronized diagnostic capture is ready."
fi

set +e
"$REPRODUCE" goal "${GOAL_ARGS[@]}" 2>&1 | tee "$GOAL_LOG"
goal_status=${PIPESTATUS[0]}
set -e

capture_status=0
if [[ -n "$CAPTURE_OUTPUT" ]]; then
  sleep "$CAPTURE_SETTLE_SECONDS"
  touch "$CAPTURE_STOP_FILE"
  set +e
  wait "$CAPTURE_PID"
  capture_status=$?
  set -e
  CAPTURE_PID=""

  frame_pattern="$CAPTURE_ROOT/presentation/frames/frame_%05d.jpg"
  if [[ ! -f "$CAPTURE_ROOT/presentation/frames/frame_00000.jpg" ]]; then
    echo "ERROR: capture produced no frames; see $CAPTURE_LOG" >&2
    capture_status=1
  else
    if [[ "$CAPTURE_OUTPUT" != /* ]]; then
      CAPTURE_OUTPUT="$PWD/$CAPTURE_OUTPUT"
    fi
    mkdir -p "$(dirname -- "$CAPTURE_OUTPUT")"
    ffmpeg -y -loglevel warning \
      -framerate "$CAPTURE_FPS" \
      -start_number 0 \
      -i "$frame_pattern" \
      -c:v libx264 \
      -crf 18 \
      -pix_fmt yuv420p \
      -movflags +faststart \
      "$CAPTURE_OUTPUT"
    printf 'capture video: %s\n' "$CAPTURE_OUTPUT"
  fi
fi

if [[ "$goal_status" -ne 0 ]]; then
  exit "$goal_status"
fi
exit "$capture_status"
