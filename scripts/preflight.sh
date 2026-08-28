#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${REBOT_IMAGE:-rebot-isaac:4.5}"
ISAACSIM_ROOT="${ISAACSIM_ROOT:-$HOME/isaacsim-5.1}"
STRICT=0
CHECK_MODELS=1
CHECK_ISAAC=1
FAILURES=0
WARNINGS=0

usage() {
  cat <<'EOF'
usage: preflight.sh [--strict] [--skip-models] [--skip-isaac-sim]

Checks the host prerequisites without downloading or changing anything.
Warnings become failures with --strict.
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --strict) STRICT=1 ;;
    --skip-models) CHECK_MODELS=0 ;;
    --skip-isaac-sim) CHECK_ISAAC=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

pass() {
  printf 'PASS  %s\n' "$*"
}

warn() {
  printf 'WARN  %s\n' "$*" >&2
  WARNINGS=$((WARNINGS + 1))
}

fail() {
  printf 'FAIL  %s\n' "$*" >&2
  FAILURES=$((FAILURES + 1))
}

require_command() {
  local command_name="$1"
  if command -v "$command_name" >/dev/null 2>&1; then
    pass "$command_name is available"
  else
    fail "missing required command: $command_name"
  fi
}

printf 'reArm Isaac ROS preflight\n'
printf 'repository: %s\n\n' "$REPO_ROOT"

for command_name in bash git python3 docker nvidia-smi sha256sum; do
  require_command "$command_name"
done

if command -v ffmpeg >/dev/null 2>&1; then
  pass "ffmpeg is available for diagnostic video encoding"
else
  warn "ffmpeg is missing (required only for --capture video output)"
fi

if python3 -m venv --help >/dev/null 2>&1; then
  pass "Python venv support is available"
else
  fail "python3-venv is required for the local test environment"
fi

if [[ "$(uname -m)" == "x86_64" ]]; then
  pass "host architecture is x86_64"
else
  fail "validated host architecture is x86_64; found $(uname -m)"
fi

if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  if [[ "${VERSION_ID:-}" == "22.04" ]]; then
    pass "host OS is Ubuntu 22.04"
  else
    warn "validated host OS is Ubuntu 22.04; found ${PRETTY_NAME:-unknown}"
  fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  if gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" &&
     [[ -n "$gpu_name" ]]; then
    pass "NVIDIA GPU is visible: $gpu_name"
  else
    fail "nvidia-smi could not query an NVIDIA GPU"
  fi
fi

if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    pass "Docker daemon is reachable"
    if docker image inspect "$IMAGE" >/dev/null 2>&1; then
      pass "runtime image is present: $IMAGE"
    else
      warn "runtime image is absent: $IMAGE (bootstrap will build it)"
    fi
  else
    fail "Docker daemon is not reachable by the current user"
  fi
fi

if [[ "$CHECK_ISAAC" -eq 1 ]]; then
  if [[ -x "$ISAACSIM_ROOT/isaac-sim.sh" &&
        -x "$ISAACSIM_ROOT/python.sh" ]]; then
    pass "Isaac Sim launchers found under $ISAACSIM_ROOT"
  else
    fail "Isaac Sim 5.1 launchers not found under $ISAACSIM_ROOT"
  fi
fi

available_kib="$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print $4}')"
if [[ "$available_kib" =~ ^[0-9]+$ ]]; then
  available_gib=$((available_kib / 1024 / 1024))
  if (( available_gib >= 40 )); then
    pass "available disk space is ${available_gib} GiB"
  else
    warn "less than 40 GiB is free (${available_gib} GiB)"
  fi
fi

if [[ -r /proc/meminfo ]]; then
  memory_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
  memory_gib=$((memory_kib / 1024 / 1024))
  if (( memory_gib >= 32 )); then
    pass "host memory is ${memory_gib} GiB"
  else
    warn "validated system has at least 32 GiB RAM; found ${memory_gib} GiB"
  fi
fi

if [[ "$CHECK_MODELS" -eq 1 ]]; then
  if "$REPO_ROOT/rebot_isaac_ws/docker/verify_models.sh" >/dev/null 2>&1; then
    pass "portable model assets and GPU engines are present"
  else
    warn "model bundle is incomplete (bootstrap can install it)"
  fi
fi

if "$REPO_ROOT/scripts/fetch_sources.sh" --check >/dev/null 2>&1; then
  pass "pinned source checkouts and generated vendor assets are ready"
else
  warn "source tree is not prepared (run ./scripts/fetch_sources.sh)"
fi

printf '\nsummary: %d failure(s), %d warning(s)\n' "$FAILURES" "$WARNINGS"
if (( FAILURES > 0 || (STRICT == 1 && WARNINGS > 0) )); then
  exit 1
fi
