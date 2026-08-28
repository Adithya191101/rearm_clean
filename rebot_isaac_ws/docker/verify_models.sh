#!/usr/bin/env bash
set -euo pipefail

WS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "$WS/.." && pwd)"
CHECKSUMS="$REPO_ROOT/MODEL_SHA256SUMS"
ENGINES="$REPO_ROOT/MODEL_ENGINE_PATHS"

[[ -f "$CHECKSUMS" ]] || {
  echo "ERROR: missing model checksum manifest: $CHECKSUMS" >&2
  exit 1
}
[[ -f "$ENGINES" ]] || {
  echo "ERROR: missing generated-engine manifest: $ENGINES" >&2
  exit 1
}
[[ -d "$WS/isaac_ros_assets" ]] || {
  echo "ERROR: missing model directory: $WS/isaac_ros_assets" >&2
  exit 1
}

cd "$REPO_ROOT"
sha256sum --check "$CHECKSUMS"

while IFS= read -r engine; do
  [[ -z "$engine" || "$engine" == \#* ]] && continue
  [[ -s "$engine" ]] || {
    echo "ERROR: missing or empty GPU-generated TensorRT engine: $engine" >&2
    echo "Run ./scripts/install_models.sh --accept-eula on this GPU." >&2
    exit 1
  }
  echo "$engine: present (GPU-generated)"
done < "$ENGINES"
