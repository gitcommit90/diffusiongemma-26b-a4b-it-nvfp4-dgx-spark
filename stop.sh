#!/usr/bin/env bash
set -euo pipefail
NAME=${NAME:-diffusiongemma-26b-nvfp4}
docker rm -f "$NAME" 2>/dev/null || true
echo "stopped $NAME"
