#!/usr/bin/env bash
# Usage: bash scripts/train_scripts/cacom.sh 3m seed=0 t_max=2050000
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
map_name="${1:-3m}"
if [ "$#" -gt 0 ]; then shift; fi
"${PYTHON_BIN:-python}" src/main.py --config=cacom --env-config=sc2 with \
    "env_args.map_name=${map_name}" "$@"
