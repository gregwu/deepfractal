#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/greg/micromamba/envs/rapids-fil/bin/python3
SCRIPT="${1:-demo.py}"

exec "$PYTHON" "$SCRIPT" "${@:2}"




# Run demo.py (default)
#./run.sh

# Run a different script
#./run.sh benchmark.py

# Pass arguments through
#./run.sh demo.py --some-arg value