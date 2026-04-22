#!/usr/bin/env bash
# Run 3 users through each model using the 10M dataset.
# Usage: ./demo_10m.sh
# Must be run from the repo root. Requires CODE/weights_10m/ to be populated (run CODE/train_10m.py first).

set -e
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
CLI="CODE/src/cli.py"
USERS=(42 150 1054)
TOPN=5

run_user() {
    local model=$1
    local user=$2
    echo "----------------------------------------"
    echo "  Model: ${model^^}  |  User: $user  |  Top $TOPN"
    echo "----------------------------------------"
    $PYTHON $CLI --user "$user" --model "$model" --topn "$TOPN" --dataset 10m
    echo
}

for user in "${USERS[@]}"; do
    echo "========================================"
    echo "  USER $user"
    echo "========================================"
    echo

    run_user als   "$user"
    run_user knn   "$user"
    run_user neumf "$user"
done
