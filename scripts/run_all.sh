#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"

case "$MODE" in
  quick)
    exec make quick
    ;;
  all)
    exec make all
    ;;
  cf-10m-resume)
    exec make cf-10m-resume
    ;;
  *)
    echo "Usage: scripts/run_all.sh [quick|all|cf-10m-resume]" >&2
    exit 1
    ;;
esac
