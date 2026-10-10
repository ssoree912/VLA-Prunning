#!/bin/bash
# Plan 5.1: 2 closed-loop episodes of one task per method (run after oracle_scripts/smoke_test.py passes).
# Usage (from src/openvla): bash oracle_scripts/run_smoke_episodes.sh [gpu]
cd "$(dirname "$0")/.." && source oracle_scripts/common.sh
GPU=${1:-0}
for m in full pruner oracle; do
  run_method $GPU $m smoke --task_ids 0 --num_trials_per_task 2 --init_state_offset 0
done
