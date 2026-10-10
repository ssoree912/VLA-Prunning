#!/bin/bash
# Plan 5.2 confirmation: init states disjoint from the pilot. LIBERO ships only 50 init states per task, so after the
# pilot's 0-9 there are 40 left (10-49) -> 400 episodes/method instead of the planned 500.
# Usage (from src/openvla): bash oracle_scripts/run_confirm.sh [gpu_full gpu_pruner gpu_oracle]
cd "$(dirname "$0")/.." && source oracle_scripts/common.sh
G_FULL=${1:-0}; G_PRUNER=${2:-1}; G_ORACLE=${3:-2}
mkdir -p $LOG_DIR
run_method $G_FULL   full   confirm --num_trials_per_task 40 --init_state_offset 10 --save_video False > $LOG_DIR/confirm_full.out 2>&1 &
run_method $G_PRUNER pruner confirm --num_trials_per_task 40 --init_state_offset 10 --save_video False > $LOG_DIR/confirm_pruner.out 2>&1 &
run_method $G_ORACLE oracle confirm --num_trials_per_task 40 --init_state_offset 10 --save_video False > $LOG_DIR/confirm_oracle.out 2>&1 &
wait
$PY oracle_scripts/analyze.py --log_dir $LOG_DIR --tag confirm
