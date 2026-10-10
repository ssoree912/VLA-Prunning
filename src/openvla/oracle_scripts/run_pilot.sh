#!/bin/bash
# Plan 5.2 pilot: LIBERO-Long, rho=0.125, init states 0-9 of every task (100 episodes/method), one method per GPU.
# Usage (from src/openvla): bash oracle_scripts/run_pilot.sh [gpu_full gpu_pruner gpu_oracle]
cd "$(dirname "$0")/.." && source oracle_scripts/common.sh
G_FULL=${1:-0}; G_PRUNER=${2:-1}; G_ORACLE=${3:-2}
mkdir -p $LOG_DIR
run_method $G_FULL   full   pilot --num_trials_per_task 10 --init_state_offset 0 --save_video False > $LOG_DIR/pilot_full.out 2>&1 &
run_method $G_PRUNER pruner pilot --num_trials_per_task 10 --init_state_offset 0 --save_video False > $LOG_DIR/pilot_pruner.out 2>&1 &
run_method $G_ORACLE oracle pilot --num_trials_per_task 10 --init_state_offset 0 --save_video False > $LOG_DIR/pilot_oracle.out 2>&1 &
wait
$PY oracle_scripts/analyze.py --log_dir $LOG_DIR --tag pilot
