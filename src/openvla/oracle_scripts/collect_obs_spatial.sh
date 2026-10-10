#!/bin/bash
# Full-policy rollouts on LIBERO-Spatial (init states 0-2) saving every 5th query image for offline K/label diagnosis.
# Usage (from src/openvla): bash oracle_scripts/collect_obs_spatial.sh [gpu_a gpu_b]
cd "$(dirname "$0")/.." && source oracle_scripts/common.sh
CKPT=checkpoints/openvla-7b-finetuned-libero-spatial; SUITE=libero_spatial
GA=${1:-1}; GB=${2:-2}
mkdir -p $LOG_DIR
COMMON="--num_trials_per_task 3 --init_state_offset 0 --save_video False --save_observations True --obs_every 5"
run_method $GA full obs_a $COMMON --task_ids 0,1,2,3,4 > $LOG_DIR/obs_a_full.out 2>&1 &
run_method $GB full obs_b $COMMON --task_ids 5,6,7,8,9 > $LOG_DIR/obs_b_full.out 2>&1 &
wait
