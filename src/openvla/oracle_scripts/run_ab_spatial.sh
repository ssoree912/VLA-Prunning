#!/bin/bash
# Oracle A and Oracle B on LIBERO-Spatial (init states 0-9, 100 episodes each), both on one GPU, to compare with
# the VLA-Pruner keep-12.5% run (same seed / budget). Usage (from src/openvla): bash oracle_scripts/run_ab_spatial.sh [gpu]
cd "$(dirname "$0")/.." 
export CKPT=checkpoints/openvla-7b-finetuned-libero-spatial SUITE=libero_spatial
source oracle_scripts/common.sh
GPU=${1:-3}; TAG=spatial10
mkdir -p $LOG_DIR
for m in oracle_a oracle_b; do
  ( echo "START $m $(date)"
    run_method $GPU $m $TAG --num_trials_per_task 10 --init_state_offset 0 --save_video False > $LOG_DIR/${TAG}_$m.out 2>&1
    echo "END $m exit=$? $(grep '# successes' $LOG_DIR/${TAG}_$m.out | tail -1) $(date)" ) >> $LOG_DIR/${TAG}_status.txt &
done
wait
echo "ALL DONE $(date)" >> $LOG_DIR/${TAG}_status.txt
