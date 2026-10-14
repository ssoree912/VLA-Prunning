#!/bin/bash
# 10-episode (1 trial x 10 tasks, init 0) FLOPs/latency profile of our predictor with exactly the std500_p20 settings,
# used to reconstruct per-query FLOPs of the earlier unprofiled 500-episode runs (warm-up queries dense, rest constant).
# Usage (from src/openvla): bash oracle_scripts/prof10.sh <gpu>
GPU=$1
cd /workspace/vla/oracle_pruning/VLA-Pruner/src/openvla
D=/workspace/vla/data
while read S T P W SCH; do
  export RHO_R=0.8828125 CKPT=checkpoints/openvla-7b-finetuned-$(echo $S | tr '_' '-') SUITE=$S PRED_CKPT=$P
  source oracle_scripts/common.sh
  Q=$LOG_DIR/std500_p20_status.txt
  echo "START ${T}prof10 predictor gpu$GPU $(date)" >> $Q
  PROF_OUT=/workspace/vla/eval_logs/std500/${T}prof10_predictor.prof.jsonl ORACLE_HEAD_WEIGHT=$W ORACLE_SCHEDULE=$SCH \
    run_method $GPU predictor ${T}prof10 --num_trials_per_task 1 --init_state_offset 0 --save_video False \
    > $LOG_DIR/${T}prof10_predictor.out 2>&1
  echo "END ${T}prof10 predictor exit=$? $(grep '# successes' $LOG_DIR/${T}prof10_predictor.out | tail -1) $(date)" >> $Q
done <<LIST
libero_spatial spatial50p20 $D/oracle_predictor/libero_spatial_m2_tf1_s0/best.pt 0.0 oracle_scripts/schedules/libero_spatial_m2_b1638.json
libero_goal goal50p20 $D/oracle_predictor_mix/libero_goal_m2_mix0.5_s0/A_2head/best.pt 0.4 oracle_scripts/schedules/libero_goal_m2_w0.4_b1638.json
libero_object object50p20 $D/oracle_predictor_mix/libero_object_m2_mix0.5_s0/A_2head/best.pt 0.3 oracle_scripts/schedules/libero_object_m2_w0.3_b1638.json
LIST
