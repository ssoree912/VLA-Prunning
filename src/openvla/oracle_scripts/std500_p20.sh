#!/bin/bash
# Standard 500-episode eval (50 trials x 10 tasks, init 0-49, seed 7) at the 20% token-layer budget (1638 = 19.995%):
# our predictor (w-matched 1638 schedule), VLA-Pruner (keeps 30 tokens from layer 3: fastv_r=0.8828125), full.
# Usage (from src/openvla): bash oracle_scripts/std500_p20.sh <gpu> <suite> <tag> <pred_ckpt> <w> <schedule> [methods...]
GPU=$1; S=$2; T=$3; P=$4; W=$5; SCH=$6; shift 6
METHODS=${@:-predictor pruner full}
cd /workspace/vla/oracle_pruning/VLA-Pruner/src/openvla
export RHO_R=0.8828125 CKPT=checkpoints/openvla-7b-finetuned-$(echo $S | tr '_' '-') SUITE=$S PRED_CKPT=$P
source oracle_scripts/common.sh
Q=$LOG_DIR/std500_p20_status.txt
for M in $METHODS; do
  echo "START $T $M (w=$W, budget 1638) gpu$GPU $(date)" >> $Q
  PROF_OUT=${PROF_DIR:+$PROF_DIR/${T}_$M.prof.jsonl} ORACLE_HEAD_WEIGHT=$W ORACLE_SCHEDULE=$SCH run_method $GPU $M $T --num_trials_per_task 50 --init_state_offset 0 --save_video False > $LOG_DIR/${T}_$M.out 2>&1
  echo "END $T $M exit=$? $(grep '# successes' $LOG_DIR/${T}_$M.out | tail -1) $(date)" >> $Q
done
