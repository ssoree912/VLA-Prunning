#!/bin/bash
# Resume a stopped std500_p20 run: finish task <t0> from init <i0>, then tasks t0+1..9 (50 trials each), then merge
# the original partial jsonl with the new parts into EVAL-<suite>-openvla-merged--<tag>_<method>.jsonl.
# Usage (from src/openvla): bash oracle_scripts/resume500.sh <gpu> <suite> <tag> <method> <t0> <i0> [pred_ckpt w schedule]
GPU=$1; S=$2; T=$3; M=$4; T0=$5; I0=$6; P=$7; W=$8; SCH=$9
cd /workspace/vla/oracle_pruning/VLA-Pruner/src/openvla
export RHO_R=0.8828125 CKPT=checkpoints/openvla-7b-finetuned-$(echo $S | tr '_' '-') SUITE=$S PRED_CKPT=$P
source oracle_scripts/common.sh
Q=$LOG_DIR/std500_p20_status.txt
ORIG=$(ls -t $LOG_DIR/EVAL-$S-openvla-2*--${T}_$M.jsonl | head -1)
echo "RESUME $T $M from task$T0 init$I0 gpu$GPU (orig $(wc -l < $ORIG) eps) $(date)" >> $Q
PROF_OUT=${PROF_DIR:+$PROF_DIR/${T}r1_$M.prof.jsonl} ORACLE_HEAD_WEIGHT=$W ORACLE_SCHEDULE=$SCH run_method $GPU $M ${T}r1 --task_ids $T0 --init_state_offset $I0 \
  --num_trials_per_task $((50 - I0)) --save_video False > $LOG_DIR/${T}r1_$M.out 2>&1
if [ $T0 -lt 9 ]; then
  PROF_OUT=${PROF_DIR:+$PROF_DIR/${T}r2_$M.prof.jsonl} ORACLE_HEAD_WEIGHT=$W ORACLE_SCHEDULE=$SCH run_method $GPU $M ${T}r2 --task_ids $(seq -s, $((T0 + 1)) 9) \
    --init_state_offset 0 --num_trials_per_task 50 --save_video False > $LOG_DIR/${T}r2_$M.out 2>&1
fi
OUT=$LOG_DIR/EVAL-$S-openvla-merged--${T}_$M.jsonl
cat $ORIG $(ls -t $LOG_DIR/EVAL-$S-openvla-2*--${T}r1_$M.jsonl | head -1) \
    $(ls -t $LOG_DIR/EVAL-$S-openvla-2*--${T}r2_$M.jsonl 2>/dev/null | head -1) > $OUT
echo "END $T $M (resumed) $(grep -c '"success": true' $OUT)/$(wc -l < $OUT) $(date)" >> $Q
