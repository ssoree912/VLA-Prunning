#!/bin/bash
# Base OpenVLA: full depth-predictor pipeline for one LIBERO suite on two GPUs (run once per role).
#   1. teacher labels (RLDS, split ep % 2)            helper: shard 0          main: shard 1
#   2. predictor features (PCA) + instruction labels  helper: instr shard 0    main: features, then instr shard 1
#   3. two-head ranker (mixed label 0.5)                                       main
#   4. closed-loop eval, init 0-9: predictor (two-head, w=0.4) then full     helper: tasks 0-4   main: tasks 5-9
# Usage (from src/openvla): bash oracle_scripts/suite_pipeline.sh <suite e.g. libero_object> <helper|main> <gpu> <tag e.g. object10>
S=$1; ROLE=$2; GPU=$3; TAG=$4
cd /workspace/vla/oracle_pruning/VLA-Pruner/src/openvla
PY=/opt/conda/envs/openvla/bin/python
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
D=/workspace/vla/data/pipeline_$S; mkdir -p $D
LOG=experiments/logs/oracle; Q=$LOG/${TAG}_status.txt
SH=$([ "$ROLE" = main ] && echo 1 || echo 0)
wait_for() { until [ -f $D/$1 ]; do sleep 60; done; }
note() { echo "$1 $(date)" >> $Q; }

note "START $ROLE labels shard$SH gpu$GPU"
CUDA_VISIBLE_DEVICES=$GPU nice -n 10 $PY oracle_scripts/label_rlds.py --suite $S --num_shards 2 --shard $SH > $D/labels_shard$SH.log 2>&1
note "END labels shard$SH exit=$?"; touch $D/labels_shard$SH.done
wait_for labels_shard0.done; wait_for labels_shard1.done

if [ "$ROLE" = main ]; then
  CUDA_VISIBLE_DEVICES=$GPU $PY oracle_scripts/predictor_features.py --suite $S > $D/features.log 2>&1
  note "FEATURES exit=$?"; touch $D/features.done
fi
CUDA_VISIBLE_DEVICES=$GPU $PY oracle_scripts/label_instr.py --suite $S --num_shards 2 --shard $SH > $D/instr_shard$SH.log 2>&1
note "END instr shard$SH exit=$?"; touch $D/instr_shard$SH.done
wait_for instr_shard0.done; wait_for instr_shard1.done; wait_for features.done

PRED=/workspace/vla/data/oracle_predictor_mix/${S}_m2_mix0.5_s0/A_2head/best.pt
if [ "$ROLE" = main ]; then
  CUDA_VISIBLE_DEVICES=$GPU $PY oracle_scripts/train_predictor_mix.py --suite $S --variants A --mix 0.5 --two_head > $D/train.log 2>&1
  note "TRAIN exit=$? $(grep '^variant' $D/train.log | cut -c1-160)"; touch $D/train.done
fi
wait_for train.done

export CKPT=checkpoints/openvla-7b-finetuned-$(echo $S | tr '_' '-') SUITE=$S PRED_CKPT=$PRED
source oracle_scripts/common.sh
TASKS=$([ "$ROLE" = main ] && echo 5,6,7,8,9 || echo 0,1,2,3,4)
P=$([ "$ROLE" = main ] && echo p2 || echo p1)
for M in predictor full; do
  note "START $M $P gpu$GPU"
  ORACLE_HEAD_WEIGHT=0.4 run_method $GPU $M ${TAG}_${M}_$P --task_ids $TASKS --num_trials_per_task 10 --init_state_offset 0 --save_video False > $LOG/${TAG}_${M}_$P.out 2>&1
  note "END $M $P exit=$? $(grep '# successes' $LOG/${TAG}_${M}_$P.out | tail -1)"; touch $D/eval_${M}_$P.done
done

if [ "$ROLE" = main ]; then
  for M in predictor full; do
    wait_for eval_${M}_p1.done; wait_for eval_${M}_p2.done
    OUT=$LOG/EVAL-$S-openvla-merged--${TAG}_$([ $M = predictor ] && echo predictor_2head_w04 || echo full).jsonl
    cat $(ls -t $LOG/EVAL-*--${TAG}_${M}_p1_$M.jsonl | head -1) $(ls -t $LOG/EVAL-*--${TAG}_${M}_p2_$M.jsonl | head -1) > $OUT
    note "MERGED $M $(grep -c '"success": true' $OUT)/$(wc -l < $OUT)"
  done
  note "ALL DONE"
fi
