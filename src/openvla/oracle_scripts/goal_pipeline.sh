#!/bin/bash
# Base OpenVLA LIBERO-Goal depth predictor. Labels are split over two GPUs (ep_idx % 2); the "main" role then builds the
# schedule, features, trains the ranker and runs the closed-loop eval (init states 0-9 x 10 tasks, seed 7).
# Usage (from anywhere): bash goal_pipeline.sh <helper|main> <gpu> <wait_file> <wait_pattern>
ROLE=$1; GPU=$2; WAIT_FILE=$3; WAIT_PAT=$4
S=libero_goal; SHARD=$([ "$ROLE" = main ] && echo 1 || echo 0)
cd /workspace/vla/oracle_pruning/VLA-Pruner/src/openvla
PY=/opt/conda/envs/openvla/bin/python
Q=experiments/logs/oracle/goal10_status.txt
LBL=/workspace/vla/data/oracle_labels/$S
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
until grep -q "$WAIT_PAT" "$WAIT_FILE" 2>/dev/null; do sleep 60; done
echo "START labels shard$SHARD gpu$GPU $(date)" >> $Q
CUDA_VISIBLE_DEVICES=$GPU nice -n 10 $PY oracle_scripts/label_rlds.py --suite $S --num_shards 2 --shard $SHARD \
  > /workspace/vla/data/oracle_labels/${S}_shard$SHARD.log 2>&1
echo "END labels shard$SHARD exit=$? $(date)" >> $Q
[ "$ROLE" = main ] || exit 0
until grep -q "END labels shard0" $Q; do sleep 60; done
echo "LABELS $(ls $LBL/ep*.npz | wc -l) demos $(date)" >> $Q
$PY oracle_scripts/make_schedule.py --suite $S --min_depth 2 > /workspace/vla/data/oracle_labels/${S}_schedule.log 2>&1
echo "SCHEDULE exit=$? $(tail -1 /workspace/vla/data/oracle_labels/${S}_schedule.log | cut -c1-200) $(date)" >> $Q
CUDA_VISIBLE_DEVICES=$GPU $PY oracle_scripts/predictor_features.py --suite $S > /workspace/vla/data/oracle_features_$S.log 2>&1
echo "FEATURES exit=$? $(date)" >> $Q
CUDA_VISIBLE_DEVICES=$GPU $PY oracle_scripts/train_predictor.py --suite $S --epochs 20 > /workspace/vla/data/oracle_predictor_${S}_train.log 2>&1
echo "TRAIN exit=$? $(grep '^best' /workspace/vla/data/oracle_predictor_${S}_train.log) $(date)" >> $Q
export CKPT=checkpoints/openvla-7b-finetuned-$(echo $S | tr '_' '-') SUITE=$S
source oracle_scripts/common.sh
echo "START predictor $(date)" >> $Q
run_method $GPU predictor goal10 --num_trials_per_task 10 --init_state_offset 0 --save_video False > $LOG_DIR/goal10_predictor.out 2>&1
echo "END predictor exit=$? $(grep '# successes' $LOG_DIR/goal10_predictor.out | tail -1) $(date)" >> $Q
