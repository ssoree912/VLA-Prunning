#!/bin/bash
# Our OFT predictor, 200-episode LIBERO eval (20 trials x 10 tasks, seed 7) in 3 task shards with the profiler.
# Usage: bash run_ours_eval.sh <gpu> <pred best.pt> <schedule.json> <tag> [head_weight]   (env SUITE, default libero_spatial)
GPU=$1; PRED=$2; SCHED=$3; NAME=$4; W=$5
SUITE=${SUITE:-libero_spatial}
PY=/opt/conda/envs/openvla-oft/bin/python
O=/workspace/vla/eval_logs/oft_${SUITE#libero_}200; mkdir -p $O; Q=$O/status.txt
OO=/workspace/vla/oracle_pruning/oft_oracle
CKPT=/workspace/vla/VLA-Pruner/src/openvla-oft/checkpoints/openvla-7b-oft-finetuned-$(echo $SUITE | tr '_' '-')
cd /workspace/vla/VLA-Pruner/src/openvla-oft
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU
run() {  # shard task_ids
  local TAG=oft200_${NAME}_$1
  echo "START $TAG gpu$GPU $(date)" >> $Q
  OFT_RESULTS_JSONL=$O/$TAG.jsonl OFT_METHOD=$NAME OFT_PREDICTOR_CKPT=$PRED OFT_SCHEDULE=$SCHED OFT_HEAD_WEIGHT=$W \
    OFT_TASK_IDS=$2 $PY $OO/profile_eval.py $O/$TAG.prof.jsonl $OO/run_eval.py --pretrained_checkpoint $CKPT \
    --task_suite_name $SUITE --use_vla_cache False --use_fastv False --use_vla_pruner True --fastv_k 3 \
    --fastv_r 0.9140625 --vla_pruner_layer 15 --vla_pruner_mode semantic_action --seed 7 --num_trials_per_task 20 \
    --local_log_dir $O --run_id_note $TAG > $O/$TAG.out 2>&1
  echo "END $TAG exit=$? $(grep -c '"success": true' $O/$TAG.jsonl)/$(wc -l < $O/$TAG.jsonl) $(date)" >> $Q
}
run t0123 0,1,2,3 & sleep 30; run t456 4,5,6 & sleep 30; run t789 7,8,9 & wait
