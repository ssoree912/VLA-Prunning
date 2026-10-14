#!/bin/bash
# Our OFT predictor on the OFFICIAL OpenVLA-OFT code (official.py), 200-episode LIBERO eval (20 trials x 10 tasks,
# seed 7) in 3 task shards with the shared profiler. Same stack as SpecPrune / VLA-ADP (run_oft200.sh): official repo as
# cwd + PYTHONPATH, transformers fork v4.40.1-openvla-oft + tokenizers 0.19 via PYTHONPATH, private checkpoint dir.
# Usage: bash run_ours_official.sh <gpu> <pred best.pt | ""> <schedule.json | ""> <tag> [head_weight]
#   (env SUITE, default libero_spatial; TRIALS per task, default 20; OFT_INIT_OFFSET first init state; an empty
#   predictor = official full model)
GPU=$1; PRED=$2; SCHED=$3; NAME=$4; W=$5
SUITE=${SUITE:-libero_spatial}; CK=openvla-7b-oft-finetuned-$(echo $SUITE | tr '_' '-')
PY=/opt/conda/envs/openvla-oft/bin/python
O=/workspace/vla/eval_logs/oft_${SUITE#libero_}200; mkdir -p $O; Q=$O/status.txt
OO=/workspace/vla/oracle_pruning/oft_oracle
R=/workspace/vla/baselines/openvla-oft-official
TF_OFT=/workspace/vla/baselines/_transformers_oft/src:/workspace/vla/baselines/_pydeps_oft
cd $R
export PYTHONPATH=$R:$TF_OFT MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU \
  TF_CPP_MIN_LOG_LEVEL=3
run() {  # shard task_ids
  local TAG=oft200_${NAME}_$1
  echo "START $TAG gpu$GPU $(date)" >> $Q
  OFT_RESULTS_JSONL=$O/$TAG.jsonl OFT_METHOD=$NAME OFT_PREDICTOR_CKPT=$PRED OFT_SCHEDULE=$SCHED OFT_HEAD_WEIGHT=$W \
    OFT_TASK_IDS=$2 $PY /workspace/vla/baselines/tools/vla_profile.py $O/$TAG.prof.jsonl $OO/run_eval_official.py \
    --pretrained_checkpoint /workspace/vla/models/oft-ours/$CK-ours --task_suite_name $SUITE --seed 7 \
    --num_trials_per_task ${TRIALS:-20} --local_log_dir $O --run_id_note $TAG > $O/$TAG.out 2>&1
  echo "END $TAG exit=$? $(grep -c '"success": true' $O/$TAG.jsonl)/$(wc -l < $O/$TAG.jsonl) $(date)" >> $Q
}
run t0123 0,1,2,3 & sleep 30; run t456 4,5,6 & sleep 30; run t789 7,8,9 & wait
