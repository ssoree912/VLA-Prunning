#!/bin/bash
# OFT LIBERO-Spatial baselines (init states 0-9 x 10 tasks, seed 7), sequential on one GPU.
# Usage: bash run_baselines.sh <gpu> [wait_status_file] [wait_pattern]
GPU=${1:-3}; WAIT_FILE=$2; WAIT_PAT=$3
L=/workspace/vla/oracle_pruning/oft_logs/spatial10; mkdir -p $L; Q=$L/status.txt
CKPT=/workspace/vla/VLA-Pruner/src/openvla-oft/checkpoints/openvla-7b-oft-finetuned-libero-spatial
PY=/opt/conda/envs/openvla-oft/bin/python
cd /workspace/vla/VLA-Pruner/src/openvla-oft
if [ -n "$WAIT_FILE" ]; then until grep -q "$WAIT_PAT" "$WAIT_FILE"; do sleep 60; done; fi
COMMON="--pretrained_checkpoint $CKPT --task_suite_name libero_spatial --use_vla_cache False --use_fastv False --seed 7 --num_trials_per_task 10 --local_log_dir $L"
run() {  # name extra_args...
  local name=$1; shift
  echo "START $name $(date)" >> $Q
  CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OFT_RESULTS_JSONL=$L/$name.jsonl OFT_METHOD=$name \
    $PY /workspace/vla/oracle_pruning/oft_oracle/run_eval.py $COMMON --run_id_note $name "$@" > $L/$name.out 2>&1
  echo "END $name exit=$? $(grep -c '"success": true' $L/$name.jsonl)/$(wc -l < $L/$name.jsonl) $(date)" >> $Q
}
run full --use_vla_pruner False
run pruner --use_vla_pruner True --fastv_k 3 --fastv_r 0.875 --vla_pruner_layer 15 --vla_pruner_mode semantic_action
