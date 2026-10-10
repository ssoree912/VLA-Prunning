#!/bin/bash
# OFT baselines for one LIBERO suite (init states 0-9 x 10 tasks, seed 7): full, then VLA-Pruner keep 12.5%.
# Usage: bash queue_baselines.sh <suite: spatial|goal|object|10> <gpu> [wait_file wait_pattern]
S=$1; GPU=$2; WAIT_FILE=$3; WAIT_PAT=$4
L=/workspace/vla/oracle_pruning/oft_logs/${S}10; mkdir -p $L; Q=$L/status.txt
CKPT=/workspace/vla/VLA-Pruner/src/openvla-oft/checkpoints/openvla-7b-oft-finetuned-libero-$S
PY=/opt/conda/envs/openvla-oft/bin/python
cd /workspace/vla/VLA-Pruner/src/openvla-oft
if [ -n "$WAIT_FILE" ]; then until grep -q "$WAIT_PAT" "$WAIT_FILE" 2>/dev/null; do sleep 60; done; fi
until [ -f $CKPT/model.safetensors.index.json ] && ls $CKPT/action_head--*_checkpoint.pt >/dev/null 2>&1; do sleep 60; done
COMMON="--pretrained_checkpoint $CKPT --task_suite_name libero_$S --use_vla_cache False --use_fastv False --seed 7 --num_trials_per_task 10 --local_log_dir $L"
run() {  # name extra_args...
  local name=$1; shift
  echo "START $name $(date)" >> $Q
  CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OFT_RESULTS_JSONL=$L/$name.jsonl OFT_METHOD=$name \
    $PY /workspace/vla/oracle_pruning/oft_oracle/run_eval.py $COMMON --run_id_note $name "$@" > $L/$name.out 2>&1
  echo "END $name exit=$? $(grep -c '"success": true' $L/$name.jsonl)/$(wc -l < $L/$name.jsonl) $(date)" >> $Q
}
run full --use_vla_pruner False
run pruner --use_vla_pruner True --fastv_k 3 --fastv_r 0.875 --vla_pruner_layer 15 --vla_pruner_mode semantic_action
