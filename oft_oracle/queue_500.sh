#!/bin/bash
# OFT standard-size eval (50 trials/task = init states 0-49, 500 episodes/suite): predictor and VLA-Pruner keep 12.5%.
# Usage: bash queue_500.sh <gpu> [wait_file wait_pattern]
GPU=$1; WAIT_FILE=$2; WAIT_PAT=$3
PY=/opt/conda/envs/openvla-oft/bin/python
cd /workspace/vla/VLA-Pruner/src/openvla-oft
if [ -n "$WAIT_FILE" ]; then until grep -q "$WAIT_PAT" "$WAIT_FILE" 2>/dev/null; do sleep 60; done; fi
run() {  # suite name extra_env...
  local S=$1 name=$2 pred=$3
  local L=/workspace/vla/oracle_pruning/oft_logs/${S}50; mkdir -p $L; local Q=$L/status.txt
  local CKPT=/workspace/vla/VLA-Pruner/src/openvla-oft/checkpoints/openvla-7b-oft-finetuned-libero-$S
  echo "START $name $(date)" >> $Q
  CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OFT_RESULTS_JSONL=$L/$name.jsonl OFT_METHOD=$name \
    OFT_PREDICTOR_CKPT=$pred $PY /workspace/vla/oracle_pruning/oft_oracle/run_eval.py --pretrained_checkpoint $CKPT \
    --task_suite_name libero_$S --use_vla_cache False --use_fastv False --use_vla_pruner True --fastv_k 3 --fastv_r 0.875 \
    --vla_pruner_layer 15 --vla_pruner_mode semantic_action --seed 7 --num_trials_per_task 50 --local_log_dir $L \
    --run_id_note $name > $L/$name.out 2>&1
  echo "END $name exit=$? $(grep -c '"success": true' $L/$name.jsonl)/$(wc -l < $L/$name.jsonl) $(date)" >> $Q
}
for S in spatial goal; do
  run $S predictor /workspace/vla/data/oft_predictor/libero_${S}_m2_s0/best.pt
  run $S pruner ""
done
