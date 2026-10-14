#!/bin/bash
# OFT LIBERO-Spatial full (dense) for init states 10-19 (100 eps), to extend the earlier init 0-9 run
# (oft_logs/spatial10/full.jsonl, same code path: VLA-Pruner openvla-oft, use_vla_pruner False) to the 200-episode protocol.
# Usage: bash run_full_add100.sh <gpu> <task_ids e.g. 0,1,2,3,4> <shard name>
GPU=$1; TASKS=$2; SH=$3
O=/workspace/vla/eval_logs/oft_spatial200; Q=$O/status.txt; TAG=oft200_full_i10_$SH
PY=/opt/conda/envs/openvla-oft/bin/python
CKPT=/workspace/vla/VLA-Pruner/src/openvla-oft/checkpoints/openvla-7b-oft-finetuned-libero-spatial
cd /workspace/vla/VLA-Pruner/src/openvla-oft
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU
echo "START $TAG gpu$GPU $(date)" >> $Q
OFT_RESULTS_JSONL=$O/$TAG.jsonl OFT_METHOD=full OFT_TASK_IDS=$TASKS OFT_INIT_OFFSET=10 \
  $PY /workspace/vla/oracle_pruning/oft_oracle/profile_eval.py $O/$TAG.prof.jsonl /workspace/vla/oracle_pruning/oft_oracle/run_eval.py \
  --pretrained_checkpoint $CKPT --task_suite_name libero_spatial --use_vla_cache False --use_fastv False \
  --use_vla_pruner False --seed 7 --num_trials_per_task 10 --local_log_dir $O --run_id_note $TAG > $O/$TAG.out 2>&1
echo "END $TAG exit=$? $(grep -c '"success": true' $O/$TAG.jsonl)/$(wc -l < $O/$TAG.jsonl) $(date)" >> $Q
