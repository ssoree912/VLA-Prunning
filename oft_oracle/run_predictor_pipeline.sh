#!/bin/bash
# After OFT labeling finishes (pid $1): train the ranker, then closed-loop LIBERO-Spatial eval (init 0-9 x 10 tasks).
LABEL_PID=$1; GPU=${2:-1}
L=/workspace/vla/oracle_pruning/oft_logs/spatial10; mkdir -p $L; Q=$L/status.txt
PY=/opt/conda/envs/openvla-oft/bin/python
CKPT=/workspace/vla/VLA-Pruner/src/openvla-oft/checkpoints/openvla-7b-oft-finetuned-libero-spatial
PRED=/workspace/vla/data/oft_predictor/libero_spatial_m2_s0/best.pt
while ps -p $LABEL_PID >/dev/null 2>&1; do sleep 60; done
echo "LABELS DONE $(ls /workspace/vla/data/oft_labels/libero_spatial/ep*.npz | wc -l) demos $(date)" >> $Q
cd /workspace/vla/VLA-Pruner/src/openvla-oft
echo "START train $(date)" >> $Q
CUDA_VISIBLE_DEVICES=$GPU $PY /workspace/vla/oracle_pruning/oft_oracle/train.py --suite libero_spatial --epochs 20 \
  > /workspace/vla/data/oft_predictor_libero_spatial_train.log 2>&1
echo "END train exit=$? $(grep '^best' /workspace/vla/data/oft_predictor_libero_spatial_train.log) $(date)" >> $Q
[ -f $PRED ] || exit 1
echo "START predictor $(date)" >> $Q
CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OFT_RESULTS_JSONL=$L/predictor.jsonl OFT_METHOD=predictor \
  OFT_PREDICTOR_CKPT=$PRED $PY /workspace/vla/oracle_pruning/oft_oracle/run_eval.py --pretrained_checkpoint $CKPT \
  --task_suite_name libero_spatial --use_vla_cache False --use_fastv False --use_vla_pruner True --fastv_k 3 --fastv_r 0.875 \
  --vla_pruner_layer 15 --vla_pruner_mode semantic_action --seed 7 --num_trials_per_task 10 --local_log_dir $L \
  --run_id_note predictor > $L/predictor.out 2>&1
echo "END predictor exit=$? $(grep -c '"success": true' $L/predictor.jsonl)/$(wc -l < $L/predictor.jsonl) $(date)" >> $Q
