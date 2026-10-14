#!/bin/bash
# OFT LIBERO-Spatial, our method at a 22% layer-averaged visual budget (matched to SpecPrune's official default ~22%):
#   teacher labels (RLDS demos) -> fixed schedule + ranker -> 200-episode eval (20 trials x 10 tasks, seed 7) with the
#   shared FLOPs/latency profiler. Budget (train.py): (K+1)*M + (L-K-1)*k, K=3, M=512, k=56 (fastv_r=0.890625)
#   -> 3616 / 16384 = 22.07%.
GPU=${1:-3}
PY=/opt/conda/envs/openvla-oft/bin/python
O=/workspace/vla/eval_logs/oft_spatial200; Q=$O/status.txt
CKPT=/workspace/vla/VLA-Pruner/src/openvla-oft/checkpoints/openvla-7b-oft-finetuned-libero-spatial
PRED_DIR=/workspace/vla/data/oft_predictor/libero_spatial_m2_p22_s0
cd /workspace/vla/VLA-Pruner/src/openvla-oft
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU
echo "START ours_p22 labels gpu$GPU $(date)" >> $Q
$PY /workspace/vla/oracle_pruning/oft_oracle/label_rlds.py --suite libero_spatial > $O/ours_p22_labels.log 2>&1
echo "END ours_p22 labels exit=$? $(ls /workspace/vla/data/oft_labels/libero_spatial/ep*.npz | wc -l) demos $(date)" >> $Q
$PY /workspace/vla/oracle_pruning/oft_oracle/train.py --suite libero_spatial --fastv_r 0.890625 --epochs 20 \
  --out_dir $PRED_DIR > $O/ours_p22_train.log 2>&1
echo "END ours_p22 train exit=$? $(grep '^best' $O/ours_p22_train.log) $(date)" >> $Q
[ -f $PRED_DIR/best.pt ] || exit 1
echo "START oft200_ours_p22 gpu$GPU $(date)" >> $Q
OFT_RESULTS_JSONL=$O/oft200_ours_p22.jsonl OFT_METHOD=ours_p22 OFT_PREDICTOR_CKPT=$PRED_DIR/best.pt \
  $PY /workspace/vla/baselines/tools/vla_profile.py $O/oft200_ours_p22.prof.jsonl \
  /workspace/vla/oracle_pruning/oft_oracle/run_eval.py --pretrained_checkpoint $CKPT --task_suite_name libero_spatial \
  --use_vla_cache False --use_fastv False --use_vla_pruner True --fastv_k 3 --fastv_r 0.890625 --vla_pruner_layer 15 \
  --vla_pruner_mode semantic_action --seed 7 --num_trials_per_task 20 --local_log_dir $O --run_id_note oft200_ours_p22 \
  > $O/oft200_ours_p22.out 2>&1
echo "END oft200_ours_p22 exit=$? $(grep -c '"success": true' $O/oft200_ours_p22.jsonl)/$(wc -l < $O/oft200_ours_p22.jsonl) $(date)" >> $Q
