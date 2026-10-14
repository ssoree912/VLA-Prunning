#!/bin/bash
# After the OFT LIBERO-Spatial labels finish: train the (budget-free) ranker once, build fixed schedules for 20% and 22%,
# then the 200-episode eval at 20% (matched to SpecPrune's measured 20.1%) in 3 task shards with the profiler.
GPU=${1:-3}; LABEL_PID=$2
PY=/opt/conda/envs/openvla-oft/bin/python
O=/workspace/vla/eval_logs/oft_spatial200; Q=$O/status.txt
OO=/workspace/vla/oracle_pruning/oft_oracle
CKPT=/workspace/vla/VLA-Pruner/src/openvla-oft/checkpoints/openvla-7b-oft-finetuned-libero-spatial
PRED=/workspace/vla/data/oft_predictor/libero_spatial_m2_s0
SCH=/workspace/vla/data/oft_predictor/schedules
while [ -e /proc/$LABEL_PID ] && [ "$(awk '{print $3}' /proc/$LABEL_PID/stat 2>/dev/null)" != Z ]; do sleep 60; done
cd /workspace/vla/VLA-Pruner/src/openvla-oft
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU
echo "START ours train (budget-free ranker; checkpoint schedule 20%) gpu$GPU $(date)" >> $Q
$PY $OO/train.py --suite libero_spatial --fastv_r 0.9140625 --epochs 20 --out_dir $PRED > $O/ours_train.log 2>&1
echo "END ours train exit=$? $(grep '^best' $O/ours_train.log) $(date)" >> $Q
[ -f $PRED/best.pt ] || exit 1
for r in 0.9140625 0.890625; do $PY $OO/make_schedule.py --suite libero_spatial --fastv_r $r >> $O/ours_schedules.log 2>&1; done
echo "SCHEDULES $(ls $SCH) $(date)" >> $Q
run() {  # shard tasks budget
  local TAG=oft200_ours_b$3_$1
  echo "START $TAG gpu$GPU $(date)" >> $Q
  OFT_RESULTS_JSONL=$O/$TAG.jsonl OFT_METHOD=ours_b$3 OFT_PREDICTOR_CKPT=$PRED/best.pt OFT_SCHEDULE=$SCH/libero_spatial_m2_b$3.json \
    OFT_TASK_IDS=$2 $PY $OO/profile_eval.py $O/$TAG.prof.jsonl $OO/run_eval.py --pretrained_checkpoint $CKPT \
    --task_suite_name libero_spatial --use_vla_cache False --use_fastv False --use_vla_pruner True --fastv_k 3 \
    --fastv_r 0.9140625 --vla_pruner_layer 15 --vla_pruner_mode semantic_action --seed 7 --num_trials_per_task 20 \
    --local_log_dir $O --run_id_note $TAG > $O/$TAG.out 2>&1
  echo "END $TAG exit=$? $(grep -c '"success": true' $O/$TAG.jsonl)/$(wc -l < $O/$TAG.jsonl) $(date)" >> $Q
}
run t0123 0,1,2,3 3280 & sleep 30; run t456 4,5,6 3280 & sleep 30; run t789 7,8,9 3280 & wait
