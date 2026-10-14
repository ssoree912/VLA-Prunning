#!/bin/bash
# User: PCA256 two-head -avis at w = 0.4 (schedule mix 0.4, 20% b3280) on spatial and libero_10, 500 eps each.
# spatial on GPU 1, libero_10 on GPU 3, each once that GPU is empty (after goal PCA256 w0.3 shards).
OO=/workspace/vla/oracle_pruning/oft_oracle; source $OO/gpu_slots.sh; PY=/opt/conda/envs/openvla-oft/bin/python
SD=/workspace/vla/data/oft_predictor/schedules; P=/workspace/vla/data/oft_predictor/official; Q=/workspace/vla/eval_logs/oft_bench500/status.txt
[ -f $SD/libero_spatial_official_mix0.4_m2_b3280.json ] || (cd $OO && OMP_NUM_THREADS=8 $PY make_schedule.py --suite libero_spatial \
  --fastv_r 0.9140625 --mix 0.4 --tag _official > /workspace/vla/eval_logs/oft_spatial200/ablation/schedule_mix0.4.log 2>&1)
run() {  # gpu suite
  while [ $(nproc_gpu $1) -gt 0 ]; do sleep 60; done
  echo "START ours PCA256 $2 w0.4 500 gpu$1 (user) $(date)" >> $Q
  TAG_SUFFIX=_pca256_w0.4 OURS_W=0.4 OURS_SCHED=$SD/$2_official_mix0.4_m2_b3280.json OURS_CKPT=$P/$2_twohead_no_avis_pca256/best.pt \
    bash $OO/bench500.sh $1 ours $2 0 50
}
run 1 libero_spatial & run 3 libero_10 & wait
