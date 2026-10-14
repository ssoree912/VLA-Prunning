#!/bin/bash
# User: PCA256 two-head -avis at w = 0.35 (schedule DP on the mix-0.35 label, 20% b3280), 500 eps on all four suites.
# Max 3 processes per GPU: a job (3 shards) starts only on an empty GPU. Order: libero_10, spatial, object, goal.
OO=/workspace/vla/oracle_pruning/oft_oracle; source $OO/gpu_slots.sh; PY=/opt/conda/envs/openvla-oft/bin/python
SD=/workspace/vla/data/oft_predictor/schedules; P=/workspace/vla/data/oft_predictor/official; Q=/workspace/vla/eval_logs/oft_bench500/status.txt
J=/workspace/vla/eval_logs/oft_bench500/w035_jobs.txt
for S in libero_10 libero_spatial libero_object libero_goal; do
  [ -f $SD/${S}_official_mix0.35_m2_b3280.json ] || (cd $OO && OMP_NUM_THREADS=6 $PY make_schedule.py --suite $S --fastv_r 0.9140625 \
    --mix 0.35 --tag _official > /workspace/vla/eval_logs/oft_${S#libero_}200/ablation/schedule_mix0.35.log 2>&1) &
done; wait
echo "SCHEDULES mix0.35: $(ls $SD | grep -c mix0.35) $(date)" >> $Q
printf "%s\n" libero_10 libero_spatial libero_object libero_goal > $J
worker() {  # gpu
  while [ -s $J ]; do
    while [ $(nproc_gpu $1) -gt 0 ]; do sleep 60; done
    S=$(flock $J.lock bash -c "head -n 1 $J; sed -i 1d $J"); [ -z "$S" ] && break
    echo "START ours PCA256 $S w0.35 500 gpu$1 $(date)" >> $Q
    TAG_SUFFIX=_pca256_w0.35 OURS_W=0.35 OURS_SCHED=$SD/${S}_official_mix0.35_m2_b3280.json OURS_CKPT=$P/${S}_twohead_no_avis_pca256/best.pt \
      bash $OO/bench500.sh $1 ours $S 0 50
  done
}
worker 3 & sleep 5; worker 1 & sleep 5; worker 0 & wait
echo "END w0.35 queue $(date)" >> $Q
