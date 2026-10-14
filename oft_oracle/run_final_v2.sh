#!/bin/bash
# Final plan (user):
#   GPU 3: libero_10 two-head -avis w0.5 200 (running) -> init 20-49 (to 500) -> libero_10 PCA256 500 -> PCA128 500 -> PCA64 500
#   GPU 1 (now) and GPU 0 (after VLA-Cache libero_10): queue of spatial / object / goal jobs, PCA 256 first, then 128, 64
#   (spatial w0.5, object w0.5, goal w0.4; schedule mix = w, 20% b3280). A GPU takes the first job whose ranker is trained.
# At most 3 processes per GPU (user): a job (3 shards) starts only on an empty GPU (gpu_slots.sh).
# Rankers still being trained by the run_final_queue.sh training subshell (GPU 0): object 128/64, libero_10 256/128/64.
OO=/workspace/vla/oracle_pruning/oft_oracle; source $OO/gpu_slots.sh
Q=/workspace/vla/eval_logs/oft_bench500/status.txt; SD=/workspace/vla/data/oft_predictor/schedules
P=/workspace/vla/data/oft_predictor/official; J=/workspace/vla/eval_logs/oft_bench500/final_jobs_v2.txt
logf() { echo /workspace/vla/eval_logs/oft_${1#libero_}200/ablation/twohead_no_avis_pca$2.log; }
ready() { [ -f $P/$1_twohead_no_avis_pca$2/best.pt ] && grep -q "best val KL" $(logf $1 $2) 2>/dev/null; }
evalpca() {  # gpu k suite w offset trials [tag suffix, default _pca<k>]
  echo "TAKE gpu$1: PCA$2 $3 w$4 offset $5 trials $6 $(date)" >> $Q
  TAG_SUFFIX=${7:-_pca$2} OURS_W=$4 OURS_SCHED=$SD/$3_official_mix$4_m2_b3280.json OURS_CKPT=$P/$3_twohead_no_avis_pca$2/best.pt \
    bash $OO/bench500.sh $1 ours $3 $5 $6
}
[ -s $J ] || printf "%s\n" "256 libero_spatial 0.5 20 30" "256 libero_object 0.5 0 50" "256 libero_goal 0.4 0 50" \
  "128 libero_spatial 0.5 0 50" "128 libero_object 0.5 0 50" "128 libero_goal 0.4 0 50" \
  "64 libero_spatial 0.5 0 50" "64 libero_object 0.5 0 50" "64 libero_goal 0.4 0 50" > $J
echo "START final plan v2 (GPU3 libero_10 sequence; GPU0/1 queue of $(wc -l < $J) jobs) $(date)" >> $Q
take() {
  flock $J.lock bash -c '
    P='$P'; logf() { echo /workspace/vla/eval_logs/oft_${1#libero_}200/ablation/twohead_no_avis_pca$2.log; }
    n=0; while read -r k s w o t tag; do n=$((n+1)); [ -z "$k" ] && continue
      if [ -f $P/${s}_twohead_no_avis_pca$k/best.pt ] && grep -q "best val KL" $(logf $s $k) 2>/dev/null; then
        echo "$k $s $w $o $t $tag"; sed -i "${n}d" '$J'; exit 0; fi; done < '$J''
}
worker() {  # gpu
  while [ -s $J ]; do while [ $(nproc_gpu $1) -gt 0 ]; do sleep 60; done; job=$(take); if [ -z "$job" ]; then sleep 60; continue; fi; evalpca $1 $job; done
  echo "WORKER-v2 gpu$1 done $(date)" >> $Q
}
( while [ $(grep -c "END ours_w0.5_libero_10_t" $Q) -lt 3 ]; do sleep 60; done
  while [ $(nproc_gpu 3) -gt 0 ]; do sleep 30; done
  echo "TAKE gpu3: libero_10 w0.5 init 20-49 (to 500) $(date)" >> $Q
  TAG_SUFFIX=_w0.5 OURS_W=0.5 OURS_SCHED=$SD/libero_10_official_mix0.5_m2_b3280.json bash $OO/bench500.sh 3 ours libero_10 20 30
  for k in 256 128 64; do while ! ready libero_10 $k || [ $(nproc_gpu 3) -gt 0 ]; do sleep 60; done; evalpca 3 $k libero_10 0.5 0 50; done
  echo "GPU3 libero_10 sequence done $(date)" >> $Q ) &
worker 1 &
( while [ $(grep -c "END vlacache_libero_10_t" $Q) -lt 4 ]; do sleep 120; done; worker 0 ) &  # 2 END lines are from the stopped morning run
wait
