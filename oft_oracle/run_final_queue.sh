#!/bin/bash
# Final queue (user): libero_10 stays at w0.5 (no w sweep) -> 500 eps; then PCA 256 / 128 / 64 (two-head -avis) on all
# four suites, 500 eps each (spatial w0.5, object w0.5, goal w0.4, libero_10 w0.5; schedule mix = w, 20% b3280).
# Trainings still missing run on GPU 0 once the current trainings there are done (object 128/64 hit CUDA OOM next to
# VLA-Cache; libero_10 256/128/64 new). Eval jobs: priority order, a GPU takes the first job whose ranker is trained.
OO=/workspace/vla/oracle_pruning/oft_oracle; PY=/opt/conda/envs/openvla-oft/bin/python
Q=/workspace/vla/eval_logs/oft_bench500/status.txt; SD=/workspace/vla/data/oft_predictor/schedules
P=/workspace/vla/data/oft_predictor/official; J=/workspace/vla/eval_logs/oft_bench500/final_jobs.txt
logf() { echo /workspace/vla/eval_logs/oft_${1#libero_}200/ablation/twohead_no_avis_pca$2.log; }
ready() { [ $1 = 512 ] || { [ -f $P/$2_twohead_no_avis_pca$1/best.pt ] && grep -q "best val KL" $(logf $2 $1) 2>/dev/null; }; }
train() {  # k suite
  mkdir -p $(dirname $(logf $2 $1))
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 $PY $OO/train.py --suite $2 --label_dir /workspace/vla/data/oft_labels/$2 \
    --seed 0 --two_head --mix 0.5 --drop avis --pca_dim $1 --out_dir $P/$2_twohead_no_avis_pca$1 > $(logf $2 $1) 2>&1
}
cat > $J <<'JOBS'
512 libero_10 0.5 20 30
256 libero_spatial 0.5 20 30
256 libero_object 0.5 0 50
256 libero_goal 0.4 0 50
256 libero_10 0.5 0 50
128 libero_spatial 0.5 0 50
128 libero_object 0.5 0 50
128 libero_goal 0.4 0 50
128 libero_10 0.5 0 50
64 libero_spatial 0.5 0 50
64 libero_object 0.5 0 50
64 libero_goal 0.4 0 50
64 libero_10 0.5 0 50
JOBS
echo "START final queue: $(wc -l < $J) eval jobs (libero_10 w0.5 to 500, PCA 256/128/64 x 4 suites) $(date)" >> $Q
( while pgrep -f "train.py --suite" > /dev/null; do sleep 60; done
  echo "START train object PCA128/64 + libero_10 PCA256/128/64 gpu0 $(date)" >> $Q
  train 128 libero_object & train 64 libero_object & train 256 libero_10 & wait
  train 128 libero_10 & train 64 libero_10 & wait
  echo "END train (remaining PCA rankers) $(date)" >> $Q ) &
take() {  # prints the first job whose ranker is ready and removes it from the queue
  flock $J.lock bash -c '
    P='$P'; logf() { echo /workspace/vla/eval_logs/oft_${1#libero_}200/ablation/twohead_no_avis_pca$2.log; }
    ready() { [ $1 = 512 ] || { [ -f $P/$2_twohead_no_avis_pca$1/best.pt ] && grep -q "best val KL" $(logf $2 $1) 2>/dev/null; }; }
    n=0; while read -r k s w o t; do n=$((n+1)); [ -z "$k" ] && continue
      if ready $k $s; then echo "$k $s $w $o $t"; sed -i "${n}d" '$J'; exit 0; fi; done < '$J''
}
worker() {  # gpu wait_pattern n
  while [ $(grep -c "$2" $Q) -lt $3 ]; do sleep 120; done
  local G=$1
  while [ -s $J ]; do
    job=$(take); if [ -z "$job" ]; then sleep 60; continue; fi
    set -- $job  # k suite w offset trials
    echo "TAKE gpu$G: PCA$1 $2 w$3 offset $4 trials $5 $(date)" >> $Q
    if [ $1 = 512 ]; then
      TAG_SUFFIX=_w$3 OURS_W=$3 OURS_SCHED=$SD/$2_official_mix$3_m2_b3280.json bash $OO/bench500.sh $G ours $2 $4 $5
    else
      TAG_SUFFIX=_pca$1 OURS_W=$3 OURS_SCHED=$SD/$2_official_mix$3_m2_b3280.json OURS_CKPT=$P/$2_twohead_no_avis_pca$1/best.pt \
        bash $OO/bench500.sh $G ours $2 $4 $5
    fi
  done
  echo "WORKER-final gpu$G done $(date)" >> $Q
}
worker 3 "END ours_w0.5_libero_10_t" 3 &
worker 1 "END ours_pca256_libero_spatial_t" 3 &
worker 0 "END vlacache_libero_10_t" 2 &
wait
