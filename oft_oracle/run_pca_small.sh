#!/bin/bash
# User: PCA 128 and 64 (two-head -avis) on spatial / goal / object, 500 episodes each, same per-suite settings as
# PCA 256 (spatial w0.5, object w0.5, goal w0.4; schedule mix = w, 20% b3280).
# Trains the 6 rankers on GPU 0 now (small GPU memory), then a job queue: a GPU takes the next eval once its
# run_pca256.sh sequence is done (GPU 0 after goal PCA256, GPU 1 after libero_10 w0.6, GPU 3 after libero_10 to 500).
OO=/workspace/vla/oracle_pruning/oft_oracle; PY=/opt/conda/envs/openvla-oft/bin/python
Q=/workspace/vla/eval_logs/oft_bench500/status.txt; SD=/workspace/vla/data/oft_predictor/schedules
P=/workspace/vla/data/oft_predictor/official; J=/workspace/vla/eval_logs/oft_bench500/pca_jobs.txt
echo "START train PCA 128/64 x spatial/goal/object (two-head -avis) gpu0 $(date)" >> $Q
for k in 128 64; do for S in libero_spatial libero_goal libero_object; do
  OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 $PY $OO/train.py --suite $S --label_dir /workspace/vla/data/oft_labels/$S \
    --seed 0 --two_head --mix 0.5 --drop avis --pca_dim $k --out_dir $P/${S}_twohead_no_avis_pca$k \
    > /workspace/vla/eval_logs/oft_${S#libero_}200/ablation/twohead_no_avis_pca$k.log 2>&1 &
done; done; wait
echo "END train PCA 128/64: $(ls -d $P/*_pca128 $P/*_pca64 2>/dev/null | wc -l) dirs $(date)" >> $Q
printf "%s\n" "128 libero_spatial 0.5" "128 libero_object 0.5" "128 libero_goal 0.4" "64 libero_spatial 0.5" "64 libero_object 0.5" "64 libero_goal 0.4" > $J
worker() {  # gpu wait_pattern n
  while [ $(grep -c "$2" $Q) -lt $3 ]; do sleep 120; done
  while true; do
    job=$(flock $J.lock bash -c "head -n 1 $J; sed -i 1d $J"); [ -z "$job" ] && break
    set -- $1 $job  # gpu k suite w
    echo "START ours PCA$2 $3 w$4 500 gpu$1 $(date)" >> $Q
    TAG_SUFFIX=_pca$2 OURS_W=$4 OURS_SCHED=$SD/$3_official_mix$4_m2_b3280.json OURS_CKPT=$P/$3_twohead_no_avis_pca$2/best.pt \
      bash $OO/bench500.sh $1 ours $3 0 50
    set -- $1 "$WP" "$WN"
  done
}
worker0() { WP="END ours_pca256_libero_goal_t"; WN=3; worker 0 "$WP" $WN; }
worker1() { WP="END ours_w0.6_libero_10_t"; WN=3; worker 1 "$WP" $WN; }
worker3() { WP="BEST libero_10 200ep"; WN=1; while [ $(grep -c "$WP" $Q) -lt 1 ]; do sleep 120; done
            WP="END ours_w0.[456]_libero_10_i20_t"; WN=3; worker 3 "$WP" $WN; }
worker0 & worker1 & worker3 & wait
