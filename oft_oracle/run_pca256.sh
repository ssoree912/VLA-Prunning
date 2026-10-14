#!/bin/bash
# User: PCA-256 rankers (two-head -avis) on spatial / goal / object to 500 episodes first; libero_10 w sweep after.
# Per-suite head weight / schedule as the PCA-512 runs: spatial w0.5, object w0.5, goal w0.4 (mixed-label schedule
# mix = w, 20% b3280).
#   now   : train goal and object PCA-256 on GPU 0 (small GPU memory, next to VLA-Cache)
#   GPU 1 : spatial PCA-256 200 (running) -> spatial init 20-49 -> libero_10 w0.6 200
#   GPU 0 : VLA-Cache libero_10 (running) -> goal PCA-256 500
#   GPU 3 : libero_10 w0.5 200 (running) -> object PCA-256 500 -> libero_10 w0.4 200 -> best libero_10 w to 500
OO=/workspace/vla/oracle_pruning/oft_oracle; PY=/opt/conda/envs/openvla-oft/bin/python
Q=/workspace/vla/eval_logs/oft_bench500/status.txt; SD=/workspace/vla/data/oft_predictor/schedules
P=/workspace/vla/data/oft_predictor/official; O10=/workspace/vla/eval_logs/oft_bench500/libero_10
waitn() { while [ $(grep -c "$1" $Q) -lt $2 ]; do sleep 60; done; }
waitck() { while [ ! -f $1 ] || ! grep -q "best val KL" $2 2>/dev/null; do sleep 60; done; }
for S in libero_goal libero_object; do
  L=/workspace/vla/eval_logs/oft_${S#libero_}200/ablation/twohead_no_avis_pca256.log
  OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 $PY $OO/train.py --suite $S --label_dir /workspace/vla/data/oft_labels/$S \
    --seed 0 --two_head --mix 0.5 --drop avis --pca_dim 256 --out_dir $P/${S}_twohead_no_avis_pca256 > $L 2>&1 &
done
echo "START train goal/object PCA256 (two-head -avis) gpu0 $(date)" >> $Q
run10() {  # gpu w offset trials
  echo "START ours libero_10 two-head -avis w$2 (schedule mix$2) offset $3 trials $4 gpu$1 $(date)" >> $Q
  TAG_SUFFIX=_w$2 OURS_W=$2 OURS_SCHED=$SD/libero_10_official_mix$2_m2_b3280.json bash $OO/bench500.sh $1 ours libero_10 $3 $4
}
pca() {  # gpu suite w offset trials
  echo "START ours PCA256 $2 w$3 offset $4 trials $5 gpu$1 $(date)" >> $Q
  TAG_SUFFIX=_pca256 OURS_W=$3 OURS_SCHED=$SD/$2_official_mix$3_m2_b3280.json OURS_CKPT=$P/$2_twohead_no_avis_pca256/best.pt \
    bash $OO/bench500.sh $1 ours $2 $4 $5
}
( waitn "END ours_pca256_libero_spatial_t" 3; pca 1 libero_spatial 0.5 20 30; run10 1 0.6 0 20 ) &
( waitn "END vlacache_libero_10_t" 2
  waitck $P/libero_goal_twohead_no_avis_pca256/best.pt /workspace/vla/eval_logs/oft_goal200/ablation/twohead_no_avis_pca256.log
  pca 0 libero_goal 0.4 0 50 ) &
( waitn "END ours_w0.5_libero_10_t" 3
  waitck $P/libero_object_twohead_no_avis_pca256/best.pt /workspace/vla/eval_logs/oft_object200/ablation/twohead_no_avis_pca256.log
  pca 3 libero_object 0.5 0 50; run10 3 0.4 0 20 ) &
wait
best=$(for w in 0.4 0.5 0.6; do echo "$(cat $O10/ours_w${w}_libero_10_t*.jsonl 2>/dev/null | grep -v '"q"' | grep -c '"success": true') $w"; done | sort -rn | head -n 1)
echo "BEST libero_10 200ep: $best (success w) -> extend to 500 $(date)" >> $Q
run10 3 ${best#* } 20 30
