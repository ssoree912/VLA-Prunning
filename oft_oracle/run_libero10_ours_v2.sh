#!/bin/bash
# Replaces run_libero10_ours.sh (user: PCA-256 check first on GPU 1).
#   GPU 1: after the spatial PCA-256 two-head -avis training -> spatial 200 eps (w 0.5, mix-0.5 schedule), then libero_10 w0.6 200
#   GPU 3: after libero_10 training -> libero_10 w0.5 200, then w0.4 200
#   then the best libero_10 w (200-ep successes) extended to 500 (init 20-49) on GPU 3.
OO=/workspace/vla/oracle_pruning/oft_oracle; PY=/opt/conda/envs/openvla-oft/bin/python
Q=/workspace/vla/eval_logs/oft_bench500/status.txt; QL=/workspace/vla/eval_logs/oft_10200/status.txt
SD=/workspace/vla/data/oft_predictor/schedules; O=/workspace/vla/eval_logs/oft_bench500/libero_10
P=/workspace/vla/data/oft_predictor/official
run() {  # gpu w offset trials
  echo "START ours libero_10 two-head -avis w$2 (schedule mix$2) offset $3 trials $4 gpu$1 $(date)" >> $Q
  TAG_SUFFIX=_w$2 OURS_W=$2 OURS_SCHED=$SD/libero_10_official_mix$2_m2_b3280.json bash $OO/bench500.sh $1 ours libero_10 $3 $4
}
( while ! grep -q "best val KL" /workspace/vla/eval_logs/oft_spatial200/ablation/twohead_no_avis_pca256.log 2>/dev/null; do sleep 60; done
  echo "START ours spatial PCA256 200 (two-head -avis, w0.5) gpu1 $(date)" >> $Q
  TAG_SUFFIX=_pca256 OURS_CKPT=$P/libero_spatial_twohead_no_avis_pca256/best.pt bash $OO/bench500.sh 1 ours libero_spatial 0 20
  while [ ! -f $SD/libero_10_official_mix0.6_m2_b3280.json ]; do sleep 60; done
  run 1 0.6 0 20 ) &
( while ! grep -q "END libero_10 twohead" $QL 2>/dev/null; do sleep 120; done
  for w in 0.4 0.6; do [ -f $SD/libero_10_official_mix${w}_m2_b3280.json ] || (cd $OO && OMP_NUM_THREADS=12 $PY make_schedule.py \
    --suite libero_10 --fastv_r 0.9140625 --mix $w --tag _official >> /workspace/vla/eval_logs/oft_10200/ablation/schedules_mix_sweep.log 2>&1) & done; wait
  run 3 0.5 0 20; run 3 0.4 0 20 ) &
wait
best=$(for w in 0.4 0.5 0.6; do echo "$(cat $O/ours_w${w}_libero_10_t*.jsonl 2>/dev/null | grep -v '"q"' | grep -c '"success": true') $w"; done | sort -rn | head -n 1)
echo "BEST libero_10 200ep: $best (success w) -> extend to 500 $(date)" >> $Q
run 3 ${best#* } 20 30
