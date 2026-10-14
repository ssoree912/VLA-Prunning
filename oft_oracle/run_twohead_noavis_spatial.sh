#!/bin/bash
# Two-head ranker WITHOUT avis (best single-head variant) on the official-stack spatial labels, 50/50: budget-free
# train.py --two_head --drop avis, the same mix-0.5 schedule as run_twohead_spatial.sh (DP on the mixed label, b3280),
# then 200 eps with the heads combined at w = 0.5 (layers 0-1 on SDPA, no attention extraction).
# Waits for the GPU 0 run (oft200_offnew_full_b3280) to end.
G=0; S=libero_spatial; O=/workspace/vla/eval_logs/oft_spatial200; Q=$O/status.txt
OUT=/workspace/vla/data/oft_predictor/official/${S}_twohead_no_avis
SCHED=/workspace/vla/data/oft_predictor/schedules/${S}_official_mix0.5_m2_b3280.json
PY=/opt/conda/envs/openvla-oft/bin/python; OO=/workspace/vla/oracle_pruning/oft_oracle
while [ $(grep -c "END oft200_offnew_full_b3280" $Q) -lt 3 ]; do sleep 60; done
echo "START twohead no_avis train (budget-free) gpu$G $(date)" >> $Q
OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=$G $PY $OO/train.py --suite $S \
  --label_dir /workspace/vla/data/oft_labels/$S --seed 0 --two_head --mix 0.5 --drop avis --out_dir $OUT \
  > $O/ablation/twohead_no_avis.log 2>&1
echo "END twohead no_avis train $(tail -n 1 $O/ablation/twohead_no_avis.log) $(date)" >> $Q
while [ ! -f $SCHED ]; do sleep 60; done  # made by run_twohead_spatial.sh (GPU 3)
[ -f $OUT/best.pt ] || exit 1
cd $OO && OMP_NUM_THREADS=8 bash run_ours_official.sh $G $OUT/best.pt $SCHED offnew_twohead_noavis_w0.5_b3280 0.5
