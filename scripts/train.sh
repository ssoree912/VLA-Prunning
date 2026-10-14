#!/bin/bash
# GazePrune ranker for one suite: budget-free two-head training (no avis, PCA 256) + the schedule for head weight w
# at a budget (DP on the mixed label). Output: $GAZEPRUNE_DATA/oft_predictor/{official/<suite>_twohead_no_avis_pca256,
# schedules/<suite>_official_mix<w>_m2_b<budget>.json}.
# Usage: bash scripts/train.sh <gpu> <suite> [w=0.35] [fastv_r=0.9140625 (20%: k=44, b3280)]
source "$(dirname "$0")/common.sh"
G=$1; S=$2; W=${3:-0.35}; R=${4:-0.9140625}
O=$GAZEPRUNE_LOGS/train; mkdir -p $O; export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
CUDA_VISIBLE_DEVICES=$G $PY $REPO/train/train_ranker.py --suite $S --seed 0 --two_head --mix $W --drop avis --pca_dim 256 \
  --out_dir $GAZEPRUNE_DATA/oft_predictor/official/${S}_twohead_no_avis_pca256 > $O/${S}_twohead_no_avis_pca256.log 2>&1 &
$PY $REPO/train/make_schedule.py --suite $S --fastv_r $R --mix $W --tag _official > $O/${S}_schedule_mix$W.log 2>&1 &
wait
