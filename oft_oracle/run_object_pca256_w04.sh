#!/bin/bash
# libero_object, two-head -avis PCA256, head weight w = 0.4 with the matching mix-0.4 schedule (20%, b3280), 500 eps, GPU 0.
OO=/workspace/vla/oracle_pruning/oft_oracle; PY=/opt/conda/envs/openvla-oft/bin/python; Q=/workspace/vla/eval_logs/oft_bench500/status.txt
SD=/workspace/vla/data/oft_predictor/schedules; SC=$SD/libero_object_official_mix0.4_m2_b3280.json
[ -f $SC ] || (cd $OO && OMP_NUM_THREADS=12 $PY make_schedule.py --suite libero_object --fastv_r 0.9140625 --mix 0.4 --tag _official \
  >> /workspace/vla/eval_logs/oft_object200/ablation/schedules_mix_sweep.log 2>&1)
echo "START ours PCA256 libero_object w0.4 (schedule mix0.4) 500 gpu0 $(date)" >> $Q
TAG_SUFFIX=_pca256_w0.4 OURS_W=0.4 OURS_SCHED=$SC OURS_CKPT=/workspace/vla/data/oft_predictor/official/libero_object_twohead_no_avis_pca256/best.pt \
  bash $OO/bench500.sh 0 ours libero_object 0 50
