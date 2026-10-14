#!/bin/bash
# User: libero_goal PCA256 two-head -avis at w = 0.3 (schedule mix 0.3, 20% b3280), 500 episodes, on GPU 3.
SD=/workspace/vla/data/oft_predictor/schedules; S=$SD/libero_goal_official_mix0.3_m2_b3280.json
Q=/workspace/vla/eval_logs/oft_bench500/status.txt
while [ ! -f $S ]; do sleep 30; done
echo "START ours PCA256 libero_goal w0.3 500 gpu3 $(date)" >> $Q
TAG_SUFFIX=_pca256_w0.3 OURS_W=0.3 OURS_SCHED=$S OURS_CKPT=/workspace/vla/data/oft_predictor/official/libero_goal_twohead_no_avis_pca256/best.pt \
  bash /workspace/vla/oracle_pruning/oft_oracle/bench500.sh 3 ours libero_goal 0 50
