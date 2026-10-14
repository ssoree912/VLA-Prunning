#!/bin/bash
# libero_goal two-head -avis, head weight w = 0.4 / 0.6 with the matching mixed-label schedule (DP on mix = w, 20% b3280),
# 200 episodes each: w0.4 on GPU 0 (now), w0.6 on GPU 1 (after SpecPrune nodyn libero_10 ends).
OO=/workspace/vla/oracle_pruning/oft_oracle; PY=/opt/conda/envs/openvla-oft/bin/python; Q=/workspace/vla/eval_logs/oft_bench500/status.txt
SD=/workspace/vla/data/oft_predictor/schedules
cd $OO
for w in 0.4 0.6; do [ -f $SD/libero_goal_official_mix${w}_m2_b3280.json ] || \
  OMP_NUM_THREADS=12 $PY make_schedule.py --suite libero_goal --fastv_r 0.9140625 --mix $w --tag _official \
  >> /workspace/vla/eval_logs/oft_goal200/ablation/schedules_mix_sweep.log 2>&1 & done; wait
run() {  # gpu w
  echo "START ours libero_goal 200 two-head -avis w$2 (schedule mix$2, b3280) gpu$1 $(date)" >> $Q
  TAG_SUFFIX=_w$2 OURS_W=$2 OURS_SCHED=$SD/libero_goal_official_mix$2_m2_b3280.json bash $OO/bench500.sh $1 ours libero_goal 0 20
}
run 0 0.4 &
while [ $(grep -c "END specprune_nodyn_libero_10_t" $Q) -lt 3 ]; do sleep 60; done
run 1 0.6 &
wait
