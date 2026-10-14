#!/bin/bash
# Round 3 (user): goal w0.4 / w0.5 to 500 episodes (init 20-49), VLA-Cache libero_10 500, our libero_10 labels + training.
#   GPU 0: after ours_w0.4 goal 200 ends -> ours w0.4 goal init 20-49 -> vlacache libero_10 (500, 2 shards)
#   GPU 1: after ours_w0.6 goal 200 ends -> ours w0.5 goal init 20-49
#   GPU 3: after ours object init 20-49 ends -> libero_10 official labels + two-head training + mix-0.5 schedule
OO=/workspace/vla/oracle_pruning/oft_oracle; Q=/workspace/vla/eval_logs/oft_bench500/status.txt
SD=/workspace/vla/data/oft_predictor/schedules
waitn() { while [ $(grep -c "$1" $Q) -lt $2 ]; do sleep 60; done; }
( waitn "END ours_w0.4_libero_goal_t" 3
  echo "START ours w0.4 libero_goal init 20-49 gpu0 $(date)" >> $Q
  TAG_SUFFIX=_w0.4 OURS_W=0.4 OURS_SCHED=$SD/libero_goal_official_mix0.4_m2_b3280.json bash $OO/bench500.sh 0 ours libero_goal 20 30
  echo "START vlacache libero_10 (rerun, 500) gpu0 $(date)" >> $Q
  bash $OO/bench500.sh 0 vlacache libero_10 ) &
( waitn "END ours_w0.6_libero_goal_t" 3
  echo "START ours w0.5 libero_goal init 20-49 gpu1 $(date)" >> $Q
  bash $OO/bench500.sh 1 ours libero_goal 20 30 ) &
( waitn "END ours_libero_object_i20_t" 3
  bash $OO/run_suite_ours.sh 3 libero_10 ) &
wait
echo "END round3 $(date)" >> $Q
