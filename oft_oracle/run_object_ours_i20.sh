#!/bin/bash
# Extend our libero_object run (two-head -avis, w 0.5, mix-0.5 schedule b3280) to 500 episodes: init states 20-49 on
# GPU 3 after the 200-episode run ends.
Q=/workspace/vla/eval_logs/oft_bench500/status.txt
while [ $(grep -c "END ours_libero_object_t" $Q) -lt 3 ]; do sleep 60; done
echo "START ours libero_object init 20-49 (to 500) gpu3 $(date)" >> $Q
bash /workspace/vla/oracle_pruning/oft_oracle/bench500.sh 3 ours libero_object 20 30
