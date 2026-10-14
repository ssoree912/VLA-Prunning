#!/bin/bash
# After VLA-Pruner libero_10 ends on GPU 0: our two-head -avis on libero_goal, heads at w = 0.5, mix-0.5 schedule
# (20%, b3280), 200 episodes (init states 0-19 x 10 tasks), official stack (bench500.sh ours).
Q=/workspace/vla/eval_logs/oft_bench500/status.txt
while [ $(grep -c "END vlapruner_libero_10_t" $Q) -lt 2 ]; do sleep 60; done
echo "START ours libero_goal 200 (two-head -avis, w0.5, b3280) gpu0 $(date)" >> $Q
bash /workspace/vla/oracle_pruning/oft_oracle/bench500.sh 0 ours libero_goal 0 20
