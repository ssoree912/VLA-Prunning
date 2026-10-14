#!/bin/bash
# Waits until the running libero_10 trainings have loaded their data, then strips avis from all OFT label dirs.
L=/workspace/vla/eval_logs/oft_10200/ablation; Q=/workspace/vla/eval_logs/oft_bench500/status.txt
until [ $(grep -l "loaded in" $L/twohead.log $L/twohead_no_avis.log 2>/dev/null | wc -l) -eq 2 ]; do sleep 30; done
echo "START strip avis from OFT labels (user) $(date)" >> $Q
for S in libero_spatial libero_object libero_goal libero_10; do
  nice -n 19 ionice -c3 /opt/conda/envs/openvla-oft/bin/python /workspace/vla/oracle_pruning/oft_oracle/strip_avis.py \
    /workspace/vla/data/oft_labels/$S >> /workspace/vla/eval_logs/oft_bench500/strip_avis.log 2>&1 &
done; wait
echo "END strip avis: $(tail -n 4 /workspace/vla/eval_logs/oft_bench500/strip_avis.log | tr '\n' ' ') $(du -sh /workspace/vla/data/oft_labels | cut -f1) $(date)" >> $Q
