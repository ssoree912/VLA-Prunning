#!/bin/bash
# Clean latency of every OFT method: after the benchmark queue is done (3 workers), one process at a time on an idle
# GPU, real LIBERO-Spatial rollouts (tasks 0 and 5, init states 0-2 = 6 episodes, ~100 queries; consecutive queries,
# so methods with temporal state such as SpecPrune / VLA-Cache behave as in the eval), shared profiler.
# Output: /workspace/vla/eval_logs/oft_latency/libero_spatial/<method>_libero_spatial_lat.prof.jsonl
G=${1:-0}; D=/workspace/vla/eval_logs/oft_bench500; OO=/workspace/vla/oracle_pruning/oft_oracle
export OUTROOT=/workspace/vla/eval_logs/oft_latency; mkdir -p $OUTROOT
while [ $(grep -c "^WORKER gpu.* done" $D/status.txt) -lt 3 ]; do sleep 120; done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0,1,3 | grep -q .; do sleep 60; done  # our GPUs idle
echo "START clean latency gpu$G (one process, others idle) $(date)" >> $OUTROOT/status.txt
for M in full ours specprune vlapruner vlacache; do
  bash $OO/bench500.sh $G $M libero_spatial 0 3 "lat:0,5"
done
echo "END clean latency $(date)" >> $OUTROOT/status.txt
