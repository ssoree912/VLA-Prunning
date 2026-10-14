#!/bin/bash
# GPU worker for the OFT benchmark queue: takes the first line of bench_jobs.txt ("method suite [offset trials]") under a
# lock, runs bench500.sh on its GPU, repeats until the file is empty. Optionally waits first until <wait_file> has
# <n> lines matching <pattern> (the run currently on that GPU).
# Usage: bash bench_queue.sh <gpu> [wait_file pattern n]
GPU=$1; WF=$2; WP=$3; WN=${4:-1}
D=/workspace/vla/eval_logs/oft_bench500; J=$D/bench_jobs.txt; Q=$D/status.txt; OO=/workspace/vla/oracle_pruning/oft_oracle
[ -n "$WF" ] && while [ $(grep -c "$WP" $WF) -lt $WN ]; do sleep 60; done
echo "WORKER gpu$GPU start $(date)" >> $Q
while true; do
  job=$(flock $J.lock bash -c "head -n 1 $J; sed -i 1d $J")
  [ -z "$job" ] && break
  echo "TAKE gpu$GPU: $job $(date)" >> $Q
  bash $OO/bench500.sh $GPU $job
done
echo "WORKER gpu$GPU done $(date)" >> $Q
