#!/bin/bash
# Clean latency of every OFT method, one process at a time on one GPU: real LIBERO-Spatial rollouts (tasks 0 and 5,
# init states 0-2 = 6 episodes, ~70-100 consecutive queries, so methods with temporal state behave as in the eval),
# shared profiler (FLOPs, LLM / model / get_action ms, peak GPU memory). Run it on an otherwise idle GPU.
# Output: $GAZEPRUNE_LOGS/oft_latency/libero_spatial/<method>_libero_spatial_lat.prof.jsonl
# Usage: bash scripts/latency.sh <gpu> [methods...]   (default: full ours specprune specprune_nodyn vlapruner vlacache)
source "$(dirname "$0")/common.sh"
G=$1; shift; METHODS=${@:-full ours specprune specprune_nodyn vlapruner vlacache}
export OUTROOT=$GAZEPRUNE_LOGS/oft_latency; mkdir -p $OUTROOT
for M in $METHODS; do bash $REPO/scripts/eval.sh $G $M libero_spatial 0 3 "lat:0,5"; done
