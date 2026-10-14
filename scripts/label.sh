#!/bin/bash
# Teacher labels + ranker features for one LIBERO suite on the official OFT stack (teacher/label_oft.py), 2 shards on
# one GPU (resumable: existing ep*.npz are skipped). Output: $GAZEPRUNE_DATA/oft_labels/<suite>/.
# Usage: bash scripts/label.sh <gpu> <libero_spatial|libero_object|libero_goal|libero_10> [n_shards=2]
source "$(dirname "$0")/common.sh"
G=$1; S=$2; N=${3:-2}; O=$GAZEPRUNE_LOGS/labels; mkdir -p $O
oft_env; export CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8
for s in $(seq 0 $((N - 1))); do
  $PY $REPO/teacher/label_oft.py --suite $S --num_shards $N --shard $s >> $O/${S}_shard$s.log 2>&1 & sleep 20
done; wait
