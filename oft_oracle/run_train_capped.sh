#!/bin/bash
# Remaining PCA ranker trainings on GPU 0 with at most 3 processes on the GPU (user rule): a training starts only while
# GPU 0 holds fewer than 3 compute processes (VLA-Cache shards / evals / trainings all count).
OO=/workspace/vla/oracle_pruning/oft_oracle; PY=/opt/conda/envs/openvla-oft/bin/python; P=/workspace/vla/data/oft_predictor/official
Q=/workspace/vla/eval_logs/oft_bench500/status.txt
source /workspace/vla/oracle_pruning/oft_oracle/gpu_slots.sh
nproc0() { nproc_gpu 0; }
logf() { echo /workspace/vla/eval_logs/oft_${1#libero_}200/ablation/twohead_no_avis_pca$2.log; }
echo "START capped trainings gpu0 (<= 3 procs on the GPU) $(date)" >> $Q
for job in "128 libero_spatial" "64 libero_spatial" "64 libero_goal" "128 libero_object" "64 libero_object" \
           "256 libero_10" "128 libero_10" "64 libero_10"; do
  set -- $job
  while [ $(nproc0) -ge 3 ]; do sleep 30; done
  mkdir -p $(dirname $(logf $2 $1))
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 nohup $PY $OO/train.py --suite $2 --label_dir /workspace/vla/data/oft_labels/$2 \
    --seed 0 --two_head --mix 0.5 --drop avis --pca_dim $1 --out_dir $P/$2_twohead_no_avis_pca$1 > $(logf $2 $1) 2>&1 &
  sleep 20
done
wait
echo "END capped trainings $(date)" >> $Q
