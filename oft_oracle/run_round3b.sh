#!/bin/bash
# Round 3b (user):
#   GPU 0: after ours w0.4 goal init 20-49 ends -> FLOPs / latency / speedup / memory measurement of every method, one
#          process at a time (LIBERO-Spatial rollouts, tasks 0 and 5, init 0-2 = 6 episodes, consecutive queries), then
#          the deferred VLA-Cache libero_10 500.
#   GPU 1: after ours w0.5 goal init 20-49 ends -> 2 extra libero_10 label shards from episode 190; if run_suite_ours.sh
#          (GPU 3) then finds the label set incomplete, wait for it and run the same training + schedule here.
OO=/workspace/vla/oracle_pruning/oft_oracle; PY=/opt/conda/envs/openvla-oft/bin/python
Q=/workspace/vla/eval_logs/oft_bench500/status.txt; QL=/workspace/vla/eval_logs/oft_10200/status.txt
R=/workspace/vla/baselines/openvla-oft-official; LD=/workspace/vla/data/oft_labels/libero_10
waitn() { while [ $(grep -c "$1" $Q) -lt $2 ]; do sleep 60; done; }
( waitn "END ours_w0.4_libero_goal_i20_t" 3
  export OUTROOT=/workspace/vla/eval_logs/oft_latency; mkdir -p $OUTROOT
  echo "START latency measurement gpu0 (one process at a time) $(date)" >> $OUTROOT/status.txt
  for M in full ours specprune specprune_nodyn vlapruner vlacache; do
    bash $OO/bench500.sh 0 $M libero_spatial 0 3 "lat:0,5"; done
  echo "END latency measurement $(date)" >> $OUTROOT/status.txt
  unset OUTROOT
  echo "START vlacache libero_10 (rerun, 500) gpu0 $(date)" >> $Q
  bash $OO/bench500.sh 0 vlacache libero_10 ) &
( waitn "END ours_libero_goal_i20_t" 3
  echo "START extra libero_10 label shards (from ep 190) gpu1 $(date)" >> $QL
  (cd $R && export PYTHONPATH=$R:/workspace/vla/baselines/_transformers_oft/src:/workspace/vla/baselines/_pydeps_oft \
     CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 MUJOCO_GL=osmesa OMP_NUM_THREADS=8 &&
   for s in 0 1; do $PY $OO/label_rlds_official.py --suite libero_10 --num_shards 2 --shard $s --start_ep 190 \
     >> /workspace/vla/eval_logs/oft_10200/official_labels_gpu1_shard$s.log 2>&1 & sleep 20; done; wait)
  echo "END extra libero_10 label shards $(ls $LD | grep -c '^ep.*\.npz$') files $(date)" >> $QL
  while ! grep -q "END official labels libero_10" $QL; do sleep 60; done
  if grep "END official labels libero_10" $QL | grep -qv "379 / 379"; then  # run_suite_ours.sh stopped early
    while [ $(ls $LD | grep -c '^ep.*\.npz$') -lt 379 ] || pgrep -f "label_rlds_official.py --suite libero_10" > /dev/null; do sleep 60; done
    O=/workspace/vla/eval_logs/oft_10200; OUT=/workspace/vla/data/oft_predictor/official; export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
    echo "START twohead train libero_10 (fallback on gpu1) $(date)" >> $QL
    CUDA_VISIBLE_DEVICES=1 $PY $OO/train.py --suite libero_10 --label_dir $LD --seed 0 --two_head --mix 0.5 --out_dir $OUT/libero_10_twohead > $O/ablation/twohead.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 $PY $OO/train.py --suite libero_10 --label_dir $LD --seed 0 --two_head --mix 0.5 --drop avis --out_dir $OUT/libero_10_twohead_no_avis > $O/ablation/twohead_no_avis.log 2>&1 &
    (cd $OO && $PY make_schedule.py --suite libero_10 --fastv_r 0.9140625 --mix 0.5 --tag _official > $O/ablation/schedule_mix0.5.log 2>&1)
    wait
    echo "END libero_10 twohead (fallback): no_avis $(tail -n 1 $O/ablation/twohead_no_avis.log) $(date)" >> $QL
  fi ) &
wait
