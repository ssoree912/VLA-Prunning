#!/bin/bash
# Base OpenVLA (user): two-head ranker without avis, PCA 256 (as the OFT ranker), head weight w = 0.35, 30% budget, all 4 suites.
#   1) train_predictor_mix.py --variants A --two_head --mix 0.35 --drop avis ains --pca_dim 256 (<= 3 processes on the GPU)
#   2) schedule_mix.py --w 0.35 --fastv_r 0.7734375 -> 30% budget b2450 (k=58: 3*256 + 29*58 = 2450 = 29.9%)
#   3) 100 episodes (10 trials x 10 tasks, init 0-9, seed 7) with the shared profiler, <= 3 processes on the GPU
# Usage (any cwd): bash run_base_noavis256.sh <gpu>
G=${1:-1}; R=/workspace/vla/oracle_pruning/VLA-Pruner/src/openvla; cd $R
PY=/opt/conda/envs/openvla/bin/python; OS=$R/oracle_scripts; W=0.35
LOG=$R/experiments/logs/oracle; Q=$LOG/base_noavis256_status.txt; PROF=/workspace/vla/eval_logs/base100_noavis256; mkdir -p $PROF
source /workspace/vla/oracle_pruning/oft_oracle/gpu_slots.sh
slot() { while [ $(nproc_gpu $G) -ge 3 ]; do sleep 30; done; }
echo "START base two-head -avis PCA256 w$W gpu$G $(date)" >> $Q
for S in libero_spatial libero_object libero_goal libero_10; do
  slot
  CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8 $PY $OS/train_predictor_mix.py --suite $S --variants A --two_head --mix $W --fastv_r 0.7734375 \
    --drop avis ains --pca_dim 256 --head_weights 0.3 0.35 0.4 0.5 --out_tag _noavis_pca256_b2450 > $LOG/train_${S}_noavis256.log 2>&1 &
  sleep 60
  [ -f $OS/schedules/${S}_m2_w${W}_b2450.json ] || OMP_NUM_THREADS=6 $PY $OS/schedule_mix.py --suite $S --w $W --fastv_r 0.7734375 \
    >> $LOG/schedule_${S}_w${W}.log 2>&1 &
done
wait
echo "END train + schedules: $(ls $OS/schedules | grep -c "w${W}_b2450") schedules $(date)" >> $Q
source $OS/common.sh
for S in libero_spatial libero_object libero_goal libero_10; do
  CK=/workspace/vla/data/oracle_predictor_mix/${S}_m2_mix${W}_s0_noavis_pca256_b2450/A_2head/best.pt
  [ -f $CK ] || { echo "MISSING $CK" >> $Q; continue; }
  slot
  T=base100_noavis256_w${W}_b2450_$S
  echo "START $T gpu$G $(date)" >> $Q
  ( CKPT=checkpoints/openvla-7b-finetuned-$(echo $S | tr '_' '-') SUITE=$S PRED_CKPT=$CK RHO_R=0.7734375 \
    PROF_OUT=$PROF/$T.prof.jsonl ORACLE_HEAD_WEIGHT=$W ORACLE_SCHEDULE=$OS/schedules/${S}_m2_w${W}_b2450.json \
    run_method $G predictor $T --num_trials_per_task 10 --init_state_offset 0 --save_video False > $LOG/${T}.out 2>&1
    echo "END $T exit=$? $(grep '# successes' $LOG/${T}.out | tail -1) $(date)" >> $Q ) &
  sleep 60
done
wait
echo "END base two-head -avis PCA256 w$W $(date)" >> $Q
