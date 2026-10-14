#!/bin/bash
# Base OpenVLA (user): which budget matches VLA-Cache? Two-head ranker without avis/ains, PCA 256, w = 0.35.
#   1) train once on spatial (budget-free ranker; checkpoint selection uses the 30% schedule)
#   2) schedules (DP on the w=0.35 mixed label) at 30 / 40 / 50 %: k = 58 / 87 / 115 -> 3*256 + 29k = 2450 / 3291 / 4103
#   3) LIBERO-Spatial 100 episodes (10 trials x 10 tasks, init 0-9) per budget, 3 processes on the GPU, shared profiler
# Reference (500 eps, spatial): Full 425 (85.0%), VLA-Cache 415 (83.0%, LLM FLOPs 0.77), ours 20% 398 (79.6%).
G=${1:-1}; R=/workspace/vla/oracle_pruning/VLA-Pruner/src/openvla; cd $R
PY=/opt/conda/envs/openvla/bin/python; OS=$R/oracle_scripts; W=0.35; S=libero_spatial
LOG=$R/experiments/logs/oracle; Q=$LOG/base_noavis256_status.txt; PROF=/workspace/vla/eval_logs/base100_noavis256; mkdir -p $PROF
echo "START base budget sweep (spatial, two-head -avis/-ains PCA256, w$W, 30/40/50%) gpu$G $(date)" >> $Q
CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=8 $PY $OS/train_predictor_mix.py --suite $S --variants A --two_head --mix $W --fastv_r 0.7734375 \
  --drop avis ains --pca_dim 256 --head_weights 0.3 0.35 0.4 0.5 --out_tag _noavis_pca256 > $LOG/train_${S}_noavis256.log 2>&1 &
for r in 0.7734375 0.66015625 0.55078125; do
  OMP_NUM_THREADS=6 $PY $OS/schedule_mix.py --suite $S --w $W --fastv_r $r >> $LOG/schedule_${S}_w${W}.log 2>&1 &
done; wait
CK=/workspace/vla/data/oracle_predictor_mix/${S}_m2_mix${W}_s0_noavis_pca256/A_2head/best.pt
[ -f $CK ] || { echo "MISSING $CK" >> $Q; exit 1; }
echo "END train + schedules $(ls $OS/schedules | grep "${S}_m2_w${W}_b" | tr '\n' ' ') $(date)" >> $Q
source $OS/common.sh
for pair in "0.7734375 2450" "0.66015625 3291" "0.55078125 4103"; do
  set -- $pair; T=base100_noavis256_w${W}_b$2_$S
  echo "START $T gpu$G $(date)" >> $Q
  ( CKPT=checkpoints/openvla-7b-finetuned-libero-spatial SUITE=$S PRED_CKPT=$CK RHO_R=$1 PROF_OUT=$PROF/$T.prof.jsonl \
    ORACLE_HEAD_WEIGHT=$W ORACLE_SCHEDULE=$OS/schedules/${S}_m2_w${W}_b$2.json \
    run_method $G predictor $T --num_trials_per_task 10 --init_state_offset 0 --save_video False > $LOG/${T}.out 2>&1
    echo "END $T exit=$? $(grep '# successes' $LOG/${T}.out | tail -1) $(date)" >> $Q ) &
  sleep 60
done; wait
echo "END base budget sweep $(date)" >> $Q
