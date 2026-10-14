#!/bin/bash
# LIBERO-Goal on the official OFT stack: finish the official labels (label_rlds_official.py, resumes after the 75 kept
# episodes; 2 shards on one GPU), then budget-free two-head training (mix 0.5; full inputs and -avis) and the mix-0.5
# schedules (20% b3280, SpecPrune-matched 21.5% b3532). No rollout eval. Waits for the GPU run oft200_offnew_noavis.
G=1; S=libero_goal; O=/workspace/vla/eval_logs/oft_goal200; Q=$O/status.txt; QS=/workspace/vla/eval_logs/oft_spatial200/status.txt
LD=/workspace/vla/data/oft_labels/$S; OUT=/workspace/vla/data/oft_predictor/official
PY=/opt/conda/envs/openvla-oft/bin/python; OO=/workspace/vla/oracle_pruning/oft_oracle; R=/workspace/vla/baselines/openvla-oft-official
mkdir -p $O/ablation
while [ $(grep -c "END oft200_offnew_noavis_b3280" $QS) -lt 3 ]; do sleep 60; done
echo "START official labels (resume, 2 shards) gpu$G $(date)" >> $Q
(cd $R && export PYTHONPATH=$R:/workspace/vla/baselines/_transformers_oft/src:/workspace/vla/baselines/_pydeps_oft \
   CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 MUJOCO_GL=osmesa OMP_NUM_THREADS=8 &&
 for s in 0 1; do $PY $OO/label_rlds_official.py --suite $S --num_shards 2 --shard $s >> $O/official_labels_shard$s.log 2>&1 & done; wait)
n=$(ls $LD | grep -c '^ep.*\.npz$')
echo "END official labels: $n / 428 episodes $(date)" >> $Q
[ $n -eq 428 ] || exit 1
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
echo "START twohead train (budget-free, full / -avis) + mix0.5 schedules gpu$G $(date)" >> $Q
CUDA_VISIBLE_DEVICES=$G $PY $OO/train.py --suite $S --label_dir $LD --seed 0 --two_head --mix 0.5 \
  --out_dir $OUT/${S}_twohead > $O/ablation/twohead.log 2>&1 &
CUDA_VISIBLE_DEVICES=$G $PY $OO/train.py --suite $S --label_dir $LD --seed 0 --two_head --mix 0.5 --drop avis \
  --out_dir $OUT/${S}_twohead_no_avis > $O/ablation/twohead_no_avis.log 2>&1 &
(cd $OO && for r in 0.9140625 0.896484375; do
   $PY make_schedule.py --suite $S --fastv_r $r --mix 0.5 --tag _official >> $O/ablation/schedules_mix0.5.log 2>&1; done)
wait
echo "END goal twohead: full $(tail -n 1 $O/ablation/twohead.log) | no_avis $(tail -n 1 $O/ablation/twohead_no_avis.log) | $(ls /workspace/vla/data/oft_predictor/schedules/ | grep goal_official | tr '\n' ' ') $(date)" >> $Q
