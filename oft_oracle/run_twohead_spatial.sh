#!/bin/bash
# Two-head ranker (action / instruction heads) on the official-stack spatial labels, 50/50:
#   train.py --two_head (budget-free, checkpoint = lowest val KL)  ||  make_schedule.py --mix 0.5 (DP on the mixed label,
#   20% budget b3280)  ->  200 eps on GPU 3 with the heads combined at w = 0.5. Full inputs.
G=3; S=libero_spatial; O=/workspace/vla/eval_logs/oft_spatial200; Q=$O/status.txt
OUT=/workspace/vla/data/oft_predictor/official/${S}_twohead
SCHED=/workspace/vla/data/oft_predictor/schedules/${S}_official_mix0.5_m2_b3280.json
PY=/opt/conda/envs/openvla-oft/bin/python; OO=/workspace/vla/oracle_pruning/oft_oracle
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
echo "START twohead train (budget-free) + mix0.5 schedule gpu$G $(date)" >> $Q
CUDA_VISIBLE_DEVICES=$G $PY $OO/train.py --suite $S --label_dir /workspace/vla/data/oft_labels/$S --seed 0 --two_head \
  --mix 0.5 --out_dir $OUT > $O/ablation/twohead.log 2>&1 &
[ -f $SCHED ] || (cd $OO && $PY make_schedule.py --suite $S --fastv_r 0.9140625 --mix 0.5 --tag _official \
  > $O/ablation/schedule_mix0.5.log 2>&1)
wait
echo "END twohead train $(tail -n 1 $O/ablation/twohead.log) | schedule $(ls $SCHED) $(date)" >> $Q
[ -f $OUT/best.pt ] && [ -f $SCHED ] || exit 1
cd $OO && OMP_NUM_THREADS=8 bash run_ours_official.sh $G $OUT/best.pt $SCHED offnew_twohead_w0.5_b3280 0.5
