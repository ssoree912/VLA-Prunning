#!/bin/bash
# Our predictor for one LIBERO suite on the official OFT stack: official labels (label_rlds_official.py, 2 shards on one
# GPU, resumable), then budget-free two-head training (mix 0.5; full inputs and -avis) and the mix-0.5 schedule at the
# 20% budget (b3280). No rollout eval.
# Usage: bash run_suite_ours.sh <gpu> <libero_object|libero_goal|libero_10|libero_spatial>
G=$1; S=$2; O=/workspace/vla/eval_logs/oft_${S#libero_}200; mkdir -p $O/ablation; Q=$O/status.txt
LD=/workspace/vla/data/oft_labels/$S; OUT=/workspace/vla/data/oft_predictor/official
PY=/opt/conda/envs/openvla-oft/bin/python; OO=/workspace/vla/oracle_pruning/oft_oracle; R=/workspace/vla/baselines/openvla-oft-official
N=$($PY -c "import json;d=json.load(open('/workspace/vla/data/modified_libero_rlds/${S}_no_noops/1.0.0/dataset_info.json'));print(sum(int(x) for s in d['splits'] for x in s['shardLengths']))")
echo "START official labels $S ($N episodes, 2 shards) gpu$G $(date)" >> $Q
(cd $R && export PYTHONPATH=$R:/workspace/vla/baselines/_transformers_oft/src:/workspace/vla/baselines/_pydeps_oft \
   CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 MUJOCO_GL=osmesa OMP_NUM_THREADS=8 &&
 for s in 0 1; do $PY $OO/label_rlds_official.py --suite $S --num_shards 2 --shard $s >> $O/official_labels_shard$s.log 2>&1 & sleep 20; done; wait)
n=$(ls $LD | grep -c '^ep.*\.npz$')
echo "END official labels $S: $n / $N episodes $(date)" >> $Q
[ $n -eq $N ] || exit 1
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
echo "START twohead train $S (budget-free, full / -avis) + mix0.5 schedule gpu$G $(date)" >> $Q
CUDA_VISIBLE_DEVICES=$G $PY $OO/train.py --suite $S --label_dir $LD --seed 0 --two_head --mix 0.5 \
  --out_dir $OUT/${S}_twohead > $O/ablation/twohead.log 2>&1 &
CUDA_VISIBLE_DEVICES=$G $PY $OO/train.py --suite $S --label_dir $LD --seed 0 --two_head --mix 0.5 --drop avis \
  --out_dir $OUT/${S}_twohead_no_avis > $O/ablation/twohead_no_avis.log 2>&1 &
SC=/workspace/vla/data/oft_predictor/schedules/${S}_official_mix0.5_m2_b3280.json
[ -f $SC ] || (cd $OO && $PY make_schedule.py --suite $S --fastv_r 0.9140625 --mix 0.5 --tag _official > $O/ablation/schedule_mix0.5.log 2>&1)
wait
echo "END $S twohead: full $(tail -n 1 $O/ablation/twohead.log) | no_avis $(tail -n 1 $O/ablation/twohead_no_avis.log) | $(ls $SC) $(date)" >> $Q
