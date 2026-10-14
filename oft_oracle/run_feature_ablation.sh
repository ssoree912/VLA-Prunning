#!/bin/bash
# Input-feature ablation of the OFT ranker on the official-stack labels (label_rlds_official.py), LIBERO-Spatial:
# full / -avis / -hv / -hq / -pos, same seed, same 20% schedule (b3280, computed once by the full run).
# Waits until all 432 label files exist and the labelers have exited. Offline metrics only (train.py val log).
# Usage: bash run_feature_ablation.sh <gpu_a> <gpu_b>
GA=$1; GB=$2
S=libero_spatial; LD=/workspace/vla/data/oft_labels/$S; OUT=/workspace/vla/data/oft_predictor/official
O=/workspace/vla/eval_logs/oft_spatial200/ablation; mkdir -p $O $OUT; Q=/workspace/vla/eval_logs/oft_spatial200/status.txt
PY=/opt/conda/envs/openvla-oft/bin/python; TR=/workspace/vla/oracle_pruning/oft_oracle/train.py
while [ $(ls $LD | grep -c '^ep.*\.npz$') -lt 432 ] || pgrep -f "label_rlds_official.py --suite $S" > /dev/null; do sleep 60; done
echo "START feature ablation (full/-avis/-hv/-hq/-pos) gpu$GA,$GB $(date)" >> $Q
train() {  # gpu name [drop...]
  local G=$1 N=$2; shift 2
  CUDA_VISIBLE_DEVICES=$G $PY $TR --suite $S --label_dir $LD --fastv_r 0.9140625 --seed 0 --out_dir $OUT/${S}_$N \
    ${SCHED:+--schedule_from $SCHED} ${1:+--drop "$@"} > $O/$N.log 2>&1
  echo "END ablation $N exit=$? $(tail -n 1 $O/$N.log) $(date)" >> $Q
}
train $GA full &
until [ -f $OUT/${S}_full/schedule.json ]; do sleep 20; done
SCHED=$OUT/${S}_full/schedule.json
train $GA no_avis avis & sleep 60
train $GB no_hv hv & sleep 60
train $GB no_hq hq & sleep 60
train $GA no_pos pos &
wait
echo "END feature ablation $(date)" >> $Q
