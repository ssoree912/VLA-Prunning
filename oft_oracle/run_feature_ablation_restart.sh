#!/bin/bash
# Restart of run_feature_ablation.sh for full / -hv / -hq / -pos with a CPU thread cap: 5 concurrent train.py runs with
# default threads (96 each) saturated the CPUs (load ~220, GPU idle). The -avis run of the first launch finished as is.
S=libero_spatial; LD=/workspace/vla/data/oft_labels/$S; OUT=/workspace/vla/data/oft_predictor/official
O=/workspace/vla/eval_logs/oft_spatial200/ablation; Q=/workspace/vla/eval_logs/oft_spatial200/status.txt
PY=/opt/conda/envs/openvla-oft/bin/python; TR=/workspace/vla/oracle_pruning/oft_oracle/train.py
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
echo "RESTART feature ablation full/-hv/-hq/-pos with OMP_NUM_THREADS=12 (CPU oversubscription) $(date)" >> $Q
train() {  # gpu name [drop...]
  local G=$1 N=$2; shift 2
  CUDA_VISIBLE_DEVICES=$G $PY $TR --suite $S --label_dir $LD --fastv_r 0.9140625 --seed 0 --out_dir $OUT/${S}_$N \
    --schedule_from $OUT/${S}_full/schedule.json ${1:+--drop "$@"} > $O/$N.log 2>&1
  echo "END ablation $N exit=$? $(tail -n 1 $O/$N.log) $(date)" >> $Q
}
train 0 full & sleep 30
train 0 no_hv hv & sleep 30
train 1 no_hq hq & sleep 30
train 1 no_pos pos &
wait
echo "END feature ablation $(date)" >> $Q
