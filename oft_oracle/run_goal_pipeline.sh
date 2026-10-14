#!/bin/bash
# OFT LIBERO-Goal, our method: labels with the instruction term (2 shards on one GPU) -> two-head ranker (mix 0.5) ->
# 200-episode eval at the SpecPrune-matched budget (SpecPrune goal measured 21.5%: k=53 -> 3532 / 16384 = 21.56%),
# instruction weight w = 0.4 (the base OpenVLA goal setting), 3 task shards with the profiler.
GPU=${1:-0}
S=libero_goal; PY=/opt/conda/envs/openvla-oft/bin/python
O=/workspace/vla/eval_logs/oft_goal200; mkdir -p $O; Q=$O/status.txt
OO=/workspace/vla/oracle_pruning/oft_oracle
CKPT=/workspace/vla/VLA-Pruner/src/openvla-oft/checkpoints/openvla-7b-oft-finetuned-libero-goal
PRED=/workspace/vla/data/oft_predictor/libero_goal_m2_2head_s0
cd /workspace/vla/VLA-Pruner/src/openvla-oft
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU
echo "START ours labels (action + instruction, 2 shards) gpu$GPU $(date)" >> $Q
for sh in 0 1; do
  $PY $OO/label_rlds.py --suite $S --checkpoint $CKPT --num_shards 2 --shard $sh > $O/ours_labels_shard$sh.log 2>&1 &
  sleep 60
done
wait
echo "END ours labels $(ls /workspace/vla/data/oft_labels/$S/ep*.npz | wc -l) demos $(date)" >> $Q
$PY $OO/train.py --suite $S --two_head --mix 0.5 --fastv_r 0.896484375 --epochs 20 --out_dir $PRED > $O/ours_train.log 2>&1
echo "END ours train exit=$? $(grep '^best' $O/ours_train.log) $(date)" >> $Q
[ -f $PRED/best.pt ] || exit 1
SUITE=$S bash $OO/run_ours_eval.sh $GPU $PRED/best.pt "" ours_2head_w0.4_b3532 0.4
