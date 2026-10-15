#!/bin/bash
# Equivalence check of ORACLE_FAST (sync-free kept sets) vs the original depth prefill, base OpenVLA, ours 30%
# (two-head -avis PCA256, w 0.35, b2450): every query run both ways, rollout continues with the fast actions.
# Usage: bash run_fast_check.sh <gpu> <suite> <task_ids> <trials>
G=$1; S=$2; T=$3; N=$4; cd /workspace/vla/oracle_pruning/VLA-Pruner/src/openvla; OS=oracle_scripts; D=/workspace/vla/eval_logs/base_fast_check
rm -f $D/$S.json
CUDA_VISIBLE_DEVICES=$G MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa ORACLE_HEAD_WEIGHT=0.35 ORACLE_SCHEDULE=$OS/schedules/${S}_m2_w0.35_b2450.json \
  /opt/conda/envs/openvla/bin/python $OS/check_fast_equiv.py $D/$S.json experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-$(echo $S | tr '_' '-') --task_suite_name $S --seed 7 --local_log_dir $D \
  --run_id_note fastcheck --use_fastv True --use_temporal True --use_prefil_attention True --use_text_vision_selection False \
  --fastv_k 3 --temporal_w 3 --temporal_gamma 0.8 --fastv_r 0.7734375 --use_oracle True --oracle_mode predictor \
  --oracle_predictor_ckpt /workspace/vla/data/oracle_predictor_mix/${S}_m2_mix0.35_s0_noavis_pca256/A_2head/best.pt \
  --task_ids $T --num_trials_per_task $N --init_state_offset 0 --save_video False > $D/$S.out 2>&1
