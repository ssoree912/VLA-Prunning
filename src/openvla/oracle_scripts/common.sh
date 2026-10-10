#!/bin/bash
# Shared settings for Oracle-Act-TopK vs VLA-Pruner vs Full (oracle_pruning/plan/Oracle.md). Source from src/openvla.
CKPT=${CKPT:-checkpoints/openvla-7b-finetuned-libero-10}
SUITE=${SUITE:-libero_10}
SEED=7
RHO_R=${RHO_R:-0.875}          # fastv_r = 1 - rho; rho=0.125 -> k=32 of M=256
LOG_DIR=${LOG_DIR:-./experiments/logs/oracle}
PY=${PY:-/opt/conda/envs/openvla/bin/python}
export MUJOCO_GL=${MUJOCO_GL:-osmesa} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}  # headless rendering (EGL unavailable here)

PRUNER_ARGS="--use_fastv True --use_temporal True --use_prefil_attention True --use_text_vision_selection False \
  --fastv_k 3 --temporal_w 3 --temporal_gamma 0.8 --fastv_r $RHO_R"

# run_method <gpu> <full|pruner|oracle|oracle_a|oracle_b> <tag> [extra draccus args...]
#   oracle   = Oracle-Act-TopK v0 (score layers 16..31), kept for reproducing earlier runs
#   oracle_a = Oracle A: TopK_k of the layer K..31 score, kept from layer K on
#   oracle_b = Oracle B: per-token depth under the same token-layer budget K*M + (L-K)*k
#   oracle_b_m<m> = Oracle B with no visual token removed before layer m (e.g. oracle_b_m2)
#   predictor = learned ranker at the layer-1 output + fixed schedule ($PRED_CKPT), one forward per query, no teacher
#   oracle_bfix = Oracle B-fixed: fixed per-layer counts from $SCHEDULE, tokens ranked by teacher score (no DP)
run_method() {
  local gpu=$1 method=$2 tag=$3; shift 3
  local margs
  case $method in
    full)   margs="--use_fastv False" ;;
    pruner) margs="$PRUNER_ARGS" ;;
    oracle)   margs="$PRUNER_ARGS --use_oracle True --oracle_mode topk --oracle_layers late" ;;
    oracle_a) margs="$PRUNER_ARGS --use_oracle True --oracle_mode topk --oracle_layers post_k" ;;
    oracle_b) margs="$PRUNER_ARGS --use_oracle True --oracle_mode depth" ;;
    predictor) margs="$PRUNER_ARGS --use_oracle True --oracle_mode predictor --oracle_predictor_ckpt ${PRED_CKPT:-/workspace/vla/data/oracle_predictor/${SUITE}_m2_tf1_s0/best.pt}" ;;
    oracle_b_mix_m2) margs="$PRUNER_ARGS --use_oracle True --oracle_mode depth --oracle_min_depth 2 --oracle_label_mix ${MIX:-0.5}" ;;
    oracle_bfix_mix) margs="$PRUNER_ARGS --use_oracle True --oracle_mode schedule --oracle_schedule ${SCHEDULE:-oracle_scripts/schedules/${SUITE}_m2.json} --oracle_label_mix ${MIX:-0.5}" ;;
    oracle_bfix) margs="$PRUNER_ARGS --use_oracle True --oracle_mode schedule --oracle_schedule ${SCHEDULE:-oracle_scripts/schedules/${SUITE}_m2.json}" ;;
    oracle_b_m*) margs="$PRUNER_ARGS --use_oracle True --oracle_mode depth --oracle_min_depth ${method#oracle_b_m}" ;;
  esac
  CUDA_VISIBLE_DEVICES=$gpu $PY experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint $CKPT --task_suite_name $SUITE --seed $SEED \
    --local_log_dir $LOG_DIR --run_id_note "${tag}_${method}" $margs "$@"
}
