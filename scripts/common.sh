#!/bin/bash
# Shared settings for the GazePrune OFT scripts. Roots default to the original server layout; override with env vars
# (the same ones gazeprune/paths.py reads).
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GAZEPRUNE_ROOT=${GAZEPRUNE_ROOT:-/workspace/vla}
GAZEPRUNE_DATA=${GAZEPRUNE_DATA:-$GAZEPRUNE_ROOT/data}
GAZEPRUNE_MODELS=${GAZEPRUNE_MODELS:-$GAZEPRUNE_ROOT/models}
GAZEPRUNE_LOGS=${GAZEPRUNE_LOGS:-$GAZEPRUNE_ROOT/eval_logs}
GAZEPRUNE_BASELINES=${GAZEPRUNE_BASELINES:-$GAZEPRUNE_ROOT/baselines}
export GAZEPRUNE_ROOT GAZEPRUNE_DATA GAZEPRUNE_MODELS GAZEPRUNE_LOGS GAZEPRUNE_BASELINES
PY=${PY:-/opt/conda/envs/openvla-oft/bin/python}
VLAPRUNER_OFT=${VLAPRUNER_OFT:-$GAZEPRUNE_ROOT/VLA-Pruner/src/openvla-oft}  # VLA-Pruner's OFT code (+ checkpoints)
OFT_REPO=$GAZEPRUNE_BASELINES/openvla-oft-official
TF_OFT=$GAZEPRUNE_BASELINES/_transformers_oft/src:$GAZEPRUNE_BASELINES/_pydeps_oft  # transformers v4.40.1-openvla-oft
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3
ckpt_name() { echo "openvla-7b-oft-finetuned-$(echo $1 | tr '_' '-')"; }        # <suite> -> checkpoint dir name
oft_env() { cd $OFT_REPO && export PYTHONPATH=$OFT_REPO:$TF_OFT; }               # official OFT stack
source $REPO/scripts/gpu_slots.sh
