#!/bin/bash
# OpenVLA-OFT LIBERO (SUITE, default libero_spatial), 200 episodes (20 trials x 10 tasks, seed 7), with the shared FLOPs/latency profiler.
# Each method runs from its own official repo (cwd = repo, PYTHONPATH = repo) on its own checkpoint dir
# (/workspace/vla/models/oft-<method>/..., weights symlinked) because get_vla rewrites modeling files in the checkpoint.
# Usage: bash run_oft200.sh <gpu> <adp_static|adp_dynamic|specprune> [extra args...]
GPU=$1; M=$2; shift 2
SUITE=${SUITE:-libero_spatial}; CK=openvla-7b-oft-finetuned-$(echo $SUITE | tr '_' '-')
OUT=/workspace/vla/eval_logs/oft_${SUITE#libero_}200; mkdir -p $OUT
PY=/opt/conda/envs/openvla-oft/bin/python
PROF=/workspace/vla/baselines/tools/vla_profile.py
# official OFT transformers fork (v4.40.1-openvla-oft, bidirectional attn) + tokenizers 0.19, via PYTHONPATH only
TF_OFT=/workspace/vla/baselines/_transformers_oft/src:/workspace/vla/baselines/_pydeps_oft
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU TF_CPP_MIN_LOG_LEVEL=3
TAG=oft200_$M${SHARD:+_$SHARD}  # SHARD=<name> with SPEC_TASK_IDS=<ids> runs a task subset (SpecPrune)
COMMON="--task_suite_name $SUITE --num_trials_per_task 20 --seed 7 --local_log_dir $OUT --run_id_note $TAG"
case $M in
  adp_*)
    R=/workspace/vla/baselines/VLA-ADP; cd $R; export PYTHONPATH=$R:$TF_OFT
    SCRIPT=experiments/robot/libero/run_libero_eval_prune_v2.py
    MARGS="--pretrained_checkpoint /workspace/vla/models/oft-adp/$CK-adp
      --qk_config_json experiments/robot/libero/configs/${M#adp_}_keep20.json" ;;
  specprune)
    R=/workspace/vla/baselines/SpecPrune-VLA/openvla-oft; cd $R; export PYTHONPATH=$R:$TF_OFT
    SCRIPT=experiments/robot/libero/run_libero_eval.py
    MARGS="--pretrained_checkpoint /workspace/vla/models/oft-specprune/$CK-specprune $SPECPRUNE_ARGS" ;;
esac
echo "START $TAG gpu$GPU $(date)" >> $OUT/status.txt
$PY $PROF $OUT/$TAG.prof.jsonl $SCRIPT $COMMON $MARGS "$@" > $OUT/$TAG.out 2>&1
echo "END $TAG exit=$? $(grep "# successes" $OUT/$TAG.out | tail -1) $(date)" >> $OUT/status.txt
