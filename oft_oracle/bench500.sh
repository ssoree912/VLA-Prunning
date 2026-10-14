#!/bin/bash
# OFT LIBERO benchmark runs (official settings, seed 7): one method x one suite, 3 task shards on one GPU, shared
# FLOPs/latency profiler, per-episode jsonl. Standard protocol = 50 trials x 10 tasks = 500 episodes; TRIALS / OFFSET
# run a part of it (e.g. OFFSET=20 TRIALS=30 extends a finished 200-episode run to 500).
# Usage: bash bench500.sh <gpu> <full|ours|specprune|vlapruner|vlacache> <libero_spatial|libero_object|libero_goal|libero_10>
#        [offset=0] [trials=50] [shards]
#   shards: "name:task_ids name:task_ids ..." (default 3 shards t0123 t456 t789; vlapruner / vlacache 2 shards,
#   3 of them do not fit in 49 GB), e.g. "t789:7,8,9" to redo one shard
# Each method runs from its own repo (cwd + PYTHONPATH head) on its own checkpoint dir with a unique basename (get_vla
# rewrites modeling files in it; HF caches remote code by dir basename):
#   full       official OFT (openvla-oft-official) + transformers fork v4.40.1-openvla-oft, run_eval_official.py (no predictor)
#   ours       same stack + our predictor (env OURS_CKPT, OURS_SCHED, OURS_W; default: spatial two-head -avis, mix-0.5
#              schedule b3280, w 0.5)
#   specprune  SpecPrune-VLA/openvla-oft (official default settings) + the same fork, shard_eval.py;
#              specprune_a<alpha>: same with SPEC_ALPHA=<alpha> (STATIC_PRUNE_RATIO; released default 0.8 keeps ~20% of
#              the visual tokens and fails on libero_10, cf. the repo's issue #1);
#              specprune_nodyn: released settings with dynamic pruning off (spec_diag.py, SPEC_FLAGS) -- ~26% retention,
#              LLM FLOPs ~37%, close to the paper's operating point (the author's suggestion in issue #1)
#   vlapruner  VLA-Pruner/src/openvla-oft (its own transformers 4.47 in the env), run_eval.py, 20% budget:
#              fastv_k 3, fastv_r 0.9140625 (k = 44 -> 3280 token-layers, as ours), layer 15, semantic_action
#   vlacache   vla-cache/src/openvla-oft + its transformers branch vla-cache-openvla-oft, shard_eval.py, official default
GPU=$1; M=$2; S=$3; OFFSET=${4:-0}; TRIALS=${5:-50}; SHARDS=$6
CK=openvla-7b-oft-finetuned-$(echo $S | tr '_' '-')
B=/workspace/vla/baselines; OO=/workspace/vla/oracle_pruning/oft_oracle
PY=/opt/conda/envs/openvla-oft/bin/python; PROF=$B/tools/vla_profile.py
TF_OFT=$B/_transformers_oft/src:$B/_pydeps_oft
OUTROOT=${OUTROOT:-/workspace/vla/eval_logs/oft_bench500}  # env OUTROOT: other output root (e.g. latency runs)
O=$OUTROOT/$S; mkdir -p $O; Q=$OUTROOT/status.txt
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU TF_CPP_MIN_LOG_LEVEL=3 \
  OMP_NUM_THREADS=8 OFT_INIT_OFFSET=$OFFSET OFT_METHOD=$M
unset OFT_PREDICTOR_CKPT OFT_SCHEDULE OFT_QUERY_SPAN SPEC_TASK_IDS SPEC_INIT_OFFSET
case $M in
  full)
    R=$B/openvla-oft-official; export PYTHONPATH=$R:$TF_OFT
    SCRIPT="$PROF %P $OO/run_eval_official.py"; CKPT=/workspace/vla/models/oft-ours/$CK-ours; ARGS="" ;;
  ours)
    R=$B/openvla-oft-official; export PYTHONPATH=$R:$TF_OFT
    P=/workspace/vla/data/oft_predictor
    export OFT_PREDICTOR_CKPT=${OURS_CKPT:-$P/official/${S}_twohead_no_avis/best.pt}
    export OFT_SCHEDULE=${OURS_SCHED:-$P/schedules/${S}_official_mix0.5_m2_b3280.json} OFT_HEAD_WEIGHT=${OURS_W:-0.5}
    SCRIPT="$PROF %P $OO/run_eval_official.py"; CKPT=/workspace/vla/models/oft-ours/$CK-ours; ARGS="" ;;
  specprune|specprune_a*|specprune_nodyn)
    case $M in specprune_a*) export SPEC_ALPHA=${M#specprune_a};; specprune_nodyn) export SPEC_FLAGS=dynamic_prune=0;; esac
    R=$B/SpecPrune-VLA/openvla-oft; export PYTHONPATH=$R:$TF_OFT
    SCRIPT="$PROF %P $OO/$([ $M = specprune_nodyn ] && echo spec_diag.py || echo shard_eval.py)"; CKPT=/workspace/vla/models/oft-specprune/$CK-specprune; ARGS="" ;;
  vlapruner)
    R=/workspace/vla/VLA-Pruner/src/openvla-oft; unset PYTHONPATH
    SCRIPT="$OO/profile_eval.py %P $OO/run_eval.py"; CKPT=$R/checkpoints/$CK
    ARGS="--use_vla_cache False --use_fastv False --use_vla_pruner True --fastv_k 3 --fastv_r 0.9140625
      --vla_pruner_layer 15 --vla_pruner_mode semantic_action"
    SHARDS=${SHARDS:-"t01234:0,1,2,3,4 t56789:5,6,7,8,9"} ;;
  vlacache)
    R=$B/vla-cache/src/openvla-oft; export PYTHONPATH=$R:$B/_transformers_vlacache_oft/src
    SCRIPT="$PROF %P $OO/shard_eval.py"; CKPT=$R/checkpoints/$CK-vlacache; ARGS="--use_vla_cache True"
    SHARDS=${SHARDS:-"t01234:0,1,2,3,4 t56789:5,6,7,8,9"} ;;
  *) echo "unknown method $M"; exit 1 ;;
esac
[ -e $CKPT/model.safetensors.index.json ] || { echo "missing checkpoint $CKPT"; exit 1; }
cd $R
NAME=${M}${TAG_SUFFIX}_$S$([ $OFFSET -gt 0 ] && echo _i$OFFSET)  # env TAG_SUFFIX: e.g. _w0.4
run() {  # shard task_ids
  local TAG=${NAME}_$1
  echo "START $TAG gpu$GPU $(date)" >> $Q
  OFT_RESULTS_JSONL=$O/$TAG.jsonl OFT_TASK_IDS=$2 $PY ${SCRIPT/\%P/$O/$TAG.prof.jsonl} --pretrained_checkpoint $CKPT \
    --task_suite_name $S --seed 7 --num_trials_per_task $TRIALS --local_log_dir $O --run_id_note $TAG $ARGS \
    > $O/$TAG.out 2>&1
  echo "END $TAG exit=$? $(grep -c '"success": true' $O/$TAG.jsonl)/$(wc -l < $O/$TAG.jsonl) $(date)" >> $Q
}
for sh in ${SHARDS:-t0123:0,1,2,3 t456:4,5,6 t789:7,8,9}; do run ${sh%%:*} ${sh#*:} & sleep 30; done
wait
