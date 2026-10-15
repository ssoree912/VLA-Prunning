#!/bin/bash
# Base OpenVLA queue (user), after the running 30% runs: GPUs 0, 1, 2 (GPU 3 is someone else's), at most 3 processes
# per GPU (VLA-Cache at most 2: its cache does not fit 3 per GPU). Jobs in priority order:
#   1) ours 30% (b2450, w 0.35) libero_10 init 10-49 -> with the 100-ep run (init 0-9) = 500
#   2) Full 500 where missing: object (task 5 init 38-49 + tasks 6-9, rest done), goal, libero_10
#   3) VLA-Cache 500 where missing: object, goal, libero_10 (official default; spatial done)
# Each job is one process over a task subset (sharded so GPUs fill up). Results: $LOG/<tag>.out / EVAL-*--<tag>*.txt
R=/workspace/vla/oracle_pruning/VLA-Pruner/src/openvla; cd $R; OS=$R/oracle_scripts; LOG=$R/experiments/logs/oracle
Q=$LOG/base_noavis256_status.txt; J=$LOG/base_queue_jobs.txt; PROF=/workspace/vla/eval_logs/base500; mkdir -p $PROF
VC=/workspace/vla/baselines/vla-cache
source /workspace/vla/oracle_pruning/oft_oracle/gpu_slots.sh
cat > $J <<'JOBS'
ours libero_10 0,1,2,3,4 40 10 base30_i10a_libero_10
ours libero_10 5,6,7,8,9 40 10 base30_i10b_libero_10
full libero_object 5 12 38 full500_t5i38_libero_object
full libero_object 6,7,8,9 50 0 full500_t6789_libero_object
full libero_goal 0,1,2,3,4 50 0 full500_t01234_libero_goal
full libero_goal 5,6,7,8,9 50 0 full500_t56789_libero_goal
vlacache libero_object 0,1,2,3,4 50 0 vlacache500_t01234_libero_object
vlacache libero_object 5,6,7,8,9 50 0 vlacache500_t56789_libero_object
vlacache libero_goal 0,1,2,3,4 50 0 vlacache500_t01234_libero_goal
vlacache libero_goal 5,6,7,8,9 50 0 vlacache500_t56789_libero_goal
full libero_10 0,1,2,3,4 50 0 full500_t01234_libero_10
full libero_10 5,6,7,8,9 50 0 full500_t56789_libero_10
vlacache libero_10 0,1,2,3,4 50 0 vlacache500_t01234_libero_10
vlacache libero_10 5,6,7,8,9 50 0 vlacache500_t56789_libero_10
JOBS
run_job() {  # gpu method suite tasks trials offset tag
  local G=$1 M=$2 S=$3 T=$4 N=$5 OFF=$6 TAG=$7 CK=checkpoints/openvla-7b-finetuned-$(echo $3 | tr '_' '-')
  echo "START $TAG ($M $S tasks $T trials $N offset $OFF) gpu$G $(date)" >> $Q
  case $M in
    ours) ( source $OS/common.sh; CKPT=$CK SUITE=$S RHO_R=0.7734375 PROF_OUT=$PROF/$TAG.prof.jsonl ORACLE_HEAD_WEIGHT=0.35 \
            PRED_CKPT=/workspace/vla/data/oracle_predictor_mix/${S}_m2_mix0.35_s0_noavis_pca256/A_2head/best.pt \
            ORACLE_SCHEDULE=$OS/schedules/${S}_m2_w0.35_b2450.json \
            run_method $G predictor $TAG --task_ids $T --num_trials_per_task $N --init_state_offset $OFF --save_video False ) ;;
    full) ( source $OS/common.sh; CKPT=$CK SUITE=$S PROF_OUT=$PROF/$TAG.prof.jsonl \
            run_method $G full $TAG --task_ids $T --num_trials_per_task $N --init_state_offset $OFF --save_video False ) ;;
    vlacache) ( cd $VC/src/openvla; export PYTHONPATH=$VC/src/openvla:$VC/_transformers_openvla/src:$VC/_pydeps \
            MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa VLACACHE_TASK_IDS=$T VLACACHE_INIT_OFFSET=$OFF
            CUDA_VISIBLE_DEVICES=$G /opt/conda/envs/openvla/bin/python /workspace/vla/baselines/tools/vla_profile.py \
              $PROF/$TAG.prof.jsonl experiments/robot/libero/run_libero_eval.py --pretrained_checkpoint $CK \
              --task_suite_name $S --use_vla_cache True --num_trials_per_task $N --seed 7 --run_id_note $TAG ) ;;
  esac > $LOG/$TAG.out 2>&1
  echo "END $TAG exit=$? $(grep -c 'Success: True' $LOG/$TAG.out)/$(grep -c 'Success:' $LOG/$TAG.out) $(date)" >> $Q
}
worker() {  # gpu
  local G=$1
  while [ -s $J ]; do
    local cap=3; head -n 1 $J | grep -q "^vlacache" && cap=2
    while [ $(nproc_gpu $G) -ge $cap ]; do sleep 60; done
    job=$(flock $J.lock bash -c "head -n 1 $J; sed -i 1d $J"); [ -z "$job" ] && break
    run_job $G $job & sleep 120   # let it load before counting again
  done
  wait
}
echo "START base queue (libero_10 ours 500, Full 500, VLA-Cache 500) gpus 0,1,2 $(date)" >> $Q
worker 1 & sleep 40; worker 0 & sleep 40; worker 2 & wait
echo "END base queue $(date)" >> $Q
