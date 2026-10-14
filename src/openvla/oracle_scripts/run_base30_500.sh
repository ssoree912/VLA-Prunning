#!/bin/bash
# User: base OpenVLA, new ranker at 30% (b2450, w 0.35), 500 episodes on all four suites regardless of the weak-task check.
#   spatial: weak tasks done + remaining tasks running (run_base_weak30.sh)            -> nothing to add
#   object / goal: after the weak-task run, start the remaining tasks (50 trials) unless run_base_weak30.sh already did
#   libero_10: init 0-9 done by the 100-ep runs (tasks 0-4 + part2 tasks 5-9) -> init 10-49 (40 trials), 2 task shards
# At most 3 processes per GPU (gpu_slots.sh).
R=/workspace/vla/oracle_pruning/VLA-Pruner/src/openvla; cd $R; OS=$R/oracle_scripts; W=0.35; B=2450; RHO=0.7734375
LOG=$R/experiments/logs/oracle; Q=$LOG/base_noavis256_status.txt; PROF=/workspace/vla/eval_logs/base_weak30
source $OS/common.sh; source /workspace/vla/oracle_pruning/oft_oracle/gpu_slots.sh
ours() {  # gpu suite tasks tag trials offset
  local S=$2 T=base30_${4}_$2
  while [ $(nproc_gpu $1) -ge 3 ]; do sleep 60; done
  echo "START $T tasks $3 trials $5 offset $6 gpu$1 $(date)" >> $Q
  CKPT=checkpoints/openvla-7b-finetuned-$(echo $S | tr '_' '-') SUITE=$S RHO_R=$RHO PROF_OUT=$PROF/$T.prof.jsonl \
    PRED_CKPT=/workspace/vla/data/oracle_predictor_mix/${S}_m2_mix${W}_s0_noavis_pca256/A_2head/best.pt \
    ORACLE_HEAD_WEIGHT=$W ORACLE_SCHEDULE=$OS/schedules/${S}_m2_w${W}_b$B.json \
    run_method $1 predictor $T --task_ids $3 --num_trials_per_task $5 --init_state_offset $6 --save_video False > $LOG/$T.out 2>&1
  echo "END $T exit=$? $(grep -c 'Success: True' $LOG/$T.out)/$(grep -c 'Success:' $LOG/$T.out) $(date)" >> $Q
}
rest_if_needed() {  # gpu suite rest_tasks
  while ! grep -q "END base30_weak_$2 " $Q; do sleep 60; done; sleep 120
  grep -q "START base30_rest_$2 " $Q || ours $1 $2 $3 rest 50 0
}
echo "START base 30% to 500 on all suites (user) $(date)" >> $Q
rest_if_needed 2 libero_object 2,4,5,6,7,8,9 &
rest_if_needed 0 libero_goal 1,2,4,5,6,7,8 &
# libero_10 (init 10-49): later (user: GPU 3 is in use)
wait
echo "END base 30% to 500 $(date)" >> $Q
