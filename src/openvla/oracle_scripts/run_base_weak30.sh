#!/bin/bash
# Base OpenVLA (user): our new ranker (two-head, no avis/ains, PCA 256, w 0.35) at the 30% budget (b2450) on the tasks
# where the old base predictor (std500 p20 runs) fell short of Full, 50 trials (init 0-49) each, then -- if close to
# Full -- the remaining tasks of that suite to complete 500 episodes.
#   spatial tasks 1,4   (old 37,28 / Full 48,34 of 50)          GPU 1
#   object  tasks 0,1,3 (old 37,28,30 / Full 40,34,36 of 50)    GPU 2
#   goal    tasks 0,3,9 (old 22,19,21 / 50; no Full 500 -> Full also run on these tasks)  GPU 0 (ours + Full)
# "close to Full": ours on the weak tasks >= Full - 1 episode per task (e.g. >= Full - 2 for 2 tasks).
R=/workspace/vla/oracle_pruning/VLA-Pruner/src/openvla; cd $R; OS=$R/oracle_scripts; W=0.35; B=2450; RHO=0.7734375
LOG=$R/experiments/logs/oracle; Q=$LOG/base_noavis256_status.txt; PROF=/workspace/vla/eval_logs/base_weak30; mkdir -p $PROF
source $OS/common.sh
succ() { grep -c "Success: True" $1 2>/dev/null; }
ours() {  # gpu suite tasks tag
  local S=$2 T=base30_${4}_$2
  echo "START $T tasks $3 gpu$1 $(date)" >> $Q
  CKPT=checkpoints/openvla-7b-finetuned-$(echo $S | tr '_' '-') SUITE=$S RHO_R=$RHO PROF_OUT=$PROF/$T.prof.jsonl \
    PRED_CKPT=/workspace/vla/data/oracle_predictor_mix/${S}_m2_mix${W}_s0_noavis_pca256/A_2head/best.pt \
    ORACLE_HEAD_WEIGHT=$W ORACLE_SCHEDULE=$OS/schedules/${S}_m2_w${W}_b$B.json \
    run_method $1 predictor $T --task_ids $3 --num_trials_per_task 50 --init_state_offset 0 --save_video False > $LOG/$T.out 2>&1
  echo "END $T exit=$? $(succ $LOG/$T.out)/$(grep -c 'Success:' $LOG/$T.out) $(date)" >> $Q
}
full() {  # gpu suite tasks tag
  local S=$2 T=basefull_${4}_$2
  echo "START $T tasks $3 gpu$1 $(date)" >> $Q
  CKPT=checkpoints/openvla-7b-finetuned-$(echo $S | tr '_' '-') SUITE=$S PROF_OUT=$PROF/$T.prof.jsonl \
    run_method $1 full $T --task_ids $3 --num_trials_per_task 50 --init_state_offset 0 --save_video False > $LOG/$T.out 2>&1
  echo "END $T exit=$? $(succ $LOG/$T.out)/$(grep -c 'Success:' $LOG/$T.out) $(date)" >> $Q
}
decide() {  # gpu suite weak_tasks rest_tasks full_successes n_tasks
  local o=$(succ $LOG/base30_weak_$2.out)
  if [ $o -ge $(( $5 - $6 )) ]; then
    echo "DECIDE $2: ours $o vs Full $5 on weak tasks -> run remaining tasks $4 (to 500) $(date)" >> $Q
    ours $1 $2 $4 rest
  else
    echo "DECIDE $2: ours $o vs Full $5 on weak tasks -> NOT close, stop $(date)" >> $Q
  fi
}
( ours 1 libero_spatial 1,4 weak; decide 1 libero_spatial 1,4 0,2,3,5,6,7,8,9 82 2 ) &
( ours 2 libero_object 0,1,3 weak; decide 2 libero_object 0,1,3 2,4,5,6,7,8,9 110 3 ) &
( full 0 libero_goal 0,3,9 weak & ours 0 libero_goal 0,3,9 weak; wait
  decide 0 libero_goal 0,3,9 1,2,4,5,6,7,8 $(succ $LOG/basefull_weak_libero_goal.out) 3 ) &
wait
echo "END base weak-task 30% runs $(date)" >> $Q
