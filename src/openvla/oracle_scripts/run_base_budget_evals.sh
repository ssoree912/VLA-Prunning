#!/bin/bash
# Evals of the base OpenVLA budget sweep (replaces the eval part of run_base_budget_sweep_more.sh): one suite per GPU so
# they run in parallel -- goal on GPU 0, object on GPU 2, libero_10 on GPU 3 -- each with its 3 budgets (30/40/50 %),
# 100 episodes. Waits for the suite's ranker and schedules (trainings keep running from the earlier script).
R=/workspace/vla/oracle_pruning/VLA-Pruner/src/openvla; cd $R
OS=$R/oracle_scripts; W=0.35; LOG=$R/experiments/logs/oracle; Q=$LOG/base_noavis256_status.txt
PROF=/workspace/vla/eval_logs/base100_noavis256; source /workspace/vla/oracle_pruning/oft_oracle/gpu_slots.sh
suite_evals() {  # gpu suite
  local G=$1 S=$2 CK=/workspace/vla/data/oracle_predictor_mix/$2_m2_mix${W}_s0_noavis_pca256/A_2head/best.pt
  while ! grep -q "variant A" $LOG/train_${S}_noavis256.log 2>/dev/null || [ $(ls $OS/schedules | grep -c "${S}_m2_w${W}_b") -lt 3 ]; do sleep 60; done
  while [ $(nproc_gpu $G) -gt 0 ]; do sleep 60; done
  ( source $OS/common.sh
    for pair in "0.7734375 2450" "0.66015625 3291" "0.55078125 4103"; do
      set -- $pair; T=base100_noavis256_w${W}_b$2_$S
      echo "START $T gpu$G $(date)" >> $Q
      ( CKPT=checkpoints/openvla-7b-finetuned-$(echo $S | tr '_' '-') SUITE=$S PRED_CKPT=$CK RHO_R=$1 PROF_OUT=$PROF/$T.prof.jsonl \
        ORACLE_HEAD_WEIGHT=$W ORACLE_SCHEDULE=$OS/schedules/${S}_m2_w${W}_b$2.json \
        run_method $G predictor $T --num_trials_per_task 10 --init_state_offset 0 --save_video False > $LOG/${T}.out 2>&1
        echo "END $T exit=$? $(grep '# successes' $LOG/${T}.out | tail -1) $(date)" >> $Q ) &
      sleep 60
    done; wait )
}
echo "START base budget evals goal(gpu0) object(gpu2) libero_10(gpu3) $(date)" >> $Q
suite_evals 0 libero_goal & suite_evals 2 libero_object & suite_evals 3 libero_10 & wait
echo "END base budget evals $(date)" >> $Q
