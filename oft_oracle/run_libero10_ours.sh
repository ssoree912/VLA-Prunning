#!/bin/bash
# libero_10, ours (two-head -avis): after training (run_suite_ours.sh on GPU 3), 200 episodes for each head weight
# w = 0.4 / 0.5 / 0.6 with the matching mixed-label schedule (20%, b3280), then the best w extended to 500
# (init 20-49). GPU 3 runs w0.5 then w0.4; GPU 1 runs w0.6 once its goal job (ours w0.5 init 20-49) ends.
OO=/workspace/vla/oracle_pruning/oft_oracle; PY=/opt/conda/envs/openvla-oft/bin/python
Q=/workspace/vla/eval_logs/oft_bench500/status.txt; QL=/workspace/vla/eval_logs/oft_10200/status.txt
SD=/workspace/vla/data/oft_predictor/schedules; O=/workspace/vla/eval_logs/oft_bench500/libero_10
while ! grep -q "END libero_10 twohead" $QL 2>/dev/null; do sleep 120; done
for w in 0.4 0.6; do [ -f $SD/libero_10_official_mix${w}_m2_b3280.json ] || (cd $OO && OMP_NUM_THREADS=12 $PY make_schedule.py \
  --suite libero_10 --fastv_r 0.9140625 --mix $w --tag _official >> /workspace/vla/eval_logs/oft_10200/ablation/schedules_mix_sweep.log 2>&1) & done; wait
run() {  # gpu w offset trials
  echo "START ours libero_10 two-head -avis w$2 (schedule mix$2) offset $3 trials $4 gpu$1 $(date)" >> $Q
  TAG_SUFFIX=_w$2 OURS_W=$2 OURS_SCHED=$SD/libero_10_official_mix$2_m2_b3280.json bash $OO/bench500.sh $1 ours libero_10 $3 $4
}
( run 3 0.5 0 20; run 3 0.4 0 20 ) &
( while [ $(grep -c "END ours_libero_goal_i20_t" $Q) -lt 3 ]; do sleep 60; done; run 1 0.6 0 20 ) &
wait
best=$(for w in 0.4 0.5 0.6; do echo "$(cat $O/ours_w${w}_libero_10_t*.jsonl | grep -v '"q"' | grep -c '"success": true') $w"; done | sort -rn | head -n 1)
echo "BEST libero_10 200ep: $best (success w) -> extend to 500 $(date)" >> $Q
run 3 ${best#* } 20 30
