#!/bin/bash
# Waits for the OFT object / libero_10 checkpoint downloads, makes the per-method checkpoint dirs (small files copied,
# weights symlinked, unique basenames), then starts the queue workers on GPUs 0, 1, 3 (each after its current run).
SRC=/workspace/vla/VLA-Pruner/src/openvla-oft/checkpoints; D=/workspace/vla/eval_logs/oft_bench500; Q=$D/status.txt
OO=/workspace/vla/oracle_pruning/oft_oracle
until grep -q "^DONE" /workspace/vla/eval_logs/oft_download/libero_object.log && grep -q "^DONE" /workspace/vla/eval_logs/oft_download/libero_10.log; do sleep 30; done
mk() {  # src_dir dst_dir
  mkdir -p $2
  for f in $1/*; do b=$(basename $f); case $b in *.safetensors|*checkpoint.pt|lora_adapter|README.md) ln -sfn $f $2/$b;; *.back.*) ;; *) cp -r $f $2/$b;; esac; done
}
for S in libero_spatial libero_object libero_goal libero_10; do
  CK=openvla-7b-oft-finetuned-$(echo $S | tr '_' '-')
  [ -d /workspace/vla/models/oft-ours/$CK-ours ] || mk $SRC/$CK /workspace/vla/models/oft-ours/$CK-ours
  [ -d /workspace/vla/models/oft-specprune/$CK-specprune ] || mk $SRC/$CK /workspace/vla/models/oft-specprune/$CK-specprune
  [ -d /workspace/vla/baselines/vla-cache/src/openvla-oft/checkpoints/$CK-vlacache ] || mk $SRC/$CK /workspace/vla/baselines/vla-cache/src/openvla-oft/checkpoints/$CK-vlacache
done
echo "SETUP checkpoints ready $(date)" >> $Q
S2=/workspace/vla/eval_logs/oft_spatial200/status.txt
nohup bash $OO/bench_queue.sh 0 $S2 "END oft200_specprune_i20" 3 > /dev/null 2>&1 &
nohup bash $OO/bench_queue.sh 3 $S2 "END oft200_offic_full_i20" 3 > /dev/null 2>&1 &
nohup bash $OO/bench_queue.sh 1 $S2 "END oft200_offnew_twohead_noavis_w0.5_b3280_i20" 3 > /dev/null 2>&1 &
