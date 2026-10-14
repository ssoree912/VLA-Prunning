#!/bin/bash
# Add the instruction label (s_instr) to the OFT label files while the main labeler (pid $2) is still writing them:
# repeated --add_instr passes (each skips files that already have s_instr) until the labeler is done and all files have it.
GPU=$1; LABEL_PID=$2; S=${3:-libero_spatial}
O=/workspace/vla/eval_logs/oft_spatial200; Q=$O/status.txt
cd /workspace/vla/VLA-Pruner/src/openvla-oft
export CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1
alive() { [ -e /proc/$LABEL_PID ] && [ "$(awk '{print $3}' /proc/$LABEL_PID/stat 2>/dev/null)" != Z ]; }
echo "START add_instr $S gpu$GPU $(date)" >> $Q
while true; do
  done_before=$(alive && echo 0 || echo 1)
  /opt/conda/envs/openvla-oft/bin/python /workspace/vla/oracle_pruning/oft_oracle/label_rlds.py --suite $S --add_instr \
    >> $O/add_instr_$S.log 2>&1
  [ $done_before = 1 ] && break
  alive || continue   # labeler finished during this pass: one more pass covers its last files
  sleep 120
done
n=$(/opt/conda/envs/openvla-oft/bin/python -c "import glob,numpy as np;f=glob.glob('/workspace/vla/data/oft_labels/$S/ep*.npz');print(sum('s_instr' in np.load(x).files for x in f),'/',len(f))")
echo "END add_instr $S: $n files with s_instr $(date)" >> $Q
