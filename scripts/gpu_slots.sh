#!/bin/bash
# nproc_gpu <gpu>: number of python processes bound to that GPU (CUDA_VISIBLE_DEVICES=<gpu>), counted from /proc so a
# process that is still loading data (not yet on the GPU) is counted too. Used to keep at most 3 processes per GPU.
nproc_gpu() {
  local n=0 p
  for p in $(pgrep -f "bin/python"); do
    tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && n=$((n+1))
  done
  echo $n
}
