#!/bin/bash
# tput_ctx.sh GPU...: decode throughput against prompt length (2K and 15K besides the 8K of the paper's throughput
# table) for native 1/1, 2/2, 4/4 and the loop adapter at prefill 4 / decode 1 and 2, one exclusive GPU per run
# (grid_worker.sh with MAX=1). Output: runs/throughput-sb4/<cond>-<ctx>.json, as tput_sb.sh.
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/tput_q.txt; cd $R
for x in 2048 15360; do
  for c in native1st native2st native4st S41sb S42sb; do
    echo "TPUT_DIR=throughput-sb4 /root/big-small/ouro/runs_scripts/tput_sb.sh \$GPU $c $x"
  done
done | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do setsid nohup ./grid_worker.sh $g $Q 1 > /dev/null 2>&1 < /dev/null & done
