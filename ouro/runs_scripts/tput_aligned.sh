#!/bin/bash
# tput_aligned.sh GPU: decode throughput of the aligned loop adapter (a<p><d>sb: one static prompt bank per decode loop,
# single_bank.AlignedRunner, checked against the evaluation runner by ouro/scripts/check_aligned_runner.py) at prefill 4 /
# decode 1..4 and an 8K prompt, as tput_sb.sh measures the one-cache adapter and the native model. One job at a time on
# GPU (exclusive). Output: runs/throughput-sb4/a4<d>sb.json.
G=$1; M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/tput_aligned_q.txt; cd $R
for c in a41sb a42sb a43sb a44sb; do
  echo "TPUT_DIR=throughput-sb4 CKPT=$R/X26-a-universal/checkpoint-48m.pt /root/big-small/ouro/runs_scripts/tput_sb.sh \$GPU $c"
done | flock $Q.lock sh -c "cat >> $Q"
setsid nohup ./grid_worker.sh $G $Q 1 > /dev/null 2>&1 < /dev/null &
