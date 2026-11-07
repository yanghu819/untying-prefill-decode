#!/bin/bash
# tput_batch.sh: decode cost per step against batch size at the prompt lengths of the timed tasks (512 and 1,536
# tokens, about GSM8K and MMLU-Pro), for every configuration of the quality-time comparison (Section 5): Original
# 1/1, 2/2, 4/4, Distilled 1/1 and 2/2 (merged) and the aligned adapter at 1/1, 1/2, 4/1, 4/2 (decode cost depends on
# n, not m). With the output lengths of the replayed requests this gives the time per request at batch 8-64, where
# the adapter's small per-loop operations are amortised. Appended to runs/timing_q.txt (exclusive GPUs, MAX=1).
# Output: runs/throughput-batch/<cond>-<ctx>.json (tput_sb.sh format).
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/timing_q.txt; cd $R
A=$R/X26-a-universal/checkpoint-48m.pt; E=$R/X26-E48f-p1d1/checkpoint-48m.pt; D=$R/X26-D48f-p2d2/checkpoint-48m.pt
for x in 512 1536; do
  for c in "native1st -" "native2st -" "native4st -" "D1st $E" "D2st $D" "a1x1sb $A" "a1x2sb $A" "a4x1sb $A" "a4x2sb $A"; do
    set -- $c; K=""; [ $2 != - ] && K="CKPT=$2 "
    echo "${K}TPUT_DIR=throughput-batch BATCHES='1 8 32 64' /root/big-small/ouro/runs_scripts/tput_sb.sh \$GPU $1 $x"
  done
done | flock $Q.lock sh -c "cat >> $Q"
