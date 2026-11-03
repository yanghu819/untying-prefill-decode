#!/bin/bash
# distill_fullsets.sh GPU...: full GSM8K (1,319) and MMLU (14,042) for the distilled controls trained before
# after_x26_distill.sh switched to full sets (E48, D48, H48), queued for grid_worker.sh on the given GPUs.
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/distill_q.txt; cd $R
for C in E48:X26-E48-p1d1 D48:X26-D48-p2d2 H48:X26-H48-p3d3; do
  c=${C%%:*}; [ -f $R/${C#*:}/checkpoint-48m.pt ] || continue        # only the controls trained on this box
  MA="path=$M/model-2.6b,condition=${c:0:1},checkpoint=$R/${C#*:}/checkpoint-48m.pt,max_length=16384"
  echo "./lm1.sh \$GPU o26x-$c-gsmfull - \"$MA\" \"--tasks gsm8k_cot --num_fewshot 3\""
  echo "./lm1.sh \$GPU o26x-$c-mmlufull - \"$MA\" \"--tasks mmlu --num_fewshot 5\""
done | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done
