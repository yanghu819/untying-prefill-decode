#!/bin/bash
# mbpp3_q.sh SIZE GPU...: MBPP+ under the standard 3-shot base protocol, scored on the full MBPP+ tests
# (mbpp_plus_3shot_local), for every task-map cell and native n/n of Ouro-2.6B (SIZE=26) or Ouro-1.4B (SIZE=14).
S=$1; shift; M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/mbpp3_q.txt; cd $R
[ $S = 14 ] && { MP=$M/model-1.4b; CK=$R/O14-S-universal/checkpoint-48m.pt; P=O14x; } || { MP=$M/model-2.6b; CK=$R/X26-S-universal/checkpoint-48m.pt; P=o26x; }
for c in native1 native2 native3 native4 S11 S21 S31 S41 S12 S22 S32 S42 S13 S23 S33 S43 S14 S24 S34 S44; do
  MA="path=$MP,condition=$c,max_length=16384"; [ ${c:0:1} = S ] && MA="$MA,checkpoint=$CK"
  echo "./lm1.sh \$GPU $P-$c-mbpp3 - \"$MA\" \"--tasks mbpp_plus_3shot_local --confirm_run_unsafe_code\""
done | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done
