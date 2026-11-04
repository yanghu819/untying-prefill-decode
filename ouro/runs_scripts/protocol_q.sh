#!/bin/bash
# protocol_q.sh SIZE GPU...: the two benchmarks of Ouro's base-model table (Ouro paper, Appendix C.1, Table 16) that the
# task map lacked, HellaSwag (10-shot) and Winogrande (5-shot), for every cell and native n/n of Ouro-2.6B (SIZE=26) or
# Ouro-1.4B (SIZE=14). Two jobs per GPU. BBH and MMLU-Pro keep their subsets (same protocol as Ouro; our native 4/4
# matches Ouro's full-set scores): the full sets take 8-9 GPU-hours per configuration with our looped decoder.
S=$1; shift; M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/protocol_q.txt; cd $R
[ $S = 14 ] && { MP=$M/model-1.4b; CK=$R/O14-S-universal/checkpoint-48m.pt; P=O14x; } || { MP=$M/model-2.6b; CK=$R/X26-S-universal/checkpoint-48m.pt; P=o26x; }
for c in native1 native2 native3 native4 S11 S21 S31 S41 S12 S22 S32 S42 S13 S23 S33 S43 S14 S24 S34 S44; do
  MA="path=$MP,condition=$c,max_length=16384"; [ ${c:0:1} = S ] && MA="$MA,checkpoint=$CK"
  echo "./lm1.sh \$GPU $P-$c-hellaswag - \"$MA\" \"--tasks hellaswag --num_fewshot 10\""
  echo "./lm1.sh \$GPU $P-$c-winogrande - \"$MA\" \"--tasks winogrande --num_fewshot 5\""
done | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done
