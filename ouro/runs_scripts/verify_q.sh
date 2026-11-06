#!/bin/bash
# verify_q.sh GPU...: reproducibility check of the main-table cells. Re-runs native 1/1, 2/2, 4/4 and the loop adapters
# (single cache S41/S42, aligned a41/a42) on GSM8K (full), MMLU (full) and RULER-16K from scratch under new run names
# (<run>-rv, fresh request caches), so ouro/scripts/verify_compare.py can compare them item by item with the
# originals. Backlog queue: two jobs per GPU.
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/verify_q.txt; cd $R
for c in native1 native2 native4 S41 S42 a41 a42; do
  case $c in S*) CK=$R/X26-S-universal/checkpoint-48m.pt ;; a*) CK=$R/X26-a-universal/checkpoint-48m.pt ;; *) CK=- ;; esac
  MA="path=$M/model-2.6b,condition=$c,max_length=16384"; [ $CK = - ] || MA="$MA,checkpoint=$CK"
  echo "./lm1.sh \$GPU X26-$c-gsmfull-rv - \"$MA\" \"--tasks gsm8k_cot --num_fewshot 3\""
  echo "./lm1.sh \$GPU X26-$c-mmlufull-rv - \"$MA\" \"--tasks mmlu --num_fewshot 5\""
  echo "cd /root/big-small && MODEL=$M/model-2.6b DATA=$M/bench BENCHES='ruler16k' ./ouro/runs_scripts/bench3.sh X26-${c}f-rv $c $CK \$GPU 1"
done | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done
