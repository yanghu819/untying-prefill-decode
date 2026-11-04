#!/bin/bash
# math0_q.sh GPU...: zero-shot MATH-500 (Minerva prompt without exemplars) for every Ouro-2.6B cell and native n/n, to
# test whether the 4-shot reversal (prefill 1 best at every decode depth under final-answer scoring) comes from the
# exemplars. Runs that already exist on this box are skipped. Two jobs per GPU.
# env SIZE=14: the same for Ouro-1.4B (run names O14x-<cell>-math0shot). env SKIP="S11 S22": cells queued elsewhere.
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/math0_q.txt; cd $R
if [ "${SIZE:-26}" = 14 ]; then MP=$M/model-1.4b; CK=$R/O14-S-universal/checkpoint-48m.pt; P=O14x
else MP=$M/model-2.6b; CK=$R/X26-S-universal/checkpoint-48m.pt; P=o26x; fi
for c in native1 native2 native3 native4 S11 S21 S31 S41 S12 S22 S32 S42 S13 S23 S33 S43 S14 S24 S34 S44; do
  ls lmeval/out/$P-$c-math0shot/*/samples_*.jsonl > /dev/null 2>&1 && continue
  case " $SKIP " in *" $c "*) continue ;; esac
  MA="path=$MP,condition=$c,max_length=16384"; [ ${c:0:1} = S ] && MA="$MA,checkpoint=$CK"
  echo "./lm1.sh \$GPU $P-$c-math0shot - \"$MA\" \"--tasks minerva_math500 --num_fewshot 0\""
done | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done
