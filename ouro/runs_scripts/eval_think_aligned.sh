#!/bin/bash
# eval_think_aligned.sh GPUS [14]: after T26-a-universal (or T14-a-universal for 14) (aligned loop adapter on Ouro-2.6B-Thinking) has trained, queue the
# eight benchmarks of Figure 4 at prefill 4 / decode 2 and 2 / 2 with exactly the settings of the arm-S runs
# (run names T26-<cell>-<task>_s<k>, T26x-<cell>-<task>, T26ne-<cell>-helmet) and run them two per GPU on GPUS.
# env CELLS: other cells (default a42 a22); env GSM_ONLY=1: only the GSM8K shards (Figure 5's Thinking panel).
GPUS=$1; Z=${2:-26}; M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/aligned_q.txt; MP=$M/model-2.6b-think; T=T26; cd $R
[ $Z = 14 ] && { MP=$M/model-1.4b-think; T=T14; }
until [ -f $R/$T-a-universal/TRAIN_DONE ]; do sleep 300; done
CK=$R/$T-a-universal/checkpoint-48m.pt
for c in ${CELLS:-a42 a22}; do
  MA="path=$MP,condition=$c,checkpoint=$CK,max_length=16384"
  for s in 0 1 2 3; do echo "./lm1.sh \$GPU $T-$c-gsm8k_think_s$s - \"$MA,batch_tokens=160000,max_batch=64\" \"--tasks gsm8k_think_s$s --apply_chat_template\""; done
  [ -n "$GSM_ONLY" ] && continue
  for s in 0 1 2 3 4; do echo "./lm1.sh \$GPU $T-$c-math500_think_s$s - \"$MA,batch_tokens=160000,max_batch=64\" \"--tasks math500_think_s$s --apply_chat_template\""; done
  echo "./lm1.sh \$GPU ${T}x-$c-mmlupro - \"$MA\" \"--tasks mmlu_pro --limit 30\""
  echo "./lm1.sh \$GPU ${T}x-$c-arc - \"$MA\" \"--tasks arc_challenge --num_fewshot 25\""
  echo "./lm1.sh \$GPU ${T}x-$c-bbh - \"$MA\" \"--tasks bbh_cot_fewshot --limit 60\""
  echo "./lm1.sh \$GPU ${T}x-$c-evalplus - \"$MA\" \"--tasks humanevalplus_chat_local,mbppplus_chat_local\""
  echo "cd /root/big-small && NO_EMPTY=1 MODEL=$MP DATA=$M/bench-mini BENCHES=helmet BATCH=65536 ./ouro/runs_scripts/bench3.sh ${T}ne-$c-helmet $c $CK \$GPU 1"
done | flock $Q.lock sh -c "cat >> $Q"
for g in ${GPUS//,/ }; do setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done
