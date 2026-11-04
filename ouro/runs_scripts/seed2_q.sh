#!/bin/bash
# seed2_q.sh PART GPU...: Table 1 with a second training seed of the loop adapter (X26-S-seed2): every cell needed to
# choose m at one and two decode loops (prefill 1-4 / decode 1-2), all task-map settings under Ouro's protocol, with
# the same tasks and settings as the seed-0 task map. PART lm: the lm-eval settings; PART rest: RULER and evalplus code;
# PART lm34 / rest34: the same for the decode 3-4 cells.
# Run names X26s2-<cell>-<bench> (RULER: X26s2-<cell>f). Two jobs per GPU.
# env SEED=3: the third seed (X26-S-seed3, run names X26s3-*); default seed 2.
# env SIZE=14: the Ouro-1.4B adapter (O14-S-seed<SEED>, run names O14s<SEED>-*; GSM8K and MMLU subsets as its task map).
PART=$1; shift; SD=${SEED:-2}; Z=${SIZE:-26}; M=/dev/shm/big-small-ouro; R=$M/runs; cd $R
if [ $Z = 14 ]; then MP=$M/model-1.4b; CK=$R/O14-S-seed$SD/checkpoint-48m.pt; P=O14s$SD; Q=$R/seed${SD}_14_q.txt
else MP=$M/model-2.6b; CK=$R/X26-S-seed$SD/checkpoint-48m.pt; P=X26s$SD; Q=$R/seed${SD}_q.txt; fi
CELLS="S11 S21 S31 S41 S12 S22 S32 S42"; case $PART in *34) CELLS="S13 S23 S33 S43 S14 S24 S34 S44"; PART=${PART%34} ;; esac
for c in $CELLS; do
  MA="path=$MP,condition=$c,checkpoint=$CK,max_length=16384"
  lm() { echo "./lm1.sh \$GPU $P-$c-$1 - \"$MA\" \"$2\""; }
  if [ $PART = lm ]; then
    if [ $Z = 14 ]; then lm gsm "--tasks gsm8k_cot --num_fewshot 3 --limit 400"; lm mmlu "--tasks mmlu --num_fewshot 5 --limit 20"
    else lm gsmfull "--tasks gsm8k_cot --num_fewshot 3"; lm mmlufull "--tasks mmlu --num_fewshot 5"; fi
    lm math "--tasks minerva_math500 --num_fewshot 4"; lm bbh "--tasks bbh_cot_fewshot --limit 60"
    lm arc "--tasks arc_challenge --num_fewshot 25"; lm mmlupro "--tasks mmlu_pro --limit 30"
    lm hellaswag "--tasks hellaswag --num_fewshot 10"; lm winogrande "--tasks winogrande --num_fewshot 5"
    lm gpqa "--tasks gpqa_diamond_n_shot_local"; lm ifeval "--tasks ifeval_local"
  else
    lm evalplus "--tasks humanevalplus_chat_local,mbppplus_chat_local"
    echo "cd /root/big-small && MODEL=$MP DATA=$M/bench BENCHES='ruler4k ruler8k ruler16k' ./ouro/runs_scripts/bench3.sh $P-${c}f $c $CK \$GPU 1"
  fi
done | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done
