#!/bin/bash
# after_x26_distill.sh ARM [first]: when the distilled native n/n control (launch_x26_distill.sh) has finished training, queue its
# evaluation on all 15 task-map settings (Ouro's protocol: evalplus chat code, HellaSwag, Winogrande), then start one
# grid_worker per GPU (env GPUS, default 0-7).
# Keys: L:o26x-<C>-<bench> (lm-eval) and B:X26-<C>f:ruler{4,8,16}k (RULER), C = E48 / D48 / H48 (E48f ... with first).
# env SIZE=14: Ouro-1.4B control, run names O14x-<C>-<bench> / O14-<C>f as its task map (GSM8K and MMLU subsets).
ARM=$1; F=$([ "$2" = first ] && echo f); S=${SIZE:-26}
case $ARM in E) NAME=X$S-E48$F-p1d1 ;; D) NAME=X$S-D48$F-p2d2 ;; H) NAME=X$S-H48$F-p3d3 ;; *) exit 2 ;; esac
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/distill_q.txt; C=${ARM}48$F; cd $R
MODEL=$([ $S = 14 ] && echo model-1.4b || echo model-2.6b); LM=$([ $S = 14 ] && echo O14x || echo o26x); BN=$([ $S = 14 ] && echo O14 || echo X26)
until [ -f $R/$NAME/TRAIN_DONE ]; do sleep 300; done
CK=$R/$NAME/checkpoint-48m.pt; [ -f $CK ] || CK=$(ls $R/$NAME/checkpoint*.pt | tail -1)
MA="path=$M/$MODEL,condition=$ARM,checkpoint=$CK,max_length=16384"
lm() { echo "./lm1.sh \$GPU $LM-$C-$1 - \"$MA\" \"$2\""; }
{
  lm math "--tasks minerva_math500 --num_fewshot 4"
  lm evalplus "--tasks humanevalplus_chat_local,mbppplus_chat_local"
  lm hellaswag "--tasks hellaswag --num_fewshot 10"; lm winogrande "--tasks winogrande --num_fewshot 5"
  lm bbh "--tasks bbh_cot_fewshot --limit 60"
  if [ $S = 14 ]; then lm gsm "--tasks gsm8k_cot --num_fewshot 3 --limit 400"; lm mmlu "--tasks mmlu --num_fewshot 5 --limit 20"
  else lm gsmfull "--tasks gsm8k_cot --num_fewshot 3"; lm mmlufull "--tasks mmlu --num_fewshot 5"; fi
  lm mmlupro "--tasks mmlu_pro --limit 30"
  lm arc "--tasks arc_challenge --num_fewshot 25"
  lm gpqa "--tasks gpqa_diamond_n_shot_local"
  lm ifeval "--tasks ifeval_local"
  echo "cd /root/big-small && MODEL=$M/$MODEL DATA=$M/bench BENCHES='ruler4k ruler8k ruler16k' ./ouro/runs_scripts/bench3.sh $BN-${C}f $ARM $CK \$GPU 1"
} | flock $Q.lock sh -c "cat >> $Q"
for g in ${GPUS:-0 1 2 3 4 5 6 7}; do
  pgrep -f "grid_worker.sh $g $Q" > /dev/null || setsid nohup ./grid_worker.sh $g $Q > /dev/null 2>&1 < /dev/null &
done
