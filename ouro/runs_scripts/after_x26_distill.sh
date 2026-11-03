#!/bin/bash
# after_x26_distill.sh ARM [first]: when the distilled native n/n control (launch_x26_distill.sh) has finished training, queue its
# evaluation on all 13 Table 1 benchmarks with exactly the task-map settings, then start one grid_worker per idle GPU.
# Keys: L:o26x-<C>-<bench> (lm-eval) and B:X26-<C>f:ruler{4,8,16}k (RULER), C = E48 / D48 / H48 (E48f ... with first).
ARM=$1; F=$([ "$2" = first ] && echo f)
case $ARM in E) NAME=X26-E48$F-p1d1 ;; D) NAME=X26-D48$F-p2d2 ;; H) NAME=X26-H48$F-p3d3 ;; *) exit 2 ;; esac
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/distill_q.txt; C=${ARM}48$F; cd $R
until [ -f $R/$NAME/TRAIN_DONE ]; do sleep 300; done
CK=$R/$NAME/checkpoint-48m.pt; [ -f $CK ] || CK=$(ls $R/$NAME/checkpoint*.pt | tail -1)
MA="path=$M/model-2.6b,condition=$ARM,checkpoint=$CK,max_length=16384"
lm() { echo "./lm1.sh \$GPU o26x-$C-$1 - \"$MA\" \"$2\""; }
{
  lm math "--tasks minerva_math500 --num_fewshot 4"
  lm code "--tasks humaneval_local,mbpp_plus_instruct_local --confirm_run_unsafe_code"
  lm bbh "--tasks bbh_cot_fewshot --limit 60"
  lm gsmfull "--tasks gsm8k_cot --num_fewshot 3"
  lm mmlufull "--tasks mmlu --num_fewshot 5"
  lm mmlupro "--tasks mmlu_pro --limit 30"
  lm arc "--tasks arc_challenge --num_fewshot 25"
  lm gpqa "--tasks gpqa_diamond_n_shot_local"
  lm ifeval "--tasks ifeval_local"
  echo "cd /root/big-small && MODEL=$M/model-2.6b DATA=$M/bench BENCHES='ruler4k ruler8k ruler16k' ./ouro/runs_scripts/bench3.sh X26-${C}f $ARM $CK \$GPU 1"
} | flock $Q.lock sh -c "cat >> $Q"
for g in 0 1 2 3 4 5 6 7; do
  pgrep -f "grid_worker.sh $g $Q" > /dev/null || setsid nohup ./grid_worker.sh $g $Q > /dev/null 2>&1 < /dev/null &
done
