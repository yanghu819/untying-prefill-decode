#!/bin/bash
# sweep_think.sh GPU...: prefill-depth sweep on Ouro-2.6B-Thinking (the advisor's Exp. 2 for the reasoning model), aligned
# adapter T26-a-universal at decode n = 2 and prefill m = 1, 2, 3, 4, 6, 8, on the Thinking benchmarks with short
# answers or log-likelihood scoring (ARC-C, BBH, MMLU-Pro, HELMET); GSM8K and MATH-500 (thousands of thinking tokens)
# are left out. Settings as eval_think_aligned.sh (run names T26x-a<m>x2-<task>, T26ne-a<m>x2-helmet); finished
# benchmarks are skipped. Queue runs/sweep_q.txt, two jobs per GPU. env GSM=1: also GSM8K (four shards, as
# eval_think_aligned.sh; run names T26-a<m>x2-gsm8k_think_s<k>), the most decode-heavy Thinking task; ONLY_GSM=1: just that.
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/sweep_q.txt; MP=$M/model-2.6b-think; CK=$R/T26-a-universal/checkpoint-48m.pt; cd $R
for m in ${MS:-1 2 3 4 6 8}; do
  c=a${m}x2; MA="path=$MP,condition=$c,checkpoint=$CK,max_length=16384"
  lm() { ls lmeval/out/T26x-$c-$1/*/samples_* > /dev/null 2>&1 || echo "./lm1.sh \$GPU T26x-$c-$1 - \"$MA\" \"$2\""; }
  [ -z "$ONLY_GSM" ] && { lm arc "--tasks arc_challenge --num_fewshot 25"; lm bbh "--tasks bbh_cot_fewshot --limit 60"
    lm mmlupro "--tasks mmlu_pro --limit 30"; }
  [ -n "$GSM" ] && for k in 0 1 2 3; do
    ls lmeval/out/T26-$c-gsm8k_think_s$k/*/samples_* > /dev/null 2>&1 || \
      echo "./lm1.sh \$GPU T26-$c-gsm8k_think_s$k - \"$MA,batch_tokens=160000,max_batch=64\" \"--tasks gsm8k_think_s$k --apply_chat_template\""
  done
  [ -n "$ONLY_GSM" ] || ls bench/T26ne-$c-helmet/s0/helmet.jsonl > /dev/null 2>&1 || echo "cd /root/big-small && NO_EMPTY=1 MODEL=$MP DATA=$M/bench-mini BENCHES=helmet BATCH=65536 ./ouro/runs_scripts/bench3.sh T26ne-$c-helmet $c $CK \$GPU 1"
done | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done
