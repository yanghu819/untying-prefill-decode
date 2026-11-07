#!/bin/bash
# time_helmet.sh GPU...: end-to-end per-request time and accuracy on HELMET (bench-mini, 1,108 prompts of ~8K tokens,
# short answers) for Ouro-2.6B-Thinking: native 2/2 and 4/4, LoRA-distilled native 2/2 (merged), and the loop adapter
# at 2/2 and 4/2 with one stored cache (S) and aligned caches (a). Batch 1, each job alone on its GPU (grid_worker.sh
# MAX=1); the answers are scored as in the full evaluation (evaluate_bench.helmet_score). This is the long-context
# workload where deeper prefill gains most (Figure 4), so it puts accuracy and request time on the same items.
# NO_EMPTY=1 as the HELMET evaluation (bench3 / eval_think_aligned.sh). Output: runs/e2e/think-<COND>-helmet.jsonl (the
# first record is a warm-up).
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/timing_q.txt; mkdir -p $R/e2e; cd $R
S=$R/T26-S-universal/checkpoint-48m.pt; A=$R/T26-a-universal/checkpoint-48m.pt; D=$R/T26-Df-native22/checkpoint-48m.pt
for c in "native2st - native2st" "native4st - native4st" "D2st $D Df2st" "S22sb $S S22sb" "S42sb $S S42sb" "a42sb $A a42sb" "a22sb $A a22sb"; do
  set -- $c
  echo "cd /root/big-small && NO_EMPTY=1 CUDA_VISIBLE_DEVICES=\$GPU HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=$M/hf-lmeval OURO_NO_CUDNN_SDP=1 PYTHONPATH=$M/lmeval-pkgs:/root/big-small environments/ouro/.venv/bin/python ouro/scripts/e2e_latency.py $M/model-2.6b-think $1 $2 helmet 1108 $R/e2e/think-$3-helmet.jsonl > $R/e2e/think-$3-helmet.log 2>&1"
done | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do setsid nohup ./grid_worker.sh $g $Q 1 > /dev/null 2>&1 < /dev/null & done
