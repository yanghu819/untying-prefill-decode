#!/bin/bash
# time_batch_think.sh GPU...: batched serving (batch_requests.py) on Ouro-2.6B-Thinking with the chat template: the
# lm-eval requests of T26-a42-<TASK>_think_s* (TASK=gsm8k: 4 shards, 4,096-token budget, default; TASK=math500: 5
# shards, 8,192 tokens), N requests chosen by hash, for Original
# 2/2 and 4/4, Distilled 2/2 (T26-Df-native22, merged) and LoopBridge 4/2 (T26-a-universal, maps folded in), at batch
# sizes $BATCHES (default 16 32). Output: runs/e2e/<OUT><B>-<name>-<gsm|math>think.jsonl (env OUT, default tbatch; N
# default 400).
M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/btime_q.txt; mkdir -p $R/e2e; cd $R; O=${OUT:-tbatch}; N=${N:-400}
T=${TASK:-gsm8k}; X=${T%8k}; X=${X%500}think; P=$R/e2e/prompts/T26-a42-${T}_think; mkdir -p $P
for d in $R/lmeval/out/T26-a42-${T}_think_s*; do s=${d##*_}; [ -e $P/$s ] || ln -s $(ls -d $d/*/ | head -1) $P/$s; done
A=$R/T26-a-universal/checkpoint-48m.pt; D=$R/T26-Df-native22/checkpoint-48m.pt
job() {  # COND CKPT NAME B
  echo "cd /root/big-small && OURO_NO_CUDNN_SDP=1 CUDA_VISIBLE_DEVICES=\$GPU HF_HUB_OFFLINE=1 PYTHONPATH=/root/big-small environments/ouro/.venv/bin/python ouro/scripts/batch_requests.py $M/model-2.6b-think $1 $2 $P $N $4 $R/e2e/$O$4-$3-$X.jsonl > $R/e2e/$O$4-$3-$X.log 2>&1"
}
{
  for b in ${BATCHES:-16 32}; do
    job native2st - native2st $b; job native4st - native4st $b; job D2st $D D2st $b; job a4x2pm $A a4x2pm $b
  done
} | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do setsid nohup ./grid_worker.sh $g $Q 1 > /dev/null 2>&1 < /dev/null & done
