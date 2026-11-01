#!/bin/bash
# zcell.sh GPU ARM PORT : trained KV-sharing baseline Z<d> on Ouro-2.6B (native 4-loop prefill, d-loop LoRA decode that
# reads one shared last-loop KV bank, BeNice-style self-live training) — train 24M -> RULER 16K (bench-mini),
# GSM8K (400), MMLU (20/subject)
G=$1 A=$2 PORT=$3; M=/dev/shm/big-small-ouro; R=$M/runs; cd $R; N=X26-$A
[ -f $R/$N/TRAIN_DONE ] || ./run_mix.sh $N $A $G $PORT 24000000 --micro 1 || exit 1
CK=$R/$N/checkpoint-24m.pt
MODEL=$M/model-2.6b DATA=$M/bench-mini BENCHES=ruler16k BATCH=24576 OURO_NO_CUDNN_SDP=1 ./bench3.sh $N $A $CK $G 1
./lm1.sh $G $N-gsm - "path=$M/model-2.6b,condition=$A,checkpoint=$CK,max_length=16384" "--tasks gsm8k_cot --num_fewshot 3 --limit 400"
./lm1.sh $G $N-mmlu - "path=$M/model-2.6b,condition=$A,checkpoint=$CK,max_length=16384" "--tasks mmlu --num_fewshot 5 --limit 20"
