#!/bin/bash
# xcell.sh GPU ARM PORT : Ouro-2.6B loop-map cell X<p><d> on one GPU — train 24M (decode model never re-reads the
# prompt; prefill costs exactly p loops) -> the three main tasks: RULER 16K (bench-mini), GSM8K (400), MMLU (20/subject)
G=$1 A=$2 PORT=$3; M=/dev/shm/big-small-ouro; R=/dev/shm/big-small-ouro/runs; cd $R; N=X26-$A
[ -f $R/$N/TRAIN_DONE ] || ./run_mix.sh $N $A $G $PORT 24000000 --micro 1 || exit 1
CK=$R/$N/checkpoint-24m.pt
MODEL=$M/model-2.6b DATA=$M/bench-mini BENCHES="ruler4k ruler8k ruler16k" BATCH=24576 OURO_NO_CUDNN_SDP=1 ./bench3.sh $N $A $CK $G 1
./lm1.sh $G $N-gsm - "path=$M/model-2.6b,condition=$A,checkpoint=$CK,max_length=16384" "--tasks gsm8k_cot --num_fewshot 3 --limit 400"
./lm1.sh $G $N-mmlu - "path=$M/model-2.6b,condition=$A,checkpoint=$CK,max_length=16384" "--tasks mmlu --num_fewshot 5 --limit 20"
