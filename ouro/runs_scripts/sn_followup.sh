#!/bin/bash
# sn_followup.sh: SenseNova read-depth tests where the thesis predicts an effect (long context, knowledge):
# RULER 4K/8K/16K (520 each) and MMLU-Pro (5-shot CoT, 70 per subject) for native1/native2/splice21 now, and for the
# trained connector C21 / C11 (same decoder, read 2 vs read 1) once launch_sn.sh has finished training.
M=/dev/shm/big-small-ouro; R=$M/runs; cd $R
line() {  # NAME COND CKPT
  echo "MODEL=$M/model-sn9b PP=$M/tf518:/root/big-small DATA=$M/bench BENCHES='ruler4k ruler8k ruler16k' BATCH=24576 OURO_NO_CUDNN_SDP=1 ./bench3.sh SN-$1 $2 $3 \$GPU 1"
  ma="path=$M/model-sn9b,condition=$2,max_length=16384,batch_tokens=250000,max_batch=64"; [ "$3" != "-" ] && ma="$ma,checkpoint=$3"
  echo "./lm1.sh \$GPU SN-$1-mmlupro $M/tf518: \"$ma\" \"--tasks mmlu_pro --num_fewshot 5 --limit 70\""
}
{ line native1 native1 -; line native2 native2 -; line splice21 splice21 -; } | flock jobs.txt.lock tee -a jobs.txt > /dev/null
echo "$(date +%T) native SN jobs queued"
until [ -f $R/SN-C-universal/TRAIN_DONE ]; do sleep 120; done
CK=$R/SN-C-universal/checkpoint.pt
{ line C21 C21 $CK; line C11 C11 $CK; } | flock jobs.txt.lock sh -c "cat - jobs.txt > /tmp/jq_snf && cat /tmp/jq_snf > jobs.txt"
echo "$(date +%T) connector SN jobs queued"
