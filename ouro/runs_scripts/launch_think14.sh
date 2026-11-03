#!/bin/bash
# Ouro-1.4B-Thinking single-bank universal loop adapter (arm S, 48M KL tokens, chat-template mix, teacher = native 4/4
# Thinking 1.4B). Reuses mix-think-48m when the tokenizer is identical to 2.6B-Thinking's, else rebuilds it.
# Reserves GPUS from the dispatcher, trains, frees them, then queues the thinking evals (GSM8K 4 shards + MATH-500 shard 0)
# for native 1/1, 2/2, 4/4 and prefill 1 / decode 1, prefill 4 / decode 1, prefill 2 / decode 2, prefill 4 / decode 2.
GPUS=$1; M=/dev/shm/big-small-ouro; R=$M/runs; cd $R; N=T14-S-universal; MOD=$M/model-1.4b-think
echo " ${GPUS//,/ }" >> /dev/shm/reserved_gpus
until grep -q THINK14_DONE /dev/shm/dlthink14.log; do sleep 60; done
DATA=$M/mix-think-48m
if ! cmp -s $MOD/tokenizer.json $M/model-2.6b-think/tokenizer.json; then
  DATA=$M/mix-think14-48m
  [ -f $DATA/manifest.json ] || (cd /root/big-small && PYTHONPATH=. environments/ouro/.venv/bin/python ouro/scripts/prepare_think_mix.py $M/mix-data-48m $DATA $MOD)
fi
for g in ${GPUS//,/ }; do
  until [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ]; do sleep 60; done
done
echo "$(date +%T) $N on $GPUS (data $DATA)"
[ -f $R/$N/TRAIN_DONE ] || MODEL=$MOD DATA=$DATA ./run_mix2.sh $N S $GPUS 29791 48000000 --micro 1 || exit 1
for g in ${GPUS//,/ }; do sed -i "s/\b$g\b//" /dev/shm/reserved_gpus; done
CK=$R/$N/checkpoint-48m.pt
for c in native1:400000 S41:160000 native2:250000 S11:160000 S22:160000 S42:160000 native4:120000; do n=${c%%:*}; bt=${c##*:}
  MA="path=$MOD,condition=$n,max_length=16384,batch_tokens=$bt,max_batch=64"; [ "${n#native}" = "$n" ] && MA="$MA,checkpoint=$CK"
  for t in gsm8k_think_s0 gsm8k_think_s1 gsm8k_think_s2 gsm8k_think_s3 math500_think_s0; do
    echo "./lm1.sh \$GPU T14-$n-$t - \"$MA\" \"--tasks $t --apply_chat_template\""
  done
done | flock jobs.txt.lock tee -a jobs.txt > /dev/null
echo "$(date +%T) evals queued"
