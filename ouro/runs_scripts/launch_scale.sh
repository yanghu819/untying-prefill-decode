#!/bin/bash
# Two 4-GPU trainings of the single-bank universal loop adapter (arm S, 48M KL tokens, same recipe), started as GPUs free up:
# (1) Ouro-1.4B (second scale), (2) Ouro-2.6B seed 2 (variance). Each queues its full-set evals on the dispatcher when done:
# GSM8K, MMLU, RULER 4K/8K/16K (520) for S4d / Sdd (d = 1..3) and, for 1.4B, natives 1/1..4/4.
M=/dev/shm/big-small-ouro; R=$M/runs; cd $R
take4() {  # wait until 4 GPUs are idle, print them comma-separated
  while true; do
    free=""
    for g in 0 1 2 3 4 5 6 7; do
      u=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      n=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | wc -l)
      [ "$n" = 0 ] && [ "$u" -lt 2000 ] && free="$free $g"
    done
    set -- $free
    [ $# -ge 4 ] && { echo "$1,$2,$3,$4"; return; }
    sleep 60
  done
}
queue() {  # queue NAME MODEL CKPT CELLS... : full-set evals on the dispatcher
  local N=$1 MOD=$2 CK=$3; shift 3
  for c in "$@"; do
    local MA="path=$MOD,condition=$c,max_length=16384"; local CKA=-
    [ "${c#native}" = "$c" ] && { MA="$MA,checkpoint=$CK"; CKA=$CK; }
    echo "MODEL=$MOD DATA=$M/bench BENCHES='ruler4k ruler8k ruler16k' BATCH=24576 OURO_NO_CUDNN_SDP=1 ./bench3.sh $N-${c}f $c $CKA \$GPU 1"
    echo "./lm1.sh \$GPU $N-$c-gsmfull - \"$MA\" \"--tasks gsm8k_cot --num_fewshot 3\""
    echo "./lm1.sh \$GPU $N-$c-mmlufull - \"$MA\" \"--tasks mmlu --num_fewshot 5\""
  done | flock jobs.txt.lock tee -a jobs.txt > /dev/null
}
until grep -q OURO14_DONE /dev/shm/dl14.log; do sleep 30; done
G=$(take4); echo "$(date +%T) O14-S-universal on $G"
( [ -f $R/O14-S-universal/TRAIN_DONE ] || MODEL=$M/model-1.4b DATA=$M/mix-data-48m ./run_mix2.sh O14-S-universal S $G 29771 48000000 --micro 1 &&
  queue O14 $M/model-1.4b $R/O14-S-universal/checkpoint-48m.pt native4 S41 native1 S11 S42 native2 S22 S43 native3 S33 && echo "$(date +%T) O14 evals queued" ) &
sleep 300
G=$(take4); echo "$(date +%T) X26-S-seed2 on $G"
( [ -f $R/X26-S-seed2/TRAIN_DONE ] || DATA=$M/mix-data-48m ./run_mix2.sh X26-S-seed2 S $G 29761 48000000 --micro 1 --seed 2 &&
  queue X26-S2 $M/model-2.6b $R/X26-S-seed2/checkpoint-48m.pt S41 S11 S42 S22 S43 S33 && echo "$(date +%T) seed-2 evals queued" ) &
wait
