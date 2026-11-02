#!/bin/bash
# Ouro-1.4B-Thinking headline set: read 4/write 2 (S42), same connector read 2/write 2 (S22), native 2/2, native 4/4.
# usage: t14_head.sh a1|a2   (one sequential queue per GPU; lm1.sh / bench3.sh are resumable)
M=/dev/shm/big-small-ouro; R=$M/runs; CK=$R/T14-S-universal/checkpoint-48m.pt; MP=$M/model-1.4b-think
cd $R
ma() { case $1 in S*) echo "path=$MP,condition=$1,max_length=16384,checkpoint=$CK$2";; *) echo "path=$MP,condition=$1,max_length=16384$2";; esac; }
math() { ./lm1.sh $1 T14-$2-math500_think_s$3 - "$(ma $2 ,batch_tokens=${4:-160000},max_batch=64)" "--tasks math500_think_s$3 --apply_chat_template"; }
lm() { ./lm1.sh $1 T14x-$2-$3 - "$(ma $2)" "$4"; }
helmet() { cd /root/big-small; NO_EMPTY=1 MODEL=$MP BENCHES=helmet BATCH=65536 ./ouro/runs_scripts/bench3.sh T14ne-$2-helmet $2 $( [ "${2:0:1}" = S ] && echo $CK || echo - ) $1 1; cd $R; }
q() { local g=$1; shift; ( for j in "$@"; do eval "$j"; done ) > /dev/null 2>&1 < /dev/null & }
if [ "$1" = a1 ]; then
  for s in 1 2 3 4; do q $((s - 1)) "math $((s - 1)) native4 $s 120000"; q $((s + 3)) "math $((s + 3)) native2 $s"; done
else
  q 0 "math 0 S42 1" "math 0 S42 2"; q 3 "math 3 S42 3" "math 3 S42 4"
  q 4 "math 4 S22 1" "math 4 S22 2"; q 5 "math 5 S22 3" "math 5 S22 4"
  A="--tasks arc_challenge --num_fewshot 25"; C="--tasks humaneval_local,mbpp_plus_instruct_local --confirm_run_unsafe_code"
  q 6 "lm 6 S42 arc '$A'" "lm 6 S22 arc '$A'" "lm 6 native2 arc '$A'" "lm 6 native4 arc '$A'" \
      "lm 6 native4 mmlupro '--tasks mmlu_pro --limit 30'" "lm 6 native4 bbh '--tasks bbh_cot_fewshot --limit 60'"
  q 7 "lm 7 S42 code '$C'" "lm 7 S22 code '$C'" "lm 7 native2 code '$C'" "lm 7 native4 code '$C'" "helmet 7 native4"
fi
echo queued
