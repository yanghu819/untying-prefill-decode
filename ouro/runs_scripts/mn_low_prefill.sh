#!/bin/bash
# Prefill m < decode n on the tasks where more prefill loops hurt: one adapter per model, cells S12 S14 S24 and S44,
# same tasks and settings as the existing runs (4-shot MATH-500 and RULER on Ouro-2.6B; 3-shot BBH on Ouro-1.4B and
# Ouro-1.4B-Thinking). One sequential queue per a2 GPU (lm1.sh / bench3.sh are resumable).
M=/dev/shm/big-small-ouro; R=$M/runs; cd $R
X26=$R/X26-S-universal/checkpoint-48m.pt; O14=$R/O14-S-universal/checkpoint-48m.pt; T14=$R/T14-S-universal/checkpoint-48m.pt
math() { ./lm1.sh $1 o26x-$2-math - "path=$M/model-2.6b,condition=$2,checkpoint=$X26,max_length=16384" "--tasks minerva_math500 --num_fewshot 4"; }
ruler() { cd /root/big-small; MODEL=$M/model-2.6b DATA=$M/bench BENCHES='ruler4k ruler8k ruler16k' ./ouro/runs_scripts/bench3.sh X26-$2f $2 $X26 $1 1; cd $R; }
bbh() { local mp=$M/model-1.4b ck=$O14 n=O14x; [ $3 = T ] && mp=$M/model-1.4b-think ck=$T14 n=T14x
        ./lm1.sh $1 $n-$2-bbh - "path=$mp,condition=$2,max_length=16384,checkpoint=$ck" "--tasks bbh_cot_fewshot --limit 60"; }
q() { local g=$1; shift; ( for j in "$@"; do eval "$j"; done ) > /dev/null 2>&1 < /dev/null & }
q 0 "math 0 S12" "math 0 S44"
q 1 "math 1 S14" "math 1 S24"
q 2 "ruler 2 S12" "ruler 2 S44"
q 3 "ruler 3 S14"
q 4 "ruler 4 S24"
q 5 "bbh 5 S12 O" "bbh 5 S14 O" "bbh 5 S24 O" "bbh 5 S44 O"
q 6 "bbh 6 S12 T" "bbh 6 S14 T"
q 7 "bbh 7 S24 T" "bbh 7 S44 T"
wait
