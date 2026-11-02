#!/bin/bash
# 4-shot MATH-500 grid on Ouro-2.6B (one adapter): the cells not run elsewhere, on a1 GPUs 4-7
M=/dev/shm/big-small-ouro; R=$M/runs; cd $R; X26=$R/X26-S-universal/checkpoint-48m.pt
math() { ./lm1.sh $1 o26x-$2-math - "path=$M/model-2.6b,condition=$2,checkpoint=$X26,max_length=16384" "--tasks minerva_math500 --num_fewshot 4"; }
q() { local g=$1; shift; ( for j in "$@"; do eval "$j"; done ) > /dev/null 2>&1 < /dev/null & }
q 4 "math 4 S13"; q 5 "math 5 S23"; q 6 "math 6 S32"; q 7 "math 7 S34"
wait
