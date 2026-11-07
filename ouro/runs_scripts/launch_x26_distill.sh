#!/bin/bash
# Same-training controls for Table 1 (base Ouro-2.6B): LoRA distillation of native n/n for n = 1, 2, 3 (arms E, D, H;
# native 4/4 is the teacher itself). Same data (mix-data-48m), 48M KL tokens, native 4/4 teacher, micro 1 and seed as
# the loop adapter X26-S-universal, so "chosen m vs distilled n/n" isolates untying from training.
# usage: launch_x26_distill.sh ARM GPUS PORT [first]     e.g. launch_x26_distill.sh E 0,1,2,3 29811 first
# "first" also supervises the first answer token, as the loop adapter does (runs X26-<ARM>48f-*); without it the
# control omits that token (the X26-<ARM>48-* runs).
# env SIZE=14: the same controls for Ouro-1.4B (runs X14-*, against the adapter O14-S-universal).
# env RANK=80: LoRA rank of the control (default 16); rank 80 matches the loop adapter's 63M trainable parameters
# (runs X26-<ARM>48<f>r80-*).
ARM=$1; GPUS=$2; PORT=$3; F=$([ "$4" = first ] && echo f); FLAG=$([ "$4" = first ] && echo --first-target)
S=${SIZE:-26}; MODEL=$([ $S = 14 ] && echo model-1.4b || echo model-2.6b)
case $ARM in E) NAME=X$S-E48$F-p1d1 ;; D) NAME=X$S-D48$F-p2d2 ;; H) NAME=X$S-H48$F-p3d3 ;; *) echo "arm must be E, D or H"; exit 2 ;; esac
[ -n "$RANK" ] && { NAME=${NAME/-p/r$RANK-p}; export OURO_LORA_RANK=$RANK; }
M=/dev/shm/big-small-ouro; R=$M/runs; cd $R
[ -f $R/$NAME/TRAIN_DONE ] || MODEL=$M/$MODEL DATA=$M/mix-data-48m ./run_mix2.sh $NAME $ARM $GPUS $PORT 48000000 --micro 1 $FLAG
