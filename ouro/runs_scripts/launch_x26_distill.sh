#!/bin/bash
# Same-training controls for Table 1 (base Ouro-2.6B): LoRA distillation of native n/n for n = 1, 2, 3 (arms E, D, H;
# native 4/4 is the teacher itself). Same data (mix-data-48m), 48M KL tokens, native 4/4 teacher, micro 1 and seed as
# the loop adapter X26-S-universal, so "chosen m vs distilled n/n" isolates untying from training.
# usage: launch_x26_distill.sh ARM GPUS PORT [first]     e.g. launch_x26_distill.sh E 0,1,2,3 29811 first
# "first" also supervises the first answer token, as the loop adapter does (runs X26-<ARM>48f-*); without it the
# control omits that token (the X26-<ARM>48-* runs).
ARM=$1; GPUS=$2; PORT=$3; F=$([ "$4" = first ] && echo f); FLAG=$([ "$4" = first ] && echo --first-target)
case $ARM in E) NAME=X26-E48$F-p1d1 ;; D) NAME=X26-D48$F-p2d2 ;; H) NAME=X26-H48$F-p3d3 ;; *) echo "arm must be E, D or H"; exit 2 ;; esac
M=/dev/shm/big-small-ouro; R=$M/runs; cd $R
[ -f $R/$NAME/TRAIN_DONE ] || MODEL=$M/model-2.6b DATA=$M/mix-data-48m ./run_mix2.sh $NAME $ARM $GPUS $PORT 48000000 --micro 1 $FLAG
