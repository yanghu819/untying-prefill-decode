#!/bin/bash
# evalplus_q.sh SET GPU...: HumanEval(+) and MBPP(+) as Ouro's base-model evaluation ran them (evalplus, chat prompt,
# greedy; tasks humanevalplus_chat_local / mbppplus_chat_local; native 4/4 reproduces Ouro's reported scores within
# 1-3 points). Score afterwards with ouro/scripts/score_evalplus.py. SET: 26 (Ouro-2.6B cells, natives, distilled
# controls; 26x: only the controls on this box) | 14 (Ouro-1.4B cells and natives) | T26 (the Ouro-2.6B-Thinking configurations of Table 7). Two jobs per GPU.
S=$1; shift; M=/dev/shm/big-small-ouro; R=$M/runs; Q=$R/evalplus_q.txt; cd $R
CELLS="native1 native2 native3 native4 S11 S21 S31 S41 S12 S22 S32 S42 S13 S23 S33 S43 S14 S24 S34 S44"
case $S in
  26|26x) MP=$M/model-2.6b; CK=$R/X26-S-universal/checkpoint-48m.pt; P=o26x
      EXTRA="E:X26-E48-p1d1:E48 D:X26-D48-p2d2:D48 H:X26-H48-p3d3:H48 E:X26-E48f-p1d1:E48f D:X26-D48f-p2d2:D48f H:X26-H48f-p3d3:H48f"
      [ $S = 26x ] && CELLS="" ;;
  14) MP=$M/model-1.4b; CK=$R/O14-S-universal/checkpoint-48m.pt; P=O14x; EXTRA="" ;;
  T26) MP=$M/model-2.6b-think; CK=$R/T26-S-universal/checkpoint-48m.pt; P=T26x; CELLS="native2 native4 S22 S42"
       EXTRA="D:T26-D-native22:D D:T26-Df-native22:Df" ;;
esac
{
  for c in $CELLS; do
    MA="path=$MP,condition=$c,max_length=16384"; [ ${c:0:1} = S ] && MA="$MA,checkpoint=$CK"
    echo "./lm1.sh \$GPU $P-$c-evalplus - \"$MA\" \"--tasks humanevalplus_chat_local,mbppplus_chat_local\""
  done
  for e in $EXTRA; do   # arm:run:name of a LoRA-distilled native control
    IFS=: read -r arm run name <<< "$e"; [ -f $R/$run/checkpoint-48m.pt ] || continue
    echo "./lm1.sh \$GPU $P-$name-evalplus - \"path=$MP,condition=$arm,checkpoint=$R/$run/checkpoint-48m.pt,max_length=16384\" \"--tasks humanevalplus_chat_local,mbppplus_chat_local\""
  done
} | flock $Q.lock sh -c "cat >> $Q"
for g in "$@"; do setsid nohup ./grid_worker.sh $g $Q 2 > /dev/null 2>&1 < /dev/null & done
