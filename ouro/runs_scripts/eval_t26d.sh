#!/bin/bash
# After T26-D-native22 finishes: the distillation baseline (arm D, native 2/2 LoRA) on the eight headline benchmarks,
# same tasks and settings as Figure 4 (one sequential queue per a1 GPU; lm1.sh / bench3.sh are resumable).
M=/dev/shm/big-small-ouro; R=$M/runs; CK=$R/T26-D-native22/checkpoint-48m.pt; MP=$M/model-2.6b-think
cd $R
until [ -f $R/T26-D-native22/TRAIN_DONE ]; do sleep 300; done
ls $CK || CK=$(ls $R/T26-D-native22/checkpoint*.pt | tail -1)
MA="path=$MP,condition=D,checkpoint=$CK,max_length=16384"
think() { ./lm1.sh $1 T26-D-$2 - "$MA,batch_tokens=160000,max_batch=64" "--tasks $2 --apply_chat_template"; }
lm() { ./lm1.sh $1 T26x-D-$2 - "$MA" "$3"; }
helmet() { cd /root/big-small; NO_EMPTY=1 MODEL=$MP DATA=$M/bench-mini BENCHES=helmet BATCH=65536 ./ouro/runs_scripts/bench3.sh T26ne-D-helmet D $CK $1 1; cd $R; }
q() { local g=$1; shift; ( for j in "$@"; do eval "$j"; done ) > /dev/null 2>&1 < /dev/null & }
q 0 "think 0 math500_think_s0" "think 0 gsm8k_think_s0"
q 1 "think 1 math500_think_s1" "think 1 gsm8k_think_s1"
q 2 "think 2 math500_think_s2" "think 2 gsm8k_think_s2"
q 3 "think 3 math500_think_s3" "think 3 gsm8k_think_s3"
q 4 "think 4 math500_think_s4"
q 5 "helmet 5"
q 6 "lm 6 mmlupro '--tasks mmlu_pro --limit 30'" "lm 6 arc '--tasks arc_challenge --num_fewshot 25'"
q 7 "lm 7 bbh '--tasks bbh_cot_fewshot --limit 60'" "lm 7 code '--tasks humaneval_local,mbpp_plus_instruct_local --confirm_run_unsafe_code'"
wait
