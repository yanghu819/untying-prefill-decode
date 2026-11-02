#!/bin/bash
# SenseNova-Loop-9B: (B) where deep reading lives in a hybrid — the writer gets deep attention KV only (A) or deep
# Gated-DeltaNet conv + recurrent state only (G), untrained (splice21A/G) and with the trained connector (C21A/G);
# (A) headline set at write 1: native 1/1, connector read 1 / write 1 (C11) and read 2 / write 1 (C21), native 2/2.
# One sequential queue per GPU on a1 (lm1.sh / bench3.sh are resumable).
M=/dev/shm/big-small-ouro; R=$M/runs; CK=$R/SN-C-universal/checkpoint.pt; MP=$M/model-sn9b
cd $R
ck() { case $1 in C*) echo "$CK";; *) echo "-";; esac; }
ma() { local a="path=$MP,condition=$1,max_length=16384,batch_tokens=250000,max_batch=64"; [ "$(ck $1)" != "-" ] && a="$a,checkpoint=$CK"; echo "$a"; }
ruler() { MODEL=$MP PP=$M/tf518:/root/big-small DATA=$M/bench BENCHES='ruler4k ruler8k ruler16k' BATCH=24576 OURO_NO_CUDNN_SDP=1 ./bench3.sh SNnt-$2 $2 $(ck $2) $1 1; }
helmet() { MODEL=$MP PP=$M/tf518:/root/big-small DATA=$M/bench BENCHES=helmet BATCH=24576 OURO_NO_CUDNN_SDP=1 ./bench3.sh SNnt-$2-helmet $2 $(ck $2) $1 1; }
mmlupro() { OURO_NO_CUDNN_SDP=1 ./lm1.sh $1 SN-$2-mmlupro $M/tf518: "$(ma $2)" "--tasks mmlu_pro --num_fewshot 5 --limit 70"; }
arc() { OURO_NO_CUDNN_SDP=1 ./lm1.sh $1 SN-$2-arc $M/tf518: "$(ma $2)" "--tasks arc_challenge --num_fewshot 25"; }
code() { OURO_NO_CUDNN_SDP=1 ./lm1.sh $1 SN-$2-code $M/tf518: "$(ma $2)" "--tasks humaneval_local,mbpp_plus_instruct_local --confirm_run_unsafe_code"; }
math() { OURO_NO_CUDNN_SDP=1 ./lm1.sh $1 SN-$2-math500_think_s$3 $M/tf518: "$(ma $2)" "--tasks math500_think_s$3 --apply_chat_template"; }
q() { local g=$1; shift; ( for j in "$@"; do eval "$j"; done ) > /dev/null 2>&1 < /dev/null & }
q 0 "ruler 0 splice21A" "mmlupro 0 splice21A" "helmet 0 splice21A" "arc 0 native1" "code 0 native1"
q 1 "ruler 1 splice21G" "mmlupro 1 splice21G" "helmet 1 splice21G" "arc 1 native2" "code 1 native2"
q 2 "ruler 2 C21A" "mmlupro 2 C21A" "helmet 2 C21A" "arc 2 C11" "code 2 C11"
q 3 "ruler 3 C21G" "mmlupro 3 C21G" "helmet 3 C21G" "arc 3 C21" "code 3 C21"
q 4 "math 4 C21 0" "math 4 C21 1" "math 4 C21 2"
q 5 "math 5 C21 3" "math 5 C21 4" "math 5 C11 0"
q 6 "math 6 C11 1" "math 6 C11 2" "math 6 C11 3" "math 6 C11 4"
q 7 "math 7 native2 0" "math 7 native2 1" "math 7 native2 2"
echo queued
