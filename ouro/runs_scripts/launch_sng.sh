#!/bin/bash
# SenseNova-Loop-9B loop adapter trained with the Gated-DeltaNet-only deep state (--mix G): same recipe as SN-C-universal
# (launch_sn.sh: 3 GPUs, sn-mix, 20M KL tokens, p 1/2 at 50/50, seed 0) except that read-2 samples give the decoder the
# second visit's GDN state and the first visit's attention KV. Then evaluates C21G (in distribution) and C11.
M=/dev/shm/big-small-ouro; R=$M/runs; N=SNG-C-universal; GPUS=${1:-4,5,6}
cd /root/big-small
[ -f $R/$N/TRAIN_DONE ] || {
  CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=$M/tf518:/root/big-small environments/ouro/.venv/bin/python -m torch.distributed.run \
    --nproc_per_node 3 --master_port 29943 -m ouro.sensenova_bridge --model-path $M/model-sn9b --data $M/sn-mix/train.jsonl \
    --output $R/$N --kl-tokens 20e6 --tokens-per-step 65536 --mix G > $R/$N.log 2>&1 && touch $R/$N/TRAIN_DONE; } || exit 1
cd $R
CK=$R/$N/checkpoint.pt; MP=$M/model-sn9b
ma() { echo "path=$MP,condition=$1,max_length=16384,batch_tokens=250000,max_batch=64,checkpoint=$CK"; }
ruler() { MODEL=$MP PP=$M/tf518:/root/big-small DATA=$M/bench BENCHES='ruler4k ruler8k ruler16k' BATCH=24576 OURO_NO_CUDNN_SDP=1 ./bench3.sh SNGnt-$2 $2 $CK $1 1; }
helmet() { MODEL=$MP PP=$M/tf518:/root/big-small DATA=$M/bench BENCHES=helmet BATCH=24576 OURO_NO_CUDNN_SDP=1 ./bench3.sh SNGnt-$2-helmet $2 $CK $1 1; }
bbh() { MODEL=$MP PP=$M/tf518:/root/big-small DATA=$M/bench BENCHES=bbh BATCH=24576 OURO_NO_CUDNN_SDP=1 ./bench3.sh SNGnt-$2-bbh $2 $CK $1 1; }
mmlupro() { OURO_NO_CUDNN_SDP=1 ./lm1.sh $1 SNG-$2-mmlupro $M/tf518: "$(ma $2)" "--tasks mmlu_pro --num_fewshot 5 --limit 70"; }
arc() { OURO_NO_CUDNN_SDP=1 ./lm1.sh $1 SNG-$2-arc $M/tf518: "$(ma $2)" "--tasks arc_challenge --num_fewshot 25"; }
gsm() { OURO_NO_CUDNN_SDP=1 ./lm1.sh $1 SNG-$2-gsm8k_think_s$3 $M/tf518: "$(ma $2)" "--tasks gsm8k_think_s$3 --apply_chat_template"; }
q() { local g=$1; shift; ( for j in "$@"; do eval "$j"; done ) > /dev/null 2>&1 < /dev/null & }
G=(${GPUS//,/ }) ; G+=(7)
q ${G[0]} "ruler ${G[0]} C21G" "mmlupro ${G[0]} C21G" "gsm ${G[0]} C21G 0" "gsm ${G[0]} C21G 1"
q ${G[1]} "ruler ${G[1]} C11" "mmlupro ${G[1]} C11" "gsm ${G[1]} C11 0" "gsm ${G[1]} C11 1"
q ${G[2]} "helmet ${G[2]} C21G" "arc ${G[2]} C21G" "gsm ${G[2]} C21G 2" "gsm ${G[2]} C21G 3"
q ${G[3]} "helmet ${G[3]} C11" "arc ${G[3]} C11" "bbh ${G[3]} C21G" "bbh ${G[3]} C11" "gsm ${G[3]} C11 2" "gsm ${G[3]} C11 3"
wait
