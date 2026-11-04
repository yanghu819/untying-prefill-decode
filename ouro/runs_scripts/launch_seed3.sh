#!/bin/bash
# launch_seed3.sh GPUS PORT: a third training seed of the Ouro-2.6B loop adapter (X26-S-universal recipe, --seed 3), once all
# of GPUS are idle; then its Table 1 cells (decode 1-2, every setting) via seed2_q.sh with SEED=3 on the same GPUs.
G=$1; PORT=$2; M=/dev/shm/big-small-ouro; R=$M/runs; N=X26-S-seed3; cd $R
idle() { for g in ${G//,/ }; do [ -z "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader)" ] || return 1; done; }
until idle && sleep 120 && idle; do sleep 120; done
echo "$(date +%T) $N on $G"
[ -f $R/$N/TRAIN_DONE ] || MODEL=$M/model-2.6b DATA=$M/mix-data-48m ./run_mix2.sh $N S $G $PORT 48000000 --micro 1 --seed 3 || exit 1
SEED=3 ./seed2_q.sh lm ${G//,/ }; SEED=3 ./seed2_q.sh rest
