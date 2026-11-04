#!/bin/bash
# launch_seed3.sh GPUS PORT [SEED] [SIZE]: another training seed of the loop adapter (X26-S-universal / O14-S-universal recipe,
# --seed SEED, default 3; SIZE 26 or 14), once all of GPUS are idle; then its Table 1 cells (decode 1-2, every setting)
# via seed2_q.sh on the same GPUs.
G=$1; PORT=$2; SD=${3:-3}; Z=${4:-26}; M=/dev/shm/big-small-ouro; R=$M/runs; cd $R
[ $Z = 14 ] && { N=O14-S-seed$SD; MODEL=$M/model-1.4b; } || { N=X26-S-seed$SD; MODEL=$M/model-2.6b; }
idle() { for g in ${G//,/ }; do [ -z "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader)" ] || return 1; done; }
until idle && sleep 120 && idle; do sleep 120; done
echo "$(date +%T) $N on $G"
[ -f $R/$N/TRAIN_DONE ] || MODEL=$MODEL DATA=$M/mix-data-48m ./run_mix2.sh $N S $G $PORT 48000000 --micro 1 --seed $SD || exit 1
SEED=$SD SIZE=$Z ./seed2_q.sh lm ${G//,/ }; SEED=$SD SIZE=$Z ./seed2_q.sh rest
