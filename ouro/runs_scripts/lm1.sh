#!/bin/bash
# lm1.sh GPU NAME PYTHONPATH_PREFIX|- "MODEL_ARGS" "TASK_ARGS" : one lm-eval job (our runner) on one exclusive GPU,
# resumable through its sqlite request cache.
G=$1 NAME=$2 PP=$3 MA=$4 TA=$5; [ "$PP" = "-" ] && PP=
M=/dev/shm/big-small-ouro; Q=$M/runs/lmeval; cd /root/big-small; mkdir -p $Q/cache $Q/out $Q/logs
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 HF_ALLOW_CODE_EVAL=1 HF_HOME=$M/hf-lmeval PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$PP$M/lmeval-pkgs:/root/big-small environments/ouro/.venv/bin/python ouro/run_lmeval.py \
  --model ouro --model_args $MA $TA --include_path ouro/lmeval_tasks --use_cache $Q/cache/$NAME --output_path $Q/out/$NAME --log_samples \
  > $Q/logs/$NAME.log 2>&1
