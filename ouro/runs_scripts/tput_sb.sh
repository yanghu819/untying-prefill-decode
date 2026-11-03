#!/bin/bash
# tput_sb.sh GPU COND [CTX]   (env: BATCHES, STEPS = decode steps per trial, TPUT_DIR, CKPT) : decode throughput / memory vs batch at CTX (default 8K) context (exclusive GPU), merged LoRA;
# COND = native<d> | S<p><d>sb (single-bank) | O<p><d>sb (single prompt bank + shared answer bank; CKPT=arm O) | S<p><d>
# (same loop adapter, one materialised bank per decode loop)
G=$1 C=$2 X=${3:-8192}; M=/dev/shm/big-small-ouro; O=$M/runs/${TPUT_DIR:-throughput-sb2}; mkdir -p $O; cd /root/big-small
CK=""; [ "${C#native}" = "$C" ] && CK="--checkpoint ${CKPT:-$M/runs/X26-S-universal/checkpoint-48m.pt} --merge-lora"
T=$C; [ $X != 8192 ] && T=$C-$X
CUDA_VISIBLE_DEVICES=$G STATIC_MAX_NEW=${STEPS:-32} OURO_NO_CUDNN_SDP=1 HF_HUB_OFFLINE=1 environments/ouro/.venv/bin/python -m ouro.bench_throughput --model-path $M/model-2.6b --condition $C $CK --prompts $M/bench/ruler16k.jsonl --contexts $X --batches ${BATCHES:-1 4 8 12 16 20 24 32 40 48 56 64 80 96 112 128} --steps ${STEPS:-32} --row-prefill --output $O/$T.json > $O/$T.log 2>&1
