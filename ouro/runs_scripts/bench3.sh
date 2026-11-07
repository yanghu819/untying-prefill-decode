#!/bin/bash
# usage: bench3.sh (bench2.sh + DATA=bench dir, e.g. bench-mini) (bench.sh + PP=PYTHONPATH prefix, MODEL=model dir) NAME COND CKPT|- GPUS SHARDS   evaluate_bench.py shards with stall watchdog (resumable).
# env: BENCHES (default: all), BATCH (batched greedy token budget per step; 0 = one row at a time)
cd /root/big-small; M=/dev/shm/big-small-ouro; R=$M/runs; O=$R/bench/$1; mkdir -p $O
COND=$2; CK=""; [ "$3" != "-" ] && CK="--checkpoint $3"; IFS=, read -ra G <<< "$4"; N=${5:-8}
B=${BENCHES:-"bbh math500 helmet ruler4k ruler8k ruler16k longbench mmlu"}
for s in $(seq 0 $((N - 1))); do
  $R/watchdog.sh 30 $O/s$s $O/s$s.log env PYTHONPATH=${PP:-} CUDA_VISIBLE_DEVICES=${G[$((s % ${#G[@]}))]} HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True environments/ouro/.venv/bin/python -m ouro.evaluate_bench eval --model-path ${MODEL:-$M/model} --data ${DATA:-$M/bench} --output $O/s$s --condition $COND $CK --shard-index $s --num-shards $N --benches $B --batch-tokens ${BATCH:-0} &
  P+=($!)
done
ok=1; for p in "${P[@]}"; do wait $p || ok=0; done   # DONE only if every shard succeeded (an OOM shard leaves no DONE)
[ $ok = 1 ] && touch $O/DONE-$(echo $B | tr " " _)
