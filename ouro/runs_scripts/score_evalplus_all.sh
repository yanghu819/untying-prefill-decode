#!/bin/bash
# score_evalplus_all.sh: score every lm-eval run of this box whose evalplus chat samples (HumanEval+, MBPP+) have no
# evalplus_*_chat.jsonl yet or a newer sample file, with ouro/scripts/score_evalplus.py (CPU; runs model-written code
# in evalplus's sandbox). Run after new code evaluations land; paper_numbers.py export reads the scored files.
R=/dev/shm/big-small-ouro/runs; M=/dev/shm/big-small-ouro; cd $R/lmeval/out
todo=""
for d in */; do
  d=${d%/}
  s=$(ls $d/*/samples_humanevalplus_chat_local_*.jsonl 2>/dev/null | tail -1); [ -n "$s" ] || continue
  [ -f $d/evalplus_humaneval_chat.jsonl ] && [ $d/evalplus_humaneval_chat.jsonl -nt "$s" ] && continue
  todo="$todo $d"
done
echo "to score:$todo"
[ -n "$todo" ] && cd /root/big-small && PYTHONPATH=$M/lmeval-pkgs:/root/big-small HF_HUB_OFFLINE=1 \
  environments/ouro/.venv/bin/python ouro/scripts/score_evalplus.py $R $todo
