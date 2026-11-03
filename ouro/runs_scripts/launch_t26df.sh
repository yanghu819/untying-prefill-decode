#!/bin/bash
# launch_t26df.sh GPUS PORT: the Ouro-2.6B-Thinking distillation control with the first answer token supervised
# (arm D, --first-target; otherwise as launch_t26d.sh), once all of GPUS are idle; then its eight Table 7 benchmarks,
# queued for grid_worker.sh with the settings of eval_t26d.sh (run names T26-Df-*, T26x-Df-*, T26ne-Df-helmet).
G=$1; PORT=$2; M=/dev/shm/big-small-ouro; R=$M/runs; N=T26-Df-native22; Q=$R/t26df_q.txt; MP=$M/model-2.6b-think; cd $R
idle() { for g in ${G//,/ }; do [ -z "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader)" ] || return 1; done; }
until idle && sleep 120 && idle; do sleep 120; done
echo "$(date +%T) $N on $G"
[ -f $R/$N/TRAIN_DONE ] || MODEL=$MP DATA=$M/mix-think-48m ./run_mix2.sh $N D $G $PORT 48000000 --micro 1 --first-target || exit 1
CK=$R/$N/checkpoint-48m.pt; [ -f $CK ] || CK=$(ls $R/$N/checkpoint*.pt | tail -1)
MA="path=$MP,condition=D,checkpoint=$CK,max_length=16384"
{
  for s in 0 1 2 3 4; do echo "./lm1.sh \$GPU T26-Df-math500_think_s$s - \"$MA,batch_tokens=160000,max_batch=64\" \"--tasks math500_think_s$s --apply_chat_template\""; done
  for s in 0 1 2 3; do echo "./lm1.sh \$GPU T26-Df-gsm8k_think_s$s - \"$MA,batch_tokens=160000,max_batch=64\" \"--tasks gsm8k_think_s$s --apply_chat_template\""; done
  echo "./lm1.sh \$GPU T26x-Df-mmlupro - \"$MA\" \"--tasks mmlu_pro --limit 30\""
  echo "./lm1.sh \$GPU T26x-Df-arc - \"$MA\" \"--tasks arc_challenge --num_fewshot 25\""
  echo "./lm1.sh \$GPU T26x-Df-bbh - \"$MA\" \"--tasks bbh_cot_fewshot --limit 60\""
  echo "./lm1.sh \$GPU T26x-Df-code - \"$MA\" \"--tasks humaneval_local,mbpp_plus_instruct_local --confirm_run_unsafe_code\""
  echo "cd /root/big-small && NO_EMPTY=1 MODEL=$MP DATA=$M/bench-mini BENCHES=helmet BATCH=65536 ./ouro/runs_scripts/bench3.sh T26ne-Df-helmet D $CK \$GPU 1"
} | flock $Q.lock sh -c "cat >> $Q"
for g in ${G//,/ }; do setsid nohup ./grid_worker.sh $g $Q > /dev/null 2>&1 < /dev/null & done
