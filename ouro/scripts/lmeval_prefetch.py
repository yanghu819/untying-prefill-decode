"""Download every dataset the lm-eval queue needs (retrying through hf-mirror), so workers can run offline."""
import time
from lm_eval.tasks import TaskManager, get_task_dict
tm = TaskManager(include_path="/root/big-small/ouro/lmeval_tasks")
for task in ["gsm8k", "gsm8k_cot", "bbh_cot_fewshot", "minerva_math500", "mmlu", "arc_challenge", "hellaswag", "winogrande", "mmlu_pro"]:
    for attempt in range(20):
        try:
            get_task_dict([task], tm); print("ok", task, flush=True); break
        except Exception as e:
            print("retry", task, attempt, repr(e)[:120], flush=True); time.sleep(10)
