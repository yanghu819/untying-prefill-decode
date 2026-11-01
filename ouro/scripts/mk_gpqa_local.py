"""Offline GPQA for lm-eval: lm-eval's own gpqa cot_n_shot / n_shot configs (prompts, few-shot, filters, metrics),
data from the ModelScope mirror csv (AI-ModelScope/gpqa) instead of the gated HF repo."""
import re, shutil
from pathlib import Path
M = "/dev/shm/big-small-ouro"; T = Path(f"{M}/lmeval-pkgs/lm_eval/tasks/gpqa"); O = Path("/root/big-small/ouro/lmeval_tasks")
for kind, subset in (("cot_n_shot", "diamond"), ("n_shot", "main"), ("n_shot", "diamond")):
    out = O / f"gpqa_{kind}_local"; out.mkdir(parents=True, exist_ok=True)
    shutil.copy(T / kind / "utils.py", out / "utils.py")
    base = (T / kind / f"_gpqa_{kind}_yaml").read_text()
    base = re.sub(r"^dataset_path: .*$", "dataset_path: csv", base, flags=re.M)
    base = re.sub(r"^tag: .*$", "tag: gpqa_local", base, flags=re.M)
    (out / f"_gpqa_{kind}_local_yaml").write_text(base)
    task = (T / kind / f"gpqa_{subset}_{kind}.yaml").read_text()
    task = re.sub(r"^include: .*$", f"include: _gpqa_{kind}_local_yaml", task, flags=re.M)
    task = re.sub(r"^task: (.*)$", r"task: \1_local", task, flags=re.M)
    task = re.sub(r"^dataset_name: .*\n", "", task, flags=re.M)
    task += f"dataset_kwargs:\n  data_files:\n    train: {M}/gpqa-data/gpqa_{subset}.csv\n"
    (out / f"gpqa_{subset}_{kind}_local.yaml").write_text(task)
    print(out / f"gpqa_{subset}_{kind}_local.yaml"); print(task)
