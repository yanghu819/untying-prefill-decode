"""Offline LongBench (THUDM, English tasks) for lm-eval: lm-eval's own prompts / metrics, local jsonl data from the
downloaded data.zip, greedy decoding (the shipped configs sample at temperature 1; LongBench reports greedy)."""
import re, shutil, zipfile
from pathlib import Path
M = "/dev/shm/big-small-ouro"; T = Path(f"{M}/lmeval-pkgs/lm_eval/tasks/longbench"); O = Path("/root/big-small/ouro/lmeval_tasks/longbench_local")
DATA = Path(f"{M}/longbench-data")
zp = next(Path(f"{M}/hf-lmeval/hub/datasets--THUDM--LongBench/snapshots").glob("*/data.zip"))
if not DATA.exists():
    zipfile.ZipFile(zp).extractall(DATA)
files = {p.stem: p for p in DATA.rglob("*.jsonl")}
EN = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum", "multi_news",
      "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
O.mkdir(parents=True, exist_ok=True)
for f in ("metrics.py", "utils.py"):
    if (T / f).exists(): shutil.copy(T / f, O / f)
for name in EN:
    s = (T / f"{name}.yaml").read_text()
    s = re.sub(r"^tag:\n(  - .*\n)+", "tag:\n  - longbench_local\n", s, flags=re.M)
    s = re.sub(r"^task: longbench_(.*)$", r"task: lbl_\1", s, flags=re.M)
    s = re.sub(r"^dataset_path: .*$", "dataset_path: json", s, flags=re.M)
    s = re.sub(r"^dataset_name: .*\n", "", s, flags=re.M)
    s = re.sub(r"^  temperature: .*$", "  temperature: 0", s, flags=re.M)
    s = re.sub(r"^  do_sample: .*$", "  do_sample: false", s, flags=re.M)
    s += f"dataset_kwargs:\n  data_files:\n    test: {files[name]}\n"
    (O / f"{name}.yaml").write_text(s)
print(len(EN), "tasks ->", O, "| data files:", len(files))
