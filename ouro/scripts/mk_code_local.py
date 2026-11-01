"""Offline copies of lm-eval's humaneval_instruct / mbpp_plus_instruct (same prompts, filters, metrics), reading the
already-downloaded parquet files instead of the Hub (datasets offline mode cannot resolve these repos)."""
import re, shutil
from pathlib import Path
M = "/dev/shm/big-small-ouro"; T = Path(f"{M}/lmeval-pkgs/lm_eval/tasks"); O = Path("/root/big-small/ouro/lmeval_tasks")
HE = f"{M}/hf-lmeval/hub/datasets--openai--openai_humaneval/snapshots/7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544/openai_humaneval/test-00000-of-00001.parquet"
MB = f"{M}/hf-lmeval/hub/datasets--evalplus--mbppplus/snapshots/b2d74c91837c3f2a20c1299ae98133cbe7cfa077/data/test-00000-of-00001-d5781c9c51e02795.parquet"
def local(src_dir, files, data, renames):
    out = O / (src_dir + "_local"); out.mkdir(parents=True, exist_ok=True)
    shutil.copy(T / src_dir / "utils.py", out / "utils.py")
    for f in files:
        s = (T / src_dir / f).read_text()
        for a, b in renames.items():
            s = re.sub(rf"^(task|include): {re.escape(a)}$", lambda m: f"{m.group(1)}: {b}", s, flags=re.M)
        s = re.sub(r"^dataset_path: .*$", "dataset_path: parquet", s, flags=re.M)
        s = re.sub(r"^dataset_name: .*\n", "", s, flags=re.M)
        if "dataset_path: parquet" in s:
            s += f"\ndataset_kwargs:\n  data_files:\n    test: {data}\n"
        (out / renames.get(f, f)).write_text(s)
    print(out, sorted(p.name for p in out.iterdir()))
local("humaneval", ["humaneval.yaml", "humaneval_instruct.yaml"], HE,
      {"humaneval": "humaneval_local", "humaneval.yaml": "humaneval_local.yaml",
       "humaneval_instruct": "humaneval_instruct_local", "humaneval_instruct.yaml": "humaneval_instruct_local.yaml"})
local("mbpp", ["mbpp.yaml", "mbpp_instruct.yaml", "mbpp_plus_instruct.yaml"], MB,
      {"mbpp": "mbpp_local", "mbpp.yaml": "mbpp_local.yaml", "mbpp_instruct": "mbpp_instruct_local",
       "mbpp_instruct.yaml": "mbpp_instruct_local.yaml", "mbpp_plus_instruct": "mbpp_plus_instruct_local",
       "mbpp_plus_instruct.yaml": "mbpp_plus_instruct_local.yaml"})
