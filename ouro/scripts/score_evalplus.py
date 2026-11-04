"""Score evalplus-protocol generations (tasks humanevalplus_local / mbppplus_local) with evalplus itself, as Ouro's
base-model evaluation does: solution = prompt + completion, evalplus.sanitize, then evalplus.evaluate on the base and
the plus tests. Writes <run>/evalplus_<dataset>.jsonl (doc_id, task_id, base, plus) next to the lm-eval samples;
paper_numbers.py reads metric `base` / `plus` of key L:<run>:evalplus_<dataset>:<base|plus>.

usage: score_evalplus.py RUNS_DIR RUN...     (executes model-written code; run inside the evaluation container)
"""
import glob
import json
import os
import subprocess
import sys
import tempfile

from evalplus.sanitize import sanitize

DATASETS = {"humanevalplus_local": "humaneval", "mbppplus_local": "mbpp",            # direct completion
            "humanevalplus_chat_local": "humaneval", "mbppplus_chat_local": "mbpp"}  # evalplus chat prompt


def main():
    runs, names = sys.argv[1], sys.argv[2:]
    for name in names:
        for task, dataset in DATASETS.items():
            files = glob.glob(f"{runs}/lmeval/out/{name}/*/samples_{task}_*.jsonl")
            if not files:
                continue
            rows = [json.loads(line) for line in open(files[0])]
            with tempfile.TemporaryDirectory() as tmp:
                samples = os.path.join(tmp, "samples.jsonl")
                with open(samples, "w") as f:
                    for r in rows:
                        doc = r["doc"]
                        gen = r["resps"][0][0]   # evalplus: chat output alone, direct completion after its prompt
                        code = sanitize(gen if "chat" in task else doc["prompt"] + gen, entrypoint=doc["entry_point"])
                        f.write(json.dumps(dict(task_id=doc["task_id"], solution=code)) + "\n")
                subprocess.run([sys.executable, "-m", "evalplus.evaluate", "--dataset", dataset, "--samples", samples,
                                "--parallel", "32", "--i-just-wanna-run"], check=True, capture_output=True)
                res = json.load(open(samples.replace(".jsonl", "_eval_results.json")))["eval"]
            ok = lambda v, k: float(v[0][k] == "pass")
            by_id = {t: (ok(v, "base_status"), ok(v, "plus_status")) for t, v in res.items()}
            tag = dataset + ("_chat" if "chat" in task else "")
            with open(f"{runs}/lmeval/out/{name}/evalplus_{tag}.jsonl", "w") as f:
                for r in rows:
                    b, p = by_id[r["doc"]["task_id"]]
                    f.write(json.dumps(dict(doc_id=r["doc_id"], task_id=r["doc"]["task_id"], base=b, plus=p)) + "\n")
            n = len(rows)
            print(f"{name:24s} {tag:14s} base {100 * sum(v[0] for v in by_id.values()) / n:5.1f}  "
                  f"plus {100 * sum(v[1] for v in by_id.values()) / n:5.1f}  (n={n})")


if __name__ == "__main__":
    main()
