"""Rescore saved MBPP+ generations (task mbpp_plus_instruct_local, which lm-eval scored on the first base assertion only)
with the full MBPP+ tests (the dataset's `test` field). Writes <run>/mbppplus.jsonl (doc_id, pass) next to the samples;
paper_numbers.py reads it as metric `mbppplus` of key L:<run>:mbppplus.

usage: rescore_mbpp_plus.py RUNS_DIR RUN...      (needs HF_ALLOW_CODE_EVAL=1; executes model-written code)
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import evaluate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lmeval_tasks", "mbpp_local"))
from utils import extract_code_blocks  # the same code extraction lm-eval used for these generations

code_eval = evaluate.load("code_eval")


def score(args):
    code, test = args
    out = code_eval.compute(references=[test], predictions=[[code]], k=[1], num_workers=1, timeout=20.0)
    return float(out[0]["pass@1"])


def main():
    runs, names = sys.argv[1], sys.argv[2:]
    for name in names:
        files = glob.glob(f"{runs}/lmeval/out/{name}/*/samples_mbpp_plus_instruct_local_*.jsonl")
        if not files:
            continue
        rows = [json.loads(line) for line in open(files[0])]
        jobs = [(extract_code_blocks(r["resps"][0][0]), r["doc"]["test"]) for r in rows]
        with ProcessPoolExecutor(max_workers=32) as pool:
            passed = list(pool.map(score, jobs))
        with open(f"{runs}/lmeval/out/{name}/mbppplus.jsonl", "w") as f:
            for r, p in zip(rows, passed):
                f.write(json.dumps(dict(doc_id=r["doc_id"], **{"pass": p})) + "\n")
        print(f"{name:24s} base first assertion {100 * sum(float(r['pass_at_1']) for r in rows) / len(rows):5.1f}  "
              f"full MBPP+ tests {100 * sum(passed) / len(rows):5.1f}  (n={len(rows)})")


if __name__ == "__main__":
    main()
