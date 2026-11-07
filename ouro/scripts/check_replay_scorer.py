"""Check that paper_numbers.replay_score (used to score the replayed answers of the time study) reproduces the
evaluation's own scores: rescore the evaluated responses of lm-eval runs and compare item by item.

usage: python scripts/check_replay_scorer.py LMEVAL_OUT_DIR RUN...   (e.g. .../lmeval/out X26-a41-gsmfull X26-a41-bbh)
2026-11-07 (X26-a41): GSM8K 1319/1319, BBH 1620/1620, MMLU-Pro 420/420 agree; MATH-500 uses boxed() in both.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import paper_numbers as P  # noqa: E402

out_dir, runs = sys.argv[1], sys.argv[2:]
for run in runs:
    agree = n = 0
    for f in glob.glob(f"{out_dir}/{run}/*/samples_*.jsonl"):
        task = os.path.basename(f)[:-len(".jsonl")].split("samples_")[1].rsplit("_", 1)[0]
        for r in map(json.loads, open(f)):
            if r.get("filter") not in (None, "strict-match", "get-answer", "custom-extract") or "exact_match" not in r:
                continue
            agree += P.replay_score(f"{task}/{r['doc_id']}", r["resps"][0][0], r["target"]) == float(r["exact_match"])
            n += 1
    print(f"{run}: {agree}/{n} agree")
