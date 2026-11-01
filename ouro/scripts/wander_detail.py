"""Shallow-writer wandering, decomposed (Ouro-2.6B-Thinking GSM8K, lm-eval samples): per configuration the share of answers
that hit the 4,096-token cap, that stopped (EOS) without a \\boxed answer, correct answers without \\boxed, and hesitation
markers per 1,000 generated tokens (so shorter answers are not credited mechanically).

usage: wander_detail.py TOKENIZER OUT_DIR [CONFIG=GLOB ...]   (GLOB relative to OUT_DIR, e.g. native1=think-native1)
"""
import glob
import json
import re
import sys

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
root = sys.argv[2]
MARK = re.compile(r"(?i)\balternatively\b|\bwait\b|\bre-?read\b|\bon second thought\b|\blet me (re)?check\b")
rows = {}
for spec in sys.argv[3:]:
    name, pat = spec.split("=")
    R = []
    for f in glob.glob(f"{root}/{pat}-gsm8k_think_s*/*/samples_*.jsonl"):
        R += [json.loads(line) for line in open(f)]
    texts = [r["resps"][0][0] for r in R]
    ntok = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    boxed = ["\\boxed" in t for t in texts]
    cap = [n >= 4090 for n in ntok]
    n = len(R)
    rows[name] = dict(
        n=n, acc=100 * sum(r["math_verify"] for r in R) / n,
        cap_hit=100 * sum(cap) / n,
        eos_unboxed=100 * sum((not c) and (not b) for c, b in zip(cap, boxed)) / n,
        correct_unboxed=100 * sum(r["math_verify"] and not b for r, b in zip(R, boxed)) / n,
        tokens=sum(ntok) / n,
        markers_per_1k=1000 * sum(len(MARK.findall(t)) for t in texts) / sum(ntok),
        markers_per_1k_finished=1000 * sum(len(MARK.findall(t)) for t, c in zip(texts, cap) if not c) / max(1, sum(n_ for n_, c in zip(ntok, cap) if not c)),
    )
    print(name, json.dumps({k: round(v, 2) for k, v in rows[name].items()}), flush=True)
json.dump(rows, open("/tmp/wander_detail.json", "w"), indent=1)
