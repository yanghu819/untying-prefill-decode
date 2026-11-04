"""Overlap audit of the adaptation data against the evaluation questions (13-gram token matches, as in GPT-3's
contamination analysis).

usage: overlap_audit.py DATA_DIR MODEL_DIR RUNS_DIR OUT.json RUN...
  DATA_DIR  adaptation mixture (samples.npy, gsm_train.jsonl, fineweb_train.bin; every sample the training used)
  RUN       lm-eval output directories under RUNS_DIR/lmeval/out (one per benchmark; their docs give the questions)
Writes, per run, the item keys (task/doc_id, as in paper_numbers.py) whose question shares any 13-token span with the
training data, the number of items checked, and the items too short to test (fewer than 13 tokens: unassessed).
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from ouro.prepare_adapter_data import AdapterDataset

N = 13
FIELDS = ("question", "Question", "problem", "input", "prompt", "instruction", "ctx", "sentence")   # question text per task
P = np.uint64(1000003)


def ngram_hashes(ids):
    """Polynomial hashes (mod 2^64) of every N-token window of ids."""
    a = np.asarray(ids, dtype=np.uint64)
    if len(a) < N:
        return np.empty(0, dtype=np.uint64)
    h = np.zeros(len(a) - N + 1, dtype=np.uint64)
    for k in range(N):
        h = h * P + a[k: len(a) - N + 1 + k]
    return h


def main():
    data, model, runs, out, names = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5:]
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    items = {}                                                  # key -> n-gram hashes of its question
    for name in names:
        for f in glob.glob(f"{runs}/lmeval/out/{name}/*/samples_*.jsonl"):
            task = Path(f).name.split("samples_")[1].rsplit("_", 1)[0]
            for line in open(f):
                r = json.loads(line)
                text = next((r["doc"][k] for k in FIELDS if isinstance(r["doc"].get(k), str)), "")
                items[(name, f"{task}/{r['doc_id']}")] = ngram_hashes(tok.encode(text, add_special_tokens=False))
    probe = np.unique(np.concatenate([h for h in items.values() if len(h)]))
    seen = np.zeros(len(probe), dtype=bool)
    ds = AdapterDataset(data)
    done, chunk, size = set(), [], 0
    for i in range(len(ds)):                                    # every distinct sequence the training used
        source, item, _ = map(int, ds.samples[i])
        if (source, item) in done:
            continue
        done.add((source, item))
        s = ds[i]
        chunk.append(ngram_hashes(s["prefix_ids"] + s["suffix_ids"])); size += len(chunk[-1])
        if size > 50_000_000 or i == len(ds) - 1:               # one membership test per ~50M training n-grams
            seen |= np.isin(probe, np.concatenate(chunk))
            chunk, size = [], 0
    if chunk:
        seen |= np.isin(probe, np.concatenate(chunk))
    hit = set(probe[seen].tolist())
    report = {}
    for (name, key), h in items.items():
        r = report.setdefault(name, dict(items=0, overlapping=[], unassessed=[]))
        r["items"] += 1
        if not len(h):
            r["unassessed"].append(key)
        elif any(int(x) in hit for x in h):
            r["overlapping"].append(key)
    out.write_text(json.dumps(dict(n=N, training_sequences=len(done), runs=report), indent=1))
    for name, r in report.items():
        print(f"{name:28s} {len(r['overlapping']):5d} of {r['items']:6d} items share a {N}-gram with the training data; "
              f"{len(r['unassessed'])} too short to test")


if __name__ == "__main__":
    main()
