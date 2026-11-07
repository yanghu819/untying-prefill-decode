"""Per-request time of real lm-evaluation-harness requests (batch 1, one idle GPU): replay the exact prompts and stop
rules of a finished evaluation run and time prefill (time to first token: the p-loop prefill and the first-token
logits), decode (every further token) and the whole answer, with the input and output lengths. The replayed answer is
compared with the evaluated one and stored with the target, so paper_numbers.py scores the replayed answer itself with
the evaluation's rule (item id `<task>/<doc_id>`): quality and time then come from the same requests.

usage: time_requests.py MODEL_DIR COND CKPT|- SAMPLES_DIR N OUT.jsonl
  COND: native<d>st | D<d>st (LoRA-distilled, merged; with CKPT) | S<p><d>sb | a<p><d>sb | a<p>x<d>sb (bench_throughput.make)
  SAMPLES_DIR: lm-eval output directory of the same task (its samples_<task>_*.jsonl files; the newest per task)
  N: requests per task file group, chosen by a hash of the item id (the same items for every condition)
"""
import glob
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from transformers import AutoTokenizer

from ouro.bench_throughput import make

model_dir, cond, ckpt, samples_dir, n, out = sys.argv[1:7]
n, out = int(n), Path(out)
MAX_LENGTH = 16384   # as the evaluation (lmeval_ouro.py, max_length=16384)

newest = {}
for f in glob.glob(os.path.join(samples_dir, "*", "samples_*.jsonl")):
    task, stamp = os.path.basename(f)[:-len(".jsonl")].split("samples_")[1].rsplit("_", 1)
    newest[task] = max(newest.get(task, (stamp, f)), (stamp, f))
requests = {}
for task, (_, f) in newest.items():
    for r in map(json.loads, open(f)):
        requests.setdefault(f"{task}/{r['doc_id']}", r)   # one record per filter: the prompt and response are shared
key = lambda i: hashlib.md5(i.encode()).hexdigest()
chosen = sorted(requests, key=key)[:n]

tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
gen_cfg = Path(model_dir) / "generation_config.json"
eos = json.load(open(gen_cfg)).get("eos_token_id") if gen_cfg.exists() else None
stop_ids = {e for e in (eos if isinstance(eos, list) else [eos]) + [tok.eos_token_id] if e is not None}
runner = make(cond, model_dir, None if ckpt == "-" else ckpt, merge=ckpt != "-")


def timed(fn):
    torch.cuda.synchronize(); t = time.perf_counter(); r = fn(); torch.cuda.synchronize()
    return r, time.perf_counter() - t


def answer(ids, until, max_new):
    x = torch.tensor([ids], device="cuda")
    (logits, cache), prefill = timed(lambda: runner.prefill(x))
    out_ids, text, reason = [], "", "length"
    t0 = time.perf_counter()
    nxt = int(logits.argmax(-1))
    while True:
        if nxt in stop_ids:
            reason = "eos"; break
        out_ids.append(nxt)
        text = tok.decode(out_ids, skip_special_tokens=True)
        if any(u in text for u in until):
            text = text[:min(text.index(u) for u in until if u in text)]; reason = "stop"; break
        if len(out_ids) == max_new:
            break
        logits, cache = runner.decode(torch.tensor([[nxt]], device="cuda"), cache)
        nxt = int(logits.argmax(-1))
    torch.cuda.synchronize()
    del cache
    return text, len(out_ids), reason, prefill, time.perf_counter() - t0


def prompt(r):
    context, kw = r["arguments"]["gen_args_0"]["arg_0"], r["arguments"]["gen_args_0"]["arg_1"]
    until = kw.get("until", [])
    until = [until] if isinstance(until, str) else list(until)
    max_new = int(kw.get("max_gen_toks", 256))
    ids, limit = tok.encode(context, add_special_tokens=False), MAX_LENGTH - max_new
    if len(ids) > limit:
        ids = ids[:limit // 2] + ids[len(ids) - (limit - limit // 2):]
    return ids, until, max_new


with torch.inference_mode(), open(out, "w") as f:
    answer(*prompt(requests[chosen[0]]))   # warm-up, not recorded
    for i in chosen:
        r = requests[i]
        ids, until, max_new = prompt(r)
        text, new, reason, prefill, decode = answer(ids, until, max_new)
        ref = r["resps"][0][0]
        f.write(json.dumps(dict(id=i, cond=cond, prompt_tokens=len(ids), new_tokens=new, stop=reason,
                                prefill_s=round(prefill, 4), decode_s=round(decode, 4), total_s=round(prefill + decode, 4),
                                match=text.strip() == ref.strip(), text=text, target=r["target"])) + "\n")
        f.flush()
