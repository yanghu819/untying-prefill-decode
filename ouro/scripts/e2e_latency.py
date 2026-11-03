"""End-to-end per-request measurement (batch 1, one idle GPU): time to first token (includes the p-loop prefill and, for
the loop adapter, the decoder's pass over the last prompt token), per-step decode time, total answer time, generated tokens,
stop reason and correctness — for real greedy answers, not a fixed-token replay.

usage: e2e_latency.py MODEL_DIR COND [CKPT|-] TASK N OUT.jsonl   (env START=k: items k .. k+N-1, default 0)
  COND: native<d>st (static-cache native baseline) | S<p><d>sb (fused single-bank loop adapter)
  TASK: gsm8k_think | math500_think (chat template, zero-shot, \\boxed{} answer, math_verify; caps 4,096 / 8,192) | ruler16k (long prompt, short answer;
        timing only — accuracy comes from the full-set evaluations)
"""
import json
import os
from pathlib import Path
import sys
import time

import torch
from transformers import AutoTokenizer


model_dir, cond, ckpt, task, n, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), Path(sys.argv[6])
start = int(os.environ.get("START", 0))
family = json.load(open(Path(model_dir) / "config.json"))["model_type"]
if family == "sensenova_loop":  # native<c> | splice<p><d> | C<p>1 (+ checkpoint); needs transformers 5.18 on PYTHONPATH
    from ouro.sensenova_runner import SenseNovaRunner, tokenizer
    runner = SenseNovaRunner(model_dir, cond, None if ckpt == "-" else ckpt)
    tok = tokenizer(model_dir)
else:
    from ouro.bench_throughput import make  # Ouro runners (transformers 4.x)
    runner = make(cond, model_dir, None if ckpt == "-" else ckpt, merge=ckpt != "-")
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


def chat_ids(msg):
    if family == "sensenova_loop":  # transformers 5 returns a mapping for tokenize=True; render text, then encode
        return tok(tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False), add_special_tokens=False)["input_ids"]
    return tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=True)
gen_cfg = json.load(open(Path(model_dir) / "generation_config.json")) if (Path(model_dir) / "generation_config.json").exists() else {}
eos = gen_cfg.get("eos_token_id", tok.eos_token_id)
stop = set(eos if isinstance(eos, list) else [eos]) | {tok.eos_token_id}


def items():
    if task == "gsm8k_think":
        import datasets
        rows = datasets.load_dataset("gsm8k", "main", split="test")
        for i in range(start, start + n):
            r = rows[i]
            msg = [{"role": "user", "content": r["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
            ids = chat_ids(msg)
            yield i, ids, r["answer"].split("####")[-1].strip().replace(",", ""), 4096
    elif task == "math500_think":
        import datasets
        rows = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
        for i in range(start, start + n):
            r = rows[i]
            msg = [{"role": "user", "content": r["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
            ids = chat_ids(msg)
            yield i, ids, r["answer"], 8192
    else:
        for line in list(open(f"/dev/shm/big-small-ouro/bench/{task}.jsonl"))[start:start + n]:
            r = json.loads(line)
            yield r["id"], r["prompt_ids"], None, 64


def score(gold, text):
    if gold is None:
        return None
    from math_verify import parse, verify
    try:
        return float(verify(parse("$" + gold + "$"), parse(text)))
    except Exception:
        return 0.0


def timed(fn):
    torch.cuda.synchronize(); t = time.perf_counter(); r = fn(); torch.cuda.synchronize()
    return r, time.perf_counter() - t


with torch.inference_mode(), open(out, "w") as f:
    for idx, ids, gold, max_new in items():
        x = torch.tensor([ids], device="cuda")
        (logits, cache), ttft = timed(lambda: runner.prefill(x))
        out_ids, t0 = [], time.perf_counter()
        nxt = int(logits.argmax(-1))
        reason = "length"
        for _ in range(max_new):
            out_ids.append(nxt)
            if nxt in stop:
                reason = "eos"
                break
            if len(out_ids) == max_new:  # cap reached: no further decode step
                break
            logits, cache = runner.decode(torch.tensor([[nxt]], device="cuda"), cache)
            nxt = int(logits.argmax(-1))
        torch.cuda.synchronize(); gen = time.perf_counter() - t0
        text = tok.decode(out_ids, skip_special_tokens=True)
        rec = dict(id=idx, cond=cond, prompt_tokens=len(ids), new_tokens=len(out_ids), stop=reason, ttft_s=round(ttft, 4),
                   gen_s=round(gen, 3), total_s=round(ttft + gen, 3), ms_per_token=round(1000 * gen / max(1, len(out_ids)), 2),
                   correct=score(gold, text))
        f.write(json.dumps(rec) + "\n"); f.flush()
        del cache
