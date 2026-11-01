"""greedy_batch with LoopedRunner: batch of 4 (left-padded, masked) vs one row at a time, on RULER-4K rows."""
import json, sys
import torch
from transformers import AutoTokenizer
from ouro.evaluate_bench import greedy_batch
from ouro.looped_bridge import LoopedRunner, tokenizer

path, cond = sys.argv[1], sys.argv[2]
ckpt = sys.argv[3] if len(sys.argv) > 3 else None
runner = LoopedRunner(path, cond, ckpt)
tok = tokenizer(path)
src = AutoTokenizer.from_pretrained("/dev/shm/big-small-ouro/model", trust_remote_code=True)
rows = [json.loads(l) for l in open("/dev/shm/big-small-ouro/bench/ruler4k.jsonl")][:4]
for r in rows:
    r["prompt_ids"] = tok.encode(src.decode(r["prompt_ids"]), add_special_tokens=False)
    r["max_new_tokens"] = 24
with torch.inference_mode():
    batched = greedy_batch(runner, tok, rows)
    single = [greedy_batch(runner, tok, [r])[0] for r in rows]
for r, b, s in zip(rows, batched, single):
    print(r["answers"], "| B:", repr(b[0][:60]), "| 1:", repr(s[0][:60]))
