"""Check that the fast aligned decoder (single_bank.AlignedRunner, used for throughput) produces the same greedy tokens
as the evaluation runner (adapter_model.AdapterRunner with arm a) at a few cells.

    python check_aligned_runner.py MODEL_DIR CHECKPOINT
"""
import sys

import torch
from transformers import AutoTokenizer

from ouro.adapter_model import load_adapter_runner
from ouro.single_bank import AlignedRunner

model_dir, ckpt = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
prompts = ["Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
           "How many clips did Natalia sell altogether in April and May?\nA:",
           "The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"]


def greedy(runner, ids, steps=32):
    logits, cache = runner.prefill(ids)
    out = []
    for _ in range(steps):
        nxt = int(logits.argmax(-1))
        out.append(nxt)
        logits, cache = runner.decode(torch.tensor([[nxt]], device="cuda"), cache)
    return out


ref = load_adapter_runner(model_dir, ckpt)
for p, d in ((4, 2), (4, 4), (2, 1)):
    ref.system.prefill_steps, ref.system.decode_steps = p, d
    fast = AlignedRunner(ref.system, p, d)
    same = total = 0
    for text in prompts:
        ids = torch.tensor([tok.encode(text)], device="cuda")
        a, b = greedy(ref, ids), greedy(fast, ids)
        same += sum(x == y for x, y in zip(a, b)); total += len(a)
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        print(f"a{p}{d} first difference at token {first}")
    print(f"a{p}{d}: {same}/{total} greedy tokens agree")
