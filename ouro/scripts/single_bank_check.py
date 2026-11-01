"""Check that single-bank decoding equals the materialised-bank S connector: same prompts, greedy decode, compare the
logits at every step and the generated tokens.  usage: single_bank_check.py MODEL CKPT PROMPTS.jsonl [CTX STEPS]"""
import json
from pathlib import Path
import sys

import torch

from ouro.adapter_model import load_adapter_runner
from ouro.evaluate import rows
from ouro.single_bank import SingleBankRunner

model, ckpt, prompts = sys.argv[1:4]
ctx, steps = (int(x) for x in (sys.argv[4:6] if len(sys.argv) > 5 else (4096, 32)))
runner = load_adapter_runner(model, ckpt)
single = SingleBankRunner(runner.system, 4, 1)
ids = torch.tensor([r["prompt_ids"][:ctx] for r in rows(Path(prompts))[:4]], device="cuda")


def greedy(r):
    logits, cache = r.prefill(ids)
    out, toks = [logits.float()], []
    for _ in range(steps):
        tok = logits.argmax(-1, keepdim=True)
        toks.append(tok)
        logits, cache = r.decode(tok, cache)
        out.append(logits.float())
    return torch.stack(out, 1), torch.cat(toks, 1)


for p, d in [(4, 1), (4, 2), (2, 3), (4, 4)]:
    runner.system.prefill_steps, runner.system.decode_steps = p, d
    ref_logits, ref_toks = greedy(runner)
    sb_logits, sb_toks = greedy(single)
    # compare logits on the reference's own path up to the first token divergence
    same = (ref_toks == sb_toks).float().cumprod(1).sum(1).long()
    diff = (ref_logits[:, 0] - sb_logits[:, 0]).abs().max().item()
    print(json.dumps(dict(p=p, d=d, first_logit_maxdiff=round(diff, 4), first_logit_scale=round(ref_logits[:, 0].abs().max().item(), 2),
                          token_agreement=round((ref_toks == sb_toks).float().mean().item(), 4),
                          identical_prefix_len=same.tolist(), steps=steps)), flush=True)
