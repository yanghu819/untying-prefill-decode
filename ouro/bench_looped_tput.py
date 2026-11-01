"""Decode throughput and decode-phase KV for HF looped families (looped_bridge.LoopedRunner) on one GPU.

Conditions: native{n} or a looped_bridge checkpoint (E/F). Real RULER-16K prompt text re-encoded with the
model tokenizer and truncated to each context; batch rows are copies. Each row is prefilled alone and the caches
are concatenated, so only the decode phase limits the batch (prefill runs on the big model in deployment).
Reports decode ms/step, tokens/s, decode KV GiB (all cached tensors) and peak memory; OOM ends a context's sweep.
"""

import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from ouro.looped_bridge import LoopedRunner, tokenizer


def cache_tensors(cache):
    out = []
    for name, bank in vars(cache).items():
        if isinstance(bank, list):
            out += [t for t in bank if torch.is_tensor(t)]
    for layer in getattr(cache, "layers", None) or []:
        out += [getattr(layer, n) for n in ("keys", "values") if torch.is_tensor(getattr(layer, n, None))]
    return out


def merge(caches):
    """Concatenate per-row prompt caches along the batch dimension (in place on the first)."""
    first = caches[0]
    for name, bank in vars(first).items():
        if isinstance(bank, list) and bank and torch.is_tensor(bank[0]):
            setattr(first, name, [torch.cat([vars(c)[name][j] for c in caches]) if torch.is_tensor(t) else t
                                  for j, t in enumerate(bank)])
    for j, layer in enumerate(getattr(first, "layers", None) or []):
        for n in ("keys", "values"):
            if torch.is_tensor(getattr(layer, n, None)):
                setattr(layer, n, torch.cat([getattr(c.layers[j], n) for c in caches]))
    return first


def sync_time(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--prompts", type=Path, default=Path("/dev/shm/big-small-ouro/bench/ruler16k.jsonl"))
    parser.add_argument("--contexts", type=int, nargs="+", default=[8192])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runner = LoopedRunner(args.model_path, args.condition, args.checkpoint)
    tok = tokenizer(args.model_path)
    src = AutoTokenizer.from_pretrained("/dev/shm/big-small-ouro/model", trust_remote_code=True)
    text = src.decode(json.loads(open(args.prompts).readline())["prompt_ids"])
    ids = tok.encode(text, add_special_tokens=False)
    report = []
    for ctx in args.contexts:
        row = torch.tensor([ids[:ctx]], device="cuda")
        for batch in args.batches:
            rec = dict(condition=args.condition, context=ctx, batch=batch)
            try:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                caches = []
                for _ in range(batch):
                    logits, c = runner.prefill(row)
                    caches.append(c)
                cache = merge(caches)
                del caches
                torch.cuda.empty_cache()
                kv = sum(t.numel() * t.element_size() for t in cache_tensors(cache))
                tokens = logits.argmax(-1).view(1, 1).expand(batch, 1).contiguous()
                for _ in range(2):
                    _, cache = runner.decode(tokens, cache)

                def run(c=cache):
                    for _ in range(args.steps):
                        _, c = runner.decode(tokens, c)
                    return c
                cache, seconds = sync_time(run)
                rec.update(decode_ms_per_step=round(1000 * seconds / args.steps, 2),
                           tokens_per_s=round(batch * args.steps / seconds, 1),
                           decode_kv_gib=round(kv / 2**30, 2), kv_gib_per_seq=round(kv / batch / 2**30, 3),
                           peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1))
                del cache
            except torch.OutOfMemoryError:
                rec["oom"] = True
            torch.cuda.empty_cache()
            print(json.dumps(rec), flush=True)
            report.append(rec)
            if rec.get("oom"):
                break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
