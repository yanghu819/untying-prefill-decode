"""Decode throughput vs batch size on one idle GPU, up to the largest batch that fits.

Conditions: native1/2/4 (per-loop banks) or a checkpoint (E/F/D/G arms). Real RULER-16K prompt
tokens truncated to each context; batch rows are copies of real prompts. Decode replays --steps
greedy steps; reports prefill seconds, decode ms/step, decoded tokens/s, decode-phase KV GiB and peak
memory. A batch that runs out of memory is recorded as "oom" and larger batches are skipped.
--row-prefill prefills one row at a time and concatenates the per-row caches into the decode batch, so
only the decode phase's memory limits the batch (prefill runs elsewhere in a P/D or big/small deployment).
"""

import argparse
import json
from pathlib import Path
import re
import time

import torch

from ouro.adapter_model import load_adapter_runner, merge_reader_lora
from ouro.evaluate import rows
from ouro.model import OuroRunner, Variant, cache_bytes, load_model
from ouro.single_bank import AlignedRunner, NativeContiguousRunner, NativeStaticRunner, SingleBankRunner


def make(condition, model_path, checkpoint, merge=False):
    if checkpoint and condition[0] in "DEH" and condition.endswith("st"):  # D2st: LoRA-distilled native 2/2, LoRA merged,
        runner = load_adapter_runner(model_path, checkpoint)            # timed exactly like native<d>st
        return NativeStaticRunner(merge_reader_lora(runner.system.reader), int(condition[1]))
    if checkpoint:
        runner = load_adapter_runner(model_path, checkpoint)
        if merge:
            merge_reader_lora(runner.system.reader)
        deep = re.fullmatch(r"([SOa])(\d+)x(\d+)sb", condition)  # a16x1sb: prefill depths beyond 9 (depth sweep)
        p, d = (int(deep[2]), int(deep[3])) if deep else (int(condition[1]), int(condition[2])) if len(condition) > 2 else (0, 0)
        if condition.endswith("sb") and condition[0] == "a":  # a<p><d>sb: aligned, one static prompt bank per decode loop
            return AlignedRunner(runner.system, p, d)
        if condition.endswith("sb"):  # S<p><d>sb: prompt KV stored once; O<p><d>sb: prompt and answer KV stored once
            return SingleBankRunner(runner.system, p, d)
        if len(condition) == 3 and condition[0] in "UVSWT":  # universal connector at cell (p, d), materialised banks
            runner.system.prefill_steps, runner.system.decode_steps = int(condition[1]), int(condition[2])
        return runner
    if condition.endswith("sc"):  # native<d>sc: preallocated contiguous per-loop cache, one SDPA call (strongest native)
        return NativeContiguousRunner(load_model(model_path), int(condition[6]))
    if condition.endswith("st"):  # native<d>st: native model, static per-loop prompt banks + the fused decode attention
        return NativeStaticRunner(load_model(model_path), int(condition[6]))
    runner = OuroRunner(load_model(model_path), "p1-d1")
    loops = {"native1": 1, "native2": 2, "native3": 3, "native4": 4}[condition]
    runner.variant = Variant(loops, loops, "per_loop")
    return runner


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
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[8192, 15360])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128])
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-prefill", action="store_true")
    parser.add_argument("--merge-lora", action="store_true", help="fold the decode model's LoRA into its weights")
    args = parser.parse_args()
    runner = make(args.condition, args.model_path, args.checkpoint, args.merge_lora)
    source = [r["prompt_ids"] for r in rows(args.prompts)[:16]]
    report = []
    for ctx in args.contexts:
        for batch in args.batches:
            ids = torch.tensor([source[i % len(source)][:ctx] for i in range(batch)], device="cuda")
            tok = ids[:, -1:].clone()
            row = dict(condition=args.condition, context=ctx, batch=batch)
            try:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                if args.row_prefill:
                    def rowwise():
                        # copy each row's cache into preallocated batch tensors, so memory never holds two copies
                        merged, last = None, None
                        for i in range(batch):
                            last, c = runner.prefill(ids[i:i + 1])
                            banks = [b for b in ("key_cache", "value_cache", "prompt_keys", "prompt_values") if hasattr(c, b)]
                            if merged is None:
                                merged = c
                                for bank in banks:
                                    setattr(merged, bank, [None if t is None else t.new_empty((batch, *t.shape[1:])).index_copy_(0, torch.tensor([0], device=t.device), t)
                                                           for t in c.__dict__[bank]])
                            else:
                                for bank in banks:
                                    for full, t in zip(merged.__dict__[bank], c.__dict__[bank]):
                                        if full is not None:
                                            full[i:i + 1].copy_(t)
                            del c
                            torch.cuda.empty_cache()
                        if hasattr(merged, "delta_keys"):  # prompt deltas are only read on the first cache fill
                            merged.delta_keys = merged.delta_values = []
                        return last, merged
                    (logits, cache), prefill_s = sync_time(rowwise)
                else:
                    (logits, cache), prefill_s = sync_time(lambda: runner.prefill(ids))
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()  # peak below = decode phase only
                kv = cache_bytes(cache)["logical_bytes"]
                for _ in range(2):
                    _, cache = runner.decode(tok, cache)
                state = [cache]
                del cache

                def run():
                    for _ in range(args.steps):
                        _, state[0] = runner.decode(tok, state[0])
                (_, decode_s) = sync_time(run)
                cache = state.pop()  # the closure holds no cache reference after this (it used to keep the previous
                del run              # trial's KV alive through the next trial's prefill and decode, inflating peaks)
                row.update(prefill_s=round(prefill_s, 3), decode_ms_per_step=round(1000 * decode_s / args.steps, 2),
                           tokens_per_s=round(batch * args.steps / decode_s, 1), decode_kv_gib=round(kv / 2**30, 2),
                           decode_peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1))
                del cache, logits
            except torch.OutOfMemoryError:
                row["oom"] = True
            torch.cuda.empty_cache()
            print(json.dumps(row), flush=True)
            report.append(row)
            if row.get("oom"):
                break
    args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
