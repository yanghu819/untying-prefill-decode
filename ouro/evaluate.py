"""Four frozen Ouro prefill/decode configurations on fixed paired data."""

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from ouro.model import OuroRunner, cache_bytes, load_model
from ouro.score import score_generation


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tokens(ids):
    return torch.tensor([ids], dtype=torch.long, device="cuda")


@torch.inference_mode()
def continuation(runner, row):
    logits, cache = runner.prefill(tokens(row["prompt_ids"]))
    losses = []
    for i, target in enumerate(row["target_ids"]):
        y = tokens([target]).view(-1)
        losses.append(float(F.cross_entropy(logits.float(), y)))
        if i + 1 < len(row["target_ids"]):
            logits, cache = runner.decode(y.view(1, 1), cache)
    return dict(id=row["id"], target_tokens=len(losses), nll=sum(losses) / len(losses),
                nll_sum=sum(losses), first_token_nll=losses[0],
                remaining_tokens_nll=sum(losses[1:]) / (len(losses) - 1))


@torch.inference_mode()
def generate(runner, tokenizer, row):
    logits, cache = runner.prefill(tokens(row["prompt_ids"]))
    ids = []
    for i in range(row["max_new_tokens"]):
        next_token = logits.argmax(dim=-1).view(1, 1)
        value = int(next_token.item())
        if value == tokenizer.eos_token_id:
            break
        ids.append(value)
        if i + 1 < row["max_new_tokens"]:
            logits, cache = runner.decode(next_token, cache)
    prediction = tokenizer.decode(ids, skip_special_tokens=True)
    score = score_generation(row["task"], prediction, row["answers"])
    return dict(id=row["id"], task=row["task"], prompt_tokens=len(row["prompt_ids"]),
                generated_tokens=len(ids), prediction=prediction, score=score,
                reached_limit=len(ids) == row["max_new_tokens"])


@torch.inference_mode()
def benchmark(runner, batch_rows, context):
    # Same real document prefixes for every variant. No sampling, EOS stopping,
    # compilation, Graph capture, or timing of a cache read in place of prefill.
    ids = torch.tensor([r["prompt_ids"][:context] for r in batch_rows],
                       dtype=torch.long, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    logits, cache = runner.prefill(ids)
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - t0
    prefill_peak = torch.cuda.max_memory_allocated()
    historical_cache = cache_bytes(cache)
    allocated_after_prefill = torch.cuda.memory_allocated()
    # Three warm-up steps are outside latency samples, but their added cache
    # positions are stated explicitly. Every timed step appends a real token.
    for _ in range(3):
        logits, cache = runner.decode(logits.argmax(-1).view(-1, 1), cache)
    torch.cuda.synchronize()
    timed_start_cache = cache_bytes(cache)
    torch.cuda.reset_peak_memory_stats()
    milliseconds = []
    start = time.perf_counter()
    for _ in range(256):
        step_start = time.perf_counter()
        logits, cache = runner.decode(logits.argmax(-1).view(-1, 1), cache)
        torch.cuda.synchronize()
        milliseconds.append((time.perf_counter() - step_start) * 1000)
    elapsed = time.perf_counter() - start
    result = dict(context=context, batch=16, warmup_steps=3, measured_steps=256,
                  prompt_to_decode_history_start=context + 3,
                  final_history_length=context + 259,
                  prefill_seconds=prefill_seconds,
                  prefill_peak_allocated_bytes=prefill_peak,
                  allocated_after_prefill_bytes=allocated_after_prefill,
                  decode_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  decode_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  historical_cache=historical_cache, timed_start_cache=timed_start_cache,
                  final_cache=cache_bytes(cache), decode_seconds=elapsed,
                  decode_step_ms_median=float(np.median(milliseconds)),
                  decode_step_ms_p90=float(np.quantile(milliseconds, .9)),
                  generated_tokens_per_second=16 * 256 / elapsed,
                  step_ms=milliseconds,
                  timing_scope="Synchronized wall time of full B16 step including argmax, model, head, and KV append; no Graph/compile; median is not divided by16.")
    del ids, logits, cache
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", choices=["p4-d4", "p1-d1", "p4-d1", "p4-d2"], required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(31847)
    model = load_model(args.model_path, device="cuda", dtype=torch.bfloat16, attention="sdpa")
    model.requires_grad_(False).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    runner = OuroRunner(model, args.variant)
    save(args.output / "config.json", dict(
        variant=args.variant, model_path=args.model_path, seed=31847,
        started_at_utc=now(), training=False, optimizer_updates=0,
        dtype="bfloat16", attention="torch SDPA", graph=False,
        torch_version=torch.__version__, transformers_version=__import__("transformers").__version__,
        gpu=torch.cuda.get_device_name(), parameter_count=sum(p.numel() for p in model.parameters()),
        data_manifest=json.loads((args.data / "manifest.json").read_text())))
    counts = {}
    for dataset in ["pg19", "ruler-4096", "ruler-8192", "qasper"]:
        source = rows(args.data / f"{dataset}.jsonl")
        out_rows = []
        with (args.output / f"{dataset}.jsonl").open("w") as f:
            for row in source:
                t0 = time.perf_counter()
                result = continuation(runner, row) if dataset == "pg19" else generate(runner, tokenizer, row)
                result["seconds"] = time.perf_counter() - t0
                out_rows.append(result)
                f.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                f.flush()
                save(args.output / "progress.json", dict(
                    variant=args.variant, dataset=dataset, completed=len(out_rows),
                    total=len(source), observed_at_utc=now()))
        if dataset == "pg19":
            total_tokens = sum(r["target_tokens"] for r in out_rows)
            counts[dataset] = dict(samples=len(out_rows), prediction_tokens=total_tokens,
                                  nll=sum(r["nll_sum"] for r in out_rows) / total_tokens)
        else:
            tasks = sorted({r["task"] for r in out_rows})
            task_scores = {t: float(np.mean([r["score"] for r in out_rows if r["task"] == t])) * 100
                           for t in tasks}
            counts[dataset] = dict(samples=len(out_rows), task_scores=task_scores,
                                  macro_score=float(np.mean(list(task_scores.values()))),
                                  truncated=sum(r["reached_limit"] for r in out_rows))
        save(args.output / "quality.json", counts)
        gc.collect()
        torch.cuda.empty_cache()
    bench_rows = rows(args.data / "benchmark.jsonl")
    timing = []
    for context in (2048, 8192):
        timing.append(benchmark(runner, bench_rows, context))
        save(args.output / "benchmark.json", timing)
    save(args.output / "result.json", dict(status="completed", variant=args.variant,
                                          completed_at_utc=now(), quality=counts,
                                          benchmark=timing, optimizer_updates=0))
    print(json.dumps(dict(variant=args.variant, status="completed", quality=counts)), flush=True)


if __name__ == "__main__":
    main()
