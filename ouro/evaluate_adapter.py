"""Fixed paired evaluation and decode timing for the KITE-style Ouro adapters.

Quality uses the previously prepared inputs and the original scoring functions.
Run quality shards on each arm's assigned GPUs. Run --timing separately on the
same physical GPU for every condition, after training has paused at 50M.
"""

import argparse
from datetime import datetime, timezone
import gc
import glob
import json
from pathlib import Path
import re
import time

import numpy as np

import torch
from transformers import AutoTokenizer

from ouro.evaluate import continuation, generate, rows, save
from ouro.evaluate_gsm8k import generate_one, strict_score
from ouro.evaluate_prefix_suffix import split_document
from ouro.model import MODEL_ID, MODEL_REVISION, OuroRunner, Variant, cache_bytes, load_model


DATASETS = ("pg19", "gsm8k", "ruler-4096", "ruler-8192", "qasper")
FULL_DATASETS = ("full-ruler-4096", "full-ruler-8192", "full-ruler-16384", "longbench-qasper", "longbench-hotpotqa", "longbench-2wikimqa")
DATASETS += FULL_DATASETS
CONDITIONS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "native1", "native2", "native3", "native4", "zero41", "zero42", "zero43", "zero44")
COUNTS = dict(pg19=32, gsm8k=1319, **{"ruler-4096": 128, "ruler-8192": 128}, qasper=64)

COUNTS.update({name: 6500 if "ruler" in name else 200 for name in FULL_DATASETS})

def now():
    return datetime.now(timezone.utc).isoformat()


class RawRunner:
    """Untrained read-p / write-d: the frozen model prefills all but the last prompt token with p loops and keeps only
    the last loop's KV; the same frozen model then runs every remaining token (incl. the last prompt token, so it also
    emits the first answer token) with d loops, each loop reading that single KV bank."""

    def __init__(self, model, p, d, aligned=False):
        self.model, self.p, self.d, self.aligned = model, p, d, aligned
        self.variant = Variant(p, d, "aligned" if aligned else "last_loop")
        self.variant_name = f"{'ali' if aligned else 'raw'}{p}{d}"
        self.time_adapter, self.last_adapter_seconds = False, 0.0

    def decode_cache(self, cache, depth):
        """Last-loop mode: every decode loop reads prefill loop p. Aligned mode (ali<p><d>, p >= d): decode loop k
        reads prefill loop p - d + k through its own per-loop cache, so ali<n><n> is exactly native n/n."""
        from ouro.model import LastLoopCache, NativeCache
        if not self.aligned:
            return LastLoopCache(cache, depth, self.p, self.d)
        out = NativeCache(self.d * depth)
        lo = (self.p - self.d) * depth
        out.key_cache, out.value_cache = cache.key_cache[lo:lo + self.d * depth], cache.value_cache[lo:lo + self.d * depth]
        out._seen_tokens = cache.get_seq_length(lo)
        return out

    @torch.inference_mode()
    def prefill(self, input_ids, attention_mask=None):
        from ouro.adapter_model import backbone
        from ouro.model import NativeCache
        depth = self.model.config.num_hidden_layers
        _, cache = backbone(self.model, input_ids[:, :-1], NativeCache(self.p * depth), self.p,
                            None if attention_mask is None else attention_mask[:, :-1])
        hidden, cache = backbone(self.model, input_ids[:, -1:], self.decode_cache(cache, depth), self.d, attention_mask)
        return self.model.lm_head(hidden[:, -1:])[:, 0], cache

    @torch.inference_mode()
    def decode(self, input_ids, cache, attention_mask=None):
        from ouro.adapter_model import backbone
        hidden, cache = backbone(self.model, input_ids, cache, self.d, attention_mask)
        return self.model.lm_head(hidden[:, -1:])[:, 0], cache


def make_runner(args):
    if len(args.condition) == 5 and args.condition[:3] in ("raw", "ali"):  # untrained read p / write d
        model = load_model(args.model_path, device="cuda", dtype=torch.bfloat16, attention="sdpa")
        return RawRunner(model, int(args.condition[3]), int(args.condition[4]), aligned=args.condition.startswith("ali"))
    if (len(args.condition) == 3 and args.condition[0] in "XNQ") or (len(args.condition) == 2 and args.condition[0] in "ZY"):
        # loop-map connector checkpoint X<p><d>, trained KV-sharing baseline Z<d>
        from ouro.adapter_model import load_adapter_runner

        runner = load_adapter_runner(args.model_path, args.checkpoint, device="cuda", dtype=torch.bfloat16, attention="sdpa")
        if runner.arm != args.condition:
            raise ValueError(f"Checkpoint arm {runner.arm} does not match {args.condition}")
        runner.time_adapter = args.timing
        return runner
    deep = re.fullmatch(r"([UVSWTMOPa])(\d+)x(\d+)", args.condition)  # a16x1: prefill depths beyond 9 (depth sweep)
    if deep or (len(args.condition) == 3 and args.condition[0] in "UVSWTMOPa"):  # universal connector at cell (p, d), e.g. U41 / V14
        from ouro.adapter_model import load_adapter_runner

        runner = load_adapter_runner(args.model_path, args.checkpoint, device="cuda", dtype=torch.bfloat16, attention="sdpa")
        if runner.arm != args.condition[0]:
            raise ValueError(f"Checkpoint arm {runner.arm} is not the universal connector {args.condition[0]}")
        p, d = (int(deep[2]), int(deep[3])) if deep else (int(args.condition[1]), int(args.condition[2]))
        runner.system.prefill_steps, runner.system.decode_steps = p, d
        runner.variant = Variant(p, d, "adapter")
        runner.variant_name = f"adapter-{args.condition}"
        runner.time_adapter = args.timing
        return runner
    if args.condition in ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"):
        from ouro.adapter_model import load_adapter_runner

        runner = load_adapter_runner(args.model_path, args.checkpoint, device="cuda",
                                     dtype=torch.bfloat16, attention="sdpa")
        if runner.arm != args.condition:
            raise ValueError(f"Checkpoint arm {runner.arm} does not match {args.condition}")
        runner.time_adapter = args.timing
        return runner
    model = load_model(args.model_path, device="cuda", dtype=torch.bfloat16,
                       attention="sdpa")
    if args.condition == "native3":
        runner = OuroRunner(model, "p1-d1")
        runner.variant = Variant(3, 3, "per_loop")
        runner.variant_name = "native3"
        return runner
    if args.condition == "native2":
        # The old p2-d2 name means shared LAST-LOOP history. It is not native2.
        runner = OuroRunner(model, "p1-d1")
        runner.variant = Variant(2, 2, "per_loop")
        runner.variant_name = "native2"
        return runner
    # zero4d: untrained p4 prefill + d-loop decode, every loop reading one shared last-loop KV (Ouro KV sharing)
    variant = {"native1": "p1-d1", "native4": "p4-d4", "zero41": "p4-d1", "zero42": "p4-d2", "zero43": "p4-d3",
               "zero44": "p4-d4-shared"}[args.condition]
    return OuroRunner(model, variant)


def source_rows(dataset, data, gsm_data, full_data=None):
    directory = gsm_data if dataset == "gsm8k" else data
    filename = dataset
    if dataset in FULL_DATASETS:
        directory = full_data
        filename = dataset.removeprefix("full-")
    source = rows(directory / f"{filename}.jsonl")
    if dataset == "pg19":
        source = [split_document(row) for row in source]
    # Missing or duplicate rows would silently change the fixed subset.
    if len(source) != COUNTS[dataset] or len({r["id"] for r in source}) != len(source):
        raise ValueError(f"{dataset}: expected {COUNTS[dataset]} unique fixed examples")
    return source


def aggregate(dataset, results):
    if dataset == "pg19":
        total = sum(row["target_tokens"] for row in results)
        return dict(samples=len(results), prediction_tokens=total,
                    nll=sum(row["nll_sum"] for row in results) / total,
                    first_token_nll=sum(row["first_token_nll"] for row in results) / len(results),
                    remaining_511_tokens_nll=sum(
                        row["remaining_tokens_nll"] * (row["target_tokens"] - 1)
                        for row in results) / (total - len(results)))
    if dataset == "gsm8k":
        return dict(samples=len(results), correct=sum(r["correct"] for r in results),
                    score=100 * sum(r["correct"] for r in results) / len(results),
                    unmatched=sum(not r["matched"] for r in results),
                    truncated=sum(r["truncated"] for r in results),
                    generated_tokens=sum(r["generated_tokens"] for r in results))
    by_task = {task: [r["score"] for r in results if r["task"] == task]
               for task in sorted({r["task"] for r in results})}
    task_scores = {task: 100 * sum(values) / len(values) for task, values in by_task.items()}
    return dict(samples=len(results), task_scores=task_scores,
                score=sum(task_scores.values()) / len(task_scores),
                truncated=sum(r["reached_limit"] for r in results),
                generated_tokens=sum(r["generated_tokens"] for r in results))


def done_rows(path):
    """Complete rows already written to a results file; a partial last line is dropped."""
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[row["id"]] = row
    return done


@torch.inference_mode()
def quality(args, runner, tokenizer, destination):
    counts = {}
    gsm_manifest = json.loads((args.gsm_data / "manifest.json").read_text()) if "gsm8k" in args.datasets else None
    for dataset in args.datasets:
        source = source_rows(dataset, args.data, args.gsm_data, args.full_data)[args.shard_index::args.num_shards]
        # Rows finished by another (e.g. stalled) shard over the same ids are left to that shard.
        elsewhere = set()
        for pattern in args.skip_dirs:
            for path in glob.glob(f"{pattern}/{dataset}.jsonl"):
                elsewhere.update(done_rows(Path(path)))
        source = [row for row in source if row["id"] not in elsewhere]
        path = destination / f"{dataset}.jsonl"
        done = done_rows(path) if args.resume else {}
        if args.resume:
            path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in done.values()))
        outputs = [done[row["id"]] for row in source if row["id"] in done]
        with path.open("a" if args.resume else "x") as stream:
            for row in source:
                if row["id"] in done:
                    continue
                started = time.perf_counter()
                if dataset == "pg19":
                    result = continuation(runner, row)
                    result["source_id"] = row["source_id"]
                elif dataset == "gsm8k":
                    result = generate_one(runner, tokenizer, row, gsm_manifest["generation"])
                    result.update(strict_score(result["text"], row["answer"], gsm_manifest["strict_match"]))
                else:
                    result = generate(runner, tokenizer, row)
                result["seconds"] = time.perf_counter() - started
                outputs.append(result)
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                save(destination / "progress.json", dict(
                    condition=args.condition, dataset=dataset, completed=len(outputs),
                    shard_total=len(source), shard_index=args.shard_index,
                    num_shards=args.num_shards, observed_at_utc=now()))
        if outputs:  # every row may belong to another shard
            counts[dataset] = aggregate(dataset, outputs)
        save(destination / "quality.json", counts)
        gc.collect()
        torch.cuda.empty_cache()
    return counts


@torch.inference_mode()
def benchmark(runner, source, context, batch):
    ids = torch.tensor([r["prompt_ids"][:context] for r in source[:batch]],
                       dtype=torch.long, device="cuda")
    # Replaying the same real-text tokens removes condition-specific sampling
    # and early stopping from the cost comparison. This measures cost, not NLL.
    replay = torch.tensor([r["prompt_ids"][-259:] for r in source[:batch]],
                          dtype=torch.long, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    logits, cache = runner.prefill(ids)
    torch.cuda.synchronize()
    prefill_including_adapter = time.perf_counter() - started
    adapter_seconds = getattr(runner, "last_adapter_seconds", 0.0)
    prefill_peak = torch.cuda.max_memory_allocated()
    historical_cache = cache_bytes(cache)
    allocated_after_prefill = torch.cuda.memory_allocated()
    for index in range(3):
        logits, cache = runner.decode(replay[:, index:index + 1], cache)
    torch.cuda.synchronize()
    timed_start_cache = cache_bytes(cache)
    torch.cuda.reset_peak_memory_stats()
    milliseconds = []
    started = time.perf_counter()
    for index in range(3, 259):
        step_start = time.perf_counter()
        logits, cache = runner.decode(replay[:, index:index + 1], cache)
        torch.cuda.synchronize()
        milliseconds.append(1000 * (time.perf_counter() - step_start))
    elapsed = time.perf_counter() - started
    result = dict(
        context=context, batch=batch, warmup_steps=3, measured_steps=256,
        prompt_to_decode_history_start=context + 3, final_history_length=context + 259,
        prefill_seconds=prefill_including_adapter - adapter_seconds,
        adapter_seconds=adapter_seconds,
        prefill_including_adapter_seconds=prefill_including_adapter,
        prefill_peak_allocated_bytes=prefill_peak,
        allocated_after_prefill_bytes=allocated_after_prefill,
        decode_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        decode_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        historical_cache=historical_cache, timed_start_cache=timed_start_cache,
        final_cache=cache_bytes(cache), decode_seconds=elapsed,
        decode_step_ms_median=float(np.median(milliseconds)),
        decode_step_ms_p90=float(np.quantile(milliseconds, .9)),
        decoded_tokens_per_second=batch * 256 / elapsed, step_ms=milliseconds,
        timing_scope="Synchronized wall time, full decoder and vocabulary head, append of real growing KV; fixed replay token input, no argmax/sampling, compile or Graph. Step latency is not divided by batch.",
        replay="Last259 real-text token IDs of each prepared8192-token benchmark document, identical across conditions. These are a fixed throughput workload, not the document's unseen continuation.",
        lora="Unmerged reader attention LoRA, when present; its actual compute overhead is included.",
        pd_scope="Writer and reader are on the same GPU; this measurement excludes inter-GPU KV transfer.",
    )
    del ids, replay, logits, cache
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--full-data", type=Path)
    parser.add_argument("--gsm-data", type=Path, default=Path("/dev/shm/big-small-ouro/gsm8k"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS[:5]))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--timing", action="store_true", help="Only controlled timing, after quality evaluation; use same physical GPU for all conditions")
    parser.add_argument("--resume", action="store_true", help="Reuse the shard directory and skip rows already written there")
    parser.add_argument("--skip-dirs", nargs="*", default=[], help="Shard directory globs whose finished rows this run skips")
    args = parser.parse_args()
    if args.condition in ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L") and args.checkpoint is None:
        parser.error("A/B/C/D evaluation requires its trained checkpoint")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("Require 0 <= shard-index < num-shards")
    destination = args.output / ("timing" if args.timing else f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}")
    destination.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(4)
    torch.manual_seed(31847)
    runner = make_runner(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    save(destination / "config.json", dict(
        condition=args.condition, checkpoint=str(args.checkpoint) if args.checkpoint else None,
        checkpoint_metadata=getattr(runner, "checkpoint_metadata", None),
        started_at_utc=now(), model_id=MODEL_ID, model_revision=MODEL_REVISION,
        model_path=args.model_path, seed=31847, training=False,
        prefill_loops=runner.variant.prefill_steps, decode_loops=runner.variant.decode_steps,
        cache_mode=runner.variant.cache_mode, batch_size=1 if not args.timing else [1, 16],
        datasets=args.datasets if not args.timing else [], shard_index=args.shard_index,
        num_shards=args.num_shards, data=str(args.data), gsm_data=str(args.gsm_data),
        dtype="bfloat16", attention="torch SDPA", graph=False,
        torch_version=torch.__version__, transformers_version=__import__("transformers").__version__,
        gpu=torch.cuda.get_device_name(), gpu_device_index=torch.cuda.current_device(),
        gpu_visible_devices=__import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        first_token="Final prefill-loop logits; A/B/C adapter affects remaining tokens, D LoRA also affects prefill.",
        pg19="Original32 documents,1536 prefix+512 targets; report full512 and last511 separately.",
        data_scope="Original fixed subsets and/or explicitly selected full RULER13x500 at4K/8K/16K and LongBench200 per task; see datasets and data manifest.",
        full_data=str(args.full_data) if args.full_data else None,
        raw_data_policy="No source prompts, documents, answers or token arrays saved here; model text, IDs and scores only.",
    ))
    started = time.perf_counter()
    if args.timing:
        source = rows(args.data / "benchmark.jsonl")
        timing = []
        for context in (2048, 8192):
            for batch in (1, 16):
                timing.append(benchmark(runner, source, context, batch))
                save(destination / "benchmark.json", timing)
        result = dict(benchmark=timing)
    else:
        result = dict(quality=quality(args, runner, tokenizer, destination))
    result.update(status="completed", condition=args.condition, completed_at_utc=now(),
                  elapsed_seconds=time.perf_counter() - started, optimizer_updates=0)
    save(destination / "result.json", result)
    print(json.dumps(dict(condition=args.condition, status="completed", output=str(destination))), flush=True)


if __name__ == "__main__":
    main()
