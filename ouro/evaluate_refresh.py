"""GSM8K with chunked re-prefill: shallow per-token decode, deep chunk refresh.

Prompt: writer runs 4 loops, adapter maps last-loop KV for the 1-loop reader.
Decode: reader generates token by token with one loop. When a chunk closes
(newline token, or every N tokens), the writer runs 4 loops over the chunk in
parallel, its last-loop KV goes through the same adapter and replaces the
reader's shallow KV for those positions; the writer's logits at the chunk end
predict the next token. The writer cache is exactly native 4/4 over everything
committed so far, so refresh=none reproduces the plain adapter arm.
"""

import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from ouro.adapter_model import PrefixState, backbone, load_adapter_runner
from ouro.evaluate import now, rows, save, tokens
from ouro.evaluate_gsm8k import strict_score


@torch.inference_mode()
def refresh_chunk(system, chunk, writer_cache, reader_cache, committed):
    depth = system.writer.config.num_hidden_layers
    system.capture_enabled = True
    hidden, writer_cache = backbone(system.writer, chunk, writer_cache, 4)
    system.capture_enabled = False
    k = chunk.shape[1]
    offset = 3 * depth
    state = PrefixState(
        list(system.raw_keys),
        [writer_cache.value_cache[offset + layer][:, :, -k:] for layer in range(depth)],
        writer_cache.cos,
        writer_cache.sin,
    )
    system.raw_keys = [None] * depth
    mapped = system.adapter(state)
    for layer in range(depth):
        reader_cache.key_cache[layer] = torch.cat(
            (reader_cache.key_cache[layer][:, :, :committed], mapped.key_cache[layer]), dim=2)
        reader_cache.value_cache[layer] = torch.cat(
            (reader_cache.value_cache[layer][:, :, :committed], mapped.value_cache[layer]), dim=2)
    reader_cache._seen_tokens = committed + k
    logits = system.writer.lm_head(hidden[:, -1:])[:, 0]
    return logits, writer_cache, reader_cache, committed + k


@torch.inference_mode()
def generate_refresh(runner, tokenizer, row, generation, mode, every):
    system = runner.system
    prompt = tokens(row["prompt_ids"])
    logits, prefix, writer_cache = system.writer_prefix(prompt, 4)
    reader_cache = system.adapter(prefix)
    committed = prompt.shape[1]
    generated, pending = [], []
    text, stop_reason = "", "max_new_tokens"
    reader_calls = refreshes = writer_tokens = 0
    for index in range(generation["max_new_tokens"]):
        token = logits.argmax(dim=-1).view(1, 1)
        value = int(token.item())
        generated.append(value)
        if value == generation["eos_token_id"]:
            stop_reason = "eos"
            break
        text = tokenizer.decode(generated, skip_special_tokens=False)
        stops = [(text.find(m), m) for m in generation["until"]]
        stops = [(p, m) for p, m in stops if p >= 0]
        if stops:
            position, marker = min(stops)
            text = text[:position]
            stop_reason = f"stop:{marker}"
            break
        if index + 1 >= generation["max_new_tokens"]:
            break
        pending.append(value)
        close = (mode == "newline" and "\n" in tokenizer.decode([value])) or (
            mode == "every" and len(pending) >= every)
        if close:
            chunk = torch.tensor([pending], device=prompt.device)
            logits, writer_cache, reader_cache, committed = refresh_chunk(
                system, chunk, writer_cache, reader_cache, committed)
            refreshes += 1
            writer_tokens += len(pending)
            pending = []
        else:
            out, reader_cache = system.reader_suffix(token, reader_cache)
            logits = out[:, -1]
            reader_calls += 1
    return dict(
        id=row["id"], text=text, generated_tokens=len(generated),
        reader_calls=reader_calls, refreshes=refreshes, writer_tokens=writer_tokens,
        stop_reason=stop_reason, truncated=stop_reason == "max_new_tokens",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("none", "newline", "every"), required=True)
    parser.add_argument("--every", type=int, default=32)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1319)
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    source = rows(args.data / "gsm8k.jsonl")[args.start : args.start + args.count]
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(31847)
    runner = load_adapter_runner(args.model_path, args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    save(args.output / "config.json", dict(
        started_at_utc=now(), arm=runner.arm, checkpoint=args.checkpoint,
        mode=args.mode, every=args.every, start=args.start, count=len(source),
        metadata=runner.checkpoint_metadata,
    ))
    outputs, started = [], time.perf_counter()
    with (args.output / "results.jsonl").open("w") as handle:
        for row in source:
            t0 = time.perf_counter()
            result = generate_refresh(runner, tokenizer, row, manifest["generation"], args.mode, args.every)
            result.update(strict_score(result["text"], row["answer"], manifest["strict_match"]))
            result["seconds"] = time.perf_counter() - t0
            outputs.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
    summary = dict(
        mode=args.mode, every=args.every, samples=len(outputs),
        correct=sum(r["correct"] for r in outputs),
        accuracy=sum(r["correct"] for r in outputs) / len(outputs),
        truncated=sum(r["truncated"] for r in outputs),
        generated_tokens=sum(r["generated_tokens"] for r in outputs),
        reader_calls=sum(r["reader_calls"] for r in outputs),
        writer_tokens=sum(r["writer_tokens"] for r in outputs),
        refreshes=sum(r["refreshes"] for r in outputs),
        seconds=time.perf_counter() - started, completed_at_utc=now(),
    )
    save(args.output / "result.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
