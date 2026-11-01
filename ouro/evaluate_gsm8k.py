"""Fixed 3-shot greedy GSM8K test evaluation for the four requested Ouro paths."""

import argparse
import json
from pathlib import Path
import re
import time

import torch
from transformers import AutoTokenizer

from ouro.evaluate import now, rows, save, tokens
from ouro.model import MODEL_ID, MODEL_REVISION, VARIANTS, OuroRunner, load_model


GSM_VARIANTS = ("p4-d4", "p1-d1", "p4-d1", "p4-d2")


def strict_score(text, answer, spec):
    matches = re.findall(spec["regex_pattern"], text)
    prediction = matches[0].strip() if matches else spec["no_match"]

    def normalize(value):
        value = str(value).lower() if spec["ignore_case"] else str(value)
        for pattern in spec["regexes_to_ignore"]:
            value = re.sub(pattern, "", value)
        return value.strip()

    return dict(
        prediction=prediction, matched=bool(matches),
        correct=bool(matches) and normalize(prediction) == normalize(answer),
    )


@torch.inference_mode()
def generate_one(runner, tokenizer, row, generation):
    logits, cache = runner.prefill(tokens(row["prompt_ids"]))
    generated = []
    text = ""
    stop_reason = "max_new_tokens"
    decode_calls = 0
    for index in range(generation["max_new_tokens"]):
        token = logits.argmax(dim=-1).view(1, 1)
        value = int(token.item())
        generated.append(value)
        if value == generation["eos_token_id"]:
            stop_reason = "eos"
            break
        text = tokenizer.decode(generated, skip_special_tokens=False)
        stops = [(text.find(marker), marker) for marker in generation["until"]]
        stops = [(position, marker) for position, marker in stops if position >= 0]
        if stops:
            position, marker = min(stops)
            text = text[:position]
            stop_reason = f"stop:{marker}"
            break
        if index + 1 < generation["max_new_tokens"]:
            logits, cache = runner.decode(token, cache)
            decode_calls += 1
    return dict(
        id=row["id"], prompt_tokens=len(row["prompt_ids"]), text=text,
        generated_tokens=len(generated), decode_calls=decode_calls,
        stop_reason=stop_reason, truncated=stop_reason == "max_new_tokens",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=GSM_VARIANTS, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    source = rows(args.data / "gsm8k.jsonl")
    assert len(source) == manifest["samples"] == 1319
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(31847)
    model = load_model(args.model_path, device="cuda", dtype=torch.bfloat16, attention="sdpa")
    runner = OuroRunner(model, args.variant)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    variant = VARIANTS[args.variant]
    save(args.output / "config.json", dict(
        started_at_utc=now(), variant=args.variant, model_id=MODEL_ID,
        model_revision=MODEL_REVISION, model_path=args.model_path,
        prefill_steps=variant.prefill_steps, decode_steps=variant.decode_steps,
        cache_mode=variant.cache_mode, training=False, optimizer_updates=0,
        dtype="bfloat16", attention="torch SDPA", batch_size=1, graph=False,
        torch_version=torch.__version__,
        transformers_version=__import__("transformers").__version__,
        gpu=torch.cuda.get_device_name(), data_manifest=manifest,
        first_token="Final prefill logits; subsequent tokens use configured decode loops",
        result_scope="Full GSM8K test, fixed 3-shot prompt, greedy max512; no tuning or retries",
    ))
    started = time.perf_counter()
    outputs = []
    with (args.output / "results.jsonl").open("w") as handle:
        for row in source:
            row_start = time.perf_counter()
            result = generate_one(runner, tokenizer, row, manifest["generation"])
            result.update(strict_score(result["text"], row["answer"], manifest["strict_match"]))
            result["seconds"] = time.perf_counter() - row_start
            outputs.append(result)
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            save(args.output / "progress.json", dict(
                variant=args.variant, completed=len(outputs), total=len(source),
                correct=sum(r["correct"] for r in outputs),
                unmatched=sum(not r["matched"] for r in outputs),
                truncated=sum(r["truncated"] for r in outputs),
                generated_tokens=sum(r["generated_tokens"] for r in outputs),
                elapsed_seconds=time.perf_counter() - started, observed_at_utc=now(),
            ))
    correct = sum(row["correct"] for row in outputs)
    result = dict(
        status="completed", variant=args.variant, completed_at_utc=now(),
        samples=len(outputs), correct=correct, accuracy=correct / len(outputs),
        unmatched=sum(not row["matched"] for row in outputs),
        truncated=sum(row["truncated"] for row in outputs),
        generated_tokens=sum(row["generated_tokens"] for row in outputs),
        evaluation_seconds=time.perf_counter() - started, optimizer_updates=0,
        scoring="gsm8k_cot strict-match; first matching answer, no flexible fallback",
    )
    save(args.output / "result.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
