"""Frozen Ouro NLL on the prepared PG19 documents: 1536 prefix + 512 suffix."""

import argparse
import json
from pathlib import Path
import time

import torch

from ouro.evaluate import continuation, now, rows, save
from ouro.model import MODEL_ID, MODEL_REVISION, VARIANTS, OuroRunner, load_model


PREFIX_TOKENS = 1536
SUFFIX_TOKENS = 512


def split_document(row):
    """Retain the original document and use its first 2048 token IDs."""
    ids = row["prompt_ids"] + row["target_ids"]
    if len(ids) < PREFIX_TOKENS + SUFFIX_TOKENS:
        raise ValueError(f'{row["id"]}: prepared document has fewer than 2048 tokens')
    return dict(
        id=row["id"],
        source_id=row["source_id"],
        prompt_ids=ids[:PREFIX_TOKENS],
        target_ids=ids[PREFIX_TOKENS : PREFIX_TOKENS + SUFFIX_TOKENS],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data", type=Path, required=True, help="Prepared Ouro data directory")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = [split_document(row) for row in rows(args.data / "pg19.jsonl")]
    torch.set_num_threads(4)
    torch.manual_seed(31847)
    model = load_model(args.model_path, device="cuda", dtype=torch.bfloat16, attention="sdpa")
    runner = OuroRunner(model, args.variant)
    variant = VARIANTS[args.variant]
    save(args.output / "config.json", dict(
        variant=args.variant, model_id=MODEL_ID, model_revision=MODEL_REVISION,
        model_path=args.model_path, started_at_utc=now(),
        training=False, optimizer_updates=0, seed=31847,
        prefill_steps=variant.prefill_steps, decode_steps=variant.decode_steps,
        cache_mode=variant.cache_mode, batch_size=1,
        dtype="bfloat16", attention="torch SDPA", graph=False,
        parameter_count=sum(p.numel() for p in model.parameters()),
        torch_version=torch.__version__,
        transformers_version=__import__("transformers").__version__,
        gpu=torch.cuda.get_device_name(),
        dataset="PG19 test; same previously prepared documents and tokenizer",
        data_file=str(args.data / "pg19.jsonl"),
        source_ids=[row["source_id"] for row in source],
        document_ids=[row["id"] for row in source],
        samples=len(source), prefix_tokens=PREFIX_TOKENS,
        suffix_prediction_tokens_per_sample=SUFFIX_TOKENS,
        first_target_raw_token_index=PREFIX_TOKENS,
        last_target_raw_token_index=PREFIX_TOKENS + SUFFIX_TOKENS - 1,
        first_target_from="Final prefill-loop logits at prefix position 1535",
        subsequent_targets_from="511 sequential teacher-forced decode calls",
        note="Continuation token NLL, not PG19 word perplexity or a training result.",
    ))
    outputs = []
    started = time.perf_counter()
    with (args.output / "pg19.jsonl").open("w") as f:
        for row in source:
            step_started = time.perf_counter()
            result = continuation(runner, row)
            result["source_id"] = row["source_id"]
            result["seconds"] = time.perf_counter() - step_started
            outputs.append(result)
            f.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            save(args.output / "progress.json", dict(
                variant=args.variant, completed=len(outputs), total=len(source),
                prediction_tokens=sum(r["target_tokens"] for r in outputs),
                observed_at_utc=now(),
            ))
    prediction_tokens = sum(row["target_tokens"] for row in outputs)
    quality = dict(
        samples=len(outputs), prediction_tokens=prediction_tokens,
        prefix_tokens=PREFIX_TOKENS, suffix_tokens=SUFFIX_TOKENS,
        suffix_nll=sum(row["nll_sum"] for row in outputs) / prediction_tokens,
        first_token_nll=sum(row["first_token_nll"] for row in outputs) / len(outputs),
        remaining_511_tokens_nll=sum(
            row["remaining_tokens_nll"] * (row["target_tokens"] - 1) for row in outputs
        ) / (prediction_tokens - len(outputs)),
    )
    save(args.output / "quality.json", quality)
    result = dict(
        status="completed", variant=args.variant, completed_at_utc=now(),
        quality=quality, optimizer_updates=0,
        evaluation_seconds=time.perf_counter() - started,
    )
    save(args.output / "result.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
