"""Diverse distillation stream (replaces the 50% GSM8K-augmentation math stream).

Sources, mixed by KL (suffix) tokens:
- OpenThoughts-114k: problem -> reasoning trace (first --max-target tokens of the trace)
- Tulu-3 SFT mixture: first user turn -> first assistant turn (general chat, code, math, safety, ...)
- LongAlpaca-12k: long document + instruction -> answer (prompt keeps its last --max-prompt tokens)
- FineWeb-edu: 7680-token document prefix -> next 512 tokens (fresh windows, never used before)
No GSM8K/MATH test or train rows are added on purpose; text is plain (no chat template):
"User: {prompt}\n\nAssistant:" -> " {response}<eos>".
Output keeps the AdapterDataset layout: pool rows in gsm_train.jsonl (source 1), FineWeb windows
as source 0 read from the shared fineweb_train.bin with prefix_tokens_override=7680.
"""

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def chat(prompt):
    return f"User: {prompt.strip()}\n\nAssistant:"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--base-data", type=Path, required=True, help="directory with fineweb_train.bin and manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=40_000_000)
    parser.add_argument("--fractions", type=float, nargs=4, default=[0.3, 0.3, 0.1, 0.3],
                        help="openthoughts tulu longalpaca fineweb, by KL tokens")
    parser.add_argument("--max-target", type=int, default=1024)
    parser.add_argument("--max-prompt", type=int, default=7680)
    parser.add_argument("--fineweb-start", type=int, default=230001)
    parser.add_argument("--global-batch", type=int, default=24)
    parser.add_argument("--seed", type=int, default=31847)
    args = parser.parse_args()
    slow = "nanbeige" in json.loads((Path(args.model_path) / "config.json").read_text()).get("model_type", "")
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True, use_fast=not slow)
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    pools = {"openthoughts": [], "tulu": [], "longalpaca": []}

    def add(name, prompt, response):
        p = enc(chat(prompt))[-args.max_prompt:]
        t = (enc(" " + response.strip()) + [tok.eos_token_id])[: args.max_target]
        if len(t) >= 16 and len(p) >= 4:
            pools[name].append(dict(source=name, prompt_ids=p, target_ids=t, effective_tokens=len(t) - 1))

    for path in sorted(args.raw.glob("open-thoughts_*.parquet")):
        for conv in pd.read_parquet(path)["conversations"]:
            turns = {c["from"]: c["value"] for c in conv}
            if "user" in turns and "assistant" in turns:
                add("openthoughts", turns["user"], turns["assistant"])
    for path in sorted(args.raw.glob("allenai_tulu*.parquet")):
        for msgs in pd.read_parquet(path)["messages"]:
            user = next((m["content"] for m in msgs if m["role"] == "user"), None)
            reply = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
            if user and reply:
                add("tulu", user, reply)
    for row in json.loads((args.raw / "Yukang_LongAlpaca-12k__LongAlpaca-12k.json").read_text()):
        add("longalpaca", row["instruction"], row["output"])

    frac = dict(zip(list(pools) + ["fineweb"], args.fractions))
    rng = random.Random(args.seed)
    rows, pool, offsets = [], [], {}
    for name, items in pools.items():
        rng.shuffle(items)
        offsets[name] = len(pool)
        pool += items
    names = [n for n, f in zip(list(pools) + ["fineweb"], args.fractions) if f > 0]  # fineweb 0: other tokenizers
    used = {n: 0 for n in list(pools) + ["fineweb"]}
    cursor = {n: 0 for n in pools}
    fineweb = args.fineweb_start
    total = 0
    while total < args.tokens or len(rows) % args.global_batch:
        # next source = the one furthest below its target share of KL tokens
        name = min(names, key=lambda n: used[n] - frac[n] * (total + 1))
        if name == "fineweb":
            rows.append((0, fineweb, 511)); fineweb += 1; n = 511
        else:
            item = offsets[name] + cursor[name] % len(pools[name]); cursor[name] += 1
            n = pool[item]["effective_tokens"]; rows.append((1, item, n))
        used[name] += n
        total += n
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "gsm_train.jsonl").open("w") as handle:
        for row in pool:
            handle.write(json.dumps(row) + "\n")
    np.save(args.output / "samples.npy", np.asarray(rows, dtype=np.int64))
    fw = args.output / "fineweb_train.bin"
    if not fw.exists() and frac["fineweb"] > 0:
        os.symlink((args.base_data / "fineweb_train.bin").resolve(), fw)
    manifest = json.loads((args.base_data / "manifest.json").read_text())
    manifest.pop("math_continuation", None)
    manifest.update(prefix_tokens_override=7680, mix=dict(
        sources={n: dict(pool=len(pools[n]), passes=cursor[n] / max(1, len(pools[n]))) for n in pools},
        kl_tokens=dict(used), tokenizer=args.model_path, fractions=frac,
        fineweb_windows=[args.fineweb_start, fineweb - 1], samples=len(rows), max_target=args.max_target,
        max_prompt=args.max_prompt, format="User: {prompt}\\n\\nAssistant: {response}<eos>, no chat template"))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["mix"], indent=2))


if __name__ == "__main__":
    main()
