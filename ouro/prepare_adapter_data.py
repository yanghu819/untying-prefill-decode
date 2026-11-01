"""Prepare the shared, token-budgeted training stream for the Ouro KV adapters.

Only manifest.json is a safe result artifact. Corpus tokens, GSM8K training
records, and source-document selection stay in the remote data directory.
"""

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from itertools import islice
import json
import os
from pathlib import Path
import random

import numpy as np


FINEWEB_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
GSM_DATASET = "openai/gsm8k"
GSM_REVISION = "740312add88f781978c0658806c59bc2815b9866"
MODEL_ID = "ByteDance/Ouro-1.4B"
MODEL_REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
HARNESS_REVISION = "d6de81643928d653435c431bae19945d41d32520"
PREFIX_TOKENS = 1536
SUFFIX_TOKENS = 512
WINDOW_TOKENS = PREFIX_TOKENS + SUFFIX_TOKENS


class AdapterDataset:
    """Indexed, already ordered samples; ranks partition each global block."""

    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        self.samples = np.load(directory / "samples.npy", mmap_mode="r")
        self.effective_tokens = self.samples[:, 2]
        self.fineweb = np.memmap(directory / "fineweb_train.bin", dtype="<u4", mode="r")
        self.gsm = [json.loads(line) for line in (directory / "gsm_train.jsonl").read_text().splitlines()]
        # Optional long-context variant: same suffix, longer prefix ending at the same place.
        self.prefix_tokens = int(self.manifest.get("prefix_tokens_override", PREFIX_TOKENS))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        source, item, effective = map(int, self.samples[index])
        if source == 0:
            start = item * WINDOW_TOKENS
            prefix = self.fineweb[start + PREFIX_TOKENS - self.prefix_tokens : start + PREFIX_TOKENS].tolist()
            suffix = self.fineweb[start + PREFIX_TOKENS : start + WINDOW_TOKENS].tolist()
            name = "fineweb"
        else:
            row = self.gsm[item]
            prefix, suffix = row["prompt_ids"], row["target_ids"]
            name = "gsm"
        return dict(prefix_ids=prefix, suffix_ids=suffix, source=name,
                    effective_tokens=effective)


def prepare_gsm(tokenizer, task_yaml, cache_dir, output):
    """Use train only, retaining each entire worked answer and its final value."""
    from datasets import load_dataset
    import yaml

    task = yaml.safe_load(task_yaml.read_text())
    demos = task["fewshot_config"]["samples"][:3]
    prefix = "\n\n".join(f'Q: {row["question"]}\nA: {row["target"]}' for row in demos) + "\n\n"
    demo_questions = {" ".join(row["question"].split()) for row in demos}
    source = load_dataset(GSM_DATASET, name="main", split="train",
                          revision=GSM_REVISION, cache_dir=str(cache_dir / "datasets"))
    prepared, excluded = [], []
    for source_index, row in enumerate(source):
        if " ".join(row["question"].split()) in demo_questions:
            excluded.append(source_index)
            continue
        rationale, final = row["answer"].rsplit("####", 1)
        # Preserve all rationale (including calculation annotations). Only the
        # final-answer marker changes to match the fixed evaluation's format.
        target = " " + rationale.rstrip() + "\nThe answer is " + final.strip() + "."
        prompt_ids = tokenizer.encode(prefix + f'Q: {row["question"]}\nA:',
                                      add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        target_ids.append(tokenizer.eos_token_id)
        prepared.append(dict(source_index=source_index, prompt_ids=prompt_ids,
                             target_ids=target_ids, effective_tokens=len(target_ids) - 1))
    with (output / "gsm_train.jsonl").open("w") as handle:
        for row in prepared:
            handle.write(json.dumps(row) + "\n")
    lengths = [len(row["target_ids"]) for row in prepared]
    return prepared, dict(
        dataset=GSM_DATASET, revision=GSM_REVISION, config="main", split="train",
        original_samples=len(source), retained_samples=len(prepared),
        excluded_demo_overlap_samples=len(excluded),
        fewshot=3, fewshot_indices=[0, 1, 2], harness_revision=HARNESS_REVISION,
        task_yaml=str(task_yaml), prompt_format="three fixed demonstrations, then Q: {question}\\nA:",
        target_format="complete original rationale; #### value becomes The answer is value.; append EOS",
        answer_truncation=False, add_special_tokens=False, chat_template=False,
        minimum_target_tokens=min(lengths), maximum_target_tokens=max(lengths),
        minimum_prompt_tokens=min(len(row["prompt_ids"]) for row in prepared),
        maximum_prompt_tokens=max(len(row["prompt_ids"]) for row in prepared),
        full_pool_effective_kl_tokens=sum(row["effective_tokens"] for row in prepared),
    )


def make_schedule(gsm, budget, global_batch, seed):
    """Balance complete samples toward 1% GSM valid KL tokens, never trim them."""
    rng = random.Random(seed)
    deck = list(range(len(gsm)))
    rng.shuffle(deck)
    cursor, fineweb_index, total_tokens, gsm_tokens = 0, 0, 0, 0
    samples = []
    while total_tokens < budget or len(samples) % global_batch:
        if cursor == len(deck):
            rng.shuffle(deck)
            cursor = 0
        gsm_index = deck[cursor]
        gsm_count = gsm[gsm_index]["effective_tokens"]
        # Multiplying the 1:99 target by 100 makes the choice integer-exact.
        after_gsm = abs(100 * (gsm_tokens + gsm_count) - (total_tokens + gsm_count))
        after_fineweb = abs(100 * gsm_tokens - (total_tokens + SUFFIX_TOKENS - 1))
        if after_gsm < after_fineweb:
            samples.append((1, gsm_index, gsm_count))
            gsm_tokens += gsm_count
            total_tokens += gsm_count
            cursor += 1
        else:
            samples.append((0, fineweb_index, SUFFIX_TOKENS - 1))
            fineweb_index += 1
            total_tokens += SUFFIX_TOKENS - 1
    schedule = np.asarray(samples, dtype=np.int64)
    usage = Counter(int(row[1]) for row in samples if row[0] == 1)
    return schedule, dict(
        seed=seed, global_batch=global_batch, optimizer_updates=len(samples) // global_batch,
        samples=len(samples), requested_effective_kl_tokens=budget,
        effective_kl_tokens=total_tokens, overshoot_effective_kl_tokens=total_tokens - budget,
        source_samples=dict(fineweb=fineweb_index, gsm=sum(usage.values())),
        source_effective_kl_tokens=dict(fineweb=total_tokens - gsm_tokens, gsm=gsm_tokens),
        observed_gsm_fraction=gsm_tokens / total_tokens,
        target_gsm_fraction=0.01, gsm_unique_samples=len(usage),
        gsm_maximum_sample_uses=max(usage.values(), default=0),
        gsm_effective_pool_passes=gsm_tokens / sum(row["effective_tokens"] for row in gsm),
        fineweb_window_order="Published document order, consecutive non-overlapping 2048-token windows",
        gsm_order="Python Random(seed).shuffle of all retained train rows; reshuffle only on exhaustion",
        source_selection="Choose next whole GSM/FineWeb sample minimizing deviation from 1% GSM effective KL tokens",
        stop="First complete global batch with effective KL tokens >= requested budget",
        kl_positions="suffix[1:]; first suffix target excluded because it is predicted by unadapted prefill",
    )


def prepare_fineweb(tokenizer, required_tokens, cache_dir, output, batch_size):
    from datasets import load_dataset

    documents = iter(load_dataset(
        FINEWEB_DATASET, name="sample-10BT", split="train", revision=FINEWEB_REVISION,
        streaming=True, cache_dir=str(cache_dir / "datasets"),
    ))
    written, ordinal, fetched, source_tokens, final_trim = 0, 0, 0, 0, 0
    with closing(documents), (output / "fineweb_train.bin").open("wb") as stream, \
            (output / "fineweb_selection.jsonl").open("w") as selection:
        while written < required_tokens:
            batch = list(islice(documents, batch_size))
            if not batch:
                raise RuntimeError("The pinned FineWeb stream ended before the staged token budget")
            fetched += len(batch)
            encoded = tokenizer([row["text"] for row in batch], add_special_tokens=False,
                                return_attention_mask=False, return_token_type_ids=False,
                                padding=False, truncation=False)["input_ids"]
            for row, ids in zip(batch, encoded):
                ids.append(tokenizer.eos_token_id)
                take = min(len(ids), required_tokens - written)
                np.asarray(ids[:take], dtype="<u4").tofile(stream)
                selection.write(json.dumps(dict(
                    ordinal=ordinal, source_id=row["id"], tokens_including_eos=len(ids),
                    written_tokens=take,
                )) + "\n")
                written += take
                source_tokens += len(ids)
                final_trim = len(ids) - take
                ordinal += 1
                if written == required_tokens:
                    break
            print(json.dumps(dict(stage="fineweb", selected_documents=ordinal,
                                  written_tokens=written, required_tokens=required_tokens)), flush=True)
    return dict(
        dataset=FINEWEB_DATASET, revision=FINEWEB_REVISION, config="sample-10BT", split="train",
        order="Published streaming order, no shuffle", selected_documents=ordinal,
        fetched_and_tokenized_documents=fetched, unselected_read_ahead_documents=fetched - ordinal,
        source_tokens_including_eos=source_tokens, written_tokens=written,
        final_document_unused_tokens=final_trim, dtype="little-endian uint32",
        document_separator_id=tokenizer.eos_token_id,
        packing="Concatenate complete document token IDs plus EOS; trim final document to staged boundary",
        prefix_tokens=PREFIX_TOKENS, suffix_tokens=SUFFIX_TOKENS,
        effective_kl_tokens_per_window=SUFFIX_TOKENS - 1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--task-yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--effective-kl-tokens", type=int, default=50_000_000)
    parser.add_argument("--global-batch", type=int, default=24)
    parser.add_argument("--seed", type=int, default=31847)
    parser.add_argument("--tokenizer-batch-size", type=int, default=128)
    args = parser.parse_args()
    os.environ["HF_HOME"] = str(args.cache_dir)
    os.environ["HF_HUB_CACHE"] = str(args.cache_dir / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(args.cache_dir / "datasets")
    from transformers import AutoTokenizer

    args.output.mkdir(parents=True, exist_ok=False)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True,
                                             trust_remote_code=True)
    gsm, gsm_spec = prepare_gsm(tokenizer, args.task_yaml, args.cache_dir, args.output)
    schedule, schedule_spec = make_schedule(gsm, args.effective_kl_tokens, args.global_batch, args.seed)
    np.save(args.output / "samples.npy", schedule)
    print(json.dumps(dict(stage="schedule", **schedule_spec)), flush=True)
    fineweb_spec = prepare_fineweb(
        tokenizer, schedule_spec["source_samples"]["fineweb"] * WINDOW_TOKENS,
        args.cache_dir, args.output, args.tokenizer_batch_size,
    )
    gsm_input_tokens = sum(len(gsm[int(i)]["prompt_ids"]) + len(gsm[int(i)]["target_ids"])
                           for source, i, _ in schedule if source == 1)
    manifest = dict(
        status="prepared", prepared_at_utc=datetime.now(timezone.utc).isoformat(),
        tokenizer=dict(model=MODEL_ID, revision=MODEL_REVISION, path=str(args.model_path),
                       local_files_only=True, retokenized_for_ouro=True),
        schedule=schedule_spec, fineweb=fineweb_spec, gsm=gsm_spec,
        input_tokens=fineweb_spec["written_tokens"] + gsm_input_tokens,
        input_tokens_by_source=dict(fineweb=fineweb_spec["written_tokens"], gsm=gsm_input_tokens),
        files=dict(fineweb="fineweb_train.bin", gsm="gsm_train.jsonl", schedule="samples.npy"),
        schedule_columns=["source: 0=fineweb, 1=gsm", "source_item_index", "effective_kl_tokens"],
        arms="A/B/C consume the same ordered global batches; no rank-local or arm-local sampling",
        limitations=["No benchmark files enter training; no semantic decontamination claim.",
                     "This directory stages only the first requested milestone; later budget requires a separate continuation."],
        safe_to_collect="manifest.json only; source text, tokens, selection and schedule remain remote",
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(status="prepared", manifest=str(args.output / "manifest.json"),
                          effective_kl_tokens=schedule_spec["effective_kl_tokens"])), flush=True)


if __name__ == "__main__":
    main()
