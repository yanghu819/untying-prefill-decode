"""Prepare fixed Ouro development subsets on the remote machine (CPU only).

Raw documents, synthetic prompts, answers and token IDs stay under --output.
Only manifest.json is intended for the research repository.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import runpy
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile

from ouro.score import RULER_TASKS


RULER_REVISION = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
LONGBENCH_CODE_REVISION = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
LONGBENCH_DATA_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
PG19_CODE_REVISION = "b0e1f2925359688af19f8784c23c3deeda52d3ea"
SEED = 31847


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path


def write_jsonl(path, rows):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def encode(tokenizer, text):
    # The base model receives plain continuation prompts, without chat wrappers.
    return tokenizer.encode(text, add_special_tokens=False)


def prepare_pg19(tokenizer, output):
    listing_url = "https://storage.googleapis.com/storage/v1/b/deepmind-gutenberg/o?prefix=test%2F&maxResults=200"
    with urllib.request.urlopen(listing_url) as response:
        listing = json.load(response)
    objects = sorted((x for x in listing["items"] if x["name"].endswith(".txt")),
                     key=lambda x: x["name"])
    quality, benchmark, selected = [], [], []
    for source_index, obj in enumerate(objects):
        path = download(obj["mediaLink"], output / "raw" / "pg19" / Path(obj["name"]).name)
        ids = encode(tokenizer, path.read_text())
        source_id = f'{obj["name"]}@{obj["generation"]}'
        use_quality = len(quality) < 32 and len(ids) >= 2304
        use_benchmark = len(benchmark) < 16 and len(ids) >= 8192
        base = {"source_id": source_id, "source_index": source_index}
        if use_quality:
            quality.append({
                **base, "id": f'pg19/{Path(obj["name"]).stem}', "task": "pg19",
                "prompt_ids": ids[:2048], "target_ids": ids[2048:2304],
            })
        if use_benchmark:
            benchmark.append({
                **base, "id": f'benchmark/{Path(obj["name"]).stem}', "task": "benchmark",
                "prompt_ids": ids[:8192], "prompt_tokens": 8192,
            })
        selected.append({
            **base, "source_url": obj["mediaLink"], "document_tokens": len(ids),
            "quality": use_quality, "benchmark": use_benchmark,
        })
        if len(quality) == 32 and len(benchmark) == 16:
            break
    if len(quality) != 32 or len(benchmark) != 16:
        raise ValueError("PG19 test split did not supply the predefined sample counts")
    write_jsonl(output / "pg19.jsonl", quality)
    write_jsonl(output / "benchmark.jsonl", benchmark)
    return {
        "source": "https://github.com/google-deepmind/pg19",
        "source_revision": PG19_CODE_REVISION, "license": "Apache-2.0",
        "split": "test", "ordering": "GCS object name lexicographic",
        "quality_count": 32, "prompt_tokens": 2048, "target_tokens": 256,
        "prediction_tokens": 8192, "benchmark_count": 16,
        "benchmark_prefix_tokens": 8192, "selected_documents": selected,
        "metric": "Ouro subword continuation NLL, not official full PG19 word perplexity",
    }


def prepare_ruler(tokenizer, model_path, output, resume=False):
    import yaml

    upstream = output / "upstream" / "RULER"
    files = [
        "LICENSE", "scripts/synthetic.yaml", "scripts/data/tokenizer.py",
        "scripts/data/manifest_utils.py", "scripts/data/synthetic/constants.py",
        "scripts/data/synthetic/niah.py", "scripts/data/synthetic/variable_tracking.py",
        "scripts/data/synthetic/common_words_extraction.py",
    ]
    for name in files:
        if not resume or not (upstream / name).exists():
            download(f"https://raw.githubusercontent.com/NVIDIA/RULER/{RULER_REVISION}/{name}",
                     upstream / name)
    # This upstream asset is stored in Git LFS. The raw URL returns its pointer,
    # which caused the original CWE preparation to stop before any samples.
    words_name = "scripts/data/synthetic/json/english_words.json"
    words_url = f"https://media.githubusercontent.com/media/NVIDIA/RULER/{RULER_REVISION}/{words_name}"
    download(words_url, upstream / words_name)
    tasks = yaml.safe_load((upstream / "scripts/synthetic.yaml").read_text())
    constants = runpy.run_path(str(upstream / "scripts/data/synthetic/constants.py"))["TASKS"]
    summary = {
        "source": "https://github.com/NVIDIA/RULER", "source_revision": RULER_REVISION,
        "license": "Apache-2.0", "seed": SEED, "tasks": list(RULER_TASKS),
        "samples_per_task_length": 32, "context_limits": [4096, 8192],
        "length_definition": "Official max_seq_length includes generation reserve; record actual prompt lengths.",
        "template": "Official base task template plus official answer_prefix; no chat/thinking template.",
        "metric": "Official string_match_all: case-insensitive substring recall over all reference strings.",
        "adaptation": "Four-task development subset; niah_multikey_2 uses needle distractors. It is not multikey_1 or full RULER.",
        "english_words_source": words_url,
        "subsets": [],
    }
    for context in (4096, 8192):
        rows = []
        for task in RULER_TASKS:
            config = tasks[task]
            definition = constants[config["task"]]
            generated = output / "raw" / "ruler" / str(context)
            command = [
                sys.executable, str(upstream / "scripts/data/synthetic" / (config["task"] + ".py")),
                "--save_dir", str(generated), "--save_name", task, "--subset", "validation",
                "--tokenizer_path", model_path, "--tokenizer_type", "hf",
                "--max_seq_length", str(context),
                "--tokens_to_generate", str(definition["tokens_to_generate"]),
                "--num_samples", "32", "--random_seed", str(SEED),
                "--template", definition["template"] + definition.get("answer_prefix", ""),
            ]
            for name, value in config["args"].items():
                command.extend(["--" + name, str(value)])
            source_file = generated / task / "validation.jsonl"
            if not resume or not source_file.exists():
                log = output / "logs" / f"ruler-{context}-{task}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("a" if resume else "w") as stream:
                    if resume:
                        stream.write("\nResuming after correcting the pinned Git LFS asset URL.\n")
                        stream.flush()
                    subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
            else:
                print(f"Reusing original {context}/{task} samples", flush=True)
            raw = [json.loads(line) for line in source_file.read_text().splitlines()]
            if len(raw) != 32:
                raise ValueError(f"Official generator returned {len(raw)} examples for {task}")
            lengths = []
            for index, item in enumerate(raw):
                # Official current RULER stores the completion prefix separately.
                prompt = item["input"] + item.get("answer_prefix", "")
                ids = encode(tokenizer, prompt)
                max_new_tokens = definition["tokens_to_generate"]
                if len(ids) + max_new_tokens > context:
                    raise ValueError(f"{task}/{index}: actual Ouro tokens exceed the fixed context budget")
                lengths.append(len(ids))
                rows.append({
                    "id": f"ruler/{context}/{task}/{index}", "task": task,
                    "context_length": context, "prompt_ids": ids,
                    "answers": item["outputs"], "max_new_tokens": max_new_tokens,
                    "source_id": f"{RULER_REVISION}/{SEED}/{context}/{task}/{index}",
                    "source_index": index, "prompt_tokens": len(ids),
                })
            summary["subsets"].append({
                "task": task, "context_length": context, "count": len(raw),
                "max_new_tokens": definition["tokens_to_generate"],
                "prompt_tokens": lengths, "generator_args": config["args"],
            })
        write_jsonl(output / f"ruler-{context}.jsonl", rows)
    return summary


def prepare_qasper(tokenizer, output):
    base = f"https://raw.githubusercontent.com/THUDM/LongBench/{LONGBENCH_CODE_REVISION}"
    upstream = output / "upstream" / "LongBench"
    for name in ("LICENSE", "LongBench/config/dataset2prompt.json", "LongBench/config/dataset2maxlen.json"):
        download(f"{base}/{name}", upstream / name)
    template = json.loads((upstream / "LongBench/config/dataset2prompt.json").read_text())["qasper"]
    max_new_tokens = json.loads((upstream / "LongBench/config/dataset2maxlen.json").read_text())["qasper"]
    archive = download(
        f"https://huggingface.co/datasets/zai-org/LongBench/resolve/{LONGBENCH_DATA_REVISION}/data.zip",
        output / "raw" / "longbench-data.zip",
    )
    with zipfile.ZipFile(archive) as z:
        records = [json.loads(line) for line in z.read("data/qasper.jsonl").decode().splitlines()][:64]
    rows, metadata = [], []
    before, after = template.split("{context}")
    for index, record in enumerate(records):
        prefix = before.format(**record)
        suffix = after.format(**record)
        context_ids = encode(tokenizer, record["context"])
        prompt = prefix + record["context"] + suffix
        original_length = len(encode(tokenizer, prompt))
        kept = len(context_ids)
        if original_length > 8192:
            # LongBench's head+tail strategy, applied only to the article so
            # the complete question and both fixed instructions are retained.
            kept = max(0, 8192 - len(encode(tokenizer, prefix + suffix)))
            while True:
                left, right = (kept + 1) // 2, kept // 2
                article = tokenizer.decode(context_ids[:left], skip_special_tokens=True)
                if right:
                    article += tokenizer.decode(context_ids[-right:], skip_special_tokens=True)
                prompt = prefix + article + suffix
                length = len(encode(tokenizer, prompt))
                if length <= 8192:
                    break
                kept -= length - 8192
                if kept < 0:
                    raise ValueError("Qasper question/template alone exceeds 8192 tokens")
        ids = encode(tokenizer, prompt)
        source_id = record["_id"]
        rows.append({
            "id": f"qasper/{source_id}", "task": "qasper", "context_length": 8192,
            "prompt_ids": ids, "answers": record["answers"],
            "max_new_tokens": max_new_tokens, "source_id": source_id,
            "source_index": index, "prompt_tokens": len(ids),
        })
        metadata.append({
            "id": source_id, "source_index": index,
            "original_prompt_tokens": original_length, "prompt_tokens": len(ids),
            "original_article_tokens": len(context_ids), "kept_article_tokens": kept,
            "truncated": original_length > 8192,
        })
    if len(rows) != 64:
        raise ValueError("LongBench Qasper did not supply the predefined 64 examples")
    write_jsonl(output / "qasper.jsonl", rows)
    return {
        "source": "https://huggingface.co/datasets/zai-org/LongBench",
        "source_revision": LONGBENCH_DATA_REVISION,
        "code_source": "https://github.com/THUDM/LongBench",
        "code_revision": LONGBENCH_CODE_REVISION, "code_license": "MIT",
        "underlying_dataset": "https://huggingface.co/datasets/allenai/qasper",
        "underlying_data_license": "CC-BY-4.0",
        "selection": "First 64 original qasper test JSONL records, before observing model outputs",
        "max_prompt_tokens": 8192, "max_new_tokens": max_new_tokens, "count": 64,
        "template": "Official LongBench Qasper plain prompt; no chat/thinking wrapper",
        "truncation": "Head+tail article truncation, retaining full question and template. Adaptation of LongBench whole-prompt middle truncation.",
        "metric": "Official LongBench max-reference English token F1",
        "examples": metadata,
    }


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse completed manifest stages and original generated RULER files.")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    manifest = {
        "status": "preparing", "model_path": args.model_path, "seed": SEED,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Fixed development subsets; not full-benchmark or paper-reproduction scores",
        "tokenization": "Ouro tokenizer; add_special_tokens=False; no chat template",
        "raw_data_policy": "Only this manifest is safe to copy into the repository; raw prompts, answers, documents and token IDs remain remote.",
    }
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text())
    for name, prepare in [
        ("pg19", lambda: prepare_pg19(tokenizer, output)),
        ("ruler", lambda: prepare_ruler(tokenizer, args.model_path, output, args.resume)),
        ("qasper", lambda: prepare_qasper(tokenizer, output)),
    ]:
        if args.resume and name in manifest:
            print(f"Reusing completed {name}", flush=True)
            continue
        print(f"Preparing {name}", flush=True)
        manifest[name] = prepare()
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["status"] = "prepared"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "prepared", "pg19": 32, "ruler": 256, "qasper": 64,
                      "benchmark": 16}), flush=True)


if __name__ == "__main__":
    main()
