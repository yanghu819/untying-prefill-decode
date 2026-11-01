"""Full RULER and three full LongBench QA tasks; remote-only inputs.

Before first preparation: python -m nltk.downloader -d /root/nltk_data punkt_tab
The official essay generator uses this NLTK sentence tokenizer resource.
"""
import argparse
from pathlib import Path
import json
import runpy
import subprocess
import sys
import zipfile
from ouro.data import (download, encode, write_jsonl, RULER_REVISION,
                       LONGBENCH_CODE_REVISION, LONGBENCH_DATA_REVISION, SEED)
FULL_RULER_TASKS = ("niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1",
                    "niah_multikey_2", "niah_multikey_3", "niah_multivalue", "niah_multiquery",
                    "vt", "cwe", "fwe", "qa_1", "qa_2")

def prepare_ruler(tokenizer, model_path, output, resume=False):
    import yaml

    upstream = output / "upstream" / "RULER"
    files = [
        "LICENSE", "scripts/synthetic.yaml", "scripts/data/tokenizer.py",
        "scripts/data/manifest_utils.py", "scripts/data/synthetic/constants.py",
        "scripts/data/synthetic/niah.py", "scripts/data/synthetic/variable_tracking.py",
        "scripts/data/synthetic/common_words_extraction.py",
        "scripts/data/synthetic/freq_words_extraction.py", "scripts/data/synthetic/qa.py",
    ]
    for name in files:
        if not resume or not (upstream / name).exists():
            download(f"https://raw.githubusercontent.com/NVIDIA/RULER/{RULER_REVISION}/{name}",
                     upstream / name)
    # This upstream asset is stored in Git LFS. The raw URL returns its pointer,
    # which caused the original CWE preparation to stop before any samples.
    words_name = "scripts/data/synthetic/json/english_words.json"
    words_url = f"https://media.githubusercontent.com/media/NVIDIA/RULER/{RULER_REVISION}/{words_name}"
    if not (upstream / words_name).exists():
        download(words_url, upstream / words_name)
    assets = {
        "squad.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
        "hotpotqa.json": "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json",
    }
    for asset, url in assets.items():
        path = upstream / "scripts/data/synthetic/json" / asset
        if not path.exists():
            download(url, path)
    essay_dir = upstream / "scripts/data/synthetic/json"
    if not (essay_dir / "PaulGrahamEssays.json").exists():
        for name in ("download_paulgraham_essay.py", "PaulGrahamEssays_URLs.txt"):
            path = essay_dir / name
            if not path.exists():
                download(f"https://raw.githubusercontent.com/NVIDIA/RULER/{RULER_REVISION}/scripts/data/synthetic/json/{name}", path)
        # HTTPS reaches the same source pages through the existing proxy.
        urls = essay_dir / "PaulGrahamEssays_URLs.txt"
        urls.write_text(urls.read_text().replace("http://www.paulgraham.com/", "https://www.paulgraham.com/"))
        log = output / "logs/paulgraham-download.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as stream:
            subprocess.run([sys.executable, "download_paulgraham_essay.py"], cwd=essay_dir,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        if "Fail download" in log.read_text():
            (essay_dir / "PaulGrahamEssays.json").rename(essay_dir / "PaulGrahamEssays.incomplete.json")
            raise RuntimeError("Official essay downloader skipped source pages; inspect its log before generating RULER")
    tasks = yaml.safe_load((upstream / "scripts/synthetic.yaml").read_text())
    constants = runpy.run_path(str(upstream / "scripts/data/synthetic/constants.py"))["TASKS"]
    summary = {
        "source": "https://github.com/NVIDIA/RULER", "source_revision": RULER_REVISION,
        "license": "Apache-2.0", "seed": SEED, "tasks": list(FULL_RULER_TASKS),
        "samples_per_task_length": 500, "context_limits": [4096, 8192, 16384],
        "length_definition": "Official max_seq_length includes generation reserve; record actual prompt lengths.",
        "template": "Official base task template plus official answer_prefix; no chat/thinking template.",
        "metric": "Official string_match_all for retrieval; string_match_part for QA aliases.",
        "adaptation": "Full 13 tasks, 500 samples each, generated once and shared by all conditions.",
        "english_words_source": words_url,
        "subsets": [],
    }
    for context in (4096, 8192, 16384):
        rows = []
        for task in FULL_RULER_TASKS:
            config = tasks[task]
            definition = constants[config["task"]]
            generated = output / "raw" / "ruler" / str(context)
            command = [
                sys.executable, str(upstream / "scripts/data/synthetic" / (config["task"] + ".py")),
                "--save_dir", str(generated), "--save_name", task, "--subset", "validation",
                "--tokenizer_path", model_path, "--tokenizer_type", "hf",
                "--max_seq_length", str(context),
                "--tokens_to_generate", str(definition["tokens_to_generate"]),
                "--num_samples", "500", "--random_seed", str(SEED),
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
                        stream.write("\nContinuing an unfinished generator; existing complete samples are reused.\n")
                        stream.flush()
                    subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
            else:
                print(f"Reusing original {context}/{task} samples", flush=True)
            raw = [json.loads(line) for line in source_file.read_text().splitlines()]
            if len(raw) != 500:
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


def prepare_longbench(tokenizer, output, task, archive):
    base = f"https://raw.githubusercontent.com/THUDM/LongBench/{LONGBENCH_CODE_REVISION}"
    upstream = output / "upstream" / "LongBench"
    for name in ("LICENSE", "LongBench/config/dataset2prompt.json", "LongBench/config/dataset2maxlen.json"):
        download(f"{base}/{name}", upstream / name)
    template = json.loads((upstream / "LongBench/config/dataset2prompt.json").read_text())[task]
    max_new_tokens = json.loads((upstream / "LongBench/config/dataset2maxlen.json").read_text())[task]
    with zipfile.ZipFile(archive) as z:
        records = [json.loads(line) for line in z.read(f"data/{task}.jsonl").decode().splitlines()]
    rows, metadata = [], []
    before, after = template.split("{context}")
    for index, record in enumerate(records):
        prefix = before.format(**record)
        suffix = after.format(**record)
        context_ids = encode(tokenizer, record["context"])
        prompt = prefix + record["context"] + suffix
        original_length = len(encode(tokenizer, prompt))
        kept = len(context_ids)
        if original_length > 16384:
            # LongBench's head+tail strategy, applied only to the article so
            # the complete question and both fixed instructions are retained.
            kept = max(0, 16384 - len(encode(tokenizer, prefix + suffix)))
            while True:
                left, right = (kept + 1) // 2, kept // 2
                article = tokenizer.decode(context_ids[:left], skip_special_tokens=True)
                if right:
                    article += tokenizer.decode(context_ids[-right:], skip_special_tokens=True)
                prompt = prefix + article + suffix
                length = len(encode(tokenizer, prompt))
                if length <= 16384:
                    break
                kept -= length - 16384
                if kept < 0:
                    raise ValueError("Qasper question/template alone exceeds 16384 tokens")
        ids = encode(tokenizer, prompt)
        source_id = record["_id"]
        rows.append({
            "id": f"{task}/{source_id}", "task": task, "context_length": 16384,
            "prompt_ids": ids, "answers": record["answers"],
            "max_new_tokens": max_new_tokens, "source_id": source_id,
            "source_index": index, "prompt_tokens": len(ids),
        })
        metadata.append({
            "id": source_id, "source_index": index,
            "original_prompt_tokens": original_length, "prompt_tokens": len(ids),
            "original_article_tokens": len(context_ids), "kept_article_tokens": kept,
            "truncated": original_length > 16384,
        })
    if len(rows) != 200:
        raise ValueError("LongBench Qasper did not supply the full 200 examples")
    write_jsonl(output / f"longbench-{task}.jsonl", rows)
    return {
        "source": "https://huggingface.co/datasets/zai-org/LongBench",
        "source_revision": LONGBENCH_DATA_REVISION,
        "code_source": "https://github.com/THUDM/LongBench",
        "code_revision": LONGBENCH_CODE_REVISION, "code_license": "MIT",
        "task": task,
        "selection": "Full original test JSONL, all 200 records",
        "max_prompt_tokens": 16384, "max_new_tokens": max_new_tokens, "count": 200,
        "template": "Official per-task LongBench plain prompt; no chat/thinking wrapper",
        "truncation": "Head+tail article truncation, retaining full question and template. Adaptation of LongBench whole-prompt middle truncation.",
        "metric": "Official LongBench max-reference English token F1",
        "examples": metadata,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--original-data", type=Path, required=True)
    args = p.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.original_data / "raw/longbench-data.zip"
    marker = args.output / "longbench-manifest.json"
    if marker.exists():
        manifest = json.loads(marker.read_text())
    else:
        manifest = {"longbench": {task: prepare_longbench(tokenizer, args.output, task, archive)
                                 for task in ("qasper", "hotpotqa", "2wikimqa")}}
    for task, info in manifest["longbench"].items():
        info.pop("underlying_dataset", None)
        info.pop("underlying_data_license", None)
        info.update(task=task, template="Official per-task LongBench plain prompt; no chat/thinking wrapper")
    (args.output / "longbench-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["ruler"] = prepare_ruler(tokenizer, args.model_path, args.output, resume=True)
    manifest["status"] = "prepared"
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

if __name__ == "__main__":
    main()
