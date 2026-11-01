"""HELMET (Yen et al., ICLR 2025) short-context subset for base models, at 8K and 16K tokens.

Categories: RAG (NQ, TriviaQA, HotpotQA, PopQA; 2-shot, 100 questions at every gold-document
depth), synthetic recall (JSON KV, 100), passage re-ranking (MS MARCO, all 41 queries x 3 depths,
2-shot) and many-shot ICL (TREC coarse/fine, Banking77, CLINC150, NLU; balanced demos, labels
mapped to random integers). Prompt templates, shot counts, generation lengths, newline stopping,
context truncation (drop the end of the context so prompt + generation fits the length) and
metrics follow the HELMET code (data.py, model_utils.tokenize, utils.py) without a chat template.
Question subsets are drawn with seed 42 (`--raw` holds the files selected from HELMET data.tar.gz).
"""

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import random

import pandas as pd
from transformers import AutoTokenizer

RAG = dict(nq="nq-{}-multikilt_1000_k{}_dep6", triviaqa="triviaqa-{}-multikilt_1000_k{}_dep6",
           hotpotqa="hotpotqa-{}-multikilt_1000_k{}_dep3", popqa="popqa_test_1000_k{}_dep6")
DOCS = {8192: 50, 16384: 105}
KV = {8192: 105, 16384: 220}
RERANK = {8192: 50, 16384: 130}
ICL_SHOTS = {"trec_coarse": (400, 800), "trec_fine": (400, 800), "banking77": (360, 720),
             "clinic150": (440, 880), "nlu": (510, 1020)}


def load(path):
    return [json.loads(line) for line in open(path)]


def field(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def fit(tok, template, item, length, gen):
    """HELMET truncation: cut the end of the context until prompt + generation <= length."""
    ids = tok.encode(template.format(**item), add_special_tokens=False)
    over = len(ids) - (length - gen)
    if over > 0:
        ctx = tok.encode(item["context"], add_special_tokens=False)
        item = dict(item, context=tok.decode(ctx[:-over]))
        ids = tok.encode(template.format(**item), add_special_tokens=False)
    return ids


def seed_of(text, offset=0):
    return (int(hashlib.sha256(str(text).encode("utf-8")).hexdigest(), 16) + offset) % 2**31


def rag_rows(tok, raw, name, length):
    template = ("Use the given documents to write a concise and short answer to the question. Write your answer in the "
                "following format:\nAnswer: [answer]\n\n{demos}{context}\n\nQuestion: {question}\nAnswer:")
    pattern = RAG[name]
    test = load(raw / f"{pattern.format('dev', DOCS[length]) if name != 'popqa' else pattern.format(DOCS[length])}.jsonl")
    demos = load(raw / f"{pattern.format('train', 3) if name != 'popqa' else pattern.format(3)}.jsonl")
    key = "id" if "id" in test[0] else "question"
    passage = lambda c: f"Document (Title: {c['title']}): {c['text']}"
    out = []
    for i, r in enumerate(test):
        pool = [d for d in demos if d[key] != r[key]] if name == "popqa" else demos
        pool = list(pool)
        random.Random(seed_of(r[key])).shuffle(pool)
        chosen, seen = [], set()
        for d in pool:
            if d[key] not in seen:
                seen.add(d[key]); chosen.append(d)
            if len(chosen) == 2:
                break
        demo_text = "\n\n".join("\n\n".join(passage(c) for c in field(d["ctxs"])) + f"\n\nQuestion: {d['question']}\nAnswer: {field(d['answers'])[0]}"
                                for d in chosen) + "\n\n"
        item = dict(demos=demo_text, context="\n\n".join(passage(c) for c in field(r["ctxs"])), question=r["question"])
        out.append(dict(id=f"helmet/rag_{name}/{length}/{i}", task=f"rag_{name}", length=length,
                        prompt_ids=fit(tok, template, item, length, 20), max_new_tokens=20, until=["\n"],
                        answers=field(r["answers"]), metric="subem"))
    return out


def kv_rows(tok, raw, length):
    template = ("{context}\n\nExtract the value corresponding to the specified key in the JSON object below.\n\n"
                "{demos}Key: {question}\nCorresponding value:")
    out = []
    for i, r in enumerate(load(raw / f"test_k{KV[length]}_dep6.jsonl")):
        demos = field(r["demos"])[:2]
        item = dict(context=r["context"], question=r["question"],
                    demos="\n\n".join(f"Key: {k}\nCorresponding value: {v}" for k, v in demos) + "\n\n")
        out.append(dict(id=f"helmet/recall_jsonkv/{length}/{i}", task="recall_jsonkv", length=length,
                        prompt_ids=fit(tok, template, item, length, 100), max_new_tokens=100, until=[],
                        answers=[r["answer"]], metric="subem"))
    return out


def rerank_rows(tok, raw, length):
    template = ("You are provided with a list of documents, each indicated by their ID. Rank each document based on their "
                "relevance to the question in descending order from most relelvant to least relevant texts. Include all "
                "documents in the rankings. Write your answer using the unique IDs, with the following format:\nRanking: "
                "ID3 > ID1 > ID2\n\n{demos}{context}\n\nQuery: {question}\nRanking:")
    test = load(raw / f"test_reranking_data_k{RERANK[length]}_dep3.jsonl")
    demos = load(raw / "test_reranking_data_k10_dep3.jsonl")
    out = []
    for i, r in enumerate(test):
        ctxs = field(r["ctxs"])
        fmt = (lambda c: f"[ID: {c['id']}] Document (Title: {c['title']}): {c['text']}") if "title" in ctxs[0] else \
              (lambda c: f"[ID: {c['id']}] Document: {c['text']}")
        pool = [d for d in demos if d["qid"] != r["qid"]]
        random.Random(seed_of(r["qid"])).shuffle(pool)
        chosen, seen = [], set()
        for d in pool:
            if d["qid"] not in seen:
                seen.add(d["qid"]); chosen.append(d)
            if len(chosen) == 2:
                break
        demo_text = ""
        for d in chosen:
            dc = field(d["ctxs"])
            ranking = " > ".join(c["id"] for c in sorted(dc, key=lambda c: c["label"], reverse=True))
            demo_text += "\n\n".join(fmt(c) for c in dc) + f"\n\nQuery: {d['query']}\nRanking: {ranking}" + "\n\n"
        item = dict(demos=demo_text, context="\n\n".join(fmt(c) for c in ctxs), question=r["query"])
        out.append(dict(id=f"helmet/rerank_msmarco/{length}/{i}", task="rerank_msmarco", length=length,
                        prompt_ids=fit(tok, template, item, length, 200), max_new_tokens=200, until=["\n"],
                        qrel={c["id"]: int(c["label"]) for c in ctxs}, metric="ndcg10"))
    return out


def icl_data(raw, name):
    if name.startswith("trec"):
        def read(p):
            rows = []
            for line in open(p, encoding="latin-1"):
                label, text = line.strip().split(" ", 1)
                rows.append(dict(text=text, label=label if name == "trec_fine" else label.split(":")[0]))
            return rows
        return read(raw / "trec_train.label"), read(raw / "trec_test.label")
    if name == "banking77":
        read = lambda p: [dict(text=r["text"], label=r["category"]) for r in csv.DictReader(open(p))]
        return read(raw / "banking_train.csv"), read(raw / "banking_test.csv")
    if name == "clinic150":
        read = lambda p: [dict(text=r.text, label=int(r.intent)) for r in pd.read_parquet(p).itertuples()]
        return read(raw / "clinc_train.parquet"), read(raw / "clinc_validation.parquet")
    rows = [dict(text=r["answer_normalised"] or r["answer"], label=f"{r['scenario']}_{r['intent']}")
            for r in csv.DictReader(open(raw / "nlu.csv"), delimiter=";") if (r["answer_normalised"] or r["answer"])]
    random.Random(42).shuffle(rows)
    cut = int(0.9 * len(rows))
    return rows[:cut], rows[cut:]


def balance(data, shots, seed):
    """HELMET balance_labels: consecutive rounds that each hold every label once, labels shuffled per round."""
    rand = random.Random(seed)
    by = {}
    for x in data:
        by.setdefault(x["label"], []).append(x)
    rounds = math.ceil(shots / len(by))
    sets = [[] for _ in range(rounds)]
    for samples in by.values():
        idx = rand.sample(range(len(samples)), rounds % len(samples))
        while len(idx) < rounds:
            idx += rand.sample(range(len(samples)), min(rounds - len(idx), len(samples)))
        for i, j in enumerate(idx):
            sets[i].append(samples[j])
    for s in sets:
        rand.shuffle(s)
    return [x for s in sets for x in s][:shots]


def icl_rows(tok, raw, name, length, max_test):
    template = ("Use the provided mapping from the text to label to assign a label to the text. Only output "
                "\"label: {{label}}\" and nothing else. \n\n{context}\n\n{question}\nlabel:")
    train, test = icl_data(raw, name)
    test = list(test)
    random.Random(42).shuffle(test)
    test = balance(test, max_test, 42) if len(test) > max_test else test
    labels = sorted({x["label"] for x in train}, key=str)
    shots = ICL_SHOTS[name][0 if length == 8192 else 1]
    out = []
    for i, r in enumerate(test):
        seed = seed_of(r["text"], 42)
        demos = balance(train, shots, seed)
        mapping = list(range(len(labels)))
        random.Random(seed).shuffle(mapping)
        to = {l: mapping[k] for k, l in enumerate(labels)}
        item = dict(context="\n\n".join(f"{d['text']}\nlabel: {to[d['label']]}" for d in demos), question=r["text"])
        out.append(dict(id=f"helmet/icl_{name}/{length}/{i}", task=f"icl_{name}", length=length,
                        prompt_ids=fit(tok, template, item, length, 20), max_new_tokens=20, until=["\n"],
                        answers=[str(to[r["label"]])], metric="icl_em"))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--icl-test", type=int, default=500)
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    rows = []
    for length in (8192, 16384):
        for name in RAG:
            rows += rag_rows(tok, args.raw, name, length)
        rows += kv_rows(tok, args.raw, length) + rerank_rows(tok, args.raw, length)
        for name in ICL_SHOTS:
            rows += icl_rows(tok, args.raw, name, length, args.icl_test)
    args.output.write_text("".join(json.dumps(r) + "\n" for r in rows))
    stats = {}
    for r in rows:
        s = stats.setdefault(f"{r['task']}/{r['length']}", [0, 0, 10**9])
        s[0] += 1; s[1] = max(s[1], len(r["prompt_ids"])); s[2] = min(s[2], len(r["prompt_ids"]))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
