"""Standard few-shot benchmarks for every Ouro condition/checkpoint: BBH (3-shot CoT,
Suzgun et al. prompts), MATH-500 (Minerva 4-shot), MMLU (5-shot, next-token choice).

`prepare`: build fixed rows (Ouro token ids) from the raw files in --raw.
`eval`: greedy generation (BBH, MATH-500) or first-token choice (MMLU) with the same
runner API as evaluate_adapter (prefill -> first-token logits, then decode).
Scoring follows lm-eval-harness: BBH exact match on the text after "the answer is",
MATH-500 Minerva answer normalization + exact match (sympy equivalence not used).
"""

import argparse
import glob
import json
from pathlib import Path
import math
import os
import re
import string
from types import SimpleNamespace

import pandas as pd
import torch
from transformers import AutoTokenizer

from ouro.evaluate import rows
from ouro.evaluate_adapter import done_rows, make_runner

MINERVA_SHOTS = [
    ("Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.",
     "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct."),
    ("If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
     "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct."),
    ("Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
     "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct."),
    ("If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
     "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct."),
]

# Minerva normalization (Lewkowycz et al. 2022, appendix D), as in lm-eval minerva_math.
SUBSTITUTIONS = [("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""), (r"\ ", ""), (" ", ""), ("mbox", "text"),
                 (",\\text{and}", ","), ("\\text{and}", ","), ("\\text{m}", "\\text{}")]
REMOVED = ["square", "ways", "integers", "dollars", "mph", "inches", "ft", "hours", "km", "units", "\\ldots", "sue",
           "points", "feet", "minutes", "digits", "cents", "degrees", "cm", "gm", "pounds", "meters", "meals", "edges",
           "students", "childrentickets", "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}", "\\text{}^2",
           "\\text{}^3", "\\text{\n}", "\\text{}", r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"',
           "\\dots"]


def normalize(answer):
    answer = answer.split("=")[-1]
    for expr in REMOVED:
        if expr.isalpha():
            answer = re.sub(rf"(?<!\\)\b{expr}\b", "", answer)
    for before, after in SUBSTITUTIONS:
        answer = answer.replace(before, after)
    for expr in REMOVED:
        if expr.isalpha():
            parts = re.split(r"(\\[a-zA-Z]+)", answer)
            for i in range(0, len(parts), 2):
                parts[i] = parts[i].replace(expr, "")
            answer = "".join(parts)
        else:
            answer = answer.replace(expr, "")
    answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", answer)
    answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", answer)
    answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", answer)
    answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", answer)
    answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", answer)
    answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", answer)
    answer = re.sub(r"(sqrt)(\[[^\]]*\])(\w)", "\\1\\2{\\3}", answer)
    answer = re.sub(r"(sqrt)(?!\[)([^{])", "sqrt{\\2}", answer)
    answer = answer.replace("$", "")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+", answer):
        answer = answer.replace(",", "")
    return answer


def minerva_answer(text):
    m = re.search(r"Final Answer: The final answer is(.*?). I hope it is correct.", text + "I hope it is correct.")
    return normalize(m.group(1).strip()) if m else "[invalidanswer]"


def bbh_answer(text):
    m = re.search(r"the answer is (.*)", text)
    if not m:
        return "[invalidanswer]"
    answer = m.group(1).strip().split("\n")[0].strip()
    return answer[:-1] if answer.endswith(".") else answer


def helmet_norm(s):
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def helmet_parse(output, prefix):
    for pat in (re.compile(f"(?:{prefix})(.*)(?:\n|$)", flags=re.IGNORECASE), re.compile(r"(?:^)(.*)(?:\n|$)")):
        m = pat.search(output)
        if m is not None:
            return re.sub(f"^{re.escape(prefix)}", "", m[1].strip(), flags=re.IGNORECASE).strip()
    return ""


def helmet_rankings(output):
    output = re.sub(r"[\[\]:]", "", output).lower().replace("id", "")
    longest = max((m.group(0) for m in re.finditer(r"(\d+)(?:\s*>\s*(\d+))*", output)), key=len, default="")
    ranks = [n.strip() for n in longest.split(">") if n.strip().isdigit()]
    out = {}
    for i, r in enumerate(ranks):
        out.setdefault(r, len(ranks) - i)
    return out


def helmet_score(row, text):
    """HELMET metrics: RAG/recall substring EM, ICL exact match on the parsed label, re-rank NDCG@10 (trec_eval gains)."""
    if row["metric"] == "subem":
        return float(any(helmet_norm(a) in helmet_norm(text) for a in row["answers"]))
    if row["metric"] == "icl_em":
        pred = helmet_norm(helmet_parse(text, "label:"))
        return float(any(pred == helmet_norm(a) for a in row["answers"]))
    run = sorted(helmet_rankings(text).items(), key=lambda kv: -kv[1])[:10]
    dcg = sum(row["qrel"].get(d, 0) / math.log2(i + 2) for i, (d, _) in enumerate(run))
    ideal = sorted(row["qrel"].values(), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def prepare(args):
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    args.output.mkdir(parents=True, exist_ok=True)
    out = []
    for path in sorted(glob.glob(f"{args.raw}/bbh/*/test-*.parquet")):
        task = Path(path).parent.name
        shots = Path(f"{args.raw}/bbh-cot/{task}.txt").read_text().split("-----\n", 1)[1].strip()
        for i, r in pd.read_parquet(path).iterrows():
            prompt = f"{shots}\n\nQ: {r['input']}\nA: Let's think step by step."
            out.append(dict(id=f"bbh/{task}/{i}", task=task, prompt_ids=enc(prompt), answer=r["target"],
                            max_new_tokens=256, until=["Q:", "\n\n\n"]))
    (args.output / "bbh.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    shots = "".join(f"Problem:\n{p}\n\nSolution:\n{s}\n\n" for p, s in MINERVA_SHOTS)
    out = []
    for line in open(f"{args.raw}/MATH-500/test.jsonl"):
        r = json.loads(line)
        out.append(dict(id=f"math500/{r['unique_id']}", task=r["subject"], level=r["level"],
                        prompt_ids=enc(shots + f"Problem:\n{r['problem']}\n\nSolution:"), answer=normalize(r["answer"]),
                        max_new_tokens=1024, until=["Problem:"]))
    (args.output / "math500.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    dev = pd.read_parquet(f"{args.raw}/mmlu/all/dev-00000-of-00001.parquet")
    test = pd.read_parquet(f"{args.raw}/mmlu/all/test-00000-of-00001.parquet")
    fmt = lambda r: r["question"].strip() + "".join(f"\n{l}. {c}" for l, c in zip("ABCD", r["choices"])) + "\nAnswer:"
    out = []
    for subject, group in test.groupby("subject", sort=True):
        head = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
        head += "".join(fmt(r) + f" {'ABCD'[r['answer']]}\n\n" for _, r in dev[dev.subject == subject].iterrows())
        for i, (_, r) in enumerate(group.iterrows()):
            out.append(dict(id=f"mmlu/{subject}/{i}", task=subject, prompt_ids=enc(head + fmt(r)), answer="ABCD"[r["answer"]]))
    (args.output / "mmlu.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    print(json.dumps({n: sum(1 for _ in open(args.output / f"{n}.jsonl")) for n in ("bbh", "math500", "mmlu")}))


@torch.inference_mode()
def greedy(runner, tok, row):
    logits, cache = runner.prefill(torch.tensor([row["prompt_ids"]], device="cuda"))
    gen, text, reason = [], "", "max_new_tokens"
    for i in range(row["max_new_tokens"]):
        token = logits.argmax(-1).view(1, 1)
        if int(token) == tok.eos_token_id:
            reason = "eos"
            break
        gen.append(int(token))
        text = tok.decode(gen, skip_special_tokens=True).lstrip()  # leading whitespace never stops generation
        hits = [(text.find(s), s) for s in row["until"] if text.find(s) >= 0]
        if hits:
            p, s = min(hits)
            text, reason = text[:p], f"stop:{s}"
            break
        if i + 1 < row["max_new_tokens"]:
            logits, cache = runner.decode(token, cache)
    return text, len(gen), reason


@torch.inference_mode()
def select_rows(cache, keep, n):
    """Keep batch rows `keep` (cuda index tensor) of a prompt-filled cache (native, last-loop or delta)."""
    pick = lambda t: t.index_select(0, keep) if torch.is_tensor(t) and t.dim() == 4 and t.shape[0] == n else t
    for name in ("key_cache", "value_cache", "delta_keys", "delta_values", "shared_key_cache", "shared_value_cache",
                 "local_key_cache", "local_value_cache",
                 "prompt_keys", "prompt_values", "hist_k", "hist_v"):  # arm O LiveCache: prompt bank + shared answer history
        bank = getattr(cache, name, None)
        if isinstance(bank, list):
            setattr(cache, name, [pick(t) for t in bank])
    shallow = getattr(cache, "shallow", None)  # arm M: first-loop prompt bank read by the shallow branch
    if shallow is not None:
        cache.shallow = tuple([pick(t) for t in bank] for bank in shallow)
    for layer in getattr(cache, "layers", None) or []:  # transformers >= 4.56 DynamicCache layers
        for name in ("keys", "values"):
            if hasattr(layer, name):
                setattr(layer, name, pick(getattr(layer, name)))
        for name in ("conv_states", "recurrent_states"):  # linear-attention (GDN) layers: dicts of batch-first states
            states = getattr(layer, name, None)
            if isinstance(states, dict):
                for i, t in states.items():
                    if torch.is_tensor(t) and t.shape[0] == n:
                        states[i] = t.index_select(0, keep)


def greedy_batch(runner, tok, batch):
    """Left-padded batched greedy decoding; each row stops on EOS, its own stop strings or max_new_tokens.

    Finished rows leave the batch (their cache rows are dropped), so a batch never waits on idle rows.
    Stop strings are searched in the decoded tail of the generation (from the first non-whitespace
    token), and the full text is decoded only when a stop may have appeared; leading whitespace never stops.
    """
    n, width = len(batch), max(len(r["prompt_ids"]) for r in batch)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((n, width), pad, dtype=torch.long)
    mask = torch.zeros((n, width), dtype=torch.long)
    for i, r in enumerate(batch):
        ids[i, width - len(r["prompt_ids"]):] = torch.tensor(r["prompt_ids"])
        mask[i, width - len(r["prompt_ids"]):] = 1
    ids, mask = ids.cuda(), mask.cuda()
    single = n == 1  # no padding: skip the explicit mask so SDPA keeps its memory-efficient causal kernel
    logits, cache = runner.prefill(ids, None if single else mask)
    gen = [[] for _ in batch]
    start = [None] * n
    out = [None] * n
    active = list(range(n))  # batch rows still decoding, in cache order
    # NO_EMPTY=1: a row may not stop before it has produced a non-whitespace token (thinking checkpoints on raw prompts
    # otherwise often emit "\n" + EOS); applied identically to every configuration
    no_empty = os.environ.get("NO_EMPTY", "0") == "1"
    stop_ids = [t for t in getattr(tok, "stop_ids", (tok.eos_token_id,)) if t is not None]
    for step in range(max(r["max_new_tokens"] for r in batch)):
        if no_empty:
            rows = [j for j, i in enumerate(active) if start[i] is None]
            if rows:
                logits = logits.clone()
                logits[torch.tensor(rows, device=logits.device)[:, None], torch.tensor(stop_ids, device=logits.device)[None]] = float("-inf")
        tokens = logits.argmax(-1)
        for j, i in enumerate(active):
            r, t = batch[i], int(tokens[j])
            if t in getattr(tok, "stop_ids", (tok.eos_token_id,)):
                out[i] = (tok.decode(gen[i], skip_special_tokens=True), len(gen[i]), "eos")
                continue
            gen[i].append(t)
            if start[i] is None and tok.decode([t], skip_special_tokens=True).strip():
                start[i] = len(gen[i]) - 1
            if start[i] is not None and r["until"]:
                tail = tok.decode(gen[i][max(start[i], len(gen[i]) - 8):], skip_special_tokens=True)
                if any(x in tail for x in r["until"]):
                    full = tok.decode(gen[i], skip_special_tokens=True)
                    lead = len(full) - len(full.lstrip())  # leading whitespace never stops generation
                    q, x = min((full.find(x, lead), x) for x in r["until"] if full.find(x, lead) >= 0)
                    # raw rows (lm-eval) keep leading whitespace: code indentation must survive, as with HF generate
                    out[i] = (full[:q] if r.get("raw") else full[lead:q], len(gen[i]), f"stop:{x}")
                    continue
            if len(gen[i]) >= r["max_new_tokens"]:
                full = tok.decode(gen[i], skip_special_tokens=True)
                out[i] = (full if r.get("raw") else full.lstrip(), len(gen[i]), "max_new_tokens")
        keep = [j for j, i in enumerate(active) if out[i] is None]
        if not keep:
            break
        if len(keep) < len(active):
            index = torch.tensor(keep, device=tokens.device)
            select_rows(cache, index, len(active))
            tokens, mask, active = tokens[index], mask[index], [active[j] for j in keep]
        mask = torch.cat([mask, mask.new_ones((len(active), 1))], dim=1)
        logits, cache = runner.decode(tokens.view(-1, 1), cache, None if single else mask)
    return out


def score_row(bench, row, text, n, reason):
    if bench.startswith(("ruler", "longbench")):
        from ouro.score import score_generation
        return dict(id=row["id"], task=row["task"], text=text, generated_tokens=n, stop_reason=reason,
                    score=score_generation(row["task"], text, row["answers"]))
    if bench == "helmet":
        return dict(id=row["id"], task=row["task"], length=row["length"], text=text, generated_tokens=n,
                    stop_reason=reason, score=helmet_score(row, text))
    pred = bbh_answer(text) if bench == "bbh" else minerva_answer(text)
    return dict(id=row["id"], task=row["task"], text=text, generated_tokens=n, stop_reason=reason,
                pred=pred, correct=pred == row["answer"])


def evaluate(args):
    if os.environ.get("OURO_NO_CUDNN_SDP"):  # cuDNN SDPA fails on some 16K single-row shapes
        torch.backends.cuda.enable_cudnn_sdp(False)
    family = json.loads((Path(args.model_path) / "config.json").read_text())["model_type"]
    reencode = None
    if family == "ouro":
        runner = make_runner(SimpleNamespace(condition=args.condition, checkpoint=args.checkpoint,
                                             model_path=args.model_path, timing=False))
        tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    else:  # other looped families: same benchmark rows (Ouro token ids) decoded to text and re-encoded
        if family == "sensenova_loop":  # hybrid GDN / full attention; needs transformers 5.18 on PYTHONPATH
            from ouro.sensenova_runner import SenseNovaRunner, tokenizer
            runner = SenseNovaRunner(args.model_path, args.condition, args.checkpoint)
        else:
            from ouro.looped_bridge import LoopedRunner, tokenizer
            runner = LoopedRunner(args.model_path, args.condition, args.checkpoint)
        tok = tokenizer(args.model_path)
        source = AutoTokenizer.from_pretrained(args.source_tokenizer, local_files_only=True, trust_remote_code=True)
        reencode = lambda ids: tok.encode(source.decode(ids), add_special_tokens=False)
        if family == "sensenova_loop":  # thinking model: answer directly (Qwen3 non-thinking form), or a short answer
            no_think = tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False)  # budget is spent thinking
            reencode = lambda ids: tok.encode(source.decode(ids), add_special_tokens=False) + no_think
    choice_ids = [tok.encode(f" {c}", add_special_tokens=False)[-1] for c in "ABCD"]
    args.output.mkdir(parents=True, exist_ok=True)
    for bench in args.benches:
        data = rows(args.data / f"{bench}.jsonl")[args.shard_index::args.num_shards]
        if reencode:
            for row in data:
                row["prompt_ids"] = reencode(row["prompt_ids"])
        path = args.output / f"{bench}.jsonl"
        done = done_rows(path)
        path.write_text("".join(json.dumps(r) + "\n" for r in done.values()))
        todo = [row for row in data if row["id"] not in done]
        with path.open("a") as h, torch.inference_mode():
            if args.batch_tokens and bench != "mmlu":
                todo.sort(key=lambda r: len(r["prompt_ids"]) + r["max_new_tokens"])
                i = 0
                while i < len(todo):
                    span = len(todo[i]["prompt_ids"]) + todo[i]["max_new_tokens"]
                    size = max(1, min(args.max_batch, args.batch_tokens // span))
                    batch = todo[i:i + size]
                    for row, (text, n, reason) in zip(batch, greedy_batch(runner, tok, batch)):
                        h.write(json.dumps(score_row(bench, row, text, n, reason), ensure_ascii=False) + "\n")
                    h.flush()
                    i += size
                continue
            for row in todo:
                if bench == "mmlu":
                    logits, _ = runner.prefill(torch.tensor([row["prompt_ids"]], device="cuda"))
                    pred = "ABCD"[int(logits[0, choice_ids].argmax())]
                    r = dict(id=row["id"], task=row["task"], pred=pred, correct=pred == row["answer"])
                else:
                    r = score_row(bench, row, *greedy(runner, tok, row))
                h.write(json.dumps(r, ensure_ascii=False) + "\n")
                h.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "eval"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benches", nargs="+", default=["mmlu", "math500", "bbh"])
    parser.add_argument("--condition")
    parser.add_argument("--checkpoint")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-tokens", type=int, default=0, help="batched greedy for generation benches: rows x (prompt+new) budget")
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--source-tokenizer", default="/dev/shm/big-small-ouro/model",
                        help="tokenizer of the stored prompt_ids (re-encoded for non-Ouro families)")
    args = parser.parse_args()
    prepare(args) if args.mode == "prepare" else evaluate(args)


if __name__ == "__main__":
    main()
