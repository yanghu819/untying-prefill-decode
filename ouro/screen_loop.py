"""Gate screen for public looped LMs (no training).

For each checkpoint: native full-loop quality (does it reproduce its own claims?), native 1-loop quality
(is there headroom between shallow and deep loops?), and a zero-shot splice (full-loop prefill, 1-loop
decode that reads the last loop's prompt KV) to see whether shallow decoding can use deep KV without a
bridge. Rows come from the fixed Ouro benchmark files (token ids decoded with the Ouro tokenizer and
re-encoded for the model), so subsets are identical across models.

Families and their loop knobs:
  nanbeige  config.num_loops (per-loop KV at cache index layer + loop * L)
  huginn    forward(num_steps=...) on the recurrent core
  iquest    config.loop_num / plt: config.plt_num_loops
Conditions: full, one, splice (splice only where the cache layout is known: nanbeige).
"""

import argparse
import json
from pathlib import Path
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FAMILY_KNOB = {"nanbeige": "num_loops", "iquest": "loop_num", "plt": "plt_num_loops"}


class Looped:
    def __init__(self, path, family, full, shallow=1):
        self.family, self.full, self.shallow = family, full, shallow
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=family != "nanbeige")
        self.model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                          device_map="cuda").eval()
        self.layers = getattr(self.model.config, "num_hidden_layers", None)

    def _set(self, loops):
        if self.family in FAMILY_KNOB:
            # the knob lives on the config and, for some families, is copied onto modules at init
            knob = FAMILY_KNOB[self.family]
            setattr(self.model.config, knob, loops)
            for module in self.model.modules():
                if hasattr(module, knob) and not callable(getattr(module, knob)):
                    setattr(module, knob, loops)

    def step(self, ids, cache, loops):
        self._set(loops)
        kw = dict(num_steps=loops) if self.family == "huginn" else {}
        out = self.model(input_ids=ids, past_key_values=cache, use_cache=True, **kw)
        return out.logits[:, -1].float(), out.past_key_values

    @torch.inference_mode()
    def generate(self, ids, max_new, stops, condition):
        pre = self.full if condition in ("full", "splice") else self.shallow
        dec = self.full if condition == "full" else self.shallow
        logits, cache = self.step(torch.tensor([ids], device="cuda"), None, pre)
        if condition == "splice":
            # read the last loop's prompt KV with a one-loop decoder (layer i <- layer i + (full-1) * L)
            off = (self.full - 1) * self.layers
            if isinstance(cache, tuple):  # legacy ((k, v) per cache index)
                cache = tuple(cache[i + off] for i in range(self.layers))
            else:
                cache.layers = [cache.layers[i + off] for i in range(self.layers)]
        gen, text = [], ""
        for _ in range(max_new):
            t = int(logits.argmax(-1))
            if t in self.eos:
                break
            gen.append(t)
            text = self.tok.decode(gen, skip_special_tokens=True).lstrip()
            hits = [text.find(s) for s in stops if s and text.find(s) >= 0]
            if hits:
                text = text[: min(hits)]
                break
            logits, cache = self.step(torch.tensor([[t]], device="cuda"), cache, dec)
        return text

    @property
    def eos(self):
        e = self.model.generation_config.eos_token_id if self.model.generation_config else self.tok.eos_token_id
        return set(e if isinstance(e, list) else [e])

    @torch.inference_mode()
    def first_token(self, ids, condition):
        logits, _ = self.step(torch.tensor([ids], device="cuda"), None, self.full if condition != "one" else self.shallow)
        return logits[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--family", required=True, choices=["nanbeige", "huginn", "iquest", "plt"])
    parser.add_argument("--full-loops", type=int, required=True)
    parser.add_argument("--shallow-loops", type=int, default=1, help="loops for the 'one' condition (huginn: 4)")
    parser.add_argument("--ouro-tokenizer", default="/dev/shm/big-small-ouro/model")
    parser.add_argument("--bench", type=Path, default=Path("/dev/shm/big-small-ouro/bench"))
    parser.add_argument("--gsm", type=Path, default=Path("/dev/shm/big-small-ouro/gsm8k"))
    parser.add_argument("--conditions", nargs="+", default=["full", "one", "splice"])
    parser.add_argument("--tasks", nargs="+", default=["gsm8k", "ruler4k", "mmlu"])
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from ouro.evaluate_gsm8k import strict_score
    from ouro.score import score_generation
    otok = AutoTokenizer.from_pretrained(args.ouro_tokenizer, local_files_only=True, trust_remote_code=True)
    m = Looped(args.model_path, args.family, args.full_loops, args.shallow_loops)
    enc = lambda ids: m.tok.encode(otok.decode(ids), add_special_tokens=False)
    rng = random.Random(0)
    load = lambda p: [json.loads(l) for l in open(p)]
    data = {}
    if "gsm8k" in args.tasks:
        gen = json.loads((args.gsm / "manifest.json").read_text())
        rows = load(args.gsm / "gsm8k.jsonl"); rng.shuffle(rows); data["gsm8k"] = rows[: args.n]
    if "ruler4k" in args.tasks:
        rows = load(args.bench / "ruler4k.jsonl"); rng.shuffle(rows); data["ruler4k"] = rows[: args.n]
    if "mmlu" in args.tasks:
        rows = load(args.bench / "mmlu.jsonl"); rng.shuffle(rows); data["mmlu"] = rows[: args.n * 2]
    choice = [m.tok.encode(f" {c}", add_special_tokens=False)[-1] for c in "ABCD"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {}
    for cond in args.conditions:
        for task, rows in data.items():
            if task == "mmlu" and cond == "splice":
                continue  # first token comes from the prefill; identical to full
            t0, scores = time.perf_counter(), []
            for r in rows:
                ids = enc(r["prompt_ids"])
                if task == "gsm8k":
                    text = m.generate(ids, gen["generation"]["max_new_tokens"], gen["generation"]["until"], cond)
                    scores.append(float(strict_score(text, r["answer"], gen["strict_match"])["correct"]))
                elif task == "ruler4k":
                    text = m.generate(ids, r["max_new_tokens"], [], cond)
                    scores.append(score_generation(r["task"], text, r["answers"]))
                else:
                    lg = m.first_token(ids, cond)
                    scores.append(float("ABCD"[int(lg[choice].argmax())] == r["answer"]))
            report[f"{task}/{cond}"] = dict(score=round(100 * sum(scores) / len(scores), 1), n=len(scores),
                                            seconds=round(time.perf_counter() - t0))
            print(json.dumps({f"{task}/{cond}": report[f"{task}/{cond}"]}), flush=True)
            args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
