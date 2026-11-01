"""lm-eval-harness model "ouro": official task prompts, few-shot sets, filters and metrics on our runners.

model_args: path=<model dir>,condition=native1|native2|native4|E|F|D|G|J|K,checkpoint=<pt>,
            (non-Ouro looped families via ouro.looped_bridge.LoopedRunner: native{n} | E | F)
            batch_tokens=65536,max_batch=64,max_length=32768
Every condition goes through the same runner API used by our own benchmarks:
- generate_until: batched greedy decoding (evaluate_bench.greedy_batch); the prompt runs through
  runner.prefill (deep / delta path), every generated token after the first through runner.decode.
- loglikelihood: the context runs through runner.prefill, the continuation is teacher-forced
  through runner.decode one token at a time, i.e. scored by the decode model exactly as it would
  be generated. For native per-loop conditions this equals a single full forward.
Tokenization follows lm-eval's HFLM: no special tokens, trailing context whitespace moves to the
continuation, the continuation is the tail of encode(context + continuation).
Run through ouro/run_lmeval.py (which imports this module) with --model ouro.
"""

import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoTokenizer

from ouro.evaluate_adapter import make_runner
from ouro.evaluate_bench import greedy_batch


@register_model("ouro")
class OuroLM(LM):
    def __init__(self, path, condition="native4", checkpoint=None, batch_tokens=65536, max_batch=64,
                 max_length=32768, **kwargs):
        super().__init__()
        family = json.loads((Path(path) / "config.json").read_text())["model_type"]
        if family == "ouro":
            self.tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
            self.runner = make_runner(SimpleNamespace(condition=condition, model_path=path, checkpoint=checkpoint,
                                                      timing=False))
        elif family == "sensenova_loop":  # hybrid GDN / full attention; needs transformers 5.18 on PYTHONPATH
            from ouro.sensenova_runner import SenseNovaRunner, tokenizer
            self.tok = tokenizer(path)
            self.runner = SenseNovaRunner(path, condition, checkpoint)
        else:  # other looped families (Nanbeige, IQuest-Loop): native{n} or a looped_bridge checkpoint
            from ouro.looped_bridge import LoopedRunner, tokenizer
            self.tok = tokenizer(path)
            self.runner = LoopedRunner(path, condition, checkpoint)
        self.batch_tokens, self.max_batch, self.max_length = int(batch_tokens), int(max_batch), int(max_length)
        gen = Path(path) / "generation_config.json"
        eos = json.loads(gen.read_text()).get("eos_token_id") if gen.exists() else None
        eos = eos if isinstance(eos, list) else [eos]
        self.tok.stop_ids = {e for e in eos + [self.tok.eos_token_id] if e is not None}  # read by greedy_batch

    @property
    def tokenizer_name(self):
        return self.tok.name_or_path.replace("/", "__")

    def chat_template(self, chat_template=False):
        return getattr(self.tok, "chat_template", None) or ""

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tok.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=add_generation_prompt)

    def enc(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def batches(self, items, span):
        """Length-sorted batches under the token budget; yields (original indices, items)."""
        order = sorted(range(len(items)), key=lambda i: -span(items[i]))
        i = 0
        while i < len(order):
            n = max(1, min(self.max_batch, self.batch_tokens // max(1, span(items[order[i]]))))
            yield order[i:i + n], [items[j] for j in order[i:i + n]]
            i += n

    def generate_until(self, requests, disable_tqdm=False):
        rows = []
        for req in requests:
            context, kw = req.args
            until = kw.get("until", [])
            until = [until] if isinstance(until, str) else list(until)
            max_new = int(kw.get("max_gen_toks", 256))
            if kw.get("do_sample") and kw.get("temperature", 0) > 0:
                raise ValueError("only greedy decoding is supported")
            ids, limit = self.enc(context), self.max_length - max_new
            if len(ids) > limit:  # LongBench-style middle truncation: keep the instruction head and the question tail
                ids = ids[:limit // 2] + ids[len(ids) - (limit - limit // 2):]
            rows.append(dict(prompt_ids=ids, until=until, max_new_tokens=max_new, raw=True))
        out = [None] * len(rows)
        bar = tqdm(total=len(rows), disable=disable_tqdm, desc="generate_until")
        for idx, batch in self.batches(rows, lambda r: len(r["prompt_ids"]) + r["max_new_tokens"]):
            t0 = time.time()
            print(f"[ouro] batch start rows={len(batch)} max_prompt={max(len(r['prompt_ids']) for r in batch)}",
                  file=sys.stderr, flush=True)
            results = greedy_batch(self.runner, self.tok, batch)
            print(f"[ouro] batch done {time.time() - t0:.1f}s generated={[n for _, n, _ in results]}",
                  file=sys.stderr, flush=True)
            for i, (text, _, _) in zip(idx, results):
                out[i] = text
                self.cache_hook.add_partial("generate_until", requests[i].args, text)
            bar.update(len(batch))
        return out

    def pair(self, context, continuation):
        spaces = len(context) - len(context.rstrip())
        if spaces:
            context, continuation = context[:-spaces], context[-spaces:] + continuation
        if not context:
            return [self.tok.eos_token_id], self.enc(continuation)
        whole, ctx = self.enc(context + continuation), self.enc(context)
        return ctx, whole[len(ctx):]

    @torch.inference_mode()
    def score_batch(self, batch):
        n = len(batch)
        width = max(len(c) for c, _ in batch)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        ids = torch.full((n, width), pad, dtype=torch.long)
        mask = torch.zeros((n, width), dtype=torch.long)
        for i, (c, _) in enumerate(batch):
            ids[i, width - len(c):] = torch.tensor(c)
            mask[i, width - len(c):] = 1
        ids, mask = ids.cuda(), mask.cuda()
        single = n == 1
        logits, cache = self.runner.prefill(ids, None if single else mask)
        total, greedy = [0.0] * n, [True] * n
        for t in range(max(len(x) for _, x in batch)):
            logp = torch.log_softmax(logits.float(), dim=-1)
            best = logp.argmax(-1)
            feed = torch.full((n, 1), pad, dtype=torch.long, device="cuda")
            for i, (_, x) in enumerate(batch):
                if t < len(x):
                    total[i] += float(logp[i, x[t]])
                    greedy[i] &= int(best[i]) == x[t]
                    feed[i, 0] = x[t]
            if t + 1 == max(len(x) for _, x in batch):
                break
            mask = torch.cat([mask, mask.new_ones((n, 1))], dim=1)
            logits, cache = self.runner.decode(feed, cache, None if single else mask)
        return list(zip(total, greedy))

    def loglikelihood(self, requests, disable_tqdm=False):
        items = []
        for req in requests:
            c, x = self.pair(*req.args)
            items.append((c[-(self.max_length - len(x)):], x))
        out = [None] * len(items)
        bar = tqdm(total=len(items), disable=disable_tqdm, desc="loglikelihood")
        for idx, batch in self.batches(items, lambda it: len(it[0]) + len(it[1])):
            for i, res in zip(idx, self.score_batch(batch)):
                out[i] = res
                self.cache_hook.add_partial("loglikelihood", requests[i].args, res)
            bar.update(len(batch))
        return out

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        raise NotImplementedError("perplexity tasks are not used")
