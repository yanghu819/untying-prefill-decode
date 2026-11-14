"""Batched serving time of real lm-evaluation-harness requests (continuous batching, one idle GPU): the requests of
time_requests.py (same items, prompts and stop rules) served with up to B in flight. A new request is prefilled alone
(batch 1) as soon as a slot frees, which stalls decoding meanwhile (no chunked prefill); it then decodes in lockstep with
the others, and a finished request leaves the batch at once. Each slot keeps one request's KV for every decode loop in a
fixed region of one flat buffer per (loop, layer); decode attention is one flash-attention varlen call per layer that
reads only each request's own tokens (seqused_k). Decoding is the same for every condition: the n-loop model on
per-loop prompt caches (Original n/n; Distilled, LoRA merged; Ours with the maps folded into the caches, pm).

usage: batch_requests.py MODEL_DIR COND CKPT|- SAMPLES_DIR N B OUT.jsonl
  COND, CKPT, SAMPLES_DIR, N: as time_requests.py (native<d>st | D<d>st | a<m>x<n>pm)
Output: one line per request as time_requests.py (prefill_s: its batch-1 prefill; decode_s: admission to finish minus
prefill_s; total_s: admission to finish) and OUT with .summary.json: wall time of serving all N requests.
"""
import glob
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from transformers import AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ouro.adapter_model import set_steps
from ouro.bench_throughput import make
from ouro.model import NativeCache

model_dir, cond, ckpt, samples_dir, n, B, out = sys.argv[1:8]
n, B, out = int(n), int(B), Path(out)
MAX_LENGTH = 16384   # as the evaluation (lmeval_ouro.py, max_length=16384)
TAIL = 32            # stop strings are searched in the text of the last TAIL tokens (they span far fewer)

newest = {}
for f in glob.glob(os.path.join(samples_dir, "*", "samples_*.jsonl")):
    task, stamp = os.path.basename(f)[:-len(".jsonl")].split("samples_")[1].rsplit("_", 1)
    newest[task] = max(newest.get(task, (stamp, f)), (stamp, f))
requests = {}
for task, (_, f) in newest.items():
    for r in map(json.loads, open(f)):
        requests.setdefault(f"{task}/{r['doc_id']}", r)
key = lambda i: hashlib.md5(i.encode()).hexdigest()
chosen = sorted(requests, key=key)[:n]

tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
gen_cfg = Path(model_dir) / "generation_config.json"
eos = json.load(open(gen_cfg)).get("eos_token_id") if gen_cfg.exists() else None
stop_ids = {e for e in (eos if isinstance(eos, list) else [eos]) + [tok.eos_token_id] if e is not None}
runner = make(cond, model_dir, None if ckpt == "-" else ckpt, merge=ckpt != "-")
model = runner.model
loops = runner.loops if hasattr(runner, "loops") else runner.system.decode_steps
cfg = model.config


def prompt(r):
    context, kw = r["arguments"]["gen_args_0"]["arg_0"], r["arguments"]["gen_args_0"]["arg_1"]
    until = kw.get("until", [])
    until = [until] if isinstance(until, str) else list(until)
    max_new = int(kw.get("max_gen_toks", 256))
    ids, limit = tok.encode(context, add_special_tokens=False), MAX_LENGTH - max_new
    if len(ids) > limit:
        ids = ids[:limit // 2] + ids[len(ids) - (limit - limit // 2):]
    return ids, until, max_new


class SlotCache(NativeCache):
    """KV of up to B requests for every (loop, layer) bank: bank b is a flat [B * S, H, D] buffer in which slot r owns
    rows r*S .. r*S + S - 1 and fill[r] rows are valid. update() writes each live request's current token after its
    valid rows; slot_attention reads the valid rows plus that token."""

    def __init__(self, banks, slots, size):
        super().__init__(None)
        heads, dim = cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
        dim = getattr(cfg, "head_dim", None) or dim
        kw = dict(dtype=model.lm_head.weight.dtype, device="cuda")
        self.k = [torch.empty(slots * size, heads, dim, **kw) for _ in range(banks)]
        self.v = [torch.empty(slots * size, heads, dim, **kw) for _ in range(banks)]
        self.key_cache, self.value_cache = [], []
        self.size, self.fill, self.index = size, [0] * slots, 0

    def load(self, r, c):
        """Copy a batch-1 prefill's per-loop prompt caches and its last prompt token (time_requests' runner) to slot r."""
        base = r * self.size
        for b in range(len(self.k)):
            m = c.prompt_keys[b].shape[2]
            self.k[b][base:base + m].copy_(c.prompt_keys[b][0].transpose(0, 1))
            self.v[b][base:base + m].copy_(c.prompt_values[b][0].transpose(0, 1))
            self.k[b][base + m].copy_(c.key_cache[b][0, :, -1])
            self.v[b][base + m].copy_(c.value_cache[b][0, :, -1])
        self.fill[r] = m + 1

    def plan(self, live):
        idx = torch.tensor(live, device="cuda")
        fill = torch.tensor([self.fill[r] for r in live], device="cuda")
        self.write_at = idx * self.size + fill
        self.cu_q = torch.arange(len(live) + 1, dtype=torch.int32, device="cuda")
        start = idx * self.size
        self.cu_k = torch.cat((start, start[-1:] + self.size)).to(torch.int32)
        self.used = (fill + 1).to(torch.int32)
        self.max_k = max(self.fill[r] for r in live) + 1
        self.positions = fill[:, None]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.index = layer_idx
        self.k[layer_idx].index_copy_(0, self.write_at, key_states[:, :, 0])
        self.v[layer_idx].index_copy_(0, self.write_at, value_states[:, :, 0])
        return key_states, value_states

    def get_seq_length(self, layer_idx=0):
        return 0   # positions are passed explicitly and no mask is built


CACHE = None


def slot_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    c = CACHE
    out = torch.ops.aten._flash_attention_forward(
        query[:, :, 0], c.k[c.index], c.v[c.index], c.cu_q, c.cu_k, 1, c.max_k, 0.0, False, False,
        scale=scaling, seqused_k=c.used)[0]
    return out[:, None], None


def step(tokens, live):
    """One decode step of the live slots: [L] token ids -> [L, vocab] logits."""
    CACHE.plan(live)
    set_steps(model, loops)
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    ALL_ATTENTION_FUNCTIONS["sdpa"] = slot_attention
    try:
        outputs, _, _ = model.model(input_ids=tokens[:, None], position_ids=CACHE.positions, past_key_values=CACHE,
                                    attention_mask={"full_attention": None, "sliding_attention": None}, use_cache=True)
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original
    for r in live:
        CACHE.fill[r] += 1
    return model.lm_head(outputs.last_hidden_state[:, -1])


def serve(order, slots, record, cap=None):
    """Serve the requests `order` with up to `slots` in flight (answers cut at `cap` tokens if given); record(state)
    for each finished one. Returns the wall seconds."""
    global CACHE
    states = [prompt(requests[i]) for i in order]
    size = max(len(ids) + max_new for ids, _, max_new in states) + 1
    CACHE = SlotCache(loops * cfg.num_hidden_layers, slots, size)
    pending = list(zip(order, states))
    active = [None] * slots

    def advance(st, nxt):   # time_requests.answer's stop rules; True when the request is done
        if nxt in stop_ids:
            st["stop"] = "eos"; return True
        st["out"].append(nxt)
        tail = tok.decode(st["out"][-TAIL:], skip_special_tokens=True)
        if any(u in tail for u in st["until"]):
            st["stop"] = "stop"; return True
        return len(st["out"]) == st["max_new"]

    def finish(r):
        st = active[r]
        active[r] = None
        st["done"] = time.perf_counter()
        record(st)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while pending or any(a is not None for a in active):
        for r in range(slots):
            while active[r] is None and pending:
                i, (ids, until, max_new) = pending.pop(0)
                st = dict(id=i, ids=ids, until=until, max_new=min(max_new, cap or max_new), out=[], stop="length",
                          admit=time.perf_counter())
                logits, c = runner.prefill(torch.tensor([ids], device="cuda"))
                CACHE.load(r, c)
                del c
                nxt = int(logits.argmax(-1))
                st["prefill"] = time.perf_counter() - st["admit"]
                active[r] = st
                if advance(st, nxt):
                    finish(r)
        live = [r for r in range(slots) if active[r] is not None]
        if not live:
            continue
        logits = step(torch.tensor([active[r]["out"][-1] for r in live], device="cuda"), live)
        for r, nxt in zip(live, logits.argmax(-1).tolist()):
            if advance(active[r], nxt):
                finish(r)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    del CACHE
    CACHE = None
    torch.cuda.empty_cache()
    return wall


def text_of(st):
    text = tok.decode(st["out"], skip_special_tokens=True)
    if st["stop"] == "stop":
        text = text[:min(text.index(u) for u in st["until"] if u in text)]
    return text


if __name__ == "__main__":
  with torch.inference_mode():
    serve(chosen[:min(B, len(chosen))], B, lambda st: None, cap=16)   # warm-up, not recorded
    rows = []

    def record(st):
        r = requests[st["id"]]
        text = text_of(st)
        rows.append(dict(id=st["id"], cond=cond, batch=B, prompt_tokens=len(st["ids"]), new_tokens=len(st["out"]),
                         stop=st["stop"], prefill_s=round(st["prefill"], 4),
                         decode_s=round(st["done"] - st["admit"] - st["prefill"], 4),
                         total_s=round(st["done"] - st["admit"], 4), match=text.strip() == r["resps"][0][0].strip(),
                         text=text, target=r["target"]))

    torch.cuda.reset_peak_memory_stats()
    wall = serve(chosen, B, record)
    with open(out, "w") as f:
        for row in sorted(rows, key=lambda x: chosen.index(x["id"])):
            f.write(json.dumps(row) + "\n")
    summary = dict(cond=cond, batch=B, n=len(rows), wall_s=round(wall, 3),
                   prefill_s=round(sum(x["prefill_s"] for x in rows), 3),
                   new_tokens=sum(x["new_tokens"] for x in rows),
                   peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1))
    Path(str(out)[:-len(".jsonl")] + ".summary.json").write_text(json.dumps(summary) + "\n")
    print(json.dumps(summary))
