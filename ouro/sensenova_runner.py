"""SenseNova-Loop-9B-Preview (hybrid GDN / full-attention looped model) behind the prefill/decode runner API used by
lmeval_ouro.py and evaluate_bench.py. Needs transformers 5.18 (PYTHONPATH=/dev/shm/big-small-ouro/tf518:...).

The checkpoint runs physical layers [start, end) `count` times; every visit has its own cache slot (KV for full-attention
layers, conv + recurrent state for GDN layers), indexed by execution order. Conditions:
- native<c>: the model with the loop executed c times for prefill and decode.
- splice<p><d>: prefill with p visits, then decode with d visits; decode visit k of the looped block reads the cache slots
  of prefill visit p - d + k + 1 (the last d prefill visits), outside the loop the slots are shared. Untrained.
- C<p>1 (+ checkpoint): the trained connector of ouro/sensenova_bridge.py — prefill p visits (LoRA off), splice, decode
  with 1 visit and LoRA on; the decoder always runs the last prompt token.
- Suffix A or G (splice21A, C21G, ...): where the deep state lives. A: full-attention layers of the looped block read the
  deep prefill visit's KV, Gated-DeltaNet layers start from the shallow visit's (same-depth) conv + recurrent state;
  G: the reverse. Without a suffix both come from the deep visit.
"""

import copy

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from transformers.cache_utils import DynamicCache


def load(path):
    model = AutoModelForImageTextToText.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16,
                                                        device_map="cuda", attn_implementation="sdpa")
    # mHC routers run tens of tiny Sinkhorn steps per layer per token (launch-bound at decode): fuse them
    import sys
    code = sys.modules[type(model).__module__]
    if hasattr(code, "sinkhorn") and not getattr(code.sinkhorn, "_fused", False):
        code.sinkhorn = torch.compile(code.sinkhorn, dynamic=True)
        code.sinkhorn._fused = True
    return model.eval().requires_grad_(False)


def tokenizer(path):
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def text_model(model):
    for m in model.modules():
        if hasattr(m, "_loop_plan"):
            return m
    raise ValueError("no SenseNova loop text model found")


def plan(text, count):
    s = text.config  # physical config
    start, end = text._loop_bounds
    return tuple(range(start)) + tuple(range(start, end)) * count + tuple(range(end, s.num_hidden_layers))


def set_count(text, count):
    """Re-plan the execution for `count` loop visits (weights unchanged)."""
    if not hasattr(text, "_loop_bounds"):
        p = text._loop_plan
        n = text.config.num_hidden_layers
        start = next(i for i, x in enumerate(p) if p.count(x) > 1)
        end = start + sum(1 for x in set(p) if p.count(x) > 1)
        text._loop_bounds = (start, end)
        assert all(p.count(x) == 1 for x in range(n) if not start <= x < end)
    memo = text.__dict__.setdefault("_loop_memo", {})  # decode calls this every step: build each plan once
    if count not in memo:
        cfg = copy.deepcopy(text.config)
        cfg.num_hidden_layers = len(plan(text, count))
        cfg.layer_types = [text.config.layer_types[i] for i in plan(text, count)]
        memo[count] = (plan(text, count), cfg)
    text._loop_plan, text._loop_cache_config = memo[count]


def new_cache(text):
    cache = DynamicCache(config=text._loop_cache_config)
    cache._loop_execution_plan = text._loop_plan
    return cache


def splice(text, cache, p, d, mix=""):
    """Build a d-visit decode cache from a p-visit prefill cache (slot objects shared, not copied). mix "A" / "G": only
    the full-attention ("A") or only the Gated-DeltaNet ("G") layers of the looped block read the deep visit; the other
    kind reads the same-depth visit k."""
    start, end = text._loop_bounds
    width = end - start
    types = text.config.layer_types  # physical layers
    set_count(text, d)
    out = new_cache(text)
    src = []
    for j in range(start):
        src.append(j)
    for k in range(d):
        deep = p - d + k  # 0-based prefill visit read by decode visit k
        for i in range(width):
            attn = types[start + i] == "full_attention"
            visit = k if (mix == "A" and not attn) or (mix == "G" and attn) else deep
            src.append(start + visit * width + i)
    src += [start + p * width + i for i in range(text.config.num_hidden_layers - end)]
    assert len(src) == len(out.layers)
    out.layers = [cache.layers[i] for i in src]
    return out


class SenseNovaRunner:
    def __init__(self, path, condition, checkpoint=None):
        self.model = load(path)
        self.text = text_model(self.model)
        set_count(self.text, 1)
        self.lora = None
        if condition.startswith("C"):
            from ouro.sensenova_bridge import add_lora, load_lora
            self.lora = {"on": False}
            load_lora(add_lora(self.text, self.lora), checkpoint)
            self.p, self.d, self.mix = int(condition[1]), int(condition[2]), condition[3:]
        elif condition.startswith("native"):
            self.p = self.d = int(condition[6:])
        elif condition.startswith("splice"):
            self.p, self.d, self.mix = int(condition[6]), int(condition[7]), condition[8:]
        else:
            raise ValueError(condition)
        self.mix = getattr(self, "mix", "")
        assert self.mix in ("", "A", "G"), condition
        self.loops = self.d

    def _run(self, ids, cache, mask=None):
        out = self.text(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True)
        return self.model.lm_head(out.last_hidden_state[:, -1]).float(), out.past_key_values

    @torch.inference_mode()
    def prefill(self, ids, mask=None):
        set_count(self.text, self.p)
        if self.lora is not None:
            self.lora["on"] = False
        if self.p == self.d and self.lora is None:
            return self._run(ids, new_cache(self.text), mask)
        # read the prompt (all but its last token) with p visits; the decode model then runs the last prompt token,
        # so every output token, including the first, comes from the d-visit decoder
        _, cache = self._run(ids[:, :-1], new_cache(self.text), None if mask is None else mask[:, :-1])
        if self.p != self.d:
            cache = splice(self.text, cache, self.p, self.d, self.mix)
        return self.decode(ids[:, -1:], cache, mask)

    @torch.inference_mode()
    def decode(self, ids, cache, mask=None):
        set_count(self.text, self.d)
        cache._loop_execution_plan = self.text._loop_plan
        if self.lora is not None:
            self.lora["on"] = True
        return self._run(ids, cache, mask)
