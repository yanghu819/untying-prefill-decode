"""Delta-KV bridge for HF looped LMs whose attention hands RoPE cos/sin to Cache.update
(Nanbeige4.2 `num_loops`, IQuest-Loop `loop_num`); family-agnostic port of the Ouro arms E/F.

One frozen backbone serves as teacher, prefill model and (with LoRA switched on) decode model:
- teacher / prefill model: native `full` loops, LoRA off.
- decode model: native `shallow` loops, rank-16 LoRA on q/k/v/o (on only in decode mode).
- arm F adds the bridge: the prefill model's last-loop prompt K (inverse-RoPE'd to content coordinates)
  and V go through a zero-init per-layer, per-KV-head linear map; the result (K re-rotated) is added to
  the decode model's prompt KV on first fill of every decode loop. Zero init = arm E = native shallow.
- arm E: decode model without the bridge (native shallow loops + LoRA).
- arm X (loop map, the Ouro arm X port): the decode model never re-reads the prompt. Its prompt KV for every
  decode loop k is a zero-init (identity + delta) linear map of the prefill model's last-loop KV of all but the last
  prompt token; it then runs the last prompt token and the continuation itself, so it emits every output token.
  Prefill costs exactly `full` loops (so full < shallow is allowed); the teacher is always the native max-loop model.
- arm O (OverFill-style baseline): the decode model reads the prefill model's last-loop prompt KV directly (replacing
  its own prompt KV, no map), only the decode model is trained.
Training: KL(teacher || student) summed over every target position (the first target, predicted at the
last prompt position, included), normalised by the global batch's KL tokens. Teacher = native full loops.
The decode model's own KV is never fed back to the prefill model.

Data: prepare_mix.py output for the family's tokenizer (pool rows in gsm_train.jsonl, order in samples.npy).
Run: torchrun --nproc-per-node N ouro/looped_bridge.py --model-path ... --data ... --output ... --arm F
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

KNOB = {"nanbeige": "num_loops", "iquestloopcoder": "loop_num"}


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def head_rope(cos, sin):
    return (cos[:, None], sin[:, None]) if cos.dim() == 3 else (cos, sin)


def nan_safe_sdpa():
    """Left-padded batches: a padding query sees no key, SDPA returns NaN there, and the NaN reaches later
    layers through V (0 * NaN). Zero those outputs; real positions are unchanged. (IQuest uses this interface.)"""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if getattr(original, "nan_safe", False):
        return

    def safe(*args, **kwargs):
        out, weights = original(*args, **kwargs)
        return torch.nan_to_num(out, nan=0.0), weights
    safe.nan_safe = True
    ALL_ATTENTION_FUNCTIONS["sdpa"] = safe


def load(path):
    nan_safe_sdpa()
    # cuDNN SDPA backward returns NaN gradients for the mapped prompt KV (same failure as Ouro's OURO_NO_CUDNN_SDP)
    torch.backends.cuda.enable_cudnn_sdp(False)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                 device_map="cuda", attn_implementation="sdpa")
    return model.eval().requires_grad_(False)


def tokenizer(path):
    slow = "nanbeige" in json.loads((Path(path) / "config.json").read_text())["model_type"]
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=not slow)


def set_loops(model, loops):
    knob = KNOB[model.config.model_type]
    setattr(model.config, knob, loops)
    for module in model.modules():
        if hasattr(module, knob) and not callable(getattr(module, knob)):
            setattr(module, knob, loops)


def cache_factory(model):
    """Cache class of the family with two hooks on update(): capture one loop's KV, add a prompt delta."""
    depth = model.config.num_hidden_layers
    if model.config.model_type == "iquestloopcoder":
        base = sys.modules[type(model.model).__module__].IQuestLoopCoderCache
        make = lambda loops: (model.config.loop_window_size, depth, loops)
    else:
        base, make = DynamicCache, lambda loops: ()

    class BridgeCache(base):
        if base is not DynamicCache:
            def get_mask_sizes(self, cache_position, layer_idx=0):
                # IQuest's cache keeps `layers` empty, so HF's default returns the query length only and a
                # left-padded batch loses its padding mask during decode; the mask must span all cached tokens.
                return self.get_seq_length() + cache_position.shape[0], 0

        def update(self, key, value, layer_idx, cache_kwargs=None):
            kw = cache_kwargs or {}
            loop, layer = kw.get("loop_idx", layer_idx // depth), layer_idx % depth
            captured = getattr(self, "captured", None)
            if captured is not None and loop == self.capture_loop:
                cos, sin = head_rope(kw["cos"], kw["sin"])
                captured[layer] = (key * cos - rotate_half(key) * sin, value)  # content (pre-RoPE) key
                self.rope = (kw["cos"], kw["sin"])
            delta = getattr(self, "delta", None)
            filled = self.__dict__.setdefault("filled", set())
            if delta is not None and (loop, layer) not in filled:
                dk, dv = delta[layer]
                n = dk.shape[2]
                if getattr(self, "replace", False):  # arm O: read the prefill model's KV instead of its own
                    key = torch.cat((dk.to(key.dtype), key[:, :, n:]), dim=2)
                    value = torch.cat((dv.to(value.dtype), value[:, :, n:]), dim=2)
                else:
                    key = torch.cat((key[:, :, :n] + dk.to(key.dtype), key[:, :, n:]), dim=2)
                    value = torch.cat((value[:, :, :n] + dv.to(value.dtype), value[:, :, n:]), dim=2)
            filled.add((loop, layer))
            return super().update(key, value, layer_idx, cache_kwargs)

    def new(loops, capture_loop=None, delta=None, replace=False):
        cache = BridgeCache(*make(loops))
        cache.replace = replace
        if capture_loop is not None:
            cache.captured, cache.capture_loop = [None] * depth, capture_loop
        cache.delta = delta
        return cache
    return new


def run(model, ids, cache, loops, mask=None):
    set_loops(model, loops)
    out = model.model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True)
    if isinstance(out, tuple):  # IQuest returns (outputs, per-loop hidden states)
        out = out[0]
    return out.last_hidden_state


class LoRA(nn.Module):
    """Rank-16, alpha-16 residual on a frozen projection; active only while state['on']. With sets > 1 there is one
    LoRA per set and state['set'] picks it (arm R: one LoRA per prefill depth)."""

    def __init__(self, base, state, sets=1):
        super().__init__()
        self.base, self.state, self.sets = base, state, sets
        lead = () if sets == 1 else (sets,)
        self.lora_a = nn.Parameter(torch.empty(*lead, 16, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(*lead, base.out_features, 16, device=base.weight.device))
        for a in (self.lora_a,) if sets == 1 else self.lora_a:
            nn.init.kaiming_uniform_(a, a=5**0.5)

    def forward(self, x):
        out = self.base(x)
        if not self.state["on"]:
            return out
        a, b = (self.lora_a, self.lora_b) if self.sets == 1 else (self.lora_a[self.state["set"]], self.lora_b[self.state["set"]])
        return out + F.linear(F.linear(x.float(), a), b).to(out.dtype)


class Bridge(nn.Module):
    def __init__(self, model, arm, full, shallow, reader=None):
        super().__init__()
        cfg = model.config
        self.__dict__["model"] = model  # not a submodule: trainable() lists only LoRA, delta and a full-FT reader
        self.arm, self.full, self.shallow = arm, full, shallow
        self.teacher_loops = getattr(cfg, KNOB[cfg.model_type])  # the model's native (max) loop count
        self.state = {"on": False}
        self.new_cache = cache_factory(model)
        self.lora = nn.ModuleList()
        self.reader = reader  # full-FT decode model (OverFill-style): separate fp32 copy, trained; no LoRA
        if reader is not None:
            reader.float().train().requires_grad_(True)
        for layer in ([] if reader is not None else model.model.layers):
            attn = layer.self_attn
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                wrapped = LoRA(getattr(attn, name), self.state, self.teacher_loops if arm == "R" else 1)
                setattr(attn, name, wrapped)
                self.lora.append(wrapped)
        if arm == "F":
            shape = (cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, cfg.head_dim)
            self.k_delta = nn.Parameter(torch.zeros(shape, device="cuda"))
            self.v_delta = nn.Parameter(torch.zeros(shape, device="cuda"))
        if arm == "X":
            shape = (shallow, cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, cfg.head_dim)
            self.k_map = nn.Parameter(torch.zeros(shape, device="cuda"))
            self.v_map = nn.Parameter(torch.zeros(shape, device="cuda"))
        if arm == "V":  # universal loop map with a full per-head key map (one prompt KV bank per decode loop)
            shape = (self.teacher_loops, cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, cfg.head_dim)
            self.k_map = nn.Parameter(torch.zeros(shape, device="cuda"))
            self.v_map = nn.Parameter(torch.zeros(shape, device="cuda"))
        if arm in ("P", "R"):  # R = P + one decode LoRA per prefill depth
            # universal single-bank connector with maps indexed by (prefill depth p, decode loop k): one map can no longer
            # be asked to translate both a shallow and a deep prefill loop's KV (fails on Nanbeige, where they differ)
            shape = (self.teacher_loops * self.teacher_loops, cfg.num_hidden_layers, cfg.num_key_value_heads)
            self.k_rot = nn.Parameter(torch.zeros((*shape, 2, cfg.head_dim // 2), device="cuda"))
            self.v_map = nn.Parameter(torch.zeros((*shape, cfg.head_dim, cfg.head_dim), device="cuda"))
        if arm == "S":
            # universal single-bank connector: one set of maps for every (prefill p, decode d) up to the native loop
            # count; key map = one complex scale per RoPE frequency pair (commutes with RoPE, so deployment can fold it
            # into the query and keep a single prompt KV bank), value map = full per-head linear map
            shape = (self.teacher_loops, cfg.num_hidden_layers, cfg.num_key_value_heads)
            self.k_rot = nn.Parameter(torch.zeros((*shape, 2, cfg.head_dim // 2), device="cuda"))
            self.v_map = nn.Parameter(torch.zeros((*shape, cfg.head_dim, cfg.head_dim), device="cuda"))

    def trainable(self):
        return {n: p for n, p in self.named_parameters() if p.requires_grad}

    def deltas(self, cache, n):
        """Per-layer (dk, dv) for the first n positions of the prefill model's captured last-loop KV."""
        cos, sin = head_rope(*cache.rope)
        cos, sin = cos[..., :n, :].float(), sin[..., :n, :].float()
        out = []
        for layer, (raw, value) in enumerate(cache.captured):
            k = torch.einsum("bhti,hoi->bhto", raw[:, :, :n].float(), self.k_delta[layer])
            v = torch.einsum("bhti,hoi->bhto", value[:, :, :n].float(), self.v_delta[layer])
            out.append((k * cos + rotate_half(k) * sin, v))
        return out

    def deep_kv(self, cache, n):
        """Arm O: the prefill model's last-loop prompt KV unchanged (keys back in RoPE coordinates)."""
        cos, sin = head_rope(*cache.rope)
        cos, sin = cos[..., :n, :], sin[..., :n, :]
        return [(raw[:, :, :n] * cos + rotate_half(raw[:, :, :n]) * sin, value[:, :, :n]) for raw, value in cache.captured]

    def loopmap_cache(self, cache, n):
        """Arms X / S: decode-model cache whose prompt KV (first n positions, every decode loop) is the mapped last-loop
        KV of the prefill model; bank index = loop * depth + layer."""
        cos, sin = head_rope(*cache.rope)
        cos, sin = cos[..., :n, :].float(), sin[..., :n, :].float()
        depth = len(cache.captured)
        out = self.new_cache(self.shallow)
        for k in range(self.shallow):
            m = (self.full - 1) * self.teacher_loops + k if self.arm in ("P", "R") else k  # map index
            for layer, (raw, value) in enumerate(cache.captured):
                raw, value = raw[:, :, :n].float(), value[:, :, :n].float()
                if self.arm in ("S", "P", "R"):
                    a, b = self.k_rot[m, layer, :, 0, None], self.k_rot[m, layer, :, 1, None]  # [H, 1, D/2]
                    x1, x2 = raw.chunk(2, dim=-1)
                    key = torch.cat((x1 * (1 + a) - x2 * b, x1 * b + x2 * (1 + a)), dim=-1)
                else:
                    key = raw + torch.einsum("bhti,hoi->bhto", raw, self.k_map[m, layer])
                val = value + torch.einsum("bhti,hoi->bhto", value, self.v_map[m, layer])
                key = key * cos + rotate_half(key) * sin
                DynamicCache.update(out, key.to(torch.bfloat16), val.to(torch.bfloat16), k * depth + layer)
        return out

    def bridge_kv(self, cache, n):
        return self.deltas(cache, n) if self.arm == "F" else self.deep_kv(cache, n) if self.arm == "O" else None

    def prefill_model(self, ids, n, mask=None):
        """Native full-loop pass (LoRA off); returns hidden states and the capturing cache."""
        self.state["on"] = False
        with torch.no_grad():
            cache = self.new_cache(self.full, capture_loop=self.full - 1)
            hidden = run(self.model, ids, cache, self.full, mask)
        return hidden, cache

    def decode_model(self, ids, delta, mask=None):
        self.state["on"] = True
        cache = self.new_cache(self.shallow, delta=delta, replace=self.arm == "O")
        if self.reader is None:
            return run(self.model, ids, cache, self.shallow, mask), cache
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return run(self.reader, ids, cache, self.shallow, mask), cache

    def decode_step(self, ids, cache, mask=None):
        """Continue decoding on an existing decode-model cache."""
        if self.reader is None:
            return run(self.model, ids, cache, self.shallow, mask), cache
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return run(self.reader, ids, cache, self.shallow, mask), cache

    def decode_step_on(self, ids, cache, mask=None):
        self.state["on"] = True
        return self.decode_step(ids, cache, mask)

    def head(self, hidden):
        return (self.reader if self.reader is not None else self.model).lm_head(hidden)

    def loss(self, prompt, target):
        """KL(teacher || student) summed over target positions for one sample."""
        ids = torch.tensor([prompt + target[:-1]], device="cuda")
        n = len(prompt)
        if self.arm in ("X", "S", "V", "P", "R"):
            self.state["on"] = False
            with torch.no_grad():
                hidden = run(self.model, ids, self.new_cache(self.teacher_loops), self.teacher_loops)
                teacher = self.model.lm_head(hidden[:, n - 1:]).float().log_softmax(-1)
                _, cache = self.prefill_model(ids[:, :n - 1], n - 1)
            prompt_kv = self.loopmap_cache(cache, n - 1)
            del cache, hidden
            hidden, _ = self.decode_step_on(ids[:, n - 1:], prompt_kv)
            student = self.head(hidden).float().log_softmax(-1)
            kl = F.kl_div(student, teacher, reduction="sum", log_target=True)
            with torch.no_grad():
                tgt = torch.tensor(target, device="cuda")[None, :, None]
                ce, tce = -student.gather(-1, tgt).sum(), -teacher.gather(-1, tgt).sum()
            return kl, torch.stack((kl.detach(), ce, tce))
        hidden, cache = self.prefill_model(ids, n)
        with torch.no_grad():
            teacher = self.model.lm_head(hidden[:, n - 1:]).float().log_softmax(-1)
        delta = self.bridge_kv(cache, n)
        del cache, hidden
        hidden, _ = self.decode_model(ids, delta)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.reader is not None):
            student = self.head(hidden[:, n - 1:]).float().log_softmax(-1)
        kl = F.kl_div(student, teacher, reduction="sum", log_target=True)
        with torch.no_grad():
            tgt = torch.tensor(target, device="cuda")[None, :, None]
            ce, tce = -student.gather(-1, tgt).sum(), -teacher.gather(-1, tgt).sum()
        return kl, torch.stack((kl.detach(), ce, tce))


class LoopedRunner:
    """prefill/decode API of ouro.model.OuroRunner for evaluation (lm-eval, greedy_batch).

    condition: native{n} | E | F (E/F need a checkpoint from this trainer).
    """

    def __init__(self, path, condition, checkpoint=None, full=2, shallow=1):
        model = load(path)
        if condition == "splice":  # untrained: full-loop prefill, 1-loop decode reading the last loop's prompt KV
            self.model, self.bridge, self.loops, self.splice = model, None, 1, full
            self.new_cache = cache_factory(model)
            return
        self.splice = None
        if condition.startswith("native"):
            self.model, self.bridge, self.loops = model, None, int(condition[6:])
            self.new_cache = cache_factory(model)
            return
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        reader = load(path) if saved.get("full_ft") else None
        self.bridge = Bridge(model, saved["arm"], saved["full"], saved["shallow"], reader)
        state = self.bridge.trainable()
        with torch.no_grad():
            for name, value in saved["trainable_state"].items():
                state[name].copy_(value)
        self.model, self.loops = model, saved["shallow"]
        if saved["arm"] in ("S", "V", "P", "R"):  # universal connector evaluated at cell S<p><d> / V<p><d>
            if not (len(condition) == 3 and condition[0] == saved["arm"]):
                raise ValueError(f"universal connector needs a condition S<p><d>, got {condition}")
            self.bridge.full, self.bridge.shallow = int(condition[1]), int(condition[2])
            self.loops = self.bridge.shallow
            self.bridge.state["set"] = self.bridge.full - 1

    @torch.inference_mode()
    def prefill(self, ids, mask=None):
        if self.splice:
            if self.model.config.model_type != "nanbeige":
                raise NotImplementedError("splice is implemented for per-loop DynamicCache (Nanbeige)")
            cache = self.new_cache(self.splice)
            hidden = run(self.model, ids, cache, self.splice, mask)
            depth = self.model.config.num_hidden_layers
            cache.layers = cache.layers[(self.splice - 1) * depth:self.splice * depth]
            return self.model.lm_head(hidden[:, -1]), cache
        if self.bridge is None:
            cache = self.new_cache(self.loops)
            hidden = run(self.model, ids, cache, self.loops, mask)
            return self.model.lm_head(hidden[:, -1]), cache
        if self.bridge.arm in ("X", "S", "V", "P", "R"):
            n = ids.shape[1] - 1
            _, cache = self.bridge.prefill_model(ids[:, :n], n, None if mask is None else mask[:, :n])
            prompt_kv = self.bridge.loopmap_cache(cache, n)
            del cache
            hidden, cache = self.bridge.decode_step_on(ids[:, n:], prompt_kv, mask)
            return self.bridge.head(hidden[:, -1]).float(), cache
        delta = None
        if self.bridge.arm in ("F", "O") and not os.environ.get("OURO_ZERO_DELTA"):  # zero-delta: no deep KV
            _, cache = self.bridge.prefill_model(ids, ids.shape[1], mask)
            delta = self.bridge.bridge_kv(cache, ids.shape[1])
            keep = os.environ.get("OURO_DELTA_LAYERS")  # analysis: keep the delta only on layers lo-hi (inclusive)
            if keep:
                lo, hi = map(int, keep.split("-"))
                delta = [(k, v) if lo <= i <= hi else (k * 0, v * 0) for i, (k, v) in enumerate(delta)]
            del cache
        hidden, cache = self.bridge.decode_model(ids, delta, mask)
        cache.delta = None  # prompt deltas are applied on first fill only
        return self.bridge.head(hidden[:, -1]).float(), cache

    @torch.inference_mode()
    def decode(self, ids, cache, mask=None):
        if self.bridge is not None:
            self.bridge.state["on"] = True
            hidden, _ = self.bridge.decode_step(ids, cache, mask)
            return self.bridge.head(hidden[:, -1]).float(), cache
        hidden = run(self.model, ids, cache, self.loops, mask)
        return self.model.lm_head(hidden[:, -1]), cache


def learning_rate(tokens, planned):
    warmup = planned * 0.02
    if tokens <= warmup:
        return 1e-4 * tokens / warmup
    progress = min(1.0, (tokens - warmup) / (planned - warmup))
    return 1e-5 + 0.5 * (1e-4 - 1e-5) * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("E", "F", "O", "X", "S", "V", "P", "R"), required=True,
                        help="E: shallow decode + LoRA; F: + linear KV delta bridge; O: OverFill-style, decode reads the prefill model's KV directly; X: loop-map specialist; S: universal single-bank loop map; V: universal loop map, full key map; P: S with maps indexed by (p, k); R: P + one decode LoRA per prefill depth")
    parser.add_argument("--full-loops", type=int, default=2)
    parser.add_argument("--shallow-loops", type=int, default=1)
    parser.add_argument("--stop-tokens", type=int, default=12_000_000)
    parser.add_argument("--global-batch", type=int, default=24)
    parser.add_argument("--max-len", type=int, default=0, help="drop samples longer than this (0 = keep all)")
    parser.add_argument("--save-every", type=int, default=100, help="optimizer steps between checkpoints")
    parser.add_argument("--full-ft", action="store_true", help="train a full decode-model copy (OverFill-style) instead of LoRA")
    parser.add_argument("--reader-lr-scale", type=float, default=0.1)
    parser.add_argument("--s-cells", nargs="+", default=None, help="arms S / V: train only these (p, d) cells, e.g. 21 (default: all)")
    parser.add_argument("--seed", type=int, default=31847, help="init seed; non-default seeds also reshuffle global batches")
    args = parser.parse_args()
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    torch.manual_seed(args.seed)
    model = load(args.model_path)
    bridge = Bridge(model, args.arm, args.full_loops, args.shallow_loops, load(args.model_path) if args.full_ft else None)
    params = bridge.trainable()
    groups = [dict(params=[p for n, p in params.items() if not n.startswith("reader.")], scale=1.0),
              dict(params=[p for n, p in params.items() if n.startswith("reader.")], scale=args.reader_lr_scale)]
    optimizer = torch.optim.AdamW([g for g in groups if g["params"]], lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0)
    pool = [json.loads(l) for l in open(args.data / "gsm_train.jsonl")]
    order = [int(i) for s, i, _ in np.load(args.data / "samples.npy") if s == 1]
    if args.seed != 31847:
        import random
        n = len(order) // args.global_batch * args.global_batch
        blocks = [order[i:i + args.global_batch] for i in range(0, n, args.global_batch)]
        random.Random(args.seed).shuffle(blocks)
        order = [i for b in blocks for i in b]
    if args.max_len:
        order = [i for i in order if len(pool[i]["prompt_ids"]) + len(pool[i]["target_ids"]) <= args.max_len]
    args.output.mkdir(parents=True, exist_ok=True)
    ckpt, step, tokens = args.output / "checkpoint-last.pt", 0, 0
    if ckpt.exists():
        saved = torch.load(ckpt, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for name, value in saved["trainable_state"].items():
                params[name].copy_(value)
        if saved.get("optimizer"):
            optimizer.load_state_dict(saved["optimizer"])
        step, tokens = saved["step"], saved["kl_tokens"]
    log = open(args.output / "train.jsonl", "a") if rank == 0 else None
    t0 = time.time()

    def save(path, with_optimizer=True):
        # full-FT reader weights are stored in bf16; the final checkpoint carries no optimizer state
        if rank == 0:
            state = {n: (p.detach().to(torch.bfloat16) if n.startswith("reader.") else p.detach()).cpu()
                     for n, p in params.items()}
            torch.save(dict(arm=args.arm, full=args.full_loops, shallow=args.shallow_loops, step=step, kl_tokens=tokens,
                            model_path=args.model_path, full_ft=args.full_ft, trainable_state=state,
                            optimizer=optimizer.state_dict() if with_optimizer else None), str(path) + ".tmp")
            os.replace(str(path) + ".tmp", path)

    while tokens < args.stop_tokens and (step + 1) * args.global_batch <= len(order):
        batch = [pool[i] for i in order[step * args.global_batch:(step + 1) * args.global_batch]]
        if args.arm in ("S", "V", "P", "R"):  # one (p, d) cell per global batch, the same on every rank
            import random
            cell = random.Random(args.seed * 1_000_003 + step)
            if args.s_cells:
                bridge.full, bridge.shallow = (int(x) for x in cell.choice(args.s_cells))
            else:
                bridge.full, bridge.shallow = cell.randint(1, bridge.teacher_loops), cell.randint(1, bridge.teacher_loops)
            bridge.state["set"] = bridge.full - 1
        batch_tokens = sum(len(r["target_ids"]) for r in batch)
        stats = torch.zeros(3, device="cuda")
        for r in batch[rank::world]:
            kl, s = bridge.loss(r["prompt_ids"], r["target_ids"])
            (kl / batch_tokens).backward()
            stats += s
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params.values()]
        if distributed:
            flat = torch.cat([g.flatten() for g in grads] + [stats])
            dist.all_reduce(flat)
            offset = 0
            for p, g in zip(params.values(), grads):
                p.grad = flat[offset:offset + g.numel()].view_as(g); offset += g.numel()
            stats = flat[offset:]
        grad_norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(tokens, args.stop_tokens) * group.get("scale", 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        tokens += batch_tokens
        if rank == 0:
            log.write(json.dumps(dict(step=step, kl_tokens=tokens, kl=float(stats[0]) / batch_tokens,
                                      student_ce=float(stats[1]) / batch_tokens, teacher_ce=float(stats[2]) / batch_tokens,
                                      lr=optimizer.param_groups[0]["lr"], grad_norm=float(grad_norm),
                                      elapsed=round(time.time() - t0))) + "\n")
            log.flush()
        if step % args.save_every == 0:
            save(ckpt)
    save(args.output / "checkpoint-final.pt", with_optimizer=False)
    if ckpt.exists():
        ckpt.unlink()  # resume state is only needed while training
    if rank == 0:
        (args.output / "TRAIN_DONE").touch()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
