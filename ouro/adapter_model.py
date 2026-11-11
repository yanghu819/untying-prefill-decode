"""Frozen Ouro teacher, prefix adapters, and native two-loop LoRA control."""

import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from torch import nn

from ouro.model import LastLoopCache, NativeCache, Variant, load_model
from third_party.ouro.modeling_ouro import rotate_half


def set_steps(model, steps):
    model.config.total_ut_steps = steps
    model.model.total_ut_steps = steps
    model.early_exit_threshold = 1.0
    model.early_exit_step = None


def backbone(model, ids, cache, steps, attention_mask=None):
    set_steps(model, steps)
    outputs, _, _ = model.model(
        input_ids=ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
    )
    return outputs.last_hidden_state, outputs.past_key_values


class ReaderLoRA(nn.Module):
    """Rank-r, alpha-r (scale 1), zero-dropout residual on a frozen reader projection; r = 16 unless env OURO_LORA_RANK
    is set (rank ablation; set it identically for training and evaluation)."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        r = int(os.environ.get("OURO_LORA_RANK", "16"))
        self.lora_a = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

    def forward(self, x):
        native = self.base(x)
        residual = F.linear(F.linear(x.float(), self.lora_a), self.lora_b)
        return native + residual.to(native.dtype)


class PrefillLoRA(ReaderLoRA):
    """ReaderLoRA on a writer (prefill) projection that is active only while state["on"] (the student's prefill);
    the teacher pass (native 4/4) runs the same writer with it off."""

    def __init__(self, base, state):
        super().__init__(base)
        self.state = state

    def forward(self, x):
        return super().forward(x) if self.state["on"] else self.base(x)


@torch.no_grad()
def merge_reader_lora(model):
    """Fold every ReaderLoRA into its base projection (W <- W + B A) for deployment: decode then runs the plain
    d-loop model, so its cost equals native d loops; the KV delta only acts once, at prefill."""
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, ReaderLoRA):
                child.base.weight += (child.lora_b.float() @ child.lora_a.float()).to(child.base.weight.dtype)
                setattr(parent, name, child.base)
    return model


@dataclass
class PrefixState:
    raw_keys: list
    values: list
    cos: torch.Tensor
    sin: torch.Tensor
    first_keys: list = None    # loop-1 KV of the prefill (keys with RoPE applied), used by the two-source map (arm Q)
    first_values: list = None
    all_keys: list = None      # every prefill loop's KV (keys with RoPE applied; bank = loop * L + layer), arm a
    all_values: list = None


class CaptureCache(NativeCache):
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.cos, self.sin = cache_kwargs["cos"], cache_kwargs["sin"]
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


class DeltaCache(NativeCache):
    """Reader prompt cache: the reader's own KV plus a deep-KV delta on first fill."""

    def __init__(self, num_layers, delta_keys, delta_values):
        super().__init__(num_layers)
        self.delta_keys, self.delta_values = delta_keys, delta_values

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx >= len(self.key_cache) or self.key_cache[layer_idx] is None:
            layer = layer_idx % len(self.delta_keys)  # every reader loop gets the same delta
            key_states = key_states + self.delta_keys[layer].to(key_states.dtype)
            value_states = value_states + self.delta_values[layer].to(value_states.dtype)
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


class LoopMapAdapter(nn.Module):
    """Arm X<p><d>: the decode model never re-reads the prompt. Its prompt KV for every decode loop k and layer l is
    built from the prefill model's last-loop KV by a per-(k, l, kv-head) linear map, identity + zero-init delta
    (pre-RoPE keys, re-rotated). Prefill therefore costs exactly p loops, for any p (also p < d)."""

    def __init__(self, config, device, loops, shared_bank=False, two_source=False, shallow=False, aligned=False):
        super().__init__()
        # aligned (arm a): decode loop k of n reads prefill loop clip(m - n + k, 0, m - 1) instead of the last one, so
        # at m = n every decode loop reads its own loop's KV, as the native model; needs the shared-bank key map
        self.aligned = aligned
        shape = (loops, config.num_hidden_layers, config.num_key_value_heads, config.head_dim, config.head_dim)
        self.loops, self.shared_bank = loops, shared_bank
        if shared_bank:
            # key map restricted to one complex scale per RoPE frequency pair: it commutes with RoPE, so it can be folded
            # into the decode query and the prompt keeps a single KV bank for every decode loop
            self.k_rot = nn.Parameter(torch.zeros((*shape[:3], 2, config.head_dim // 2), device=device, dtype=torch.float32))
        else:
            self.k_delta = nn.Parameter(torch.zeros(shape, device=device, dtype=torch.float32))
        self.v_delta = nn.Parameter(torch.zeros(shape, device=device, dtype=torch.float32))
        self.two_source = two_source
        if two_source:
            # second source: the prefill's FIRST loop KV (shallow, closer to the literal tokens), added through a
            # RoPE-commuting key scale and a value map, both zero-initialised (starts identical to the one-source map)
            self.g_rot = nn.Parameter(torch.zeros((*shape[:3], 2, config.head_dim // 2), device=device, dtype=torch.float32))
            self.g_v = nn.Parameter(torch.zeros(shape, device=device, dtype=torch.float32))
        self.shallow = shallow
        if shallow:
            # arm M: a separate, gated attention branch over the prefill's FIRST-loop KV (the bank PLT / shared-memory
            # looped transformers share), next to the deep last-loop bank: query scale + value map per (decode loop,
            # layer, head), gate tanh(s_gate) initialised at 0 so the model starts exactly as arm S
            self.s_rot = nn.Parameter(torch.zeros((*shape[:3], 2, config.head_dim // 2), device=device, dtype=torch.float32))
            self.s_v = nn.Parameter(torch.zeros(shape, device=device, dtype=torch.float32))
            self.s_gate = nn.Parameter(torch.zeros(shape[:3], device=device, dtype=torch.float32))

    def map_key(self, raw_key, k, layer):
        if not self.shared_bank:
            return raw_key + torch.einsum("bhti,hoi->bhto", raw_key.float(), self.k_delta[k, layer]).to(raw_key.dtype)
        a, b = self.k_rot[k, layer, :, 0, None], self.k_rot[k, layer, :, 1, None]  # [H, 1, half]
        x1, x2 = raw_key.float().chunk(2, dim=-1)
        return torch.cat((x1 * (1 + a) - x2 * b, x1 * b + x2 * (1 + a)), dim=-1).to(raw_key.dtype)

    def build(self, prefix, n, loops=None):
        """NativeCache with loops * L banks for the first n prompt positions (bank index = loop * L + layer)."""
        loops = loops or self.loops
        depth = len(prefix.raw_keys)
        cos, sin = prefix.cos[:, :n].unsqueeze(1), prefix.sin[:, :n].unsqueeze(1)
        cache = IndexedCache(loops * depth)
        for k in range(loops):
            for layer, (raw_key, value) in enumerate(zip(prefix.raw_keys, prefix.values)):
                if self.aligned:   # the scale commutes with RoPE: map the stored (rotated) key of the aligned loop
                    m = len(prefix.all_keys) // depth
                    src = min(max(m - loops + k, 0), m - 1) * depth + layer
                    key = self.map_key(prefix.all_keys[src][:, :, :n], k, layer)
                    value = prefix.all_values[src][:, :, :n]
                    value = value + torch.einsum("bhti,hoi->bhto", value.float(), self.v_delta[k, layer]).to(value.dtype)
                    cache.key_cache.append(key)
                    cache.value_cache.append(value)
                    continue
                raw_key, value = raw_key[:, :, :n], value[:, :, :n]
                key = self.map_key(raw_key, k, layer)
                key = key * cos + rotate_half(key) * sin
                value = value + torch.einsum("bhti,hoi->bhto", value.float(), self.v_delta[k, layer]).to(value.dtype)
                if self.two_source:
                    fk, fv = prefix.first_keys[layer][:, :, :n].float(), prefix.first_values[layer][:, :, :n]
                    a, b = self.g_rot[k, layer, :, 0, None], self.g_rot[k, layer, :, 1, None]
                    x1, x2 = fk.chunk(2, dim=-1)
                    key = key + torch.cat((x1 * a - x2 * b, x1 * b + x2 * a), dim=-1).to(key.dtype)
                    value = value + torch.einsum("bhti,hoi->bhto", fv.float(), self.g_v[k, layer]).to(value.dtype)
                cache.key_cache.append(key)
                cache.value_cache.append(value)
        cache._seen_tokens = n
        return cache


class IndexedCache(NativeCache):
    """NativeCache that remembers the bank index of the last update (read by the shallow-branch attention)."""

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.index = layer_idx
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


class LiveCache(NativeCache):
    """Arm O (prompt KV through maps + self-live shared answer history). Decode loop k, layer l reads
    [mapped prompt bank (k, l) | shared answer history of layer l | the current token's live K/V]. The answer history is
    one bank per layer, written only by the final decode loop (the self-live view: earlier loops see the final loop's
    history of earlier tokens, and only their own live K/V for the current token). training=True: the answer history is
    given (from a no-grad pass) and nothing is appended; record_loop captures that loop's K/V for the next pass."""

    def __init__(self, prompt, loops, history=None, training=False):
        super().__init__(None)
        self.prompt_keys, self.prompt_values = prompt.key_cache, prompt.value_cache
        self.loops, self.depth = loops, len(prompt.key_cache) // loops
        self.n = prompt.key_cache[0].shape[2]
        self.hist_k = [h[0] for h in history] if history else [None] * self.depth
        self.hist_v = [h[1] for h in history] if history else [None] * self.depth
        self.training, self.record_loop, self.recorded = training, None, {}
        self.key_cache = self.value_cache = []  # cache_bytes() reads these; the banks live above
        self._seen_tokens = self.n

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        loop, layer = divmod(layer_idx, self.depth)
        if loop == self.record_loop:
            self.recorded[layer] = (key_states.detach(), value_states.detach())
        ks, vs = [self.prompt_keys[layer_idx]], [self.prompt_values[layer_idx]]
        if self.hist_k[layer] is not None:
            ks.append(self.hist_k[layer]); vs.append(self.hist_v[layer])
        ks.append(key_states); vs.append(value_states)
        if not self.training and loop == self.loops - 1:  # inference: the final loop commits this token to the shared bank
            self.hist_k[layer] = key_states if self.hist_k[layer] is None else torch.cat((self.hist_k[layer], key_states), dim=2)
            self.hist_v[layer] = value_states if self.hist_v[layer] is None else torch.cat((self.hist_v[layer], value_states), dim=2)
            self._seen_tokens = self.n + self.hist_k[layer].shape[2]
        return torch.cat(ks, dim=2), torch.cat(vs, dim=2)

    def get_seq_length(self, layer_idx=0):
        h = self.hist_k[0]
        return self.n + (0 if h is None else h.shape[2])


_SHALLOW = None


def _shallow_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    out, weights = _shallow_attention.original(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
    cache, keys, values, maps = _SHALLOW
    mode = os.environ.get("SHALLOW_DIAG", "")  # diagnostics: "off" = gate off; "deep" = feed the deep bank to the branch
    if mode == "off":
        return out, weights
    loop, layer = divmod(cache.index, len(keys))
    k_s, v_s = keys[layer], values[layer]
    if mode == "deep":  # same branch, but the source is the decode loop's own (mapped) deep prompt bank
        k_s, v_s = cache.key_cache[cache.index][:, :, :k_s.shape[2]], cache.value_cache[cache.index][:, :, :k_s.shape[2]]
    n = k_s.shape[2]
    a, b = maps.s_rot[loop, layer, :, 0, None], maps.s_rot[loop, layer, :, 1, None]  # [H, 1, D/2]
    q1, q2 = query.float().chunk(2, dim=-1)
    q = torch.cat((q1 * (1 + a) + q2 * b, q2 * (1 + a) - q1 * b), dim=-1).to(query.dtype)
    mask = None
    if attention_mask is not None:  # prompt columns come first in the cache: reuse their padding mask
        mask = attention_mask[..., :n]
    o = F.scaled_dot_product_attention(q, k_s, v_s, attn_mask=mask, scale=scaling).float()  # [B, H, q, D]
    o = o + torch.einsum("bhqi,hoi->bhqo", o, maps.s_v[loop, layer])
    o = torch.tanh(maps.s_gate[loop, layer])[None, :, None, None] * o
    return out + o.transpose(1, 2).to(out.dtype), weights


class shallow_branch:
    """Context: route the decode model's attention through _shallow_attention with the given first-loop prompt bank."""

    def __init__(self, cache, keys, values, maps):
        self.state = (cache, keys, values, maps)

    def __enter__(self):
        global _SHALLOW
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        self.table = ALL_ATTENTION_FUNCTIONS
        _shallow_attention.original = ALL_ATTENTION_FUNCTIONS["sdpa"]
        ALL_ATTENTION_FUNCTIONS["sdpa"] = _shallow_attention
        _SHALLOW = self.state

    def __exit__(self, *exc):
        global _SHALLOW
        self.table["sdpa"] = _shallow_attention.original
        _SHALLOW = None


def first_loop_bank(prefix, n):
    """First prefill loop's prompt KV (keys already RoPE'd) for the first n positions."""
    return [k[:, :, :n] for k in prefix.first_keys], [v[:, :, :n] for v in prefix.first_values]


class PrefixAdapter(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        shape = (
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            config.head_dim,
        )
        self.k_delta = nn.Parameter(torch.zeros(shape, device=device, dtype=torch.float32))
        self.v_delta = nn.Parameter(torch.zeros(shape, device=device, dtype=torch.float32))

    def forward(self, prefix):
        cache = NativeCache(len(prefix.raw_keys))
        cos, sin = prefix.cos.unsqueeze(1), prefix.sin.unsqueeze(1)
        for layer, (raw_key, value) in enumerate(zip(prefix.raw_keys, prefix.values)):
            # The learned map is I + delta in pre-RoPE content coordinates.
            key_residual = torch.einsum("bhti,hoi->bhto", raw_key.float(), self.k_delta[layer])
            mapped_key = raw_key + key_residual.to(raw_key.dtype)
            mapped_key = mapped_key * cos + rotate_half(mapped_key) * sin
            value_residual = torch.einsum("bhti,hoi->bhto", value.float(), self.v_delta[layer])
            cache.key_cache.append(mapped_key)
            cache.value_cache.append(value + value_residual.to(value.dtype))
        cache._seen_tokens = cache.get_seq_length()
        return cache

    def delta(self, prefix):
        """Arm F: zero-init additive KV correction read from writer last-loop KV."""
        cos, sin = prefix.cos.unsqueeze(1), prefix.sin.unsqueeze(1)
        keys, values = [], []
        for layer, (raw_key, value) in enumerate(zip(prefix.raw_keys, prefix.values)):
            k = torch.einsum("bhti,hoi->bhto", raw_key.float(), self.k_delta[layer])
            keys.append(k * cos + rotate_half(k) * sin)
            values.append(torch.einsum("bhti,hoi->bhto", value.float(), self.v_delta[layer]))
        return keys, values


class AdapterSystem(nn.Module):
    """A=p4/1 bridge; B=p4/1 bridge+LoRA; C=p1/1 bridge+LoRA; D=native2/2 LoRA;
    E=native1/1 LoRA; H=native3/3 LoRA; G/I=F with a 2/3-loop reader; F=native1/1 LoRA reader whose prompt KV gets an additive
    delta from the 4-loop writer's last-loop KV (zero init = native p1/1).

    The writer is also the independent native p4/4 teacher. It never receives
    adapter outputs or reader LoRA. Teacher answer-history KV stays private.
    """

    def __init__(self, writer, reader, arm, full_ft=False, prefill_lora=False):
        super().__init__()
        self.full_ft = full_ft
        self.prefill_lora = prefill_lora
        self.writer = writer.eval().requires_grad_(False)
        self.reader = reader.eval().requires_grad_(False)
        self.arm = arm
        if arm[0] in "XNQ":  # Q<p><d>: X plus a second source (prefill loop-1 KV); N<p><d>: same map, decode model without LoRA (map-only ablation); X<p><d>: any prefill / decode depth, decode model reads only mapped prefill KV
            self.prefill_steps, self.decode_steps = int(arm[1]), int(arm[2])
            self.adapter = None
            self.loop_adapter = LoopMapAdapter(writer.config, next(writer.parameters()).device, self.decode_steps, two_source=arm[0] == "Q")
        elif arm in ("V", "S", "W", "T", "M", "O", "P", "a"):  # O = S + self-live shared answer history; M = S + gated shallow (first-loop) attention branch; T = S + second source (prefill loop-1 KV)  # W = S without decoder LoRA (map only); universal loop-map connector: one map for every (p, d) in 1..4, set per batch / at eval
            # S: single-bank variant (RoPE-commuting key map), the prompt keeps one KV bank for every decode loop
            self.prefill_steps, self.decode_steps = 4, 4
            self.adapter = None
            self.loop_adapter = LoopMapAdapter(writer.config, next(writer.parameters()).device, 4, shared_bank=arm in ("S", "W", "T", "M", "O", "P", "a"),
                                               two_source=arm == "T", shallow=arm == "M", aligned=arm == "a")
        elif arm[0] in "ZY":  # trained KV-sharing baseline (Y: decode model also emits the first token) Z<d>: native p4 prefill, d-loop decode reading last-loop KV only
            self.prefill_steps, self.decode_steps = 4, int(arm[1])
            self.adapter = None
        else:
            self.prefill_steps = {"A": 4, "B": 4, "C": 1, "D": 2, "E": 1, "F": 4, "G": 4, "H": 3, "I": 4, "J": 2, "K": 3, "L": 3, "U": 4}[arm]
            self.decode_steps = {"D": 2, "G": 2, "H": 3, "I": 3, "L": 2}.get(arm, 1)
            self.adapter = None if arm in ("D", "E", "H") else PrefixAdapter(writer.config, next(writer.parameters()).device)
        if full_ft:
            # OverFill-style: the whole decode-side model is trained (no LoRA).
            self.reader.requires_grad_(True)
            self.reader.model.early_exit_gate.requires_grad_(False)  # unused at fixed loop count
        elif arm in ("B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "U", "V", "S", "T", "M", "O", "P", "a") or arm[0] in "XZYQ":
            for layer in self.reader.model.layers:
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    setattr(layer.self_attn, name, ReaderLoRA(getattr(layer.self_attn, name)))
        self.prefill_state = {"on": False}
        if prefill_lora:  # trainable prefill: LoRA on the writer's attention, on for the student's prefill only
            for layer in self.writer.model.layers:
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    setattr(layer.self_attn, name, PrefillLoRA(getattr(layer.self_attn, name), self.prefill_state))
        self.capture_enabled = False
        self.raw_keys = [None] * writer.config.num_hidden_layers
        for index, layer in enumerate(writer.model.layers):  # after any wrapping, so the captured keys include the LoRA
            layer.self_attn.k_proj.register_forward_hook(self._capture_key(index))

    def _capture_key(self, index):
        def capture(module, inputs, output):
            if self.capture_enabled:
                head_dim = self.writer.config.head_dim
                self.raw_keys[index] = output.view(*output.shape[:-1], -1, head_dim).transpose(1, 2)
        return capture

    def writer_prefix(self, ids, steps, attention_mask=None, student=True):
        """Prefill with `steps` loops. student=False is the native teacher (prefill LoRA off, never any gradient); a
        student prefill uses the prefill LoRA when the system has one and then carries gradient if grad is enabled."""
        depth = self.writer.config.num_hidden_layers
        on = student and self.prefill_lora
        with torch.set_grad_enabled(on and torch.is_grad_enabled()):
            self.prefill_state["on"] = on
            self.capture_enabled = True
            try:
                hidden, cache = backbone(self.writer, ids, CaptureCache(steps * depth), steps, attention_mask)
            finally:
                self.capture_enabled = False
                self.prefill_state["on"] = False
        return self._prefix_out(hidden, cache, steps, depth)

    def _prefix_out(self, hidden, cache, steps, depth):
        offset = (steps - 1) * depth
        prefix = PrefixState(
            list(self.raw_keys),
            list(cache.value_cache[offset : offset + depth]),
            cache.cos,
            cache.sin,
            list(cache.key_cache[:depth]),
            list(cache.value_cache[:depth]),
            list(cache.key_cache),
            list(cache.value_cache),
        )
        self.raw_keys = [None] * depth
        first_logits = self.writer.lm_head(hidden[:, -1:])[:, 0]
        return first_logits, prefix, cache

    def delta_prefix(self, ids, prefix, attention_mask=None):
        depth = self.reader.config.num_hidden_layers
        keys, values = self.adapter.delta(prefix)
        steps = self.decode_steps
        return backbone(self.reader, ids, DeltaCache(steps * depth, keys, values), steps, attention_mask)

    def reader_suffix(self, ids, prefix_cache, attention_mask=None):
        hidden, cache = backbone(self.reader, ids, prefix_cache, self.decode_steps, attention_mask)
        return self.reader.lm_head(hidden), cache

    def forward(self, prefix_ids, suffix_ids):
        if self.arm in ("D", "E", "H"):
            return self.native_lora_forward(prefix_ids, suffix_ids)
        teacher_first, teacher_prefix, teacher_cache = self.writer_prefix(prefix_ids, 4, student=False)
        if self.arm == "C":
            student_first, student_prefix, _ = self.writer_prefix(prefix_ids, 1)
        else:
            student_first, student_prefix = teacher_first, teacher_prefix
        student_cache = self.adapter(student_prefix)
        # Input suffix[0:-1] predicts suffix[1:]; the untrainable first target
        # is excluded equally from A, B, and C and never enters this KL sum.
        suffix_input = suffix_ids[:, :-1]
        with torch.no_grad():
            teacher_hidden, _ = backbone(self.writer, suffix_input, teacher_cache, 4)
            teacher_logits = self.writer.lm_head(teacher_hidden)
            teacher_logp = teacher_logits.float().log_softmax(-1)
            first_kl = F.kl_div(
                student_first.float().log_softmax(-1),
                teacher_first.float().log_softmax(-1),
                reduction="sum",
                log_target=True,
            )
        del teacher_cache, teacher_prefix, student_prefix, teacher_hidden, teacher_logits
        student_logits, _ = self.reader_suffix(suffix_input, student_cache)
        student_logp = student_logits.float().log_softmax(-1)
        kl_sum = F.kl_div(student_logp, teacher_logp, reduction="sum", log_target=True)
        with torch.no_grad():
            target = suffix_ids[:, 1:, None]
            ce_sum = -student_logp.gather(-1, target).sum()
            teacher_ce_sum = -teacher_logp.gather(-1, target).sum()
        return kl_sum, torch.stack((kl_sum.detach(), ce_sum, teacher_ce_sum, first_kl))

    def native_lora_forward(self, prefix_ids, suffix_ids):
        # Native two-loop student: both prompt and continuation use the same
        # LoRA, with distinct cache banks for each loop. Teacher stays private.
        depth = self.writer.config.num_hidden_layers
        with torch.no_grad():
            teacher_hidden, teacher_cache = backbone(self.writer, prefix_ids, NativeCache(4 * depth), 4)
            teacher_first = self.writer.lm_head(teacher_hidden[:, -1:])
            teacher_hidden, _ = backbone(self.writer, suffix_ids[:, :-1], teacher_cache, 4)
            teacher_logp = self.writer.lm_head(teacher_hidden).float().log_softmax(-1)
        del teacher_cache, teacher_hidden
        steps = self.prefill_steps
        hidden, cache = backbone(self.reader, prefix_ids, NativeCache(steps * depth), steps)
        with torch.no_grad():
            first_kl = F.kl_div(self.reader.lm_head(hidden[:, -1:]).float().log_softmax(-1),
                               teacher_first.float().log_softmax(-1), reduction="sum", log_target=True)
        logits, _ = self.reader_suffix(suffix_ids[:, :-1], cache)
        logp = logits.float().log_softmax(-1)
        kl = F.kl_div(logp, teacher_logp, reduction="sum", log_target=True)
        with torch.no_grad():
            target = suffix_ids[:, 1:, None]
            ce = -logp.gather(-1, target).sum()
            teacher_ce = -teacher_logp.gather(-1, target).sum()
        return kl, torch.stack((kl.detach(), ce, teacher_ce, first_kl))

    def trainable_state(self):
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def load_trainable_state(self, state):
        parameters = dict(self.named_parameters())
        expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
        if set(state) != expected:
            raise ValueError("Checkpoint trainable parameter names do not match its declared arm.")
        with torch.no_grad():
            for name, value in state.items():
                parameters[name].copy_(value)


class AdapterRunner:
    def __init__(self, system):
        self.system = system.eval()
        self.model = system.reader
        self.writer = system.writer
        self.arm = system.arm
        self.variant_name = f"adapter-{system.arm}"
        self.variant = Variant(system.prefill_steps, system.decode_steps, "per_loop" if system.arm in ("D", "E", "H") else "adapter")
        self.time_adapter = False
        self.last_adapter_seconds = 0.0

    @torch.inference_mode()
    def prefill(self, input_ids, attention_mask=None):
        if self.arm in ("D", "E", "H"):
            steps = self.system.prefill_steps
            hidden, cache = backbone(self.model, input_ids, NativeCache(steps * self.model.config.num_hidden_layers), steps, attention_mask)
            return self.model.lm_head(hidden[:, -1:])[:, 0], cache
        if self.arm[0] == "Z":  # KV sharing: native 4-loop prefill (first token too), decode keeps last-loop KV only
            depth = self.writer.config.num_hidden_layers
            set_steps(self.writer, 4)
            hidden, cache = backbone(self.writer, input_ids, NativeCache(4 * depth), 4, attention_mask)
            return self.writer.lm_head(hidden[:, -1:])[:, 0], LastLoopCache(cache, depth, 4, self.system.decode_steps)
        if self.arm[0] == "Y":  # KV sharing; the decode model runs the last prompt token, so it emits every output token
            depth = self.writer.config.num_hidden_layers
            _, cache = backbone(self.writer, input_ids[:, :-1], NativeCache(4 * depth),
                                4, None if attention_mask is None else attention_mask[:, :-1])
            hidden, cache = backbone(self.model, input_ids[:, -1:], LastLoopCache(cache, depth, 4, self.system.decode_steps),
                                     self.system.decode_steps, attention_mask)
            return self.model.lm_head(hidden[:, -1:])[:, 0], cache
        if self.arm[0] in "XNQ" or self.arm in ("V", "S", "W", "T", "M", "O", "P", "a"):
            # prefill model at p loops; decode model reads mapped KV of all but the last prompt token, then runs the
            # last prompt token itself (d loops) so the first output token also comes from the decode model
            _, prefix, _ = self.system.writer_prefix(input_ids, self.system.prefill_steps, attention_mask)
            n = input_ids.shape[1] - 1
            cache = self.system.loop_adapter.build(prefix, n, self.system.decode_steps)
            if self.system.loop_adapter.shallow:
                cache.shallow = first_loop_bank(prefix, n)
            if self.arm in ("O", "P") and self.system.decode_steps > 1:
                cache = LiveCache(cache, self.system.decode_steps)
            return self.decode(input_ids[:, -1:], cache, attention_mask)
        if self.arm in ("F", "G", "I", "J", "K", "L", "U"):
            if os.environ.get("OURO_ZERO_DELTA"):  # ablation: trained decode model, no prefill-model KV
                steps = self.system.decode_steps
                hidden, cache = backbone(self.model, input_ids, NativeCache(steps * self.model.config.num_hidden_layers), steps, attention_mask)
                return self.model.lm_head(hidden[:, -1:])[:, 0], cache
            # Every output token, including the first, comes from the decode model;
            # the prefill model only contributes KV.
            _, prefix, _ = self.system.writer_prefix(input_ids, self.system.prefill_steps, attention_mask)
            hidden, cache = self.system.delta_prefix(input_ids, prefix, attention_mask)
            return self.model.lm_head(hidden[:, -1:])[:, 0], cache
        first_logits, prefix, _ = self.system.writer_prefix(
            input_ids, self.system.prefill_steps, attention_mask
        )
        if self.time_adapter:
            torch.cuda.synchronize(input_ids.device)
            started = perf_counter()
        cache = self.system.adapter(prefix)
        if self.time_adapter:
            torch.cuda.synchronize(input_ids.device)
            self.last_adapter_seconds = perf_counter() - started
        return first_logits, cache

    @torch.inference_mode()
    def decode(self, input_ids, cache, attention_mask=None):
        if getattr(cache, "shallow", None) is not None:
            with shallow_branch(cache, *cache.shallow, self.system.loop_adapter):
                logits, cache = self.system.reader_suffix(input_ids, cache, attention_mask)
            return logits[:, -1], cache
        logits, cache = self.system.reader_suffix(input_ids, cache, attention_mask)
        return logits[:, -1], cache


def load_adapter_runner(model_path, checkpoint, device="cuda", dtype=torch.bfloat16, attention="sdpa"):
    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    writer = load_model(model_path, device=device, dtype=dtype, attention=attention)
    reader = load_model(model_path, device=device, dtype=dtype, attention=attention)
    system = AdapterSystem(writer, reader, saved["arm"], full_ft=saved.get("full_ft", False),
                           prefill_lora=saved.get("prefill_lora", False))
    system.load_trainable_state(saved["trainable_state"])
    runner = AdapterRunner(system)
    runner.checkpoint_metadata = {
        key: saved.get(key) for key in ("arm", "full_ft", "model_revision", "optimizer_steps", "kl_tokens", "next_sample", "cumulative", "saved_at")
    }
    return runner
