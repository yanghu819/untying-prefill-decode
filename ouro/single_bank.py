"""Single-bank decoding for the S connector (deployment form).

The prompt KV is stored once: the prefill's last-loop keys (RoPE applied) and values, [B, H, n, D] per layer, for any
decode depth d. Decode loop k reads it through its own maps without materialising a bank per loop:
- the key map is one complex scale per RoPE frequency pair, so it commutes with RoPE and is folded into the query
  (q' = conj(c) q, per KV head);
- the value map is linear, so it is applied once to the attention-weighted sum of prompt values.
Prompt and generated-token scores share one softmax (joint max / normaliser). Only the generated tokens keep a
per-loop bank (t << n). Rows must be unpadded (no attention mask is applied); decode runs one token at a time.
"""

from contextlib import contextmanager
import os

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ouro.adapter_model import AdapterRunner, backbone
from ouro.model import NativeCache
from third_party.ouro.modeling_ouro import rotate_half


class SingleBankCache(NativeCache):
    """One prompt bank per layer (prompt_keys / prompt_values) + banks for generated tokens: one per (loop, layer), or with
    shared_answer=True (arm O, self-live view) ONE per layer, written only by the last decode loop; earlier loops read that
    history plus the current token's live K/V. Total KV is then b(n + t) for any decode depth."""

    def __init__(self, keys, values, loops, per_loop=False, shared_answer=False):
        super().__init__(None)
        self.prompt_keys, self.prompt_values = keys, values
        self.per_loop = per_loop  # native baseline: one static prompt bank per decode loop (bank index = loop * depth + layer)
        self.depth = len(keys) // loops if per_loop else len(keys)
        self.n = keys[0].shape[2]
        self.loops, self.shared_answer = loops, shared_answer
        banks = self.depth if shared_answer else loops * self.depth
        self.key_cache = [None] * banks
        self.value_cache = [None] * banks
        self.index = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.index = layer_idx  # read by the attention function that runs right after
        if self.shared_answer:
            loop, layer = divmod(layer_idx, self.depth)
            if self.key_cache[layer] is not None:
                key_states = torch.cat((self.key_cache[layer], key_states), dim=2)
                value_states = torch.cat((self.value_cache[layer], value_states), dim=2)
            if loop == self.loops - 1:  # the last decode loop commits this token to the shared answer bank
                self.key_cache[layer], self.value_cache[layer] = key_states, value_states
                self._seen_tokens = self.n + key_states.shape[2]
            return key_states, value_states
        if self.key_cache[layer_idx] is not None:
            key_states = torch.cat((self.key_cache[layer_idx], key_states), dim=2)
            value_states = torch.cat((self.value_cache[layer_idx], value_states), dim=2)
        self.key_cache[layer_idx], self.value_cache[layer_idx] = key_states, value_states
        self._seen_tokens = self.n + key_states.shape[2]
        return key_states, value_states

    def get_seq_length(self, layer_idx=0):
        own = self.key_cache[(layer_idx or 0) % len(self.key_cache)]
        return self.n + (0 if own is None else own.shape[2])


_ACTIVE = None
FUSED = os.environ.get("SINGLE_BANK_FUSED", "1") == "1"  # flash-attention path (0 = reference matmul path)


def _flash(q, k, v, scaling):
    """Flash attention returning (output [B, H, L, D], log-sum-exp [B, H, L]); no mask (decode, unpadded rows)."""
    out, lse = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, False, False, scale=scaling)[:2]
    return out, lse


def single_bank_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    cache, maps = _ACTIVE
    loop, layer = divmod(cache.index, cache.depth)
    bank = cache.index if cache.per_loop else layer
    b, hq, q_len, dim = query.shape
    assert q_len == 1, "single-bank decode runs one token at a time"
    hkv = key.shape[1]
    scaling = scaling if scaling is not None else dim**-0.5
    q = query.view(b, hkv, hq // hkv, dim)
    if maps is None:  # native baseline: plain query on its own loop's static bank
        out_p, lse_p = _flash(query, cache.prompt_keys[bank], cache.prompt_values[bank], scaling)
        out_s, lse_s = _flash(query, key, value, scaling)
        top = torch.maximum(lse_p, lse_s)
        w_p, w_s = (lse_p - top).exp()[..., None], (lse_s - top).exp()[..., None]
        out = (w_p * out_p.float() + w_s * out_s.float()) / (w_p + w_s)
        return out.to(query.dtype).transpose(1, 2).contiguous(), None
    a, s = maps.k_rot[loop, layer, :, 0, None], maps.k_rot[loop, layer, :, 1, None]  # [H, 1, D/2]
    q1, q2 = q.float().chunk(2, dim=-1)
    q_prompt = torch.cat((q1 * (1 + a) + q2 * s, q2 * (1 + a) - q1 * s), dim=-1).to(query.dtype)
    if FUSED and hq == hkv:
        # two flash-attention calls (prompt bank with the mapped query, own tokens with the plain query) merged by their
        # log-sum-exp; the value map acts on the normalised prompt output (linear, so it commutes with the weighting)
        out_p, lse_p = _flash(q_prompt.view_as(query), cache.prompt_keys[bank], cache.prompt_values[bank], scaling)
        out_s, lse_s = _flash(query, key, value, scaling)
        out_p = out_p.float()
        out_p = out_p + torch.einsum("bhli,hoi->bhlo", out_p, maps.v_delta[loop, layer])
        top = torch.maximum(lse_p, lse_s)
        w_p, w_s = (lse_p - top).exp()[..., None], (lse_s - top).exp()[..., None]
        out = (w_p * out_p + w_s * out_s.float()) / (w_p + w_s)
        return out.to(query.dtype).transpose(1, 2).contiguous(), None
    score_p = (q_prompt @ cache.prompt_keys[bank].transpose(-1, -2)).float() * scaling  # [B, H, g, n]
    score_s = (q @ key.transpose(-1, -2)).float() * scaling                              # [B, H, g, t]
    top = torch.maximum(score_p.amax(-1, keepdim=True), score_s.amax(-1, keepdim=True))
    w_p, w_s = (score_p - top).exp(), (score_s - top).exp()
    norm = w_p.sum(-1, keepdim=True) + w_s.sum(-1, keepdim=True)
    out_p = (w_p.to(value.dtype) @ cache.prompt_values[bank]).float()
    out_p = out_p + torch.einsum("bhgi,hoi->bhgo", out_p, maps.v_delta[loop, layer])
    out = (out_p + (w_s.to(value.dtype) @ value).float()) / norm
    return out.to(query.dtype).reshape(b, hq, 1, dim).transpose(1, 2).contiguous(), None


@contextmanager
def active(cache, maps):
    """Route the decode model's attention through single_bank_attention (the prefill model does not run meanwhile)."""
    global _ACTIVE
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    _ACTIVE = (cache, maps)
    ALL_ATTENTION_FUNCTIONS["sdpa"] = single_bank_attention
    try:
        yield
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original
        _ACTIVE = None


class SingleBankRunner(AdapterRunner):
    """S connector at cell (p, d) with the prompt KV stored once; arm O also stores the answer once (self-live view)."""

    def __init__(self, system, p, d):
        assert system.arm in ("S", "O", "P")
        super().__init__(system)
        system.prefill_steps, system.decode_steps = p, d

    @torch.inference_mode()
    def prefill(self, input_ids, attention_mask=None):
        _, prefix, _ = self.system.writer_prefix(input_ids, self.system.prefill_steps)
        n = input_ids.shape[1] - 1
        cos, sin = prefix.cos[:, :n].unsqueeze(1), prefix.sin[:, :n].unsqueeze(1)
        keys = [k[:, :, :n] * cos + rotate_half(k[:, :, :n]) * sin for k in prefix.raw_keys]
        values = [v[:, :, :n].contiguous() for v in prefix.values]
        del prefix
        return self.decode(input_ids[:, -1:], SingleBankCache(keys, values, self.system.decode_steps,
                                                              shared_answer=self.system.arm in ("O", "P")))

    @torch.inference_mode()
    def decode(self, input_ids, cache, attention_mask=None):
        with active(cache, self.system.loop_adapter):
            hidden, cache = backbone(self.model, input_ids, cache, self.system.decode_steps)
        return self.model.lm_head(hidden[:, -1:])[:, 0], cache


class AlignedRunner(AdapterRunner):
    """Aligned connector (arm a) at cell (p, d) for throughput: decode loop k reads prefill loop clip(p - d + k) from its
    own static prompt bank (one bank per decode loop, as the native d-loop model stores), through the same fused
    attention, key scale and value map as the single-bank decoder."""

    def __init__(self, system, p, d):
        assert system.arm == "a"
        super().__init__(system)
        system.prefill_steps, system.decode_steps = p, d

    @torch.inference_mode()
    def prefill(self, input_ids, attention_mask=None):
        p, d = self.system.prefill_steps, self.system.decode_steps
        # prefill all but the last prompt token, so each bank is used whole (no slice copies); the banks of the loops
        # no decode loop reads are freed with the prefix
        _, prefix, _ = self.system.writer_prefix(input_ids[:, :-1], p)
        depth = len(prefix.raw_keys)
        src = [min(max(p - d + k, 0), p - 1) for k in range(d)]
        keys = [prefix.all_keys[j * depth + l] for j in src for l in range(depth)]
        values = [prefix.all_values[j * depth + l] for j in src for l in range(depth)]
        del prefix
        return self.decode(input_ids[:, -1:], SingleBankCache(keys, values, d, per_loop=True))

    @torch.inference_mode()
    def decode(self, input_ids, cache, attention_mask=None):
        with active(cache, self.system.loop_adapter):
            hidden, cache = backbone(self.model, input_ids, cache, self.system.decode_steps)
        return self.model.lm_head(hidden[:, -1:])[:, 0], cache


class NativeStaticRunner:
    """Throughput baseline: the unmodified d-loop model with the same static-prompt cache and fused attention as the
    single-bank decoder, but one prompt bank per loop (its own KV) and no maps. Isolates the effect of storing one bank."""

    def __init__(self, model, loops):
        self.model, self.loops = model, loops

    @torch.inference_mode()
    def prefill(self, input_ids, attention_mask=None):
        depth = self.model.config.num_hidden_layers
        _, full = backbone(self.model, input_ids[:, :-1], NativeCache(self.loops * depth), self.loops)
        cache = SingleBankCache(list(full.key_cache), list(full.value_cache), self.loops, per_loop=True)
        del full
        return self.decode(input_ids[:, -1:], cache)

    @torch.inference_mode()
    def decode(self, input_ids, cache, attention_mask=None):
        with active(cache, None):
            hidden, cache = backbone(self.model, input_ids, cache, self.loops)
        return self.model.lm_head(hidden[:, -1:])[:, 0], cache


class ContiguousNativeCache(NativeCache):
    """Strongest native baseline: per-(loop, layer) KV preallocated contiguously for prompt + max_new tokens; decode writes
    in place and attention is ONE standard SDPA call over the filled prefix (no concatenation, no merge)."""

    def __init__(self, source, max_new):
        super().__init__(None)
        self.key_cache, self.value_cache, self.length = [], [], []
        for k, v in zip(source.key_cache, source.value_cache):
            b, h, n, d = k.shape
            kb = k.new_empty(b, h, n + max_new, d); kb[:, :, :n] = k
            vb = v.new_empty(b, h, n + max_new, d); vb[:, :, :n] = v
            self.key_cache.append(kb); self.value_cache.append(vb); self.length.append(n)
        self._seen_tokens = self.length[0]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        t, q = self.length[layer_idx], key_states.shape[2]
        self.key_cache[layer_idx][:, :, t:t + q] = key_states
        self.value_cache[layer_idx][:, :, t:t + q] = value_states
        self.length[layer_idx] = t + q
        self._seen_tokens = t + q
        return self.key_cache[layer_idx][:, :, :t + q], self.value_cache[layer_idx][:, :, :t + q]

    def get_seq_length(self, layer_idx=0):
        return self.length[layer_idx or 0]


class NativeContiguousRunner:
    """native<d>sc: the unmodified d-loop model with a preallocated contiguous cache and one attention call per layer."""

    def __init__(self, model, loops, max_new=None):
        self.model, self.loops = model, loops
        self.max_new = max_new or int(os.environ.get("STATIC_MAX_NEW", "64"))

    @torch.inference_mode()
    def prefill(self, input_ids, attention_mask=None):
        depth = self.model.config.num_hidden_layers
        _, full = backbone(self.model, input_ids[:, :-1], NativeCache(self.loops * depth), self.loops)
        cache = ContiguousNativeCache(full, self.max_new + 1)
        del full
        return self.decode(input_ids[:, -1:], cache)

    @torch.inference_mode()
    def decode(self, input_ids, cache, attention_mask=None):
        hidden, cache = backbone(self.model, input_ids, cache, self.loops)
        return self.model.lm_head(hidden[:, -1:])[:, 0], cache
