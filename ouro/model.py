"""Frozen official Ouro backbone, with an optional last-loop history cache."""

from dataclasses import dataclass
from pathlib import Path

import torch

from third_party.ouro.configuration_ouro import OuroConfig
from third_party.ouro.modeling_ouro import (
    OuroForCausalLM,
    UniversalTransformerCache,
)


# default checkpoint of the single-model evaluation scripts; training and lm-eval runs take --model-path / path=
MODEL_ID = "ByteDance/Ouro-1.4B"
MODEL_REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"


@dataclass(frozen=True)
class Variant:
    prefill_steps: int
    decode_steps: int
    cache_mode: str


VARIANTS = {
    "p4-d4": Variant(4, 4, "per_loop"),
    "p1-d1": Variant(1, 1, "per_loop"),
    "p4-d1": Variant(4, 1, "last_loop"),
    "p4-d2": Variant(4, 2, "last_loop"),
    "p4-d3": Variant(4, 3, "last_loop"),
    "p4-d4-shared": Variant(4, 4, "last_loop"),  # Ouro paper's last-step KV sharing at full decode depth
    "p2-d1": Variant(2, 1, "last_loop"),
    "p3-d1": Variant(3, 1, "last_loop"),
    "p1-d2": Variant(1, 2, "last_loop"),
    "p2-d2": Variant(2, 2, "last_loop"),
    "p3-d2": Variant(3, 2, "last_loop"),
}


class NativeCache(UniversalTransformerCache):
    """The official cache update, adapted to the pinned HF 4.54.1 Cache API.

    HF's inherited key_cache/value_cache properties have no setters, while the
    official Ouro cache assigns ordinary lists. Shadow those properties and
    expose the mask extent expected by HF; leave the official update unchanged.
    """

    key_cache = None
    value_cache = None

    def get_mask_sizes(self, cache_position, layer_idx=0):
        return self.get_seq_length(layer_idx) + cache_position.shape[0], 0


class LastLoopCache(NativeCache):
    """Read committed history at every loop; append only on the final loop.

    The current token's K/V remain native to the current loop. Intermediate
    loops never overwrite the history that later loops must read.
    """

    def __init__(self, source, num_layers: int, prefill_steps: int, decode_steps: int):
        super().__init__(None)
        offset = (prefill_steps - 1) * num_layers
        self.key_cache = source.key_cache[offset : offset + num_layers]
        self.value_cache = source.value_cache[offset : offset + num_layers]
        self.num_layers = num_layers
        self.decode_steps = decode_steps
        self._seen_tokens = self.get_seq_length()

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        loop, physical_layer = divmod(layer_idx, self.num_layers)
        key = torch.cat((self.key_cache[physical_layer], key_states), dim=2)
        value = torch.cat((self.value_cache[physical_layer], value_states), dim=2)
        if loop == self.decode_steps - 1:
            self.key_cache[physical_layer] = key
            self.value_cache[physical_layer] = value
            self._seen_tokens = key.shape[2]
        return key, value


def load_model(path, device="cuda", dtype=torch.bfloat16, attention="sdpa"):
    """Load an already downloaded fixed-revision checkpoint; no network fallback."""
    path = Path(path)
    config = OuroConfig.from_pretrained(path, local_files_only=True)
    config.early_exit_threshold = 1.0
    config.early_exit_step = None
    model = OuroForCausalLM.from_pretrained(
        path,
        config=config,
        torch_dtype=dtype,
        attn_implementation=attention,
        local_files_only=True,
    )
    return model.to(device).eval().requires_grad_(False)


class OuroRunner:
    """Run the declared prefill/decode loop counts and history-cache policy."""

    def __init__(self, model, variant: str):
        self.model = model
        self.variant_name = variant
        self.variant = VARIANTS[variant]

    def _steps(self, steps):
        # Official forward computes all total_ut_steps before selecting an exit.
        # Both values must change to make a one-loop condition actually one loop.
        self.model.config.total_ut_steps = steps
        self.model.model.total_ut_steps = steps
        self.model.early_exit_threshold = 1.0
        self.model.early_exit_step = None

    @torch.inference_mode()
    def prefill(self, input_ids, attention_mask=None):
        steps = self.variant.prefill_steps
        self._steps(steps)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=NativeCache(steps * self.model.config.num_hidden_layers),
            use_cache=True,
            exit_at_step=steps - 1,
            exit_threshold=1.0,
            logits_to_keep=1,
        )
        cache = outputs.past_key_values
        if self.variant.cache_mode == "last_loop":
            cache = LastLoopCache(
                cache,
                self.model.config.num_hidden_layers,
                steps,
                self.variant.decode_steps,
            )
        # The first continuation token is predicted by the final prefill loop.
        # Only subsequent predictions consume the requested decode path.
        return outputs.logits[:, -1], cache

    @torch.inference_mode()
    def decode(self, input_ids, cache, attention_mask=None):
        if input_ids.shape[1] != 1:
            raise ValueError("Decode is token-sequential; input_ids must be [batch, 1].")
        steps = self.variant.decode_steps
        self._steps(steps)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            exit_at_step=steps - 1,
            exit_threshold=1.0,
            logits_to_keep=1,
        )
        return outputs.logits[:, -1], outputs.past_key_values


def cache_bytes(cache):
    entries = [
        tensor
        for banks in (cache.key_cache, cache.value_cache, getattr(cache, "prompt_keys", []), getattr(cache, "prompt_values", []))
        for tensor in banks
        if tensor is not None
    ]
    storage = {}
    for tensor in entries:
        backing = tensor.untyped_storage()
        storage[(tensor.device.type, tensor.device.index, backing.data_ptr())] = (
            backing.nbytes()
        )
    return {
        "logical_bytes": sum(t.numel() * t.element_size() for t in entries),
        "unique_storage_bytes": sum(storage.values()),
        "bank_count": len(entries) // 2,
        "seq_length": cache.get_seq_length(),
        "dtypes": sorted({str(t.dtype) for t in entries}),
    }
