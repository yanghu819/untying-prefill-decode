"""Ouro cache semantics against the official implementation.

Tiny models retain the original absolute full/cached tolerances. Checkpoints
use an exact same-dtype official reference; the official model's shape-dependent
full/cached differences are recorded rather than attributed to the adapter.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from ouro.model import (
    MODEL_REVISION,
    NativeCache,
    OuroConfig,
    OuroForCausalLM,
    OuroRunner,
    cache_bytes,
    load_model,
)


def max_error(a, b):
    return float((a.float() - b.float()).abs().max())


def direct(model, ids, steps, cache=None, use_cache=True, logits_to_keep=1):
    model.config.total_ut_steps = steps
    model.model.total_ut_steps = steps
    return model(
        input_ids=ids,
        past_key_values=cache,
        use_cache=use_cache,
        exit_at_step=steps - 1,
        exit_threshold=1.0,
        logits_to_keep=logits_to_keep,
    )


@torch.inference_mode()
def check(model, ids, prefix_tokens=6, official_reference_gate=False):
    rows = {}
    depth = model.config.num_hidden_layers
    prefix, suffix = ids[:, :prefix_tokens], ids[:, prefix_tokens:]
    for steps in (1, 4):
        native_cache = DynamicCache()
        native_cache.append_new_layers(steps * depth - 1)
        official = direct(model, prefix, steps, native_cache)
        runner = OuroRunner(model, f"p{steps}-d{steps}")
        calls = []
        handle = model.model.layers[0].register_forward_hook(
            lambda module, inputs, output: calls.append(1)
        )
        actual, cache = runner.prefill(prefix)
        handle.remove()
        assert len(calls) == steps
        prefill_error = max_error(actual, official.logits[:, -1])
        cache_error = max(
            max_error(a, b)
            for banks_a, banks_b in (
                (cache.key_cache, native_cache.key_cache),
                (cache.value_cache, native_cache.value_cache),
            )
            for a, b in zip(banks_a, banks_b)
        )
        print(json.dumps({"comparison": f"p{steps}-d{steps}-prefill",
                          "logits_max_error": prefill_error,
                          "cache_max_error": cache_error}), flush=True)
        assert prefill_error == cache_error == 0
        decoded, official_decoded = [], []
        native_decode_error = 0.0
        for i in range(suffix.shape[1]):
            token = suffix[:, i : i + 1]
            expected = direct(model, token, steps, native_cache)
            next_logits, cache = runner.decode(token, cache)
            native_decode_error = max(
                native_decode_error, max_error(next_logits, expected.logits[:, -1])
            )
            decoded.append(next_logits)
            official_decoded.append(expected.logits[:, -1])
        assert native_decode_error == 0
        full = direct(model, ids, steps, use_cache=False, logits_to_keep=0)
        decoded_logits = torch.stack(decoded, dim=1)
        official_logits = torch.stack(official_decoded, dim=1)
        full_logits = full.logits[:, prefix_tokens:]
        full_cached_error = max_error(
            decoded_logits, full_logits
        )
        official_full_cached_error = max_error(official_logits, full_logits)
        # Native-cache equality above is exact and is the cache-API gate.
        # Full vs tokenwise attention can use different BF16 kernel reductions.
        tolerance = 1e-5 if next(model.parameters()).dtype == torch.float32 else 0.25
        nll_tolerance = 1e-5 if next(model.parameters()).dtype == torch.float32 else 0.05
        targets = (ids[:, prefix_tokens:] + 1) % model.config.vocab_size
        decoded_nll = F.cross_entropy(
            decoded_logits.float().flatten(0, 1),
            targets.flatten(),
            reduction="none",
        )
        official_nll = F.cross_entropy(
            official_logits.float().flatten(0, 1),
            targets.flatten(),
            reduction="none",
        )
        full_nll = F.cross_entropy(
            full_logits.float().flatten(0, 1),
            targets.flatten(),
            reduction="none",
        )
        nll_error = max_error(decoded_nll, full_nll)
        official_nll_error = max_error(official_nll, full_nll)
        print(json.dumps({
            "comparison": f"p{steps}-d{steps}-full-cached",
            "gate": "exact-official-reference" if official_reference_gate else "original-absolute",
            "native_decode_max_error": native_decode_error,
            "local_logits_max_error": full_cached_error,
            "official_logits_max_error": official_full_cached_error,
            "local_target_nll_max_error": nll_error,
            "official_target_nll_max_error": official_nll_error,
            "original_logits_tolerance": tolerance,
            "original_target_nll_tolerance": nll_tolerance,
            "original_absolute_gate_passed": (
                full_cached_error <= tolerance and nll_error <= nll_tolerance
            ),
            "full_target_nll": full_nll.tolist(),
            "local_cached_target_nll": decoded_nll.tolist(),
            "official_cached_target_nll": official_nll.tolist(),
        }), flush=True)
        if official_reference_gate:
            # Native prefill, KV and every decode logit are already required
            # to match the official implementation exactly. The adapter may
            # not add to the official full/cached numerical discrepancy.
            assert full_cached_error <= official_full_cached_error
            assert nll_error <= official_nll_error
        else:
            assert full_cached_error <= tolerance
            assert nll_error <= nll_tolerance
        assert cache.get_seq_length() == ids.shape[1]
        rows[f"p{steps}-d{steps}"] = {
            "prefill_vs_official_dynamic_cache_max_error": prefill_error,
            "prefill_cache_max_error": cache_error,
            "decode_vs_official_dynamic_cache_max_error": native_decode_error,
            "full_vs_cached_max_error": full_cached_error,
            "full_vs_cached_tolerance": tolerance,
            "full_vs_cached_target_nll_max_error": nll_error,
            "full_vs_cached_target_nll_tolerance": nll_tolerance,
            "official_full_vs_cached_max_error": official_full_cached_error,
            "official_full_vs_cached_target_nll_max_error": official_nll_error,
            "original_absolute_logits_gate_passed": full_cached_error <= tolerance,
            "original_absolute_nll_gate_passed": nll_error <= nll_tolerance,
            "acceptance_gate": (
                "exact same-dtype official prefill/cache/decode; no added full/cached error"
                if official_reference_gate else "original absolute full/cached tolerances"
            ),
            "physical_layer_calls_per_prefill": len(calls),
            "cache": cache_bytes(cache),
        }

    for steps in (1, 2):
        original = direct(model, prefix, 4, NativeCache(4 * depth))
        runner = OuroRunner(model, f"p4-d{steps}")
        last_logits, cache = runner.prefill(prefix)
        first_token_error = max_error(last_logits, original.logits[:, -1])
        cache_error = max(
            max_error(a, b)
            for banks_a, banks_b in (
                (cache.key_cache, original.past_key_values.key_cache[3 * depth :]),
                (cache.value_cache, original.past_key_values.value_cache[3 * depth :]),
            )
            for a, b in zip(banks_a, banks_b)
        )
        assert first_token_error == cache_error == 0
        updates = []
        history_prefix_error = 0.0
        final_current_error = 0.0
        for token_idx in range(suffix.shape[1]):
            before_keys = list(cache.key_cache)
            before_values = list(cache.value_cache)
            before_length = cache.get_seq_length()
            native_update = cache.update
            current_updates = []

            def observed_update(key, value, layer_idx, cache_kwargs=None):
                nonlocal history_prefix_error, final_current_error
                loop, layer = divmod(layer_idx, depth)
                key_out, value_out = native_update(key, value, layer_idx, cache_kwargs)
                history_prefix_error = max(
                    history_prefix_error,
                    max_error(key_out[:, :, :before_length], before_keys[layer]),
                    max_error(value_out[:, :, :before_length], before_values[layer]),
                )
                final_current_error = max(
                    final_current_error,
                    max_error(key_out[:, :, before_length:], key),
                    max_error(value_out[:, :, before_length:], value),
                )
                assert key_out.shape[2] == before_length + 1
                if loop < steps - 1:
                    assert cache.key_cache[layer] is before_keys[layer]
                    assert cache.value_cache[layer] is before_values[layer]
                else:
                    assert cache.key_cache[layer] is key_out
                    assert cache.value_cache[layer] is value_out
                current_updates.append(layer_idx)
                return key_out, value_out

            cache.update = observed_update
            _, cache = runner.decode(suffix[:, token_idx : token_idx + 1], cache)
            cache.update = native_update
            assert current_updates == list(range(steps * depth))
            assert cache.get_seq_length() == before_length + 1
            assert len(cache.key_cache) == depth
            updates.append(len(current_updates))
        assert history_prefix_error == final_current_error == 0
        rows[f"p4-d{steps}"] = {
            "prefill_last_logits_vs_native_loop4_max_error": first_token_error,
            "retained_cache_vs_native_loop4_max_error": cache_error,
            "history_unchanged_within_decode_token_max_error": history_prefix_error,
            "attention_current_kv_is_current_loop_native_max_error": final_current_error,
            "cache_updates_per_token": updates,
            "intermediate_loops_do_not_commit": True,
            "exactly_one_append_per_token": True,
            "cache": cache_bytes(cache),
        }
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", "--model-path", dest="checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(31847)
    if args.checkpoint:
        model = load_model(args.checkpoint, device=args.device)
        ids = torch.tensor([list(range(11, 22))], device=args.device)
        results = {"checkpoint_bfloat16": check(
            model, ids, prefix_tokens=8, official_reference_gate=True
        )}
    else:
        config = OuroConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            layer_types=["full_attention"] * 2,
            total_ut_steps=4,
            early_exit_threshold=1.0,
            attention_dropout=0.0,
        )
        config._attn_implementation = "sdpa"
        model = OuroForCausalLM(config).eval().requires_grad_(False)
        ids = torch.randint(1, 64, (2, 9))
        results = {"tiny_float32": check(model, ids)}
        model.to(torch.bfloat16)
        results["tiny_bfloat16"] = check(model, ids)
    result = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "model_revision": MODEL_REVISION,
        "device": args.device,
        "checkpoint_used": str(args.checkpoint) if args.checkpoint else None,
        "optimizer_steps": 0,
        "official_backbone_source_unchanged": True,
        "runtime_adaptation": "HF 4.54.1 cache properties and get_mask_sizes only",
        "checkpoint_acceptance": (
            "Exact same-dtype/same-shape official-cache reference and unchanged last-loop "
            "history/commit checks. Original absolute full/cached errors and pass/fail "
            "are retained as numerical observations; no BF16 lossless claim."
            if args.checkpoint else None
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
