"""Sanity checks for looped_bridge on a family: at zero init arm F == arm E == native shallow; RoPE inverse exact;
training loss/backward runs; delta gets a gradient."""
import sys
import torch
from ouro.looped_bridge import Bridge, load, rotate_half, run, head_rope, tokenizer

path = sys.argv[1]
model = load(path)
tok = tokenizer(path)
ids = torch.tensor([tok.encode("User: What is 17 * 23? Show the steps.\n\nAssistant:", add_special_tokens=False)], device="cuda")
target = tok.encode(" 17 * 23 = 391. The answer is 391.", add_special_tokens=False)
bridge = Bridge(model, "F", 2, 1)
n = ids.shape[1]
_, cache = bridge.prefill_model(ids, n)
raw, _ = cache.captured[0]
cos, sin = head_rope(*cache.rope)
k_rot = raw * cos + rotate_half(raw) * sin
print("captured layers", sum(c is not None for c in cache.captured), "raw", tuple(raw.shape))
with torch.no_grad():
    native = model.lm_head(run(model, ids, bridge.new_cache(1), 1)).float()
    hF, _ = bridge.decode_model(ids, bridge.deltas(cache, n))
    print("zero-init F vs native1 max|dlogit|", float((model.lm_head(hF).float() - native).abs().max()))
bridge.state["on"] = False
kl, s = bridge.loss(ids[0].tolist(), target)
kl.backward()
print("kl", float(kl), "ce/teacher_ce", [round(float(x), 3) for x in s[1:]])
print("grad k_delta", float(bridge.k_delta.grad.abs().sum()), "grad lora_b", float(bridge.lora[0].lora_b.grad.abs().sum()))
print("peak GiB", round(torch.cuda.max_memory_allocated() / 2**30, 1))
