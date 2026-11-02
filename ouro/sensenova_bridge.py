"""Depth-split connector for SenseNova-Loop-9B-Preview (hybrid Gated-DeltaNet / full attention; physical layers
[start, end) are visited twice). Needs transformers 5.18 (PYTHONPATH=/dev/shm/big-small-ouro/tf518:...).

- Teacher: the native model (2 visits) over prompt + target.
- Prefill: the frozen model reads the prompt (all but its last token) with p in {1, 2} visits, LoRA off.
- Decode model: the same model with 1 visit and rank-16 LoRA (on only in decode) on the attention projections of the
  looped layers (full attention: q/k/v/o; GDN: in_proj_qkv / in_proj_z / out_proj). It reads the prefill cache spliced
  to one visit (sensenova_runner.splice: KV for full-attention layers, conv + recurrent state for GDN layers, from the
  LAST prefill visit), runs the last prompt token and target[:-1] in one parallel pass, and is trained with
  KL(teacher || student) summed over all target positions.
- One decoder serves both cells: p is sampled per sample (--p-mix), so read 1 / write 1 (same decoder, shallow prompt
  state) is the adaptation control for read 2 / write 1.

Run: torchrun --nproc-per-node N -m ouro.sensenova_bridge --model-path M --data train.jsonl --output DIR
"""

import argparse
import json
import math
import os
import random
import time
import types
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ouro.sensenova_runner import load, new_cache, set_count, splice, text_model

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj")


class LoRA(nn.Module):
    def __init__(self, base, state, rank=16):
        super().__init__()
        self.base, self.state = base, state
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=5 ** 0.5)

    def forward(self, x):
        y = self.base(x)
        if not self.state["on"]:
            return y
        return y + F.linear(F.linear(x.float(), self.a), self.b).to(y.dtype)


def add_lora(text, state):
    """Wrap the attention projections of the looped physical layers; returns {name: module}."""
    set_count(text, 1)  # sets text._loop_bounds
    start, end = text._loop_bounds
    wrapped = {}
    for i in range(start, end):
        for name, module in list(text.layers[i].named_modules()):
            for child_name, child in list(module.named_children()):
                if child_name in TARGETS and isinstance(child, nn.Linear):
                    lora = LoRA(child, state)
                    setattr(module, child_name, lora)
                    wrapped[f"layers.{i}.{name}.{child_name}".replace("..", ".")] = lora
    return wrapped


def lora_state(wrapped):
    return {k: dict(a=m.a.detach().cpu(), b=m.b.detach().cpu()) for k, m in wrapped.items()}


def load_lora(wrapped, path):
    saved = torch.load(path, map_location="cpu")["lora"]
    for k, m in wrapped.items():
        m.a.data.copy_(saved[k]["a"]); m.b.data.copy_(saved[k]["b"])


def _assign_recurrent(self, recurrent_states, state_idx=0, **kwargs):
    self.recurrent_states[state_idx] = recurrent_states
    return recurrent_states


def run(model, text, ids, cache):
    out = text(input_ids=ids, past_key_values=cache, use_cache=True)
    return out.last_hidden_state, out.past_key_values


def sample_loss(model, text, state, prompt, target, p, teacher_loops=2, mix=""):
    """Summed KL over target positions and their count, for one (prompt, target) pair (batch 1). mix: which deep state
    the writer gets when p > 1 ("" both, "A" attention KV only, "G" Gated-DeltaNet state only; sensenova_runner.splice)."""
    ids = torch.tensor([prompt + target[:-1]], device="cuda")
    n = len(prompt)
    with torch.no_grad():
        state["on"] = False
        set_count(text, teacher_loops)
        h, _ = run(model, text, ids, new_cache(text))
        t_logp = F.log_softmax(model.lm_head(h[0, n - 1:]).float(), -1)
        set_count(text, p)
        _, cache = run(model, text, ids[:, :n - 1], new_cache(text))
        if p != 1:
            cache = splice(text, cache, p, 1, mix)
    set_count(text, 1)
    cache._loop_execution_plan = text._loop_plan
    for layer in cache.layers:  # the GDN kernel saves the initial state for backward: replace, never copy_ into it
        if hasattr(layer, "recurrent_states"):
            layer.update_recurrent_state = types.MethodType(_assign_recurrent, layer)
    state["on"] = True
    h, _ = run(model, text, ids[:, n - 1:], cache)
    s_logp = F.log_softmax(model.lm_head(h[0]).float(), -1)
    state["on"] = False
    kl = (t_logp.exp() * (t_logp - s_logp)).sum()
    return kl, len(target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--kl-tokens", type=float, default=20e6)
    ap.add_argument("--tokens-per-step", type=int, default=65536, help="global KL tokens per optimizer step (approx.)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--p-mix", default="2:0.5,1:0.5")
    ap.add_argument("--mix", default="", choices=("", "A", "G"),
                    help="deep state for p > 1: both (default), attention KV only (A), Gated-DeltaNet state only (G)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    model = load(a.model_path)
    text = text_model(model)
    state = {"on": False}
    wrapped = add_lora(text, state)
    params = [p for m in wrapped.values() for p in (m.a, m.b)]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    cells = [(int(c.split(":")[0]), float(c.split(":")[1])) for c in a.p_mix.split(",")]
    rng = random.Random(a.seed * 1000 + rank)
    rows = [json.loads(line) for i, line in enumerate(open(a.data)) if i % world == rank]
    if rank == 0:
        n_params = sum(p.numel() for p in params)
        (out / "setup.json").write_text(json.dumps(dict(vars(a), lora_modules=len(wrapped), trainable=n_params,
                                                        world=world, rows_per_rank=len(rows)), indent=1))
        print(f"lora modules {len(wrapped)}, trainable {n_params / 1e6:.1f}M, rows/rank {len(rows)}", flush=True)
    planned_steps = math.ceil(a.kl_tokens / a.tokens_per_step)
    seen, step, i, t0 = 0, 0, 0, time.time()
    log = open(out / "train.jsonl", "a") if rank == 0 else None
    while step < planned_steps:
        local_tokens, local_kl = 0, 0.0
        opt.zero_grad(set_to_none=True)
        while local_tokens * world < a.tokens_per_step:
            r = rows[i % len(rows)]; i += 1
            p = rng.choices([c for c, _ in cells], [w for _, w in cells])[0]
            kl, n = sample_loss(model, text, state, r["prompt_ids"], r["target_ids"], p, mix=a.mix)
            (kl / (a.tokens_per_step / world)).backward()
            local_tokens += n; local_kl += kl.item()
        for prm in params:
            dist.all_reduce(prm.grad, op=dist.ReduceOp.AVG)
        stats = torch.tensor([local_tokens, local_kl], device="cuda", dtype=torch.float64)
        dist.all_reduce(stats)
        warm = min(1.0, (step + 1) / 20)
        decay = 0.5 * (1 + math.cos(math.pi * step / planned_steps))
        for g in opt.param_groups:
            g["lr"] = a.lr * warm * (0.1 + 0.9 * decay)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1; seen += int(stats[0])
        if rank == 0:
            rec = dict(step=step, kl_tokens=seen, kl=float(stats[1] / stats[0]), grad_norm=float(gn),
                       lr=opt.param_groups[0]["lr"], elapsed=round(time.time() - t0, 1))
            log.write(json.dumps(rec) + "\n"); log.flush()
            if step % 5 == 1:
                print(json.dumps(rec), flush=True)
            if step % 50 == 0 or step == planned_steps:
                torch.save(dict(lora=lora_state(wrapped), step=step, kl_tokens=seen, args=vars(a)), out / "checkpoint.pt")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
