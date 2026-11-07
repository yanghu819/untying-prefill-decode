"""Batched (padded) training for adapter arms A/B/C and the refresh-aware arm R.

Same data order, normalization and checkpoint format as train_adapter.py; each
rank processes its share of a global batch in microbatches of --micro rows.
"""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch import nn

from ouro.adapter_model import AdapterSystem
from ouro.batched import adapter_forward, delta_forward, kvshare_forward, kvshare_self_forward, loopmap_forward, loopmap_selflive_forward, native_lora_forward, refresh_forward
from ouro.refresh_model import RefreshSystem


class Batched(nn.Module):
    def __init__(self, system, refresh):
        super().__init__()
        self.system, self.refresh = system, refresh

    def forward(self, samples, blocks):
        device = next(p for p in self.system.parameters() if p.requires_grad).device
        if self.system.arm in ("D", "E", "H"):
            return native_lora_forward(self.system, samples, device, self.system.first_target)
        if self.system.arm == "O":
            return loopmap_selflive_forward(self.system, samples, device)
        if self.system.arm == "P":  # original self-live recipe: main per-loop loss + 0.1 x self-live view, fixed-point history
            return loopmap_selflive_forward(self.system, samples, device, faithful=True)
        if self.system.arm[0] in "XNQ" or self.system.arm in ("V", "S", "W", "T", "M", "a"):
            return loopmap_forward(self.system, samples, device)
        if self.system.arm[0] == "Z":
            return kvshare_forward(self.system, samples, device)
        if self.system.arm[0] == "Y":
            return kvshare_self_forward(self.system, samples, device)
        if self.system.arm in ("F", "G", "I", "J", "K", "L", "U"):
            return delta_forward(self.system, samples, device)
        if self.refresh:
            return refresh_forward(self.system, samples, blocks, device)
        return adapter_forward(self.system, samples, device)
from ouro.model import load_model
from ouro.prepare_adapter_data import AdapterDataset


U_CELLS = [(p, d) for p in range(1, 5) for d in range(1, 4) if p >= d]  # (4, 4) is the native teacher itself
V_CELLS = [(p, d) for p in range(1, 5) for d in range(1, 5)]  # loop-map connector: every cell, also p < d
if os.environ.get("OURO_CELLS"):  # e.g. "4x1,4x2,4x3,4x4" (fixed prefill 4) or a wider prefill range for the depth sweep
    V_CELLS = [tuple(int(v) for v in c.split("x")) for c in os.environ["OURO_CELLS"].split(",")]


def learning_rate(tokens, planned_tokens):
    warmup = planned_tokens * 0.02
    if tokens <= warmup:
        return 1e-4 * tokens / warmup
    progress = min(1.0, (tokens - warmup) / (planned_tokens - warmup))
    return 1e-5 + 0.5 * (1e-4 - 1e-5) * (1 + math.cos(math.pi * progress))


def main():
    if os.environ.get("OURO_NO_CUDNN_SDP"):  # cuDNN SDPA intermittently fails (mha_graph.execute) on some shapes
        torch.backends.cuda.enable_cudnn_sdp(False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "U", "V", "S", "W", "T", "R", "Z1", "Z2", "Z3", "Z4", "Y1", "Y2", "Y3", "Y4", "M", "O", "P", "a")
                        + tuple(f"{a}{p}{d}" for a in "XNQ" for p in range(1, 5) for d in range(1, 5)), required=True)
    parser.add_argument("--blocks", type=int, nargs="+", default=[4, 8, 16, 32, 1000])
    parser.add_argument("--micro", type=int, default=6)
    parser.add_argument("--stop-tokens", type=int, default=50_000_000)
    parser.add_argument("--planned-tokens", type=int, default=200_000_000)
    parser.add_argument("--global-batch", type=int, default=24)
    parser.add_argument("--seed", type=int, default=31847)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--full-ft", action="store_true", help="train every reader weight (fp32 master, bf16 autocast)")
    parser.add_argument("--reader-lr-scale", type=float, default=0.1)
    parser.add_argument("--start-sample", type=int, default=0, help="fresh run: begin the schedule here")
    parser.add_argument("--first-target", action="store_true",
                        help="arms D/E/H: also supervise the first answer token, as the loop adapter does")
    parser.add_argument("--freeze-maps", action="store_true",
                        help="ablation: keep the per-loop KV maps at their identity initialisation (LoRA only, direct sharing)")
    args = parser.parse_args()
    if args.stop_tokens > args.planned_tokens:
        raise ValueError("Requested stop exceeds the planned token budget.")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    if args.global_batch % world:
        raise ValueError("Global batch must divide evenly across this arm's ranks.")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    data = AdapterDataset(args.data)
    if len(data) % args.global_batch:
        raise ValueError("Prepared data must end on a complete global batch.")
    writer = load_model(args.model_path, device=device)
    reader = load_model(args.model_path, device=device, dtype=torch.float32 if args.full_ft else torch.bfloat16)
    system = RefreshSystem(writer, reader) if args.arm == "R" else AdapterSystem(writer, reader, args.arm, full_ft=args.full_ft)
    system.first_target = args.first_target
    trainable = [p for p in system.parameters() if p.requires_grad]
    if args.freeze_maps:  # maps stay in the checkpoint (as identity) so evaluation is unchanged, but are never updated
        frozen = {id(p) for p in system.loop_adapter.parameters()}
        trainable = [p for p in trainable if id(p) not in frozen]
    reader_ids = {id(p) for p in system.reader.parameters()}
    groups = [dict(params=[p for p in trainable if id(p) not in reader_ids], scale=1.0),
              dict(params=[p for p in trainable if id(p) in reader_ids], scale=args.reader_lr_scale)]
    if not args.full_ft:  # one group, as in older checkpoints, so their optimizer state resumes
        groups = [dict(params=trainable, scale=1.0)]
    optimizer = torch.optim.AdamW([g for g in groups if g["params"]], lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if args.full_ft else nullcontext()
    next_sample, consumed, update = args.start_sample, 0, 0
    cumulative = {"fineweb_kl_tokens": 0, "gsm_kl_tokens": 0, "input_tokens": 0}
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved["arm"] != ("B" if args.arm == "R" else args.arm):
            raise ValueError("Resume arm differs from the requested arm.")
        system.load_trainable_state(saved["trainable_state"])
        if len(saved["optimizer"]["param_groups"]) != len(optimizer.param_groups):
            # checkpoint used the two-group (bridge / reader) layout
            split = [dict(params=[p for p in trainable if id(p) not in reader_ids], scale=1.0),
                     dict(params=[p for p in trainable if id(p) in reader_ids], scale=1.0)]
            optimizer = torch.optim.AdamW([g for g in split if g["params"]], lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0)
        optimizer.load_state_dict(saved["optimizer"])
        next_sample, consumed, update = saved["next_sample"], saved["kl_tokens"], saved["optimizer_steps"]
        cumulative = saved["cumulative"]
        torch.set_rng_state(saved["rng"]["torch"].cpu())
        torch.cuda.set_rng_state(saved["rng"]["cuda"].cpu(), device)
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
    wrapper = Batched(system, args.arm == "R")
    model = DistributedDataParallel(wrapper, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else wrapper
    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        setup = {
            "event": "start", "arm": args.arm, "cells": V_CELLS, "first_target": args.first_target, "freeze_maps": args.freeze_maps, "refresh_blocks": args.blocks if args.arm == "R" else None, "micro": args.micro, "world_size": world,
            "global_batch": args.global_batch, "microbatch": 1,
            "gradient_accumulation": args.global_batch // world,
            "trainable_parameters": sum(p.numel() for p in trainable),
            "model_path": str(args.model_path), "planned_kl_tokens": args.planned_tokens,
            "stop_kl_tokens": args.stop_tokens, "resume": str(args.resume) if args.resume else None, "start_sample": args.start_sample,
            "data_manifest": data.manifest,
        }
        (args.output / "setup.json").write_text(json.dumps(setup, indent=2) + "\n")
        print(json.dumps(setup), flush=True)
    started = perf_counter()
    last_stats = None
    for start in range(next_sample, len(data), args.global_batch):
        if consumed >= args.stop_tokens:
            break
        end = start + args.global_batch
        block_tokens = int(data.effective_tokens[start:end].sum())
        lr = learning_rate(consumed + block_tokens, args.planned_tokens)
        for group in optimizer.param_groups:
            group["lr"] = lr * group.get("scale", 1.0)
        optimizer.zero_grad(set_to_none=True)
        stats = torch.zeros(7, device=device, dtype=torch.float64)
        local_indices = list(range(start + rank, end, world))
        chunks = [local_indices[k:k + args.micro] for k in range(0, len(local_indices), args.micro)]
        for micro, chunk in enumerate(chunks):
            samples = [data[index] for index in chunk]
            blocks = [random.Random(args.seed * 1_000_003 + index).choice(args.blocks) for index in chunk]
            if args.arm in ("U", "V", "S", "W", "T", "M", "O", "P", "a"):  # universal connector: one (prefill p, decode d) cell per micro-batch
                cells = U_CELLS if args.arm == "U" else V_CELLS
                system.prefill_steps, system.decode_steps = random.Random(args.seed * 7919 + chunk[0]).choice(cells)
            syncing = micro == len(chunks) - 1
            context = nullcontext() if syncing or world == 1 else model.no_sync()
            with context, autocast:
                loss_sum, metrics = model(samples, blocks)
                (loss_sum * world / block_tokens).backward()
            stats[:4] += metrics.to(torch.float64)
            for sample in samples:
                stats[4 if sample["source"] == "fineweb" else 5] += len(sample["suffix_ids"]) - 1
                stats[6] += len(sample["prefix_ids"]) + len(sample["suffix_ids"])
            del loss_sum, metrics, samples
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if world > 1:
            dist.all_reduce(stats)
        values = stats.tolist()
        consumed += block_tokens
        update += 1
        next_sample = end
        cumulative["fineweb_kl_tokens"] += int(values[4])
        cumulative["gsm_kl_tokens"] += int(values[5])
        cumulative["input_tokens"] += int(values[6])
        last_stats = {
            "event": "update", "arm": args.arm, "optimizer_step": update,
            "next_sample": next_sample, "kl_tokens": consumed, "block_kl_tokens": block_tokens,
            "kl": values[0] / block_tokens, "student_ce": values[1] / block_tokens,
            "teacher_ce": values[2] / block_tokens, "loss_positions": values[3],
            "lr": lr, "grad_norm": float(grad_norm), "elapsed_seconds": perf_counter() - started,
            "cumulative": cumulative.copy(),
        }
        if rank == 0 and (update == 1 or update % args.log_every == 0 or consumed >= args.stop_tokens):
            print(json.dumps(last_stats), flush=True)
            with (args.output / "train.jsonl").open("a") as handle:
                handle.write(json.dumps(last_stats) + "\n")
            (args.output / "progress.json").write_text(json.dumps(last_stats, indent=2) + "\n")
    if consumed < args.stop_tokens:
        raise ValueError("Prepared stream ended before the requested token observation.")
    if rank == 0:
        state = system.trainable_state()
        if args.full_ft:  # bf16 reader weights; optimizer dropped (no resume) to fit storage
            state = {k: v.to(torch.bfloat16) if k.startswith("reader.") else v for k, v in state.items()}
        checkpoint = {
            "arm": "B" if args.arm == "R" else args.arm, "trained_as": args.arm, "model_path": str(args.model_path),
            "full_ft": args.full_ft,
            "trainable_state": state, "optimizer": None if args.full_ft else optimizer.state_dict(),
            "optimizer_steps": update, "kl_tokens": consumed, "next_sample": next_sample,
            "cumulative": cumulative, "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device),
                    "python": random.getstate(), "numpy": np.random.get_state()},
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        torch.save(checkpoint, args.output / f"checkpoint-{args.stop_tokens // 1_000_000}m.pt")
        result = {
            "status": "stopped_at_requested_observation", "arm": args.arm,
            "kl_tokens": consumed, "optimizer_steps": update, "next_sample": next_sample,
            "cumulative": cumulative, "elapsed_seconds": perf_counter() - started,
            "checkpoint": str(args.output / f"checkpoint-{args.stop_tokens // 1_000_000}m.pt"), "last_update": last_stats,
            "continues_automatically": False,
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
