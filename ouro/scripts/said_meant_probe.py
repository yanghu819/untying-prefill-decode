"""'Shallow KV keeps what was said, deep KV keeps what it means': linear probes on the per-loop value cache.

The frozen native Ouro model reads FineWeb windows with 4 loops. For each (loop, layer) we take the stored value vectors
(what attention reads back from the prompt) and fit a linear classifier for four targets at every position i:
token i-2, token i-1, token i (what was said) and token i+1 (what comes next). Probe accuracy on held-out windows, over
the most frequent tokens (all other targets masked), is reported per (loop, layer, target).

usage: said_meant_probe.py MODEL OUT.json [--windows 192] [--len 1024]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ouro.adapter_model import backbone  # noqa: E402
from ouro.model import NativeCache, load_model  # noqa: E402

TARGETS = {"tok-2": -2, "tok-1": -1, "tok": 0, "next": 1}


def probe(x_tr, y_tr, x_te, y_te, classes, steps=600, lr=3e-3):
    w = torch.zeros(x_tr.shape[1], classes, device=x_tr.device, requires_grad=True)
    b = torch.zeros(classes, device=x_tr.device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=0.0)
    g = torch.Generator(device=x_tr.device).manual_seed(0)
    for _ in range(steps):
        i = torch.randint(0, x_tr.shape[0], (8192,), device=x_tr.device, generator=g)
        loss = F.cross_entropy(x_tr[i] @ w + b, y_tr[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return ((x_te @ w + b).argmax(-1) == y_te).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--data", default="/dev/shm/big-small-ouro/mix-data/fineweb_train.bin")
    ap.add_argument("--windows", type=int, default=192)
    ap.add_argument("--len", type=int, default=1024)
    ap.add_argument("--classes", type=int, default=2048)
    ap.add_argument("--layers", default="0,3,7,11,15,19,23,27,31,35,39,43,47")
    ap.add_argument("--loops", type=int, default=4)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--targets", default=",".join(TARGETS))
    a = ap.parse_args()

    tokens = np.memmap(a.data, dtype="<u4", mode="r")
    start = len(tokens) - a.windows * a.len - 1  # the tail of the stream
    ids = torch.from_numpy(np.array(tokens[start:start + a.windows * a.len], dtype=np.int64)).view(a.windows, a.len)

    model = load_model(a.model)
    L = model.config.num_hidden_layers
    layers = [int(x) for x in a.layers.split(",") if int(x) < L]
    keep = slice(16, a.len - 2)  # skip the first positions (no context) and leave room for the next-token target
    feats = {(k, l): [] for k in range(a.loops) for l in layers}
    with torch.no_grad():
        for i in range(0, a.windows, 8):
            cache = NativeCache(None)
            backbone(model, ids[i:i + 8].cuda(), cache, a.loops)
            for k in range(a.loops):
                for l in layers:
                    v = cache.value_cache[k * L + l]  # (b, kv_heads, n, head_dim)
                    feats[(k, l)].append(v.transpose(1, 2).flatten(2)[:, keep].float().cpu())
            del cache

    flat = ids.view(-1)
    split = int(a.windows * 0.8)
    counts = torch.bincount(ids[:split].reshape(-1), minlength=model.config.vocab_size)  # vocabulary from training windows
    top = counts.argsort(descending=True)[: a.classes]
    remap = torch.full((model.config.vocab_size,), -1, dtype=torch.long)
    remap[top] = torch.arange(a.classes)
    pos = torch.arange(a.len)[keep]
    rows = []
    for name, off in ((t, TARGETS[t]) for t in a.targets.split(",")):
        y = remap[ids[:, pos + off]]  # (windows, positions)
        mtr, mte = (y[:split] >= 0).flatten(), (y[split:] >= 0).flatten()
        ytr, yte = y[:split].flatten()[mtr].cuda(), y[split:].flatten()[mte].cuda()
        for (k, l), parts in feats.items():
            x = torch.cat(parts)
            x = (x - x[:split].mean((0, 1))) / (x[:split].std((0, 1)) + 1e-5)
            xtr = x[:split].flatten(0, 1)[mtr].cuda()
            xte = x[split:].flatten(0, 1)[mte].cuda()
            acc = probe(xtr, ytr, xte, yte, a.classes, steps=a.steps)
            rows.append(dict(target=name, loop=k + 1, layer=l, acc=acc))
            print(json.dumps(rows[-1]), flush=True)
            del xtr, xte
    Path(a.out).write_text(json.dumps(dict(model=a.model, windows=a.windows, len=a.len, classes=a.classes,
                                           coverage=float((remap[flat] >= 0).float().mean()), rows=rows), indent=1))


if __name__ == "__main__":
    main()
