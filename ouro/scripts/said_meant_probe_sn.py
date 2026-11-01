"""Said-vs-meant probes on SenseNova-Loop-9B (hybrid): the value cache of the full-attention layers inside the looped
block, at visit 1 vs visit 2 of the native 2-visit model, same layer and weights. Same probe, targets and split as
said_meant_probe.py; FineWeb windows are decoded with the Ouro tokenizer and re-encoded with SenseNova's.
Gated-DeltaNet layers keep one recurrent state per sequence, not per-token values, so they are not probed.

usage: said_meant_probe_sn.py SN_MODEL OURO_TOKENIZER OUT.json [--windows 384] [--len 1024]
Needs transformers 5.18 on PYTHONPATH.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ouro.sensenova_runner import load, new_cache, set_count, text_model, tokenizer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
import torch.nn.functional as F  # noqa: E402

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
    ap.add_argument("ouro_tok")
    ap.add_argument("out")
    ap.add_argument("--data", default="/dev/shm/big-small-ouro/mix-data/fineweb_train.bin")
    ap.add_argument("--windows", type=int, default=384)
    ap.add_argument("--len", type=int, default=1024)
    ap.add_argument("--classes", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=600)
    a = ap.parse_args()

    src = AutoTokenizer.from_pretrained(a.ouro_tok, trust_remote_code=True)
    tok = tokenizer(a.model)
    stream = np.memmap(a.data, dtype="<u4", mode="r")
    span = int(a.len * 1.6)  # Ouro tokens per window before re-encoding (SenseNova's tokenizer is coarser)
    start = len(stream) - a.windows * span - 1
    rows = []
    for w in range(a.windows):
        text = src.decode(np.array(stream[start + w * span: start + (w + 1) * span], dtype=np.int64))
        ids = tok(text, add_special_tokens=False)["input_ids"][: a.len]
        if len(ids) == a.len:
            rows.append(ids)
    ids = torch.tensor(rows)
    windows = len(rows)

    model = load(a.model)
    text = text_model(model)
    set_count(text, 2)
    s, e = text._loop_bounds
    width = e - s
    full = [l for l in range(s, e) if text.config.layer_types[l] == "full_attention"]
    keep = slice(16, a.len - 2)
    feats = {(v, l): [] for v in range(2) for l in full}
    with torch.no_grad():
        for i in range(0, windows, 8):
            cache = new_cache(text)
            text(input_ids=ids[i:i + 8].cuda(), past_key_values=cache, use_cache=True)
            for v in range(2):
                for l in full:
                    vals = cache.layers[s + v * width + (l - s)].values  # (b, kv_heads, n, head_dim)
                    feats[(v, l)].append(vals.transpose(1, 2).flatten(2)[:, keep].float().cpu())
            del cache

    split = int(windows * 0.8)
    counts = torch.bincount(ids[:split].reshape(-1), minlength=len(tok))
    top = counts.argsort(descending=True)[: a.classes]
    remap = torch.full((max(len(tok), int(ids.max()) + 1),), -1, dtype=torch.long)
    remap[top] = torch.arange(a.classes)
    pos = torch.arange(a.len)[keep]
    out = []
    for name, off in TARGETS.items():
        y = remap[ids[:, pos + off]]
        mtr, mte = (y[:split] >= 0).flatten(), (y[split:] >= 0).flatten()
        ytr, yte = y[:split].flatten()[mtr].cuda(), y[split:].flatten()[mte].cuda()
        for (v, l), parts in feats.items():
            x = torch.cat(parts)
            x = (x - x[:split].mean((0, 1))) / (x[:split].std((0, 1)) + 1e-5)
            acc = probe(x[:split].flatten(0, 1)[mtr].cuda(), ytr, x[split:].flatten(0, 1)[mte].cuda(), yte, a.classes,
                        steps=a.steps)
            out.append(dict(target=name, loop=v + 1, layer=l, acc=acc))
            print(json.dumps(out[-1]), flush=True)
    Path(a.out).write_text(json.dumps(dict(model=a.model, windows=windows, len=a.len, classes=a.classes, layers=full,
                                           coverage=float((remap[ids.reshape(-1)] >= 0).float().mean()), rows=out),
                                      indent=1))


if __name__ == "__main__":
    main()
