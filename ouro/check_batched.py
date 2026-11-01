"""Strict-FP32 check: padded batched forwards == per-sample forwards."""

import argparse
import json

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from ouro.adapter_model import AdapterSystem
from ouro.batched import adapter_forward, refresh_forward
from ouro.model import load_model
from ouro.prepare_adapter_data import AdapterDataset
from ouro.refresh_model import RefreshSystem


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = AdapterDataset(args.data)
    # one FineWeb row (truncated for speed) and three math rows of different lengths
    rows, i = [], 100800
    while len(rows) < 4:
        s = data[i]
        if s["source"] == "fineweb" and not any(r["source"] == "fineweb" for r in rows):
            s = dict(s, prefix_ids=s["prefix_ids"][-300:], suffix_ids=s["suffix_ids"][:60])
            rows.append(s)
        elif s["source"] != "fineweb" and sum(r["source"] != "fineweb" for r in rows) < 3:
            rows.append(s)
        i += 1
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["trainable_state"]
    w = load_model(args.model_path, dtype=torch.float32)
    r = load_model(args.model_path, dtype=torch.float32)
    report = {}
    with sdpa_kernel(SDPBackend.MATH):
        plain = AdapterSystem(w, r, "B")
        plain.load_trainable_state(saved)
        single = sum(float(plain(torch.tensor([s["prefix_ids"]], device="cuda"),
                                 torch.tensor([s["suffix_ids"]], device="cuda"))[0]) for s in rows)
        batch = float(adapter_forward(plain, rows, "cuda")[0])
        report["plain"] = dict(single=single, batched=batch, rel=abs(single - batch) / single)
        w2 = load_model(args.model_path, dtype=torch.float32)
        r2 = load_model(args.model_path, dtype=torch.float32)
        ref = RefreshSystem(w2, r2)
        ref.load_trainable_state(saved)
        blocks = [4, 8, 16, 1000]
        single = sum(float(ref(torch.tensor([s["prefix_ids"]], device="cuda"),
                               torch.tensor([s["suffix_ids"]], device="cuda"), b)[0])
                     for s, b in zip(rows, blocks))
        batch = float(refresh_forward(ref, rows, blocks, "cuda")[0])
        report["refresh"] = dict(single=single, batched=batch, rel=abs(single - batch) / single)
    report["lengths"] = [(len(s["prefix_ids"]), len(s["suffix_ids"]), s["source"]) for s in rows]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
