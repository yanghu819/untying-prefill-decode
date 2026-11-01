"""Re-wrap the diverse distillation mix (mix-data-48m) in the Ouro-Thinking chat template, same samples and schedule:
pool rows "User: {prompt}\n\nAssistant:" -> " {response}<eos>" become
apply_chat_template([user: prompt], add_generation_prompt=True) -> "{response}<|im_end|>" (truncated to the same
1024-token target cap); FineWeb windows (source 0) stay raw text and share fineweb_train.bin.
usage: prepare_think_mix.py SRC_DIR OUT_DIR TOKENIZER_DIR"""
import json
import os
from pathlib import Path
import sys

import numpy as np
from transformers import AutoTokenizer

src, out, tok_dir = (Path(a) for a in sys.argv[1:4])
tok = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
base_eos, end = 0, tok.convert_tokens_to_ids("<|im_end|>")
out.mkdir(parents=True, exist_ok=True)
rows, bad = [], 0
for line in open(src / "gsm_train.jsonl"):
    r = json.loads(line)
    prompt = tok.decode(r["prompt_ids"])
    if prompt.endswith("\n\nAssistant:"):
        prompt = prompt[:-len("\n\nAssistant:")]
    else:
        bad += 1
    if prompt.startswith("User: "):
        prompt = prompt[len("User: "):]  # left-truncated long prompts have no "User: " prefix
    finished = r["target_ids"][-1] == base_eos
    response = tok.decode(r["target_ids"][:-1] if finished else r["target_ids"]).lstrip(" ")
    p_ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True)
    t_ids = tok.encode(response, add_special_tokens=False)[:1024 - finished] + ([end] if finished else [])
    rows.append(dict(source=r["source"], prompt_ids=p_ids, target_ids=t_ids, effective_tokens=len(t_ids) - 1))
with open(out / "gsm_train.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
samples = np.load(src / "samples.npy")
for i, (s, item, _) in enumerate(samples):
    if s == 1:
        samples[i, 2] = rows[item]["effective_tokens"]
np.save(out / "samples.npy", samples)
if not (out / "fineweb_train.bin").exists():
    os.symlink((src / "fineweb_train.bin").resolve(), out / "fineweb_train.bin")
manifest = json.load(open(src / "manifest.json"))
manifest["chat_template"] = dict(tokenizer=str(tok_dir), note=__doc__.splitlines()[0], unparsed_prompts=bad)
json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
print(json.dumps(dict(rows=len(rows), unparsed_prompts=bad, kl_tokens=int(samples[:, 2].sum()))))
