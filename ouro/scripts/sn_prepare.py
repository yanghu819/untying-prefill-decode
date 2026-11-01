"""Re-tokenise the thinking mix (Ouro token ids, ChatML) for SenseNova-Loop (Qwen3.5 tokenizer, ChatML with <think>).

Each pool row is decoded with the Ouro tokenizer, split back into chat messages, re-rendered with SenseNova's chat
template (generation prompt ends in "<think>\\n"), and the target is mapped to SenseNova's format:
- openthoughts traces: "<|begin_of_thought|>" dropped, "<|end_of_thought|>" -> "</think>", the solution markers dropped;
- other answers (tulu, longalpaca): "\\n</think>\\n\\n" + answer (Qwen3 non-thinking form).
Rows are shuffled with a fixed seed; prompts longer than --max-prompt SenseNova tokens are skipped.

usage: sn_prepare.py OURO_TOKENIZER SN_MODEL POOL.jsonl OUT.jsonl [--rows 60000] [--max-prompt 6144]
"""

import argparse
import json
import random
import re

from transformers import AutoTokenizer


def messages(chatml):
    out = []
    for role, content in re.findall(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", chatml, flags=re.S):
        if role != "system":
            out.append(dict(role=role, content=content))
    return out


def target_text(source, text):
    text = text.replace("<|im_end|>", "").rstrip()
    if "<|begin_of_thought|>" in text or source == "openthoughts":
        text = text.replace("<|begin_of_thought|>", "").lstrip("\n")
        text = text.replace("<|end_of_thought|>", "</think>")
        text = text.replace("<|begin_of_solution|>", "").replace("<|end_of_solution|>", "")
        return text
    return "\n</think>\n\n" + text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ouro_tok")
    ap.add_argument("sn_model")
    ap.add_argument("pool")
    ap.add_argument("out")
    ap.add_argument("--rows", type=int, default=60000)
    ap.add_argument("--max-prompt", type=int, default=6144)
    ap.add_argument("--seed", type=int, default=20261031)
    a = ap.parse_args()
    o = AutoTokenizer.from_pretrained(a.ouro_tok, trust_remote_code=True)
    s = AutoTokenizer.from_pretrained(a.sn_model, trust_remote_code=True)
    rows = [json.loads(line) for line in open(a.pool)]
    random.Random(a.seed).shuffle(rows)
    kept = skipped = 0
    with open(a.out, "w") as f:
        for r in rows:
            msgs = messages(o.decode(r["prompt_ids"]))
            if not msgs or msgs[-1]["role"] != "user":
                skipped += 1
                continue
            prompt = s.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            pid = s(prompt, add_special_tokens=False)["input_ids"]
            tid = s(target_text(r["source"], o.decode(r["target_ids"])), add_special_tokens=False)["input_ids"][:1024]
            if len(pid) > a.max_prompt or len(tid) < 8:
                skipped += 1
                continue
            f.write(json.dumps(dict(source=r["source"], prompt_ids=pid, target_ids=tid)) + "\n")
            kept += 1
            if kept >= a.rows:
                break
    print(json.dumps(dict(kept=kept, skipped=skipped)))


if __name__ == "__main__":
    main()
