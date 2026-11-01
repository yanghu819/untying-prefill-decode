"""Aggregate the newest results for the group report (run locally on a copy of runs/lmeval/out and runs/bench):
extra benchmarks for Ouro-2.6B (+HELMET), the Thinking model (accuracy, answer length, decode cost), Ouro-1.4B and
Nanbeige equal-decode comparisons. Writes results/latest.json.  usage: latest_data.py RUNS_COPY_DIR"""
import glob
import json
import os
from pathlib import Path
import random
import sys

R = Path(sys.argv[1]); OUT = Path(__file__).resolve().parent.parent / "results" / "latest.json"


def lm(run, key=None):
    out, chars = {}, []
    for f in glob.glob(f"{R}/lmeval/out/{run}/*/samples_*.jsonl"):
        task = os.path.basename(f).split("samples_")[1].rsplit("_", 1)[0]
        for l in open(f):
            r = json.loads(l)
            v = r.get(key) if key else next((r[k] for k in ("math_verify", "exact_match", "acc_norm") if k in r), None)
            if v is None:
                v = next((x for k, x in r.items() if k.startswith("exact_match") and "flexible" in k), None)
            if v is not None:
                out[(task, r["doc_id"])] = float(v)
                chars.append(len(r["resps"][0][0]) if r.get("resps") and isinstance(r["resps"][0][0], str) else 0)
    return out, (sum(chars) / len(chars) if chars else None)


def bench(run, b):
    return {json.loads(l)["id"]: float(json.loads(l)["score"]) for f in glob.glob(f"{R}/bench/{run}/s*/{b}.jsonl") for l in open(f)}


def boot(a, b, min_n=50):
    ids = sorted(set(a) & set(b)); n = len(ids)
    if n < min_n:
        return None
    d = [a[i] - b[i] for i in ids]; rng = random.Random(0)
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
    return [round(100 * sum(a[i] for i in ids) / n, 1), round(100 * sum(b[i] for i in ids) / n, 1),
            round(100 * sum(d) / n, 1), round(100 * bs[49], 1), round(100 * bs[1949], 1), n]


def merged(prefix, task, shards):
    out, ch = {}, []
    for s in shards:
        d, c = lm(f"{prefix}-{task}_{s}", "math_verify")
        out.update({(s,) + k: v for k, v in d.items()})
        if c:
            ch += [c] * len(d)
    return out, (sum(ch) / len(ch) if ch else None)


data = {}
# 1) Ouro-2.6B extra benchmarks incl. HELMET, equal-decode pairs
B = ["bbh", "mmlupro", "arc", "math"]
cfg = ["native1", "native2", "native3", "native4", "S11", "S22", "S33", "S41", "S42", "S43"]
xb = {(c, b): lm(f"o26x-{c}-{b}")[0] for c in cfg for b in B}
for c in cfg:
    xb[(c, "helmet")] = bench(f"X26-{c}-helmet", "helmet")
pairs = {}
for d in (1, 2, 3):
    for b in B + ["helmet"]:
        pairs[f"d{d}-{b}-deep_vs_native"] = boot(xb[(f"S4{d}", b)], xb[(f"native{d}", b)])
        pairs[f"d{d}-{b}-deep_vs_sym"] = boot(xb[(f"S4{d}", b)], xb[(f"S{d}{d}", b)])
        pairs[f"d{d}-{b}-deep_vs_native4"] = boot(xb[(f"S4{d}", b)], xb[("native4", b)])
data["xb"] = dict(pairs=pairs, means={f"{c}-{b}": round(100 * sum(v.values()) / len(v), 1) for (c, b), v in xb.items() if v})

# 2) Thinking model (GSM8K 4 shards, MATH shard 0, HELMET)
S = ["s0", "s1", "s2", "s3"]
think = {}
for name, prefix in [("native1", "think-native1"), ("native2", "think-native2"), ("native3", "think-native3"), ("native4", "think-native4"),
                     ("S11", "T26-S11"), ("S41", "T26-S41"), ("S22", "T26-S22"), ("S42", "T26-S42"), ("S33", "T26-S33"), ("S43", "T26-S43")]:
    g, gc = merged(prefix, "gsm8k_think", S); m, mc = merged(prefix, "math500_think", ["s0"])
    h = bench(f"T26-{name}-helmet", "helmet")
    loops = int(name[-1])
    think[name] = dict(gsm=round(100 * sum(g.values()) / len(g), 1) if g else None, gsm_n=len(g), gsm_chars=round(gc) if gc else None,
                       math=round(100 * sum(m.values()) / len(m), 1) if m else None, math_n=len(m), math_chars=round(mc) if mc else None,
                       helmet=round(100 * sum(h.values()) / len(h), 1) if len(h) >= 1108 else None, write=loops,
                       cost=round(loops * gc / 1000, 2) if gc else None)
    think[name]["_g"], think[name]["_m"], think[name]["_h"] = g, m, h
tp = {}
for a, b in [("S41", "native1"), ("S41", "native2"), ("S41", "S11"), ("S11", "native1"), ("S42", "native2"), ("S43", "native3"), ("S43", "native4"), ("S22", "native2")]:
    tp[f"{a}-{b}"] = dict(gsm=boot(think[a]["_g"], think[b]["_g"]), math=boot(think[a]["_m"], think[b]["_m"], 30), helmet=boot(think[a]["_h"], think[b]["_h"]))
for v in think.values():
    del v["_g"], v["_m"], v["_h"]
data["think"] = dict(cells=think, pairs=tp)

# 3) Ouro-1.4B equal-decode
o14 = {}
for b in ["ruler4k", "ruler8k", "ruler16k"]:
    for a, c in [("S41", "native1"), ("S41", "S11"), ("S22", "native2"), ("S42", "S22"), ("S43", "S33"), ("S43", "native3")]:
        o14[f"{a}-{c}-{b}"] = boot(bench(f"O14-{a}f", b), bench(f"O14-{c}f", b))
for t, k in [("gsmfull", "gsm8k_cot"), ("mmlufull", "mmlu")]:
    for a, c in [("S41", "native1"), ("S41", "S11"), ("S22", "native2"), ("S41", "native4")]:
        A, C = lm(f"O14-{a}-{t}")[0], lm(f"O14-{c}-{t}")[0]
        A = {i: v for i, v in A.items() if not i[0].startswith("mmlu_pro_")}; C = {i: v for i, v in C.items() if not i[0].startswith("mmlu_pro_")}
        o14[f"{a}-{c}-{k}"] = boot(A, C)
for b in ["arc", "bbh"]:
    for a, c in [("S41", "native1"), ("S41", "S11")]:
        o14[f"{a}-{c}-{b}"] = boot(lm(f"O14x-{a}-{b}")[0], lm(f"O14x-{c}-{b}")[0])
data["o14"] = o14
json.dump(data, open(OUT, "w"), indent=1)
print(OUT, json.dumps({k: (len(v) if isinstance(v, dict) else v) for k, v in data.items()}))
