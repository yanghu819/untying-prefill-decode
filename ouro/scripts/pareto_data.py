"""Collect per-(p, d)-cell quality for the Ouro-2.6B Pareto plots into one JSON (run on the B300 box).

Long-context benches use the bench-mini ids shared by every grid cell (identical rows); lm-eval tasks take the
official-protocol results (sample sizes vary by cell: full sets for the older cells, --limit for the new ones).
"""
import glob, json, statistics

R = "/dev/shm/big-small-ouro/runs"; B = f"{R}/bench"; Q = f"{R}/lmeval/out"
# cell: (p, d, kind, bench dir, lm-eval prefix list)
CELLS = {
    "native1": (1, 1, "native", "X26-n1", ["o26n1"]),
    "native2": (2, 2, "native", "X26-n2", ["o26n2"]),
    "native3": (3, 3, "native", "X26-n3", ["o26n3"]),
    "native4": (4, 4, "native", "X26-n4", ["o26n4"]),
    "p1d1": (1, 1, "sym-LoRA", "X26-E-p1d1", ["o26E"]),
    "p2d2": (2, 2, "sym-LoRA", "X26-D-p2d2", ["o26D"]),
    "p3d3": (3, 3, "sym-LoRA", "X26-H-p3d3", ["X26-H-p3d3"]),
    "p2d1": (2, 1, "asym-delta", "X26-J", ["o26J"]),
    "p3d1": (3, 1, "asym-delta", "X26-K", ["o26K"]),
    "p4d1": (4, 1, "asym-delta", "X26-F-delta", ["o26F"]),
    "p3d2": (3, 2, "asym-delta", "X26-L-p3d2delta", ["X26-L-p3d2delta"]),
    "p4d2": (4, 2, "asym-delta", "X26-G-delta2", ["o26G"]),
    "p4d3": (4, 3, "asym-delta", "X26-I-p4d3delta", ["X26-I-p4d3delta"]),
    "kvshare-d1": (4, 1, "kv-share", "X26-zero41", []),
    "kvshare-d2": (4, 2, "kv-share", "X26-zero42", []),
    "kvshare-d4": (4, 4, "kv-share", "X26-zero44", []),
}
for p in range(1, 5):
    for d in range(1, 4):
        if p >= d:
            CELLS[f"U{p}{d}"] = (p, d, "universal", f"X26-U{p}{d}", [f"X26-U{p}{d}"])
# new loop-map connector (prefill exactly p loops, decoder emits every output token; allows p < d)
for p in range(1, 5):
    for d in range(1, 5):
        if p != d:
            CELLS[f"X{p}{d}"] = (p, d, "loopmap", f"X26-X{p}{d}", [f"X26-X{p}{d}"])
# universal loop-map connector: one map for all 16 cells
for p in range(1, 5):
    for d in range(1, 5):
        CELLS[f"V{p}{d}"] = (p, d, "universal-loopmap", f"X26-V{p}{d}", [f"X26-V{p}{d}"])
for p in range(1, 5):
    for d in range(1, 5):
        CELLS[f"S{p}{d}"] = (p, d, "universal-singlebank", f"X26-S{p}{d}", [f"X26-S{p}{d}"])
for d in range(1, 5):  # trained KV sharing: p4 prefill, d-loop decode, one shared last-loop bank
    CELLS[f"Z{d}"] = (4, d, "kv-share-trained", f"X26-Z{d}", [f"X26-Z{d}"])
for p in range(1, 5):  # untrained read p / write d (frozen model reads the prefill's last-loop KV directly)
    for d in range(1, 5):
        if p != d:
            CELLS[f"raw{p}{d}"] = (p, d, "untrained", f"X26-raw{p}{d}", [f"X26-raw{p}{d}"])
METRICS = {"gsm8k_cot": "exact_match,flexible-extract", "bbh_cot_fewshot": "exact_match,get-answer",
           "minerva_math500": "math_verify,none", "mmlu": "acc,none", "gpqa_main_n_shot_local": "acc,none"}


def bench_rows(name, bench):
    out = {}
    for f in glob.glob(f"{B}/{name}/s*/{bench}.jsonl"):
        for line in open(f):
            r = json.loads(line); out[r["id"]] = r["score"]
    return out


def lmeval(prefixes):
    res, lb = {}, []
    for pre in prefixes:
        # small pilot runs ("-short") first, so larger-sample reruns of the same task overwrite them
        for f in sorted(glob.glob(f"{Q}/{pre}-*/*/results_*.json"), key=lambda f: "-short" not in f):
            for task, v in json.load(open(f))["results"].items():
                if task in METRICS and METRICS[task] in v:
                    res[task] = round(100 * v[METRICS[task]], 2)
                if task.startswith("lbl_"):
                    lb.append(100 * next(x for k, x in v.items() if isinstance(x, float) and "stderr" not in k))
    if len(lb) == 16:
        res["longbench"] = round(statistics.mean(lb), 2)
    return res


out = {}
for bench in ("ruler4k", "ruler8k", "ruler16k", "helmet"):
    rows = {c: bench_rows(v[3], bench) for c, v in CELLS.items()}
    have = [c for c in CELLS if rows[c] and CELLS[c][2] in ("native", "sym-LoRA", "asym-delta")]
    common = set.intersection(*[set(rows[c]) for c in have]) if have else set()
    for c in CELLS:
        if rows[c] and common and common <= set(rows[c]):
            out.setdefault(c, {})[bench] = round(100 * sum(rows[c][i] for i in common) / len(common), 2)
    out.setdefault("_n", {})[bench] = len(common)
for c, v in CELLS.items():
    out.setdefault(c, {}).update(lmeval(v[4]))
    out[c].update(p=v[0], d=v[1], kind=v[2])
print(json.dumps(out, indent=1))
