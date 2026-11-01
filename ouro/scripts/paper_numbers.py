"""Every paired comparison and mean in the paper, recomputed from per-item scores.

  PN_TOK=<Ouro-2.6B-Thinking dir> python ouro/scripts/paper_numbers.py export RUNS_DIR [OUT_DIR]
      on the machine that holds the raw evaluation outputs (RUNS_DIR/lmeval/out/<run>/*/samples_*.jsonl from
      lm-evaluation-harness, RUNS_DIR/bench/<run>/s*/<bench>.jsonl from evaluate_bench.py); writes one compact file per
      item set used below to OUT_DIR (default ouro/results/items). Missing runs are skipped and listed.
  python ouro/scripts/paper_numbers.py compute
      reads ouro/results/items, writes ouro/results/paper_numbers.json and prints every number with the value printed in
      the paper; a mismatch is flagged with '!!'.

Paired bootstrap: 5,000 resamples of items (numpy, seed 0), 2.5 / 97.5 percentiles, in points (x100). Item sets:
  L:<run>:<metric>[:<filter>]   lm-evaluation-harness samples (GSM8K: flexible-extract; MMLU: acc; ARC: acc_norm;
                                MMLU-Pro / BBH: exact_match; thinking tasks: math_verify;
                                MATH-500 for base models: mvfull = math_verify on the full answer)
  T:<prefix>:<metric>           thinking-model GSM8K, four shards <prefix>_s0.._s3 merged
  B:<run>:<bench>               evaluate_bench.py outputs (RULER, HELMET, BBH for SenseNova)
"""
import glob
import gzip
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

ITEMS = Path(__file__).resolve().parents[1] / "results" / "items"
OUT = Path(__file__).resolve().parents[1] / "results" / "paper_numbers.json"
GF = "exact_match:flexible-extract"


def gsm(run):
    return f"L:{run}-gsmfull:{GF}"


def mmlu(run):
    return f"L:{run}-mmlufull:acc"


def ruler(run, k):
    return f"B:{run}f:ruler{k}k"


# (section, label, A, B, printed value (diff, lo, hi) or None); difference = A - B
CMP = []
for d, v in ((1, [(15.2, 14.3, 16.1), (8.6, 4.3, 12.8), (8.0, 5.3, 10.5)]),
             (2, [(6.0, 5.4, 6.7), (5.3, 1.3, 9.4), (-1.7, -3.6, 0.2)]),
             (3, [(0.6, 0.2, 1.1), (-0.4, -3.0, 2.2), (-0.8, -2.4, 0.8)])):
    CMP += [("Table 1", f"write {d} MMLU", mmlu(f"X26-S4{d}"), mmlu(f"X26-S{d}{d}"), v[0]),
            ("Table 1", f"write {d} RULER-16K", ruler(f"X26-S4{d}", 16), ruler(f"X26-S{d}{d}", 16), v[1]),
            ("Table 1", f"write {d} GSM8K", gsm(f"X26-S4{d}"), gsm(f"X26-S{d}{d}"), v[2])]
CMP += [
    ("Table 2", "Ouro-1.4B ARC-C", "L:O14x-S41-arc:acc_norm", "L:O14x-S11-arc:acc_norm", (10.8, 8.4, 13.1)),
    ("Table 2", "Ouro-1.4B RULER-16K", ruler("O14-S41", 16), ruler("O14-S11", 16), (1.4, -2.6, 5.4)),
    ("Table 2", "Ouro-1.4B GSM8K", gsm("O14-S41"), gsm("O14-S11"), (6.2, 3.6, 8.7)),
    ("Table 2", "Ouro-2.6B-T MMLU-Pro", "L:T26x-S41-mmlupro:exact_match", "L:T26x-S11-mmlupro:exact_match", (18.6, 13.6, 23.8)),
    ("Table 2", "Ouro-2.6B-T HELMET", "B:T26ne-S41-helmet:helmet", "B:T26ne-S11-helmet:helmet", (12.0, 9.1, 15.0)),
    ("Table 2", "Ouro-2.6B-T GSM8K", "T:T26-S41-gsm8k_think:math_verify", "T:T26-S11-gsm8k_think:math_verify", (6.6, 4.2, 9.2)),
    ("Table 2", "Ouro-1.4B-T MMLU-Pro", "L:T14x-S41-mmlupro:exact_match", "L:T14x-S11-mmlupro:exact_match", (17.1, 12.4, 22.4)),
    ("Table 2", "Ouro-1.4B-T HELMET", "B:T14ne-S41-helmet:helmet", "B:T14ne-S11-helmet:helmet", (11.9, 9.0, 14.8)),
    ("Table 2", "Ouro-1.4B-T GSM8K", "T:T14-S41-gsm8k_think:math_verify", "T:T14-S11-gsm8k_think:math_verify", (13.6, 10.5, 16.6)),
    ("Table 2", "SenseNova MMLU-Pro", "L:SN-C21-mmlupro:exact_match", "L:SN-C11-mmlupro:exact_match", (5.5, 2.9, 8.1)),
    ("Table 2", "SenseNova HELMET", "B:SNnt-C21-helmet:helmet", "B:SNnt-C11-helmet:helmet", (6.7, 5.7, 7.8)),
    ("Table 2", "SenseNova BBH", "B:SNnt-C21-bbh:bbh", "B:SNnt-C11-bbh:bbh", (7.0, 5.9, 8.0)),
    ("Table 5", "2.6B write 1 RULER-4K", ruler("X26-S41", 4), ruler("X26-S11", 4), (-3.8, -8.2, 0.3)),
    ("Table 5", "2.6B write 1 MATH 4-shot", "L:o26x-S41-math:mvfull", "L:o26x-S11-math:mvfull", (-11.2, -15.4, -6.8)),
    ("Table 5", "2.6B write 1 MATH 0-shot", "L:o26x-S41-math0shot:mvfull", "L:o26x-S11-math0shot:mvfull", (7.0, 3.0, 10.8)),
    ("Table 5", "2.6B write 2 MATH 4-shot", "L:o26x-S42-math:mvfull", "L:o26x-S22-math:mvfull", None),
    ("Table 5", "2.6B write 2 MATH 0-shot", "L:o26x-S42-math0shot:mvfull", "L:o26x-S22-math0shot:mvfull", (7.0, 3.0, 11.2)),
    ("Table 5", "1.4B write 1 MATH 4-shot", "L:o14x-S41-math4shot:mvfull", "L:o14x-S11-math4shot:mvfull", (-4.0, -8.2, 0.0)),
    ("Table 5", "1.4B write 1 MATH 0-shot", "L:o14x-S41-math0shot:mvfull", "L:o14x-S11-math0shot:mvfull", (4.8, 1.4, 8.4)),
    ("Table 6", "direct sharing 4/1 GSM8K", gsm("X26-Y1"), gsm("X26-S41"), (-17.5, -20.1, -14.9)),
    ("Table 6", "direct sharing 4/2 GSM8K", gsm("X26-Y2"), gsm("X26-S42"), (-0.1, -2.0, 1.7)),
    ("Table 6", "answer once 4/2 GSM8K", gsm("X26-O42"), gsm("X26-S42"), (0.3, -1.5, 2.1)),
    ("Table 6", "answer once 3/3 GSM8K", gsm("X26-O33"), gsm("X26-S33"), (-0.8, -2.4, 0.8)),
    ("Table 6", "answer once 4/4 GSM8K", gsm("X26-O44"), gsm("X26-S44"), (-2.6, -4.2, -1.1)),
    ("Table 4", "answer once 4/4 vs native 4/4 GSM8K", gsm("X26-O44"), f"L:X26-native4-gsmfull:{GF}", None),
    ("Table 6", "first-loop branch 4/1 RULER-4K", ruler("X26-M41", 4), ruler("X26-S41", 4), (2.3, 1.0, 3.6)),
    ("Table 6", "first-loop branch 4/1 GSM8K", gsm("X26-M41"), gsm("X26-S41"), (0.7, -0.8, 2.2)),
]
for c in ("11", "22", "31", "32", "42"):
    CMP.append(("Table 6", f"bank per loop {c[0]}/{c[1]} RULER-16K", ruler(f"X26-V{c}", 16), ruler(f"X26-S{c}", 16), None))
for seed, run in ((1, "X26"), (2, "X26s2")):
    CMP += [("Table 7", f"seed {seed} write 1 GSM8K", gsm(f"{run}-S41"), gsm(f"{run}-S11"), None),
            ("Table 7", f"seed {seed} write 1 MMLU", mmlu(f"{run}-S41"), mmlu(f"{run}-S11"), None),
            ("Table 7", f"seed {seed} write 2 GSM8K", gsm(f"{run}-S42"), gsm(f"{run}-S22"), None)]
    for k in (4, 8, 16):
        CMP.append(("Table 7", f"seed {seed} write 1 RULER-{k}K", ruler(f"{run}-S41", k), ruler(f"{run}-S11", k), None))
    CMP.append(("Table 7", f"seed {seed} write 2 RULER-4K", ruler(f"{run}-S42", 4), ruler(f"{run}-S22", 4), None))
CMP += [
    ("Sec 4.2", "2.6B-T read 4/write 2 vs native 4/4 GSM8K (full)", "T:T26-S42-gsm8k_think:math_verify",
     "T:think-native4-gsm8k_think:math_verify", (-1.0, -2.0, 0.1)),
    ("Sec 4.2", "SenseNova read 1/write 1 vs native 2/2 GSM8K (full)", "T:SN-C11-gsm8k_think:math_verify",
     "T:SN-native2-gsm8k_think:math_verify", None),
    ("Limitations", "write 3 RULER-4K vs native 3/3", ruler("X26-S43", 4), "B:X26-n3f:ruler4k", (-10.4, None, None)),
    ("Limitations", "write 3 RULER-8K vs native 3/3", ruler("X26-S43", 8), "B:X26-n3f:ruler8k", (-9.5, None, None)),
    ("Limitations", "write 3 RULER-16K vs native 3/3", ruler("X26-S43", 16), "B:X26-n3f:ruler16k", (-5.3, None, None)),
    ("Limitations", "1.4B-T BBH read 4 vs read 1 (write 1)", "L:T14ne-S41-bbh:exact_match", "L:T14ne-S11-bbh:exact_match",
     (-20.5, None, None)),
    ("Limitations", "1.4B BBH read 4 vs read 1 (write 1)", "L:O14x-S41-bbh:exact_match", "L:O14x-S11-bbh:exact_match",
     (-24.5, None, None)),
    ("Limitations", "1.4B-T BBH read 4 vs read 2 (write 2)", "L:T14ne-S42-bbh:exact_match", "L:T14ne-S22-bbh:exact_match",
     (8.9, None, None)),
    ("App B", "Nanbeige RULER-4K read 2 vs 1 (write 1)", "B:NBX-p2d1:ruler4k", "B:NBX-p1d1:ruler4k", (11.8, None, None)),
    ("App B", "Nanbeige RULER-8K", "B:NBX-p2d1:ruler8k", "B:NBX-p1d1:ruler8k", (11.7, None, None)),
    ("App B", "Nanbeige RULER-16K", "B:NBX-p2d1:ruler16k", "B:NBX-p1d1:ruler16k", (14.2, None, None)),
    ("App B", "Nanbeige MMLU", "L:NBX-p2d1-mmlufull:acc", "L:NBX-p1d1-mmlufull:acc", (12.2, None, None)),
    ("App B", "Nanbeige GSM8K", f"L:NBX-p2d1-gsmfull:{GF}", f"L:NBX-p1d1-gsmfull:{GF}", (5.2, None, None)),
]
WANDER = {"native 1/1": "think-native1", "read 1 / write 1": "T26-S11", "read 4 / write 1": "T26-S41"}
WKEYS = [f"T:{p}-gsm8k_think:{m}" for p in WANDER.values() for m in ("noanswer", "ntok", "marks", "steps", "math_verify")]
SN_NOANS = {c: f"T:SN-{c}-gsm8k_think:noanswer" for c in ("native1", "splice21", "native2")}
# means (Figure 1 grid on evaluation subsets, Sec 4.3 SenseNova without training)
MEANS = [("Sec 4.3", f"SenseNova {c} RULER-4K", f"B:SNnt-{c}:ruler4k") for c in ("native1", "splice21", "native2")]
MEANS += [("Sec 4.3", f"SenseNova {c} MMLU-Pro", f"L:SN-{c}-mmlupro:exact_match") for c in ("native1", "splice21", "native2")]
MEANS += [("Sec 4.3", f"SenseNova {c} GSM8K no final answer", k) for c, k in SN_NOANS.items()]
MEANS += [("Sec 4.2", f"GSM8K full {c}", f"T:{c}-gsm8k_think:math_verify")
          for c in ("T26-S42", "think-native4", "SN-C11", "SN-C21", "SN-native2", "SN-native1", "SN-splice21")]
for p in range(1, 5):
    for d in range(1, 5):
        MEANS += [("Figure 1", f"S{p}{d} MMLU (20/subject)", f"L:X26-S{p}{d}-mmlu:acc"),
                  ("Figure 1", f"S{p}{d} RULER-16K (260)", f"B:X26-S{p}{d}:ruler16k"),
                  ("Figure 1", f"S{p}{d} GSM8K (400)", f"L:X26-S{p}{d}-gsm:{GF}")]


def fname(key):
    return key.replace(":", "__").replace("/", "_") + ".json.gz"


def mvfull(gold, text):
    from math_verify import parse, verify
    try:
        return float(verify(parse("$" + gold + "$"), parse(text)))
    except Exception:
        return 0.0


MARK = re.compile(r"(?i)\balternatively\b|\bwait\b|\bre-?read\b|\bon second thought\b|\blet me (re)?check\b")
_TOK = []


def ntok(text):
    """Generated tokens (tokenizer of the evaluated model, env PN_TOK)."""
    if not _TOK:
        from transformers import AutoTokenizer
        _TOK.append(AutoTokenizer.from_pretrained(os.environ["PN_TOK"], trust_remote_code=True))
    return float(len(_TOK[0](text, add_special_tokens=False)["input_ids"]))


# per-answer properties of thinking-model GSM8K answers (Section 4.3, Figure 3)
TEXT = {"noanswer": lambda r: float("\\boxed" not in r["resps"][0][0]),          # no final (boxed) answer
        "ntok": lambda r: ntok(r["resps"][0][0]),
        "marks": lambda r: float(len(MARK.findall(r["resps"][0][0]))),             # hesitation markers
        "steps": lambda r: float(len([x for x in r["doc"]["answer"].split("####")[0].split("\n") if x.strip()]))}


def read_lm(runs, run, metric, filt=None):
    out = {}
    for f in glob.glob(f"{runs}/lmeval/out/{run}/*/samples_*.jsonl"):
        task = os.path.basename(f).split("samples_")[1].rsplit("_", 1)[0]
        for line in open(f):
            r = json.loads(line)
            if filt and r.get("filter") != filt:
                continue
            if metric == "mvfull":  # math_verify on the full answer against the reference answer (MATH-500)
                out[f"{task}/{r['doc_id']}"] = mvfull(r["doc"]["answer"], r["resps"][0][0])
            elif metric in TEXT:
                out[f"{task}/{r['doc_id']}"] = TEXT[metric](r)
            elif metric in r:
                out[f"{task}/{r['doc_id']}"] = float(r[metric])
    return out


def load_raw(runs, key):
    kind, name, *rest = key.split(":")
    if kind == "L":
        return read_lm(runs, name, rest[0], rest[1] if len(rest) > 1 else None)
    if kind == "T":
        out = {}
        for s in ("s0", "s1", "s2", "s3"):
            part = read_lm(runs, f"{name}_{s}", rest[0])
            out.update(part)
        return out if all(glob.glob(f"{runs}/lmeval/out/{name}_{s}/*/samples_*.jsonl") for s in ("s0", "s1", "s2", "s3")) else {}
    out = {}
    for f in glob.glob(f"{runs}/bench/{name}/s*/{rest[0]}.jsonl"):
        for line in open(f):
            r = json.loads(line)
            out[str(r["id"])] = float(r["score"] if "score" in r else r["correct"])  # bbh: "correct"
    return out


def keys():
    ks = {a for _, _, a, b, _ in CMP} | {b for _, _, a, b, _ in CMP} | {k for _, _, k in MEANS} | set(WKEYS)
    return sorted(ks)


def export(runs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for k in keys():
        s = load_raw(runs, k)
        if not s:
            missing.append(k)
            continue
        p = out_dir / fname(k)
        if p.exists() and len(json.load(gzip.open(p))["scores"]) >= len(s):
            continue
        with gzip.open(p, "wt") as f:
            json.dump(dict(key=k, n=len(s), scores=s), f, separators=(",", ":"))
    print(f"exported {len(keys()) - len(missing)} item sets to {out_dir}; missing here: {len(missing)}")
    for k in missing:
        print("  missing", k)


def load(key):
    p = ITEMS / fname(key)
    return json.load(gzip.open(p))["scores"] if p.exists() else None


def boot(a, b, reps=5000):
    ids = sorted(set(a) & set(b))
    d = np.array([a[i] - b[i] for i in ids])
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (reps, len(d)))].mean(1)
    return dict(n=len(ids), a=100 * np.mean([a[i] for i in ids]), b=100 * np.mean([b[i] for i in ids]),
                diff=100 * d.mean(), lo=100 * np.percentile(m, 2.5), hi=100 * np.percentile(m, 97.5))


E2E = Path(__file__).resolve().parents[1] / "results" / "e2e"
# (label, model prefix, task, A, B, printed ratio of mean seconds per answer A / B)
RATIOS = [("2.6B-T GSM8K read 4/write 2 vs native 4/4", "think", "gsm", "S42sb", "native4st", (0.55, 0.46, 0.64)),
          ("2.6B-T GSM8K read 4/write 2 vs read 2/write 2", "think", "gsm", "S42sb", "S22sb", (1.01, 0.91, 1.11)),
          ("2.6B-T MATH read 4/write 2 vs native 4/4", "think", "math", "S42sb", "native4st", (0.63, None, None)),
          ("SenseNova GSM8K read 1/write 1 vs native 2/2", "sn", "gsm", "C11", "native2", (0.59, 0.52, 0.68))]


def e2e(model, task, cond):
    """Per-answer records of one configuration; the first record of every file is a warm-up and is dropped."""
    out = {}
    for f in sorted(E2E.glob(f"{model}-{cond}-{task}*.jsonl")):
        rows = [json.loads(line) for line in open(f)][1:]
        out.update({r["id"]: r for r in rows})
    return out


def e2e_table():
    rows = []
    for model, task, conds in (("think", "gsm", ("native1st", "native2st", "native4st", "S41sb", "S22sb", "S42sb")),
                               ("think", "math", ("native2st", "native4st", "S42sb")),
                               ("sn", "gsm", ("native2", "C11", "C21"))):
        recs = {c: e2e(model, task, c) for c in conds}
        common = set.intersection(*(set(r) for r in recs.values() if r))  # same problems for every configuration
        for c in conds:
            r = {i: x for i, x in recs[c].items() if i in common}
            if r:
                rows.append(dict(model=model, task=task, cond=c, n=len(r), acc=round(100 * np.mean([x["correct"] for x in r.values()]), 1),
                                 seconds=round(float(np.mean([x["total_s"] for x in r.values()])), 1),
                                 tokens=round(float(np.mean([x["new_tokens"] for x in r.values()])))))
                print(f"Table 3      {model} {task} {c:10s} acc {rows[-1]['acc']:5.1f}  s/answer {rows[-1]['seconds']:6.1f}  n={len(r)}")
    ratios = []
    for label, model, task, a, b, printed in RATIOS:
        A, B = e2e(model, task, a), e2e(model, task, b)
        ids = sorted(set(A) & set(B))
        ta, tb = np.array([A[i]["total_s"] for i in ids]), np.array([B[i]["total_s"] for i in ids])
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(ids), (5000, len(ids)))
        bs = ta[idx].mean(1) / tb[idx].mean(1)
        r = dict(label=label, n=len(ids), ratio=round(ta.mean() / tb.mean(), 2), lo=round(float(np.percentile(bs, 2.5)), 2),
                 hi=round(float(np.percentile(bs, 97.5)), 2), printed=printed)
        ratios.append(r)
        print(f"Sec 4.2      {label:52s} {r['ratio']:.2f}x [{r['lo']:.2f}, {r['hi']:.2f}]  n={len(ids)}  paper {printed}")
    return rows, ratios


def wander():
    """Section 4.3 / Figure 3: answers without a final answer (overall and by reference-solution steps), length,
    budget hits (>= 4,090 of 4,096 tokens) and hesitation markers per 1,000 generated tokens."""
    out = {}
    for name, p in WANDER.items():
        m = {k: load(f"T:{p}-gsm8k_think:{k}") for k in ("noanswer", "ntok", "marks", "steps", "math_verify")}
        if any(v is None for v in m.values()):
            print(f"Sec 4.3      {name:52s} MISSING")
            continue
        ids = sorted(m["noanswer"])
        bucket = lambda i: 0 if m["steps"][i] <= 2 else 1 if m["steps"][i] <= 4 else 2  # noqa: E731
        by = [[i for i in ids if bucket(i) == b] for b in range(3)]
        tok = sum(m["ntok"][i] for i in ids)
        out[name] = dict(n=len(ids), acc=round(100 * np.mean([m["math_verify"][i] for i in ids]), 1),
                         no_answer=round(100 * np.mean([m["noanswer"][i] for i in ids]), 1),
                         budget_hit=round(100 * np.mean([m["ntok"][i] >= 4090 for i in ids]), 1),
                         tokens=round(tok / len(ids)), markers_per_1k=round(1000 * sum(m["marks"][i] for i in ids) / tok, 1),
                         bucket_n=[len(b) for b in by],
                         no_answer_by_steps=[round(100 * np.mean([m["noanswer"][i] for i in b]), 1) for b in by])
        print(f"Sec 4.3      {name:52s} {json.dumps(out[name])}")
    return out


def compute():
    res, bad = [], 0
    for sec, label, ka, kb, printed in CMP:
        a, b = load(ka), load(kb)
        if a is None or b is None:
            print(f"{sec:12s} {label:52s} MISSING {ka if a is None else kb}")
            continue
        r = boot(a, b)
        r = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}
        flag = ""
        if printed:
            got = (r["diff"], r["lo"], r["hi"])
            if any(p is not None and abs(p - g) > 0.051 for p, g in zip(printed, got)):
                flag, bad = "  !! paper: " + str(printed), bad + 1
        print(f"{sec:12s} {label:52s} {r['diff']:+6.1f} [{r['lo']:+.1f}, {r['hi']:+.1f}]  n={r['n']}  "
              f"({r['a']:.1f} vs {r['b']:.1f}){flag}")
        res.append(dict(section=sec, label=label, key_a=ka, key_b=kb, printed=printed, **r))
    means = []
    for sec, label, k in MEANS:
        s = load(k)
        if s is None:
            print(f"{sec:12s} {label:52s} MISSING")
            continue
        means.append(dict(section=sec, label=label, key=k, n=len(s), mean=round(100 * float(np.mean(list(s.values()))), 1)))
        print(f"{sec:12s} {label:52s} {means[-1]['mean']:6.1f}  n={len(s)}")
    e2e_rows, ratios = e2e_table()
    OUT.write_text(json.dumps(dict(comparisons=res, means=means, e2e=e2e_rows, e2e_ratios=ratios, wander=wander()), indent=1))
    print(f"wrote {OUT}; {bad} printed values differ")


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export(sys.argv[2], Path(sys.argv[3]) if len(sys.argv) > 3 else ITEMS)
    else:
        compute()
