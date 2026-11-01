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
EQUAL = [("GSM8K", "T:T26-S42-gsm8k_think:math_verify", "T:think-native{n}-gsm8k_think:math_verify"),
         ("MATH-500", "T:T26-S42-math500_think:math_verify", "T:think-native{n}-math500_think:math_verify"),
         ("MMLU-Pro", "L:T26x-S42-mmlupro:exact_match", "L:T26x-native{n}-mmlupro:exact_match"),
         ("ARC-C", "L:T26x-S42-arc:acc_norm", "L:T26x-native{n}-arc:acc_norm"),
         ("BBH", "L:T26x-S42-bbh:exact_match", "L:T26x-native{n}-bbh:exact_match"),
         ("HumanEval", "L:T26x-S42-code:pass@1", "L:T26x-native{n}-code:pass@1"),
         ("MBPP+", "L:T26x-S42-code:pass_at_1", "L:T26x-native{n}-code:pass_at_1"),
         ("HELMET", "B:T26ne-S42-helmet:helmet", "B:T26ne-native{n}-helmet:helmet")]

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
    ("Table 2", "Ouro-2.6B-T HELMET", "B:T26ne-S41-helmet:helmet", "B:T26ne-S11-helmet:helmet", (7.7, 2.1, 13.6)),
    ("Table 2", "Ouro-2.6B-T GSM8K", "T:T26-S41-gsm8k_think:math_verify", "T:T26-S11-gsm8k_think:math_verify", (6.6, 4.2, 9.2)),
    ("Table 2", "Ouro-1.4B-T MMLU-Pro", "L:T14x-S41-mmlupro:exact_match", "L:T14x-S11-mmlupro:exact_match", (17.1, 12.4, 22.4)),
    ("Table 2", "Ouro-1.4B-T HELMET", "B:T14ne-S41-helmet:helmet", "B:T14ne-S11-helmet:helmet", (4.6, 0.1, 8.6)),
    ("Table 2", "Ouro-1.4B-T GSM8K", "T:T14-S41-gsm8k_think:math_verify", "T:T14-S11-gsm8k_think:math_verify", (13.6, 10.5, 16.6)),
    ("Table 2", "SenseNova MMLU-Pro", "L:SN-C21-mmlupro:exact_match", "L:SN-C11-mmlupro:exact_match", (5.5, 2.9, 8.1)),
    ("Table 2", "SenseNova HELMET", "B:SNnt-C21-helmet:helmet", "B:SNnt-C11-helmet:helmet", (3.5, 0.8, 6.1)),
    ("Table 2", "SenseNova BBH", "B:SNnt-C21-bbh:bbh", "B:SNnt-C11-bbh:bbh", (7.0, 5.9, 8.0)),
    ("Table 5", "2.6B write 1 RULER-4K", ruler("X26-S41", 4), ruler("X26-S11", 4), (-3.8, -8.2, 0.3)),
    ("Table 5", "2.6B write 1 MATH 4-shot", "L:o26x-S41-math:mvfull", "L:o26x-S11-math:mvfull", (-11.2, -15.4, -6.8)),
    ("Table 5", "2.6B write 1 MATH 0-shot", "L:o26x-S41-math0shot:mvfull", "L:o26x-S11-math0shot:mvfull", (7.0, 3.0, 10.8)),
    ("Table 5", "2.6B write 2 MATH 4-shot", "L:o26x-S42-math:mvfull", "L:o26x-S22-math:mvfull", None),
    ("Table 5", "2.6B write 2 MATH 0-shot", "L:o26x-S42-math0shot:mvfull", "L:o26x-S22-math0shot:mvfull", (7.0, 3.0, 11.2)),
    ("Table 5", "1.4B write 1 MATH 4-shot", "L:o14x-S41-math4shot:mvfull", "L:o14x-S11-math4shot:mvfull", (-4.0, -8.2, 0.0)),
    ("Table 5", "1.4B write 1 MATH 0-shot", "L:o14x-S41-math0shot:mvfull", "L:o14x-S11-math0shot:mvfull", (4.8, 1.4, 8.4)),
    *[("No-map control", f"maps frozen (same recipe) {c} {name}", ka.format(c=c), kb.format(c=c), None)
      for c in ("S41", "S42", "S11", "S22")
      for name, ka, kb in (("GSM8K", gsm("X26fz-{c}"), gsm("X26-{c}")),
                           ("RULER-4K", "B:X26fz-{c}f:ruler4k", "B:X26-{c}f:ruler4k"),
                           ("RULER-16K", "B:X26fz-{c}f:ruler16k", "B:X26-{c}f:ruler16k"),
                           ("MMLU", mmlu("X26fz-{c}"), mmlu("X26-{c}")))],
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
    ("Sec 4.1", "2.6B-T HELMET read 4/write 2 vs native 4/4", "B:T26ne-S42-helmet:helmet", "B:T26ne-native4-helmet:helmet",
     (-8.7, -13.3, -4.1)),
    ("Sec 4.1", "2.6B-T HELMET read 4/write 2 vs native 2/2", "B:T26ne-S42-helmet:helmet", "B:T26ne-native2-helmet:helmet",
     None),
    ("App A", "2.6B-T answer once vs per-loop answer banks, read 4/write 2 GSM8K (full)",
     "T:T26-O-universal-O42-gsm8k_think:math_verify", "T:T26-S42-gsm8k_think:math_verify", (0.5, -0.5, 1.6)),
    *[("HELMET by category", f"{lab} {cat}", f"B:{x}:helmet#{cat}", f"B:{y}:helmet#{cat}", None)
      for lab, x, y in (("2.6B-T read 4/write 2 vs native 4/4", "T26ne-S42-helmet", "T26ne-native4-helmet"),
                        ("2.6B-T read 4/write 2 vs native 2/2", "T26ne-S42-helmet", "T26ne-native2-helmet"),
                        ("2.6B-T read 4 vs 1 (write 1)", "T26ne-S41-helmet", "T26ne-S11-helmet"),
                        ("1.4B-T read 4 vs 1 (write 1)", "T14ne-S41-helmet", "T14ne-S11-helmet"),
                        ("SenseNova read 2 vs 1 (write 1)", "SNnt-C21-helmet", "SNnt-C11-helmet"))
      for cat in ("rag", "rerank", "icl", "recall")],
    *[("Equal decode cost", f"2.6B-T read 4/write 2 vs native {n}/{n} {bench}", ka.format(c="S42", r=f"T26x"),
       kb.format(n=n), None)
      for n in (2, 4)
      for bench, ka, kb in (
          ("GSM8K", "T:T26-S42-gsm8k_think:math_verify", "T:think-native{n}-gsm8k_think:math_verify"),
          ("MATH-500", "T:T26-S42-math500_think:math_verify", "T:think-native{n}-math500_think:math_verify"),
          ("MMLU-Pro", "L:T26x-S42-mmlupro:exact_match", "L:T26x-native{n}-mmlupro:exact_match"),
          ("ARC-C", "L:T26x-S42-arc:acc_norm", "L:T26x-native{n}-arc:acc_norm"),
          ("BBH", "L:T26x-S42-bbh:exact_match", "L:T26x-native{n}-bbh:exact_match"),
          ("HumanEval", "L:T26x-S42-code:pass@1", "L:T26x-native{n}-code:pass@1"),
          ("MBPP+", "L:T26x-S42-code:pass_at_1", "L:T26x-native{n}-code:pass_at_1"),
          ("HELMET", "B:T26ne-S42-helmet:helmet", "B:T26ne-native{n}-helmet:helmet"))],
    # the control for the headline: the same connector (same weights, same training) at read 2 / write 2
    *[("Same connector", f"2.6B-T read 4/write 2 vs read 2/write 2 {bench}", ka, ka.replace("S42", "S22"),
       {"HELMET": (7.5, 1.4, 13.8), "ARC-C": (2.3, 0.9, 3.7), "BBH": (2.2, 0.1, 4.3), "MMLU-Pro": (8.1, 4.0, 12.1)}.get(bench))
      for bench, ka, _ in EQUAL],
    *[("Pareto", f"read 4/write {d} vs native {d}/{d} {name}", ka.format(d=d), kb.format(d=d), None)
      for d in range(1, 5)
      for name, ka, kb in (("MMLU (20/subject)", "L:X26-S4{d}-mmlu:acc", "L:o26n{d}-mmlu:acc"),
                           ("RULER-16K (260)", "B:X26-S4{d}:ruler16k", "B:X26-n{d}:ruler16k"),
                           ("GSM8K (400)", f"L:X26-S4{{d}}-gsm:{GF}", f"L:o26n{{d}}-gsm:{GF}"),
                           ("GSM8K Thinking (full)", "T:T26-S4{d}-gsm8k_think:math_verify",
                            "T:think-native{d}-gsm8k_think:math_verify"))],
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
MEANS += [("Figure pareto", f"GSM8K full think-native{d}", f"T:think-native{d}-gsm8k_think:math_verify") for d in (1, 2, 3)]
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
    if kind == "T":  # all shards <name>_s0, _s1, ... (GSM8K: 4, MATH-500: 5); empty unless every shard has samples
        shards = sorted({d.rsplit("_", 1)[1] for d in glob.glob(f"{runs}/lmeval/out/{name}_s[0-9]")})
        if not shards or not all(glob.glob(f"{runs}/lmeval/out/{name}_{s}/*/samples_*.jsonl") for s in shards):
            return {}
        out = {}
        for s in shards:
            out.update(read_lm(runs, f"{name}_{s}", rest[0]))
        return out
    out = {}
    for f in glob.glob(f"{runs}/bench/{name}/s*/{rest[0]}.jsonl"):
        for line in open(f):
            r = json.loads(line)
            out[str(r["id"])] = float(r["score"] if "score" in r else r["correct"])  # bbh: "correct"
    return out


def keys():
    ks = {a for _, _, a, b, _ in CMP} | {b for _, _, a, b, _ in CMP} | {k for _, _, k in MEANS} | set(WKEYS)
    return sorted({k.partition("#")[0] for k in ks})


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
    """Item scores of `key`; `key#category` keeps one HELMET category (rag, recall, rerank, icl)."""
    key, _, cat = key.partition("#")
    p = ITEMS / fname(key)
    if not p.exists():
        return None
    scores = json.load(gzip.open(p))["scores"]
    return {i: v for i, v in scores.items() if i.split("/")[1].split("_")[0] == cat} if cat else scores


def helmet_boot(a, b, reps=5000):
    """HELMET as its authors aggregate it: mean per dataset, then per category (RAG, recall, re-rank, ICL), then the
    mean over categories; bootstrap resamples items within each dataset."""
    ids = sorted(set(a) & set(b))
    groups = {}
    for i in ids:
        task = i.split("/")[1]
        groups.setdefault(task.split("_")[0], {}).setdefault(task, []).append(i)
    rng = np.random.default_rng(0)
    da, db, dd = 0.0, 0.0, np.zeros(reps)
    for cat, tasks in groups.items():
        for task, items in tasks.items():
            w = 1 / (len(groups) * len(tasks))
            x = np.array([a[i] for i in items]); y = np.array([b[i] for i in items])
            da += w * x.mean(); db += w * y.mean()
            dd += w * (x - y)[rng.integers(0, len(items), (reps, len(items)))].mean(1)
    return dict(n=len(ids), a=100 * da, b=100 * db, diff=100 * (da - db), lo=100 * np.percentile(dd, 2.5),
                hi=100 * np.percentile(dd, 97.5))


def boot(a, b, reps=5000, groups=None):
    """Paired bootstrap of mean(a - b); with `groups` (item -> group), resample groups (LevBench record groups)."""
    ids = sorted(set(a) & set(b), key=str)
    if ids and all(str(i).startswith("helmet/") for i in ids):
        return helmet_boot(a, b, reps)
    d = np.array([a[i] - b[i] for i in ids])
    rng = np.random.default_rng(0)
    if groups is None:
        m = d[rng.integers(0, len(d), (reps, len(d)))].mean(1)
    else:
        g = sorted({groups[i] for i in ids}); pos = {x: j for j, x in enumerate(g)}
        gi = np.array([pos[groups[i]] for i in ids])
        sums, cnts = np.bincount(gi, d, len(g)), np.bincount(gi, None, len(g))
        draws = rng.integers(0, len(g), (reps, len(g)))
        m = sums[draws].sum(1) / cnts[draws].sum(1)
    return dict(n=len(ids), a=100 * np.mean([a[i] for i in ids]), b=100 * np.mean([b[i] for i in ids]),
                diff=100 * d.mean(), lo=100 * np.percentile(m, 2.5), hi=100 * np.percentile(m, 97.5))


E2E = Path(__file__).resolve().parents[1] / "results" / "e2e"
# (label, model prefix, task, A, B, printed ratio of mean seconds per answer A / B)
RATIOS = [("2.6B-T GSM8K read 4/write 2 vs native 4/4", "think", "gsm", "S42sb", "native4st", (0.55, 0.46, 0.64)),
          ("2.6B-T GSM8K read 4/write 2 vs read 2/write 2", "think", "gsm", "S42sb", "S22sb", (1.01, 0.91, 1.11)),
          ("2.6B-T MATH read 4/write 2 vs native 4/4", "think", "math", "S42sb", "native4st", (0.63, None, None)),
          ("2.6B-T MATH read 4/write 2 vs native 2/2", "think", "math", "S42sb", "native2st", (1.22, None, None)),
          ("SenseNova GSM8K read 1/write 1 vs native 2/2", "sn", "gsm", "C11", "native2", (0.59, 0.52, 0.68)),
          ("2.6B-T GSM8K answer once vs per-loop answer banks (read 4/write 2, batch 1)", "think", "gsm", "O42sb", "S42sb", None)]


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
                               ("sn", "gsm", ("native2", "C11", "C21")),
                               ("think", "gsm", ("S42sb", "O42sb"))):  # appendix: answer stored once
        recs = {c: e2e(model, task, c) for c in conds}
        common = set.intersection(*(set(r) for r in recs.values() if r))  # same problems for every configuration
        for c in conds:
            r = {i: x for i, x in recs[c].items() if i in common}
            if r:
                rows.append(dict(model=model, task=task, cond=c, n=len(r), acc=round(100 * np.mean([x["correct"] for x in r.values()]), 1),
                                 seconds=round(float(np.mean([x["total_s"] for x in r.values()])), 1),
                                 tokens=round(float(np.mean([x["new_tokens"] for x in r.values()]))),
                                 ms_per_token=round(float(np.mean([x["ms_per_token"] for x in r.values()])), 1)))
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


LEV = Path(__file__).resolve().parents[1] / "results" / "lev" / "levbench"
# (label, A rows dir, B rows dir): accuracy at every decision depth, A minus B (LevBench development, Lev's depth_rows)
LEV_CMP = [("Lev-1.4B read-deep trained vs same-training control", "rd4-1.4B-read4", "ctl-1.4B-read0"),
           ("Lev-1.4B control vs original", "ctl-1.4B-read0", "Lev-1.4B-read0"),
           ("Lev-1.4B training-free read 4 vs native", "Lev-1.4B-read4", "Lev-1.4B-read0"),
           ("Lev-1.4B read-deep trained, read 4 vs read 2 (read 2 not trained)", "rd4-1.4B-read4", "rd4-1.4B-read2"),
           ("Lev-2.6B read-deep trained vs same-training control", "rd4-2.6B-read4", "ctl-2.6B-read0"),
           ("Lev-2.6B training-free read 4 vs native", "Lev-2.6B-read4", "Lev-2.6B-read0"),
           ("Lev-2.6B control vs original", "ctl-2.6B-read0", "Lev-2.6B-read0"),
           ("Lev-3B read-deep trained vs same-training control", "rd2-3B-read2", "ctl-3B-read0"),
           *[(f"{m} seed {k} read-deep trained vs same-training control", f"{a}-s{k}-read{r}", f"ctl-{b}-s{k}-read0")
             for m, a, b, r in (("Lev-1.4B", "rd4-1.4B", "1.4B", 4), ("Lev-2.6B", "rd4-2.6B", "2.6B", 4), ("Lev-3B", "rd2-3B", "3B", 2))
             for k in (1, 2, 3)],
           # compute check: the control trained for two epochs (twice the updates; our read-deep training forward
           # costs about twice the control's per update)
           ("Lev-1.4B read-deep trained vs control trained 2x", "rd4-1.4B-read4", "ctl2x-1.4B-read0"),
           ("Lev-2.6B read-deep trained vs control trained 2x", "rd4-2.6B-read4", "ctl2x-2.6B-read0"),
           ("Lev-3B read-deep trained vs control trained 2x", "rd2-3B-read2", "ctl2x-3B-read0")]


def lev():
    """Section on decision models: per-depth accuracy (argmax of the depth's logits == label) on LevBench's answerable
    questions (as Lev: unanswerable and report-only rows have uniform targets and a placeholder label), paired bootstrap
    over record groups (as Lev's analysis/adaptive_depth.py)."""
    def rows(d):
        f = sorted((LEV / d).glob("*.json"))
        if not f:
            return None
        return {(r["id"], r.get("question")): r for r in json.loads(f[0].read_text())["rows"]
                if not r.get("unanswerable") and not r.get("report_only")}
    out = []
    for label, a, b in LEV_CMP:
        A, B = rows(a), rows(b)
        if A is None or B is None:
            print(f"Lev          {label:52s} MISSING"); continue
        ids = sorted(set(A) & set(B), key=str)
        for k in range(1, len(A[ids[0]]["logits"]) + 1):
            ca = {i: float(int(np.argmax(A[i]["logits"][k - 1])) == A[i]["label"]) for i in ids}
            cb = {i: float(int(np.argmax(B[i]["logits"][k - 1])) == B[i]["label"]) for i in ids}
            r = {x: (round(v, 1) if isinstance(v, float) else v) for x, v in boot(ca, cb, groups={i: A[i]["group"] for i in ids}).items()}
            out.append(dict(label=label, depth=k, **r))
            print(f"Lev          {label:52s} d={k} {r['diff']:+5.1f} [{r['lo']:+.1f}, {r['hi']:+.1f}] ({r['a']:.1f} vs {r['b']:.1f})")
    # training seeds (seed 0 = the unsuffixed runs): per-seed gain at each decision depth, mean and sample std
    for m in ("Lev-1.4B", "Lev-2.6B", "Lev-3B"):
        main = f"{m} read-deep trained vs same-training control"
        labels = [main] + [f"{m} seed {k} read-deep trained vs same-training control" for k in (1, 2, 3)]
        for k in sorted({r["depth"] for r in out if r["label"] == main}):
            g = [r["diff"] for lb in labels for r in out if r["label"] == lb and r["depth"] == k]
            if len(g) > 1:
                out.append(dict(label=f"{m} seeds", depth=k, seeds=len(g), mean=round(float(np.mean(g)), 1),
                                sd=round(float(np.std(g, ddof=1)), 1), min=min(g), max=max(g)))
                print(f"Lev seeds    {m:52s} d={k} {len(g)} seeds: mean {np.mean(g):+5.1f} sd {np.std(g, ddof=1):.1f} "
                      f"range [{min(g):+.1f}, {max(g):+.1f}]")
    return out




def equal():
    """Figure 4: read 4 / write 2 vs native 2/2 and native 4/4 (Ouro-2.6B-Thinking), all three on the items every one
    of them has, so the three scores and both paired differences share one item set."""
    out = []
    for bench, ka, kb in EQUAL:
        o, n2, n4 = load(ka), load(kb.format(n=2)), load(kb.format(n=4))
        if o is None or n2 is None or n4 is None:
            print(f"Equal        {bench:12s} MISSING"); continue
        s22 = load(ka.replace("S42", "S22"))   # the same connector reading 2 loops (same weights, same training)
        ids = set(o) & set(n2) & set(n4)
        if s22 is not None and not ids <= set(s22):   # not yet on every shared item: report it alone, keep the triple
            print(f"Equal        {bench:12s} same connector 2/2 on {len(ids & set(s22))} of {len(ids)} shared items; not used")
            s22 = None
        o, n2, n4 = ({i: x[i] for i in ids} for x in (o, n2, n4))
        r2, r4 = boot(o, n2), boot(o, n4)
        row = dict(bench=bench, n=len(ids), ours=round(r2["a"], 1), native2=round(r2["b"], 1), native4=round(r4["b"], 1),
                   vs2=[round(r2[k], 1) for k in ("diff", "lo", "hi")], vs4=[round(r4[k], 1) for k in ("diff", "lo", "hi")])
        if s22 is not None:
            r22 = boot(o, {i: s22[i] for i in ids})
            row.update(same22=round(r22["b"], 1), vs22=[round(r22[k], 1) for k in ("diff", "lo", "hi")])
        out.append(row)
        print(f"Equal        {bench:12s} n={len(ids):5d} ours {row['ours']:5.1f} | 2/2 {row['native2']:5.1f} "
              f"{row['vs2'][0]:+5.1f} [{row['vs2'][1]:+.1f}, {row['vs2'][2]:+.1f}] | 4/4 {row['native4']:5.1f} "
              f"{row['vs4'][0]:+5.1f} [{row['vs4'][1]:+.1f}, {row['vs4'][2]:+.1f}]"
              + (f" | same 2/2 {row['same22']:5.1f} {row['vs22'][0]:+5.1f} [{row['vs22'][1]:+.1f}, {row['vs22'][2]:+.1f}]"
                 if "same22" in row else ""))
    return out


def throughput():
    """Table in Appendix (decode throughput) and the ratios in Sec 4.2: best tokens/s over batch sizes, 8K context."""
    d = Path(__file__).resolve().parents[1] / "results" / "throughput-sb4"
    best = {}
    for f in sorted(d.glob("*.json")):
        rows = [r for r in json.loads(f.read_text()) if "tokens_per_s" in r and r["context"] == 8192]
        if rows:
            b = max(rows, key=lambda r: r["tokens_per_s"])
            best[f.stem] = dict(tokens_per_s=round(b["tokens_per_s"]), batch=b["batch"])
    for k, v in best.items():
        print(f"Throughput   {k:52s} {v['tokens_per_s']:6d} tok/s (batch {v['batch']})")
    for a, b in (("S42sb", "native2st"), ("S42sb", "native4st")):
        print(f"Throughput   {a} / {b}: {best[a]['tokens_per_s'] / best[b]['tokens_per_s']:.2f}x")
    return best


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
    OUT.write_text(json.dumps(dict(comparisons=res, means=means, e2e=e2e_rows, e2e_ratios=ratios, wander=wander(), throughput=throughput(), lev=lev(), equal=equal()), indent=1))
    print(f"wrote {OUT}; {bad} printed values differ")


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export(sys.argv[2], Path(sys.argv[3]) if len(sys.argv) > 3 else ITEMS)
    else:
        compute()
