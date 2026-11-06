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
                                MATH-500 for base models: boxed = math_verify on the final (last boxed) answer)
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
GF = "exact_match:strict-match"   # GSM8K 3-shot CoT, strict match (Ouro, Table 16)


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
         ("HumanEval+", "P:T26x-S42-evalplus:humaneval_chat:plus", "P:T26x-native{n}-evalplus:humaneval_chat:plus"),
         ("MBPP+", "P:T26x-S42-evalplus:mbpp_chat:plus", "P:T26x-native{n}-evalplus:mbpp_chat:plus"),
         ("HELMET", "B:T26ne-S42-helmet:helmet", "B:T26ne-native{n}-helmet:helmet")]

# SenseNova-Loop-9B headline at decode 1 ({c}: native1, C11, C21, native2)
SN_EQUAL = [("GSM8K", "T:SN-{c}-gsm8k_think:math_verify"), ("MATH-500", "T:SN-{c}-math500_think:math_verify"),
            ("MMLU-Pro", "L:SN-{c}-mmlupro:exact_match"), ("ARC-C", "L:SN-{c}-arc:acc_norm"), ("BBH", "B:SNnt-{c}-bbh:bbh"),
            ("HumanEval", "L:SN-{c}-code:pass@1"), ("HELMET", "B:SNnt-{c}-helmet:helmet")]
# MBPP+ is left out for SenseNova: its instruct prompt forces a code prefix, and this thinking model then opens a <think>
# block inside the code in nearly every answer (native 2/2: 375 of 378), so the score measures that format failure for
# every condition (native 2/2 44.4, prefill 2 / decode 1 64.0). HumanEval (completion prompt) has no such answers.
# which part of the hybrid gets the prefill-2 state: attention KV only (A), Gated-DeltaNet state only (G), both ("")
SN_STATE = [("RULER-4K", "B:SNnt-{c}:ruler4k"), ("RULER-8K", "B:SNnt-{c}:ruler8k"), ("RULER-16K", "B:SNnt-{c}:ruler16k"),
            ("MMLU-Pro", "L:SN-{c}-mmlupro:exact_match"), ("ARC-C", "L:SN-{c}-arc:acc_norm"),
            ("HELMET", "B:SNnt-{c}-helmet:helmet")]

# hybrid-state numbers quoted in Appendix "Reading Deep in a Hybrid Model": (kind, mix, benchmark) -> printed
HYBRID_PRINTED = {("untrained", "A", "RULER-16K"): (50.2, None, None), ("untrained", "G", "RULER-16K"): (50.5, None, None),
                  ("untrained", "", "RULER-16K"): (66.6, None, None), ("untrained", "A", "MMLU-Pro"): (23.7, None, None),
                  ("untrained", "G", "MMLU-Pro"): (21.5, None, None), ("untrained", "", "MMLU-Pro"): (36.2, None, None),
                  ("trained", "A", "MMLU-Pro"): (4.6, 2.0, 7.2), ("trained", "G", "MMLU-Pro"): (2.0, -0.4, 4.5),
                  ("trained", "G", "RULER-16K"): (4.9, 2.1, 7.7), ("trained", "A", "RULER-4K"): (-4.5, -7.2, -2.0),
                  ("trained", "", "ARC-C"): (-2.9, -4.8, -1.0), ("trained", "G", "ARC-C"): (-2.7, -4.5, -0.9),
                  ("trained", "A", "ARC-C"): (0.8, None, None), ("trained", "G", "HELMET"): (9.1, 6.5, 11.6),
                  ("trained", "A", "HELMET"): (-0.2, None, None), ("trained", "A", "RULER-16K"): (0.7, None, None)}

SNG_KEYS = [("RULER-4K", "B:SNGnt-{c}:ruler4k"), ("RULER-8K", "B:SNGnt-{c}:ruler8k"), ("RULER-16K", "B:SNGnt-{c}:ruler16k"),
            ("MMLU-Pro", "L:SNG-{c}-mmlupro:exact_match"), ("ARC-C", "L:SNG-{c}-arc:acc_norm"),
            ("BBH", "B:SNGnt-{c}-bbh:bbh"), ("GSM8K", "T:SNG-{c}-gsm8k_think:math_verify"),
            ("HELMET", "B:SNGnt-{c}-helmet:helmet")]

CMP = []
# Table 6 / Section 3 (maps frozen at the identity, same recipe, minus the loop adapter): the values the paper prints
NOMAP = {("S41", "RULER-16K"): (-11.4, -14.7, -8.2), ("S41", "GSM8K"): (2.3, 0.2, 4.5), ("S41", "MMLU"): (-0.2, -0.6, 0.1),
         ("S42", "RULER-16K"): (-2.6, -4.5, -0.7), ("S42", "GSM8K"): (0.2, -1.3, 1.7), ("S22", "RULER-16K"): (-3.4, -5.6, -1.3),
         ("S11", "MMLU"): (-0.4, -0.7, -0.0), ("S22", "MMLU"): (-0.3, -0.5, -0.0)}
# Table 6 (design variants): unrestricted key maps, one prompt KV cache per decode loop, minus the loop adapter
BANK = {"11": (0.1, -1.3, 1.5), "22": (0.6, -0.8, 2.0), "31": (-0.9, -2.9, 1.2), "32": (1.2, -0.4, 2.7), "42": (-1.0, -2.6, 0.5)}
for d, v in ((1, [(15.2, 14.3, 16.1), (8.6, 4.3, 12.8), (8.5, 6.1, 10.9)]),
             (2, [(6.0, 5.4, 6.7), (5.3, 1.3, 9.4), (-0.1, -2.0, 2.0)]),
             (3, [(0.6, 0.2, 1.1), (-0.4, -3.0, 2.2), (-2.0, -3.9, -0.2)])):
    CMP += [("Table 1", f"decode {d} MMLU", mmlu(f"X26-S4{d}"), mmlu(f"X26-S{d}{d}"), v[0]),
            ("Table 1", f"decode {d} RULER-16K", ruler(f"X26-S4{d}", 16), ruler(f"X26-S{d}{d}", 16), v[1]),
            ("Table 1", f"decode {d} GSM8K", gsm(f"X26-S4{d}"), gsm(f"X26-S{d}{d}"), v[2])]
CMP += [
    ("Table 2", "Ouro-1.4B ARC-C", "L:O14x-S41-arc:acc_norm", "L:O14x-S11-arc:acc_norm", (10.8, 8.4, 13.1)),
    ("Table 2", "Ouro-1.4B RULER-16K", ruler("O14-S41", 16), ruler("O14-S11", 16), (1.4, -2.6, 5.4)),
    ("Table 2", "Ouro-1.4B GSM8K", gsm("O14-S41"), gsm("O14-S11"), (5.0, 2.7, 7.2)),
    ("Table 2", "Ouro-2.6B-T MMLU-Pro", "L:T26x-S41-mmlupro:exact_match", "L:T26x-S11-mmlupro:exact_match", (18.6, 13.6, 23.8)),
    ("Table 2", "Ouro-2.6B-T HELMET", "B:T26ne-S41-helmet:helmet", "B:T26ne-S11-helmet:helmet", (7.7, 2.1, 13.6)),
    ("Table 2", "Ouro-2.6B-T GSM8K", "T:T26-S41-gsm8k_think:math_verify", "T:T26-S11-gsm8k_think:math_verify", (6.6, 4.2, 9.2)),
    ("Table 2", "Ouro-1.4B-T MMLU-Pro", "L:T14x-S41-mmlupro:exact_match", "L:T14x-S11-mmlupro:exact_match", (17.1, 12.4, 22.4)),
    ("Table 2", "Ouro-1.4B-T HELMET", "B:T14ne-S41-helmet:helmet", "B:T14ne-S11-helmet:helmet", (4.6, 0.1, 8.6)),
    ("Table 2", "Ouro-1.4B-T GSM8K", "T:T14-S41-gsm8k_think:math_verify", "T:T14-S11-gsm8k_think:math_verify", (13.6, 10.5, 16.6)),
    ("Table 2", "SenseNova MMLU-Pro", "L:SN-C21-mmlupro:exact_match", "L:SN-C11-mmlupro:exact_match", (5.5, 2.9, 8.1)),
    ("Table 2", "SenseNova HELMET", "B:SNnt-C21-helmet:helmet", "B:SNnt-C11-helmet:helmet", (3.5, 0.8, 6.1)),
    ("Table 2", "SenseNova BBH", "B:SNnt-C21-bbh:bbh", "B:SNnt-C11-bbh:bbh", (7.0, 5.9, 8.0)),
    ("Table 5", "2.6B decode 1 RULER-4K", ruler("X26-S41", 4), ruler("X26-S11", 4), (-3.8, -8.2, 0.3)),
    ("Table 5", "2.6B decode 1 MATH 4-shot", "L:o26x-S41-math:boxed", "L:o26x-S11-math:boxed", (-12.0, -16.2, -7.8)),
    ("Table 5", "2.6B decode 1 MATH 0-shot", "L:o26x-S41-math0shot:boxed", "L:o26x-S11-math0shot:boxed", (8.0, 4.2, 11.6)),
    ("Table 5", "2.6B decode 2 MATH 4-shot", "L:o26x-S42-math:boxed", "L:o26x-S22-math:boxed", None),
    ("Table 5", "2.6B decode 2 MATH 0-shot", "L:o26x-S42-math0shot:boxed", "L:o26x-S22-math0shot:boxed", (7.4, 3.4, 11.4)),
    ("Table 5", "1.4B decode 1 MATH 4-shot", "L:o14x-S41-math4shot:boxed", "L:o14x-S11-math4shot:boxed", (-4.4, -8.4, -0.6)),
    ("Table 5", "1.4B decode 1 MATH 0-shot", "L:o14x-S41-math0shot:boxed", "L:o14x-S11-math0shot:boxed", (4.8, 1.2, 8.4)),
    *[("No-map control", f"maps frozen (same recipe) {c} {name}", ka.format(c=c), kb.format(c=c), NOMAP.get((c, name)))
      for c in ("S41", "S42", "S11", "S22")
      for name, ka, kb in (("GSM8K", gsm("X26fz-{c}"), gsm("X26-{c}")),
                           ("RULER-4K", "B:X26fz-{c}f:ruler4k", "B:X26-{c}f:ruler4k"),
                           ("RULER-16K", "B:X26fz-{c}f:ruler16k", "B:X26-{c}f:ruler16k"),
                           ("MMLU", mmlu("X26fz-{c}"), mmlu("X26-{c}")))],
    ("Table 6", "direct sharing 4/1 GSM8K", gsm("X26-Y1"), gsm("X26-S41"), (-11.0, -13.7, -8.3)),
    ("Table 6", "direct sharing 4/2 GSM8K", gsm("X26-Y2"), gsm("X26-S42"), (1.2, -0.9, 3.3)),
    ("Table 6", "answer once 4/2 GSM8K", gsm("X26-O42"), gsm("X26-S42"), (-2.5, -4.5, -0.5)),
    ("Table 6", "answer once 3/3 GSM8K", gsm("X26-O33"), gsm("X26-S33"), (-2.4, -4.2, -0.7)),
    ("Table 6", "answer once 4/4 GSM8K", gsm("X26-O44"), gsm("X26-S44"), (-3.0, -4.8, -1.3)),
    ("Table 4", "answer once 4/4 vs native 4/4 GSM8K", gsm("X26-O44"), f"L:X26-native4-gsmfull:{GF}", None),
    ("Table 6", "first-loop branch 4/1 RULER-4K", ruler("X26-M41", 4), ruler("X26-S41", 4), (2.3, 1.0, 3.6)),
    ("Table 6", "first-loop branch 4/1 GSM8K", gsm("X26-M41"), gsm("X26-S41"), (0.5, -0.9, 2.0)),
]
for c in ("11", "22", "31", "32", "42"):
    CMP.append(("Table 6", f"bank per loop {c[0]}/{c[1]} RULER-16K", ruler(f"X26-V{c}", 16), ruler(f"X26-S{c}", 16), BANK[c]))
# App. two-seed table: printed point differences (GSM8K, MMLU, RULER-4K/8K/16K at decode 1; GSM8K, RULER-4K at decode 2)
SEED_PRINTED = {1: (8.5, 15.2, -3.8, 2.5, 8.6, -0.1, -7.2), 2: (8.9, 15.2, -3.5, 2.0, 8.6, -1.1, -6.8)}
for seed, run in ((1, "X26"), (2, "X26s2")):
    g1, m1, r4, r8, r16, g2, r42 = ((v, None, None) for v in SEED_PRINTED[seed])
    CMP += [("Table 7", f"seed {seed} decode 1 GSM8K", gsm(f"{run}-S41"), gsm(f"{run}-S11"), g1),
            ("Table 7", f"seed {seed} decode 1 MMLU", mmlu(f"{run}-S41"), mmlu(f"{run}-S11"), m1),
            ("Table 7", f"seed {seed} decode 2 GSM8K", gsm(f"{run}-S42"), gsm(f"{run}-S22"), g2)]
    for k, v in ((4, r4), (8, r8), (16, r16)):
        CMP.append(("Table 7", f"seed {seed} decode 1 RULER-{k}K", ruler(f"{run}-S41", k), ruler(f"{run}-S11", k), v))
    CMP.append(("Table 7", f"seed {seed} decode 2 RULER-4K", ruler(f"{run}-S42", 4), ruler(f"{run}-S22", 4), r42))
CMP += [
    ("Sec 4.1", "2.6B-T HELMET prefill 4/decode 2 vs native 4/4", "B:T26ne-S42-helmet:helmet", "B:T26ne-native4-helmet:helmet",
     (-8.7, -13.3, -4.1)),
    ("Sec 4.1", "2.6B-T HELMET prefill 4/decode 2 vs native 2/2", "B:T26ne-S42-helmet:helmet", "B:T26ne-native2-helmet:helmet",
     None),
    ("App A", "2.6B-T answer once vs per-loop answer banks, prefill 4/decode 2 GSM8K (full)",
     "T:T26-O-universal-O42-gsm8k_think:math_verify", "T:T26-S42-gsm8k_think:math_verify", (0.5, -0.5, 1.6)),
    *[("HELMET by category", f"{lab} {cat}", f"B:{x}:helmet#{cat}", f"B:{y}:helmet#{cat}", None)
      for lab, x, y in (("2.6B-T prefill 4/decode 2 vs native 4/4", "T26ne-S42-helmet", "T26ne-native4-helmet"),
                        ("2.6B-T prefill 4/decode 2 vs native 2/2", "T26ne-S42-helmet", "T26ne-native2-helmet"),
                        ("2.6B-T prefill 4 vs 1 (decode 1)", "T26ne-S41-helmet", "T26ne-S11-helmet"),
                        ("1.4B-T prefill 4 vs 1 (decode 1)", "T14ne-S41-helmet", "T14ne-S11-helmet"),
                        ("SenseNova prefill 2 vs 1 (decode 1)", "SNnt-C21-helmet", "SNnt-C11-helmet"))
      for cat in ("rag", "rerank", "icl", "recall")],
    *[("Equal decode cost", f"2.6B-T prefill 4/decode 2 vs native {n}/{n} {bench}", ka.format(c="S42", r=f"T26x"),
       kb.format(n=n), None)
      for n in (2, 4)
      for bench, ka, kb in (
          ("GSM8K", "T:T26-S42-gsm8k_think:math_verify", "T:think-native{n}-gsm8k_think:math_verify"),
          ("MATH-500", "T:T26-S42-math500_think:math_verify", "T:think-native{n}-math500_think:math_verify"),
          ("MMLU-Pro", "L:T26x-S42-mmlupro:exact_match", "L:T26x-native{n}-mmlupro:exact_match"),
          ("ARC-C", "L:T26x-S42-arc:acc_norm", "L:T26x-native{n}-arc:acc_norm"),
          ("BBH", "L:T26x-S42-bbh:exact_match", "L:T26x-native{n}-bbh:exact_match"),
          ("HumanEval+", "P:T26x-S42-evalplus:humaneval_chat:plus", "P:T26x-native{n}-evalplus:humaneval_chat:plus"),
          ("MBPP+", "P:T26x-S42-evalplus:mbpp_chat:plus", "P:T26x-native{n}-evalplus:mbpp_chat:plus"),
          ("HELMET", "B:T26ne-S42-helmet:helmet", "B:T26ne-native{n}-helmet:helmet"))],
    *[("Hybrid state", f"SenseNova {kind} prefill-2 state {part} vs prefill 1 {bench}", key.format(c=deep + mix), key.format(c=shallow),
       HYBRID_PRINTED.get((kind, mix, bench)))
      for kind, deep, shallow in (("untrained", "splice21", "native1"), ("trained", "C21", "C11"))
      for mix, part in (("", "both"), ("A", "attention KV"), ("G", "GDN state"))
      for bench, key in SN_STATE],
    *[("Hybrid state", f"SenseNova trained prefill-2 GDN state only vs both {bench}", key.format(c="C21G"), key.format(c="C21"),
       {"RULER-4K": (6.5, None, None), "RULER-8K": (4.2, None, None), "RULER-16K": (3.7, None, None),
        "MMLU-Pro": (-3.5, None, None), "HELMET": (5.6, None, None)}.get(bench))
      for bench, key in SN_STATE],
    ("Hybrid state", "SenseNova trained prefill-2 GDN state only vs native 2/2 HELMET", "B:SNnt-C21G-helmet:helmet",
     "B:SNnt-native2-helmet:helmet", (-11.2, -13.0, -9.4)),
    # SenseNova adapter trained with the GDN-only prefill-2 state (SNG-*, --mix G): prefill loops within it, and against the
    # original adapter trained with both parts at prefill 2 (SN-* C21)
    *[("Hybrid G-trained", f"SenseNova G-trained {name} {bench}", key.format(c="C21G"), key.format(c=ref),
       {"HELMET": (7.2, 4.7, 9.7), "RULER-16K": (6.1, 3.2, 9.1), "ARC-C": (-2.8, -4.7, -0.9)}.get(bench))
      for name, ref in (("prefill 2 (GDN) vs prefill 1", "C11"),)
      for bench, key in SNG_KEYS],
    *[("Hybrid G-trained", f"SenseNova G-trained prefill 2 (GDN) vs original both-layer prefill 2 {bench}", key.format(c="C21G"),
       key.replace("SNGnt-", "SNnt-").replace("SNG-", "SN-").format(c="C21"),
       {"HELMET": (4.4, 2.7, 6.2), "RULER-4K": (6.7, None, None), "RULER-8K": (2.8, None, None),
        "RULER-16K": (2.9, None, None), "MMLU-Pro": (-2.7, -5.0, -0.3)}.get(bench)) for bench, key in SNG_KEYS],
    # fewer prefill than decode loops (m < n) on the tasks where more prefill loops hurt: same adapter, same n
    *[("Fewer prefill loops", f"{model} prefill {lo} vs {hi} (decode {n}) {bench}", key.format(c=f"S{lo}{n}"),
       key.format(c=f"S{hi}{n}"), (17.2, 13.0, 21.4) if (bench, lo, hi, n) == ("MATH-500 4-shot", 1, 4, 4) else None)
      for model, bench, key in (("Ouro-2.6B", "MATH-500 4-shot", "L:o26x-{c}-math:boxed"),
                                ("Ouro-2.6B", "RULER-4K", "B:X26-{c}f:ruler4k"), ("Ouro-2.6B", "RULER-8K", "B:X26-{c}f:ruler8k"),
                                ("Ouro-2.6B", "RULER-16K", "B:X26-{c}f:ruler16k"),
                                ("Ouro-1.4B-T", "BBH 3-shot", "L:T14x-{c}-bbh:exact_match"))
      for lo, hi, n in ((1, 2, 2), (1, 4, 4), (2, 4, 4), (1, 4, 2))],
    *[("Fewer prefill loops", f"Ouro-2.6B prefill {m} / decode {n} vs native {k}/{k} MATH-500 4-shot", f"L:o26x-S{m}{n}-math:boxed",
       f"L:o26x-native{k}-math:boxed", (11.4, 7.2, 15.6) if (m, n, k) == (1, 2, 4) else None)
      for m, n, k in ((1, 2, 2), (1, 2, 4), (1, 4, 4), (2, 4, 4), (1, 3, 4))],
    # Ouro-2.6B-Thinking at decode 2: prefill 1 against prefill 4 (same adapter) and against native 2/2
    *[("Thinking decode 2", f"2.6B-T prefill 1 / decode 2 vs prefill 4 / decode 2 {bench}", ka.replace("S42", "S12"), ka, None)
      for bench, ka, _ in EQUAL],
    *[("Thinking decode 2", f"2.6B-T prefill 1 / decode 2 vs native 2/2 {bench}", ka.replace("S42", "S12"), kb.format(n=2), None)
      for bench, ka, kb in EQUAL],
    # distillation baseline (arm D: native 2/2 layout, LoRA only, same data / tokens / teacher as the adapter)
    *[("Distillation baseline", f"2.6B-T prefill 4/decode 2 vs LoRA-distilled native 2/2 {bench}", ka, ka.replace("S42", "Df"),
       {"MMLU-Pro": (6.0, 1.7, 10.2), "ARC-C": (3.2, 1.5, 4.8)}.get(bench))
      for bench, ka, _ in EQUAL],
    *[("Distillation baseline", f"2.6B-T LoRA-distilled native 2/2 vs native 2/2 {bench}", ka.replace("S42", "Df"), kb.format(n=2), None)
      for bench, ka, kb in EQUAL],
    # the control for the headline: the same adapter (same weights, same training) at prefill 2 / decode 2
    *[("Same adapter", f"2.6B-T prefill 4/decode 2 vs prefill 2/decode 2 {bench}", ka, ka.replace("S42", "S22"),
       {"HELMET": (7.5, 1.4, 13.8), "ARC-C": (2.3, 0.9, 3.7), "BBH": (2.2, 0.1, 4.3), "MMLU-Pro": (8.1, 4.0, 12.1)}.get(bench))
      for bench, ka, _ in EQUAL],
    *[("Pareto", f"prefill 4/decode {d} vs native {d}/{d} {name}", ka.format(d=d), kb.format(d=d), None)
      for d in range(1, 5)
      for name, ka, kb in (("MMLU (20/subject)", "L:X26-S4{d}-mmlu:acc", "L:o26n{d}-mmlu:acc"),
                           ("RULER-16K (260)", "B:X26-S4{d}:ruler16k", "B:X26-n{d}:ruler16k"),
                           ("GSM8K (400)", f"L:X26-S4{{d}}-gsm:{GF}", f"L:o26n{{d}}-gsm:{GF}"),
                           ("GSM8K Thinking (full)", "T:T26-S4{d}-gsm8k_think:math_verify",
                            "T:think-native{d}-gsm8k_think:math_verify"))],
    ("Sec 4.2", "2.6B-T prefill 4/decode 2 vs native 4/4 GSM8K (full)", "T:T26-S42-gsm8k_think:math_verify",
     "T:think-native4-gsm8k_think:math_verify", (-1.0, -2.0, 0.1)),
    ("Sec 4.2", "SenseNova prefill 1/decode 1 vs native 2/2 GSM8K (full)", "T:SN-C11-gsm8k_think:math_verify",
     "T:SN-native2-gsm8k_think:math_verify", None),
    ("Limitations", "decode 3 RULER-4K vs native 3/3", ruler("X26-S43", 4), "B:X26-n3f:ruler4k", (-10.4, None, None)),
    ("Limitations", "decode 3 RULER-8K vs native 3/3", ruler("X26-S43", 8), "B:X26-n3f:ruler8k", (-9.5, None, None)),
    ("Limitations", "decode 3 RULER-16K vs native 3/3", ruler("X26-S43", 16), "B:X26-n3f:ruler16k", (-5.3, None, None)),
    ("App B", "Nanbeige RULER-4K prefill 2 vs 1 (decode 1)", "B:NBX-p2d1:ruler4k", "B:NBX-p1d1:ruler4k", (11.8, None, None)),
    ("App B", "Nanbeige RULER-8K", "B:NBX-p2d1:ruler8k", "B:NBX-p1d1:ruler8k", (11.7, None, None)),
    ("App B", "Nanbeige RULER-16K", "B:NBX-p2d1:ruler16k", "B:NBX-p1d1:ruler16k", (14.2, None, None)),
    ("App B", "Nanbeige MMLU", "L:NBX-p2d1-mmlufull:acc", "L:NBX-p1d1-mmlufull:acc", (12.2, None, None)),
    ("App B", "Nanbeige GSM8K", f"L:NBX-p2d1-gsmfull:{GF}", f"L:NBX-p1d1-gsmfull:{GF}", (-8.4, -11.5, -5.3)),
    ("App B", "Nanbeige GSM8K flexible extraction", "L:NBX-p2d1-gsmfull:exact_match:flexible-extract",
     "L:NBX-p1d1-gsmfull:exact_match:flexible-extract", (5.2, None, None)),
]
# means (Figure 1 grid on evaluation subsets)
MEANS = []
# task map: Ouro-2.6B, one loop adapter (X26-S-universal), every (prefill m, decode n) cell and native n/n
# Protocol of Ouro's base-model table (Ouro paper, Appendix C.1, Table 16): MATH-500 and GSM8K strict match, code with
# evalplus (chat prompt; HumanEval+ and MBPP+ = the plus tests), HellaSwag and Winogrande added.
TASK_MAP = [("MATH-500 4-shot", "L:o26x-{c}-math:boxed"), ("BBH 3-shot", "L:o26x-{c}-bbh:exact_match"),
            ("ARC-C", "L:o26x-{c}-arc:acc_norm"), ("MMLU-Pro", "L:o26x-{c}-mmlupro:exact_match"),
            ("HellaSwag", "L:o26x-{c}-hellaswag:acc_norm"), ("Winogrande", "L:o26x-{c}-winogrande:acc"),
            ("HumanEval+", "P:o26x-{c}-evalplus:humaneval_chat:plus"), ("MBPP+", "P:o26x-{c}-evalplus:mbpp_chat:plus"),
            ("GPQA-Diamond", "L:o26x-{c}-gpqa:acc_norm"), ("IFEval", "L:o26x-{c}-ifeval:prompt_level_strict_acc"),
            ("RULER-4K", "B:X26-{c}f:ruler4k"), ("RULER-8K", "B:X26-{c}f:ruler8k"), ("RULER-16K", "B:X26-{c}f:ruler16k")]
# replication on Ouro-1.4B (its own adapter, O14-S-universal), same settings (MMLU and GSM8K subsets)
TASK_MAP_14 = [("MMLU", "L:O14x-{c}-mmlu:acc"), ("GSM8K", "L:O14x-{c}-gsm:" + GF),
               ("MATH-500 4-shot", "L:O14x-{c}-math:boxed"), ("BBH 3-shot", "L:O14x-{c}-bbh:exact_match"),
               ("ARC-C", "L:O14x-{c}-arc:acc_norm"), ("MMLU-Pro", "L:O14x-{c}-mmlupro:exact_match"),
               ("HellaSwag", "L:O14x-{c}-hellaswag:acc_norm"), ("Winogrande", "L:O14x-{c}-winogrande:acc"),
               ("HumanEval+", "P:O14x-{c}-evalplus:humaneval_chat:plus"), ("MBPP+", "P:O14x-{c}-evalplus:mbpp_chat:plus"),
               ("GPQA-Diamond", "L:O14x-{c}-gpqa:acc_norm"), ("IFEval", "L:O14x-{c}-ifeval:prompt_level_strict_acc"),
               ("RULER-4K", "B:O14-{c}f:ruler4k"), ("RULER-8K", "B:O14-{c}f:ruler8k"), ("RULER-16K", "B:O14-{c}f:ruler16k")]
# same-training controls on Ouro-2.6B: LoRA distillation of native n/n (arms E, D, H; launch_x26_distill.sh)
DISTILLED = {1: "E48f", 2: "D48f", 3: "H48f"}   # controls that also supervise the first answer token (--first-target)
DISTILLED_COUNTS = {1: (8, 1, 15), 2: (3, 2, 15), 3: (2, 3, 15)}
DISTILLED_COUNTS_14 = {1: (6, 3, 15), 2: (4, 4, 15)}   # Ouro-1.4B controls X14-E48f, X14-D48f (App. models)
DISTILLED_KEYS = [k.replace("{c}", "{d}") for _, k in TASK_MAP + TASK_MAP_14] + ["L:o26x-{d}-mmlufull:acc",
                                                                                  "L:o26x-{d}-gsmfull:" + GF]
MEANS += [("Task map 1.4B", f"{c} {bench}", key.format(c=c)) for bench, key in TASK_MAP_14
          for c in [f"S{m}{n}" for m in range(1, 5) for n in range(1, 5)] + [f"native{n}" for n in range(1, 5)]]
MEANS += [("Task map", f"{c} {bench}", key.format(c=c)) for bench, key in TASK_MAP
          for c in [f"S{m}{n}" for m in range(1, 5) for n in range(1, 5)] + [f"native{n}" for n in range(1, 5)]]
MEANS += [("Sec 4.2", f"GSM8K full {c}", f"T:{c}-gsm8k_think:math_verify")
          for c in ("T26-S42", "think-native4", "SN-C11", "SN-C21", "SN-native2", "SN-native1", "SN-splice21")]
MEANS += [("Figure pareto", f"GSM8K full think-native{d}", f"T:think-native{d}-gsm8k_think:math_verify") for d in (1, 2, 3)]
for p in range(1, 5):
    for d in range(1, 5):
        MEANS += [("Figure pareto", f"S{p}{d} MMLU (20/subject)", f"L:X26-S{p}{d}-mmlu:acc"),     # subsets timed in
                  ("Figure pareto", f"S{p}{d} RULER-16K (260)", f"B:X26-S{p}{d}:ruler16k"),     # the Pareto figure
                  ("Figure pareto", f"S{p}{d} GSM8K (400)", f"L:X26-S{p}{d}-gsm:{GF}")]


def fname(key):
    return key.replace(":", "__").replace("/", "_") + ".json.gz"


def mvfull(gold, text):
    from math_verify import parse, verify
    try:
        return float(verify(parse("$" + gold + "$"), parse(text)))
    except Exception:
        return 0.0


def final_answer(text):
    """The last \\boxed{...} of an answer, else the Minerva closing sentence's "final answer is X"; None if neither."""
    i = text.rfind("\\boxed")
    j = text.find("{", i) if i >= 0 else -1
    if j >= 0:
        depth = 0
        for k in range(j, len(text)):
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            if depth == 0:
                return text[j + 1:k]
    m = re.findall(r"final answer is \$?(.+?)\$?\.? I hope", text)
    return m[-1] if m else None


def boxed(gold, text):
    """MATH-500 final-answer accuracy: the answer of final_answer() equivalent to the reference (math_verify). Unlike
    lm-eval's strict exact_match it does not require the exemplars' closing sentence; unlike mvfull it scores one answer."""
    a = final_answer(text)
    return 0.0 if a is None else mvfull(gold, "$" + a + "$")


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
            if metric in ("mvfull", "boxed"):  # MATH-500 against the reference answer: whole answer / final answer
                out[f"{task}/{r['doc_id']}"] = (mvfull if metric == "mvfull" else boxed)(r["doc"]["answer"], r["resps"][0][0])
            elif metric in TEXT:
                out[f"{task}/{r['doc_id']}"] = TEXT[metric](r)
            elif metric in r:
                out[f"{task}/{r['doc_id']}"] = float(r[metric])
    return out


def load_raw(runs, key):
    kind, name, *rest = key.split(":")
    if kind == "L":
        return read_lm(runs, name, rest[0], rest[1] if len(rest) > 1 else None)
    if kind == "P":  # evalplus results of score_evalplus.py: P:<run>:<dataset>:<base|plus>
        f = f"{runs}/lmeval/out/{name}/evalplus_{rest[0]}.jsonl"
        return {f"{rest[0]}/{r['doc_id']}": float(r[rest[1]]) for r in map(json.loads, open(f))} if os.path.exists(f) else {}
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
    ks = {a for _, _, a, b, _ in CMP} | {b for _, _, a, b, _ in CMP} | {k for _, _, k in MEANS}
    ks |= {k for _, _, ka, kb in equal_specs() for k in (ka, ka.replace("S42", "S22"), kb.format(n=2), kb.format(n=4))}
    ks |= {k.format(c=c) for _, k in SN_EQUAL for c in ("native1", "C11", "C21", "native2")}
    ks |= {ka.replace("S42", "S12") for _, _, ka, _ in equal_specs() if "T26" in ka}
    ks |= {k.format(c=c) for _, k in TASK_MAP_14 for c in [f"S{m}{n}" for m in range(1, 5) for n in range(1, 5)]
           + [f"native{n}" for n in range(1, 5)]}
    ks |= {k.format(d=d) for k in DISTILLED_KEYS for d in DISTILLED.values()}
    ks |= {k for _, _, key, nkey, _ in taskmap_specs() for n in range(1, 5)
           for k in [key.format(c=f"S{m}{n}") for m in range(1, 5)] + [nkey.format(n=n)]}
    ks |= {main_key(b, kind, c) for b in MAIN_BENCH for _, kind, c in MAIN_ROWS}
    ks |= {abl_key(b, pre, c) for b in MAIN_BENCH for _, pre in ABLATIONS for c in ("S41", "S42")}
    ks |= {abl_key(b, pre, c) for b in MAIN_BENCH for pre in MAIN_SEEDS for c in ("S41", "S42")}
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
RATIOS = [("2.6B-T GSM8K prefill 4/decode 2 vs native 4/4", "think", "gsm", "S42sb", "native4st", (0.59, 0.52, 0.66)),
          ("2.6B-T GSM8K prefill 4/decode 2 vs prefill 2/decode 2", "think", "gsm", "S42sb", "S22sb", (1.02, 0.92, 1.13)),
          ("2.6B-T GSM8K prefill 4/decode 2 vs LoRA-distilled native 2/2", "think", "gsm", "S42sb", "Df2st", (1.20, 1.05, 1.38)),
          ("2.6B-T MATH prefill 4/decode 2 vs native 4/4", "think", "math", "S42sb", "native4st", (0.63, None, None)),
          ("2.6B-T MATH prefill 4/decode 2 vs native 2/2", "think", "math", "S42sb", "native2st", (1.22, None, None)),
          ("SenseNova GSM8K prefill 1/decode 1 vs native 2/2", "sn", "gsm", "C11", "native2", (0.59, 0.52, 0.68)),
          ("2.6B-T GSM8K answer once vs per-loop answer banks (prefill 4/decode 2, batch 1)", "think", "gsm", "O42sb", "S42sb", None)]


def e2e(model, task, cond):
    """Per-answer records of one configuration; the first record of every file is a warm-up and is dropped."""
    out = {}
    for f in sorted(E2E.glob(f"{model}-{cond}-{task}*.jsonl")):
        rows = [json.loads(line) for line in open(f)][1:]
        out.update({r["id"]: r for r in rows})
    return out


def e2e_table():
    rows = []
    for group, model, task, conds in (
            ("main", "think", "gsm", ("native2st", "native4st", "Df2st", "S22sb", "S42sb")),           # 192 problems
            ("first64", "think", "gsm", ("native1st", "native2st", "native4st", "S41sb", "S22sb", "S42sb")),  # 64
            ("main", "think", "math", ("native2st", "native4st", "S42sb")),
            ("main", "sn", "gsm", ("native2", "C11", "C21")),
            ("answer-once", "think", "gsm", ("S42sb", "O42sb"))):  # appendix: answer stored once
        recs = {c: e2e(model, task, c) for c in conds}
        common = set.intersection(*(set(r) for r in recs.values() if r))  # same problems for every configuration
        for c in conds:
            r = {i: x for i, x in recs[c].items() if i in common}
            if r:
                rows.append(dict(group=group, model=model, task=task, cond=c, n=len(r), acc=round(100 * np.mean([x["correct"] for x in r.values()]), 1),
                                 seconds=round(float(np.mean([x["total_s"] for x in r.values()])), 1),
                                 tokens=round(float(np.mean([x["new_tokens"] for x in r.values()]))),
                                 ms_per_token=round(float(np.mean([x["ms_per_token"] for x in r.values()])), 1)))
                print(f"Table 3      {group:11s} {model} {task} {c:10s} acc {rows[-1]['acc']:5.1f}  s/answer {rows[-1]['seconds']:6.1f}  n={len(r)}")
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
        got = (r["ratio"], r["lo"], r["hi"])
        flag = "  !! paper: " + str(printed) if printed and any(p is not None and abs(p - g) > 0.0051 for p, g in zip(printed, got)) else ""
        print(f"Sec 4.2      {label:52s} {r['ratio']:.2f}x [{r['lo']:.2f}, {r['hi']:.2f}]  n={len(ids)}{flag}")
    return rows, ratios



LEV = Path(__file__).resolve().parents[1] / "results" / "lev" / "levbench"
# (label, A rows dir, B rows dir): accuracy at every decision depth, A minus B (LevBench development, Lev's depth_rows)
LEV_CMP = [("Lev-1.4B untied vs same-training control", "rd4-1.4B-read4", "ctl-1.4B-read0"),
           ("Lev-1.4B control vs original", "ctl-1.4B-read0", "Lev-1.4B-read0"),
           ("Lev-1.4B training-free prefill 4 vs native", "Lev-1.4B-read4", "Lev-1.4B-read0"),
           ("Lev-1.4B untied, prefill 4 vs prefill 2 (prefill 2 not trained)", "rd4-1.4B-read4", "rd4-1.4B-read2"),
           ("Lev-2.6B untied vs same-training control", "rd4-2.6B-read4", "ctl-2.6B-read0"),
           ("Lev-2.6B training-free prefill 4 vs native", "Lev-2.6B-read4", "Lev-2.6B-read0"),
           ("Lev-2.6B control vs original", "ctl-2.6B-read0", "Lev-2.6B-read0"),
           ("Lev-3B untied vs same-training control", "rd2-3B-read2", "ctl-3B-read0"),
           *[(f"{m} seed {k} untied vs same-training control", f"{a}-s{k}-read{r}", f"ctl-{b}-s{k}-read0")
             for m, a, b, r in (("Lev-1.4B", "rd4-1.4B", "1.4B", 4), ("Lev-2.6B", "rd4-2.6B", "2.6B", 4), ("Lev-3B", "rd2-3B", "3B", 2))
             for k in (1, 2, 3)],
           # compute check: the control trained for two epochs (twice the updates; our untied training forward
           # costs about twice the control's per update)
           ("Lev-1.4B untied vs control trained 2x", "rd4-1.4B-read4", "ctl2x-1.4B-read0"),  # d=1: +4.0 [2.1, 6.0]
           ("Lev-2.6B untied vs control trained 2x", "rd4-2.6B-read4", "ctl2x-2.6B-read0"),
           ("Lev-3B untied vs control trained 2x", "rd2-3B-read2", "ctl2x-3B-read0"),
           # the other direction: fewer prefill loops (m < n), one continued training per prefill count m (lev_mn.sh)
           *[(f"{m} prefill {k} vs same-training control", f"rd{k}-{s}-read{k}", f"ctl-{s}-read0")
             for m, s, ks in (("Lev-1.4B", "1.4B", (1, 2, 3)), ("Lev-2.6B", "2.6B", (1, 2, 3)), ("Lev-3B", "3B", (1,)))
             for k in ks],
           # m < n replicated with seeds 1-3 (lev_mn_seeds.sh) against the control of the same seed
           *[(f"{m} prefill {k} seed {e} vs same-training control", f"rd{k}-{s}-s{e}-read{k}", f"ctl-{s}-s{e}-read0")
             for m, s, ks in (("Lev-1.4B", "1.4B", (1, 2, 3)), ("Lev-2.6B", "2.6B", (1, 2, 3)), ("Lev-3B", "3B", (1,)))
             for k in ks for e in (1, 2, 3)]]

# values quoted in Section 4.5 / Appendix (checked by lev()): (label, decision depth) -> (diff, lo, hi)
U = "untied vs same-training control"
LEV_PRINTED = {(f"Lev-1.4B {U}", 1): (5.3, 3.4, 7.2), (f"Lev-2.6B {U}", 1): (7.5, 5.7, 9.5), (f"Lev-3B {U}", 1): (4.4, 2.8, 6.1),
               (f"Lev-3B {U}", 2): (-1.8, -3.0, -0.7),
               ("Lev-1.4B prefill 1 vs same-training control", 4): (-9.6, -11.2, -7.9),
               ("Lev-2.6B prefill 1 vs same-training control", 4): (-13.0, -14.7, -11.3),
               ("Lev-3B prefill 1 vs same-training control", 2): (-8.1, -9.7, -6.6)}
# Section 4.5: (pairs, significantly above the control, significantly below, point estimate below) per direction
LEV_COUNTS = {"m>n": (13, 12, 0, 0), "m<n": (13, 0, 11, 13)}
# Section 4.5: m < n pairs with four seeds, below the control in every seed, significantly below in every seed
LEV_SEEDS = (13, 13, 10)


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
            r = {x: (round(v, 3) if isinstance(v, float) else v) for x, v in boot(ca, cb, groups={i: A[i]["group"] for i in ids}).items()}
            out.append(dict(label=label, depth=k, **r))
            print(f"Lev          {label:52s} d={k} {r['diff']:+5.1f} [{r['lo']:+.1f}, {r['hi']:+.1f}] ({r['a']:.1f} vs {r['b']:.1f})")
    # both directions (Section 4.5, Table lev): every trained prefill count m against the control at every decision
    # depth n; counts of pairs significantly above / below the control for m > n, m = n and m < n
    grid = []
    for m, top in (("Lev-1.4B", 4), ("Lev-2.6B", 4), ("Lev-3B", 2)):
        names = {f"{m} untied vs same-training control": top, **{f"{m} prefill {k} vs same-training control": k for k in range(1, top)}}
        grid += [dict(model=m, m=names[r["label"]], n=r["depth"], diff=[float(r[k]) for k in ("diff", "lo", "hi")]) for r in out
                 if r["label"] in names]
    counts = {}
    for name, sel in (("m>n", lambda g: g["m"] > g["n"]), ("m=n", lambda g: g["m"] == g["n"]), ("m<n", lambda g: g["m"] < g["n"])):
        rows = [g for g in grid if sel(g)]
        counts[name] = (len(rows), int(sum(g["diff"][1] > 0 for g in rows)), int(sum(g["diff"][2] < 0 for g in rows)),
                        int(sum(g["diff"][0] < 0 for g in rows)))
        print(f"Lev both     {name}: {counts[name][0]} pairs, significantly above the control {counts[name][1]}, "
              f"significantly below {counts[name][2]}, point estimate below {counts[name][3]}")
    for name, printed in LEV_COUNTS.items():
        if counts.get(name) != printed:
            print(f"Lev both     {name}: {counts.get(name)}  !! paper: {printed}")
    for (label, n), printed in LEV_PRINTED.items():
        got = next(((r["diff"], r["lo"], r["hi"]) for r in out if r["label"] == label and r["depth"] == n), None)
        if got is None or any(abs(p - g) > 0.051 for p, g in zip(printed, got)):
            print(f"Lev          {label} d={n}: {got}  !! paper: {printed}")
    out.append(dict(label="both directions", grid=grid, counts=counts))
    # m < n over four seeds (seed 0 = the grid above): per pair, how many seeds are significantly below the control
    seeds = []
    for g in [g for g in grid if g["m"] < g["n"]]:
        labels = [f"{g['model']} prefill {g['m']} seed {e} vs same-training control" for e in (1, 2, 3)]
        d = [g["diff"]] + [[r["diff"], r["lo"], r["hi"]] for lb in labels for r in out if r.get("label") == lb and r.get("depth") == g["n"]]
        seeds.append(dict(model=g["model"], m=g["m"], n=g["n"], seeds=len(d), mean=round(float(np.mean([x[0] for x in d])), 3),
                          below=int(sum(x[2] < 0 for x in d)), negative=int(sum(x[0] < 0 for x in d))))
    if seeds:
        full = [x for x in seeds if x["seeds"] == 4]
        print(f"Lev seeds    m<n: {len(full)} of {len(seeds)} pairs with 4 seeds; every seed below in point estimate "
              f"{sum(x['negative'] == 4 for x in full)}, every seed significantly below {sum(x['below'] == 4 for x in full)}, "
              f"mean below {sum(x['mean'] < 0 for x in full)}")
        got = (len(full), sum(x["negative"] == 4 for x in full), sum(x["below"] == 4 for x in full))
        if got != LEV_SEEDS:
            print(f"Lev seeds    m<n: {got}  !! paper: {LEV_SEEDS}")
        out.append(dict(label="m<n seeds", pairs=seeds))
    # training seeds (seed 0 = the unsuffixed runs): per-seed gain at each decision depth, mean and sample std
    for m in ("Lev-1.4B", "Lev-2.6B", "Lev-3B"):
        main = f"{m} untied vs same-training control"
        labels = [main] + [f"{m} seed {k} untied vs same-training control" for k in (1, 2, 3)]
        for k in sorted({r["depth"] for r in out if r["label"] == main}):
            g = [r["diff"] for lb in labels for r in out if r["label"] == lb and r["depth"] == k]
            if len(g) > 1:
                out.append(dict(label=f"{m} seeds", depth=k, seeds=len(g), mean=round(float(np.mean(g)), 1),
                                sd=round(float(np.std(g, ddof=1)), 1), min=min(g), max=max(g)))
                print(f"Lev seeds    {m:52s} d={k} {len(g)} seeds: mean {np.mean(g):+5.1f} sd {np.std(g, ddof=1):.1f} "
                      f"range [{min(g):+.1f}, {max(g):+.1f}]")
    return out




def equal_specs():
    """(model, benchmark, prefill 4/decode 2 key, native key with {n}) for Ouro-2.6B-Thinking and Ouro-1.4B-Thinking."""
    t14 = lambda k: k.replace("T26", "T14").replace("T:think-native", "T:T14-native")   # 2.6B natives are "think-native*"
    return [("2.6B-T", *e) for e in EQUAL] + [("1.4B-T", e[0], *(t14(k) for k in e[1:])) for e in EQUAL]


def equal():
    """Figure 4: prefill 4 / decode 2 vs native 2/2 and native 4/4 (Ouro-2.6B-Thinking), all three on the items every one
    of them has, so the three scores and both paired differences share one item set."""
    out = []
    for model, bench, ka, kb in equal_specs():
        o, n2, n4 = load(ka), load(kb.format(n=2)), load(kb.format(n=4))
        if o is None or n2 is None or n4 is None:
            print(f"Equal        {model} {bench:12s} MISSING"); continue
        s22 = load(ka.replace("S42", "S22"))   # the same adapter reading 2 loops (same weights, same training)
        ids = set(o) & set(n2) & set(n4)
        if s22 is not None and not ids <= set(s22):   # not yet on every shared item: report it alone, keep the triple
            print(f"Equal        {model} {bench:12s} same adapter 2/2 on {len(ids & set(s22))} of {len(ids)} shared items; not used")
            s22 = None
        o, n2, n4 = ({i: x[i] for i in ids} for x in (o, n2, n4))
        r2, r4 = boot(o, n2), boot(o, n4)
        row = dict(model=model, bench=bench, n=len(ids), ours=round(r2["a"], 1), native2=round(r2["b"], 1), native4=round(r4["b"], 1),
                   vs2=[round(r2[k], 3) for k in ("diff", "lo", "hi")], vs4=[round(r4[k], 3) for k in ("diff", "lo", "hi")])
        if s22 is not None:
            r22 = boot(o, {i: s22[i] for i in ids})
            row.update(same22=round(r22["b"], 1), vs22=[round(r22[k], 3) for k in ("diff", "lo", "hi")])
            r = boot({i: s22[i] for i in ids}, n2)   # figure: every bar is a difference to native 2/2
            row["s22_vs2"] = [round(r[k], 3) for k in ("diff", "lo", "hi")]
        r = boot(n4, n2)
        row["n4_vs2"] = [round(r[k], 3) for k in ("diff", "lo", "hi")]
        out.append(row)
        print(f"Equal        {model} {bench:12s} n={len(ids):5d} ours {row['ours']:5.1f} | 2/2 {row['native2']:5.1f} "
              f"{row['vs2'][0]:+5.1f} [{row['vs2'][1]:+.1f}, {row['vs2'][2]:+.1f}] | 4/4 {row['native4']:5.1f} "
              f"{row['vs4'][0]:+5.1f} [{row['vs4'][1]:+.1f}, {row['vs4'][2]:+.1f}]"
              + (f" | same 2/2 {row['same22']:5.1f} {row['vs22'][0]:+5.1f} [{row['vs22'][1]:+.1f}, {row['vs22'][2]:+.1f}]"
                 if "same22" in row else ""))
    return out


def sn_equal():
    """SenseNova-Loop-9B at the decode depth of one visit: adapter prefill 2 / decode 1 (C21) against native 1/1, the same
    adapter reading one visit (C11) and native 2/2, all four on the items every one of them has."""
    out = []
    for bench, key in SN_EQUAL:
        d = {c: load(key.format(c=c)) for c in ("native1", "C11", "C21", "native2")}
        if any(v is None for v in d.values()):
            print(f"SN equal     {bench:12s} MISSING {[c for c, v in d.items() if v is None]}"); continue
        ids = set.intersection(*(set(v) for v in d.values()))
        d = {c: {i: v[i] for i in ids} for c, v in d.items()}
        row = dict(bench=bench, n=len(ids))
        for c in ("native1", "C11", "native2"):
            r = boot(d["C21"], d[c])
            row["C21"], row[c] = round(r["a"], 1), round(r["b"], 1)
            row[f"vs_{c}"] = [round(r[k], 3) for k in ("diff", "lo", "hi")]
        out.append(row)
        print(f"SN equal     {bench:12s} n={len(ids):5d} read2/write1 {row['C21']:5.1f} | 1/1 {row['native1']:5.1f} "
              f"{row['vs_native1'][0]:+5.1f} | C11 {row['C11']:5.1f} {row['vs_C11'][0]:+5.1f} [{row['vs_C11'][1]:+.1f}, "
              f"{row['vs_C11'][2]:+.1f}] | 2/2 {row['native2']:5.1f} {row['vs_native2'][0]:+5.1f} [{row['vs_native2'][1]:+.1f}, "
              f"{row['vs_native2'][2]:+.1f}]")
    return out

def taskmap_specs():
    """(model, benchmark, cell key with {c}, native key with {n}, distilled key with {d} or None) for every task map."""
    specs = [("MMLU", "L:X26-{c}-mmlufull:acc", "L:o26n{n}-mmlufull:acc", "L:o26x-{d}-mmlufull:acc"),     # 14,042
             ("GSM8K", "L:X26-{c}-gsmfull:" + GF, "L:X26-native{n}-gsmfull:" + GF, "L:o26x-{d}-gsmfull:" + GF)]  # 1,319
    specs += [(bench, key, key.replace("{c}", "native{n}"), key.replace("{c}", "{d}")) for bench, key in TASK_MAP]
    return [("Ouro-2.6B", *x) for x in specs] + [("Ouro-1.4B", bench, key, key.replace("{c}", "native{n}"),
                                                  key.replace("{c}", "{d}")) for bench, key in TASK_MAP_14]


# Main results table (Section 4.1): fixed configurations, standard benchmarks; (label, kind, condition)
MAIN_BENCH = ["MMLU", "MMLU-Pro", "BBH 3-shot", "ARC-C", "HellaSwag", "Winogrande", "GSM8K", "MATH-500 4-shot",
              "HumanEval+", "MBPP+", "RULER-16K"]
MAIN_ROWS = [("native 1/1", "native", 1), ("native 2/2", "native", 2), ("native 4/4", "native", 4),
             ("training-free 4/1", "raw", "raw41"), ("training-free 4/2", "raw", "raw42"),
             ("LoRA-distilled 1/1", "dist", 1), ("LoRA-distilled 2/2", "dist", 2),
             ("loop adapter 4/1", "cell", "S41"), ("loop adapter 4/2", "cell", "S42")]


# quoted in the abstract, introduction and Section 4.2
MAIN_AVG = {"native 1/1": 44.2, "LoRA-distilled 1/1": 49.8, "loop adapter 4/1": 56.4, "native 2/2": 63.0,
            "LoRA-distilled 2/2": 65.3, "loop adapter 4/2": 66.4, "native 4/4": 70.2,
            "training-free 4/1": 26.6, "training-free 4/2": 30.9}
MAIN_CELLS = {("MMLU", "native 1/1"): 51.5, ("MMLU", "loop adapter 4/1"): 66.6, ("BBH 3-shot", "native 1/1"): 30.9,
              ("BBH 3-shot", "loop adapter 4/1"): 51.7, ("HumanEval+", "native 1/1"): 44.5,
              ("HumanEval+", "loop adapter 4/1"): 64.6, ("GSM8K", "training-free 4/2"): 11.7,
              ("GSM8K", "loop adapter 4/2"): 66.7}


def main_key(bench, kind, c):
    """Item-score key of one main-table entry (Ouro-2.6B, the task-map settings)."""
    spec = next(x for x in taskmap_specs() if x[0] == "Ouro-2.6B" and x[1] == bench)
    _, _, key, nkey, dkey = spec
    if kind == "native":
        return nkey.format(n=c)
    if kind == "dist":
        return dkey.format(d=DISTILLED[c])
    if kind == "raw":   # raw runs are named X26-raw<m><n>-<task> for every benchmark
        return key.replace("o26x-{c}", "X26-{c}").format(c=c)
    return key.format(c=c)


MAIN_SEEDS = ("X26s2", "X26s3")   # further training seeds of the loop adapter (seed 0: X26-S-universal)


def main_table():
    """Accuracy of each main-table configuration, each benchmark on the items all rows share; None if a run is missing.
    The loop-adapter rows also give each training seed (key '<label> seeds': seed 0, then MAIN_SEEDS), on the same
    items. avg: mean over the benchmarks when the row is complete."""
    out = []
    for bench in MAIN_BENCH:
        sets = {label: load(main_key(bench, kind, c)) for label, kind, c in MAIN_ROWS}
        seeds = {label: [load(abl_key(bench, pre, c)) for pre in MAIN_SEEDS] for label, kind, c in MAIN_ROWS
                 if kind == "cell"}
        have = [v for v in sets.values() if v] + [v for vs in seeds.values() for v in vs if v]
        ids = set.intersection(*(set(v) for v in have)) if have else set()
        mean = lambda v: round(100 * float(np.mean([v[i] for i in ids])), 2) if v else None
        row = dict(bench=bench, items=len(ids), **{label: mean(v) for label, v in sets.items()})
        row.update({f"{label} seeds": [row[label]] + [mean(v) for v in vs] for label, vs in seeds.items()})
        out.append(row)
    for label, _, _ in MAIN_ROWS:
        vals = [r[label] for r in out]
        avg = None if None in vals else float(np.mean(vals))
        want = MAIN_AVG.get(label)
        flag = "  !! paper: " + str(want) if want is not None and (avg is None or abs(avg - want) > 0.051) else ""
        print(f"Main table   {label:20s} " + " ".join("  -- " if v is None else f"{v:5.1f}" for v in vals)
              + ("" if avg is None else f"  avg {avg:5.1f}") + flag)
    for label in [l for l, kind, _ in MAIN_ROWS if kind == "cell"]:
        per_seed = list(zip(*[r[f"{label} seeds"] for r in out]))
        avgs = [None if None in v else float(np.mean(v)) for v in per_seed]
        if None not in avgs:
            got = (round(float(np.mean(avgs)), 1), round(float(np.std(avgs, ddof=1)), 1))
            want = {"loop adapter 4/1": (56.4, 0.0), "loop adapter 4/2": (66.6, 0.2)}.get(label)   # Table 1 caption
            print(f"Main table   {label:20s} seeds: " + " ".join(f"{a:5.1f}" for a in avgs)
                  + f"  mean {got[0]:5.1f} sd {got[1]:.1f}" + (f"  !! paper: {want}" if want and got != want else ""))
    T = {r["bench"]: r for r in out}
    for (bench, label), want in MAIN_CELLS.items():
        got = T[bench][label]
        if got is None or abs(got - want) > 0.051:
            print(f"Main table   {bench} {label}: {got}  !! paper: {want}")
    return out


# Ablations (Section 4.2): training tokens at rank 16 and LoRA rank at 12M tokens; (label, run prefix or None = the
# 48M rank-16 adapter of the main table)
ABLATIONS = [("6M tokens", "X26t6"), ("12M tokens", "X26t12"), ("24M tokens", "X26t24"), ("48M tokens", None),
             ("rank 4, 12M", "X26t12r4"), ("rank 64, 12M", "X26t12r64")]


ABL_PRINTED = {("6M tokens", "S41"): 53.7, ("6M tokens", "S42"): 65.2, ("48M tokens", "S41"): 56.4,
               ("48M tokens", "S42"): 66.4}   # quoted in Section 4.2


def abl_key(bench, prefix, c):
    key = main_key(bench, "cell", c)
    if prefix is None:
        return key
    return re.sub(r"(?<=[:])(o26x|X26)-(?=S4)", prefix + "-", key)


def ablations():
    """Main-table average of each ablation run at prefill 4 / decode 1 and 2, every benchmark on the items all runs
    share; None until a run is complete."""
    out = []
    for c in ("S41", "S42"):
        per = {label: [] for label, _ in ABLATIONS}
        for bench in MAIN_BENCH:
            sets = {label: load(abl_key(bench, pre, c)) for label, pre in ABLATIONS}
            have = [v for v in sets.values() if v]
            ids = set.intersection(*(set(v) for v in have)) if have else set()
            for label, v in sets.items():
                per[label].append(100 * float(np.mean([v[i] for i in ids])) if v else None)
        for label, _ in ABLATIONS:
            vals = per[label]
            avg = None if None in vals else round(float(np.mean(vals)), 2)
            out.append(dict(cell=c, label=label, avg=avg, scores=[None if v is None else round(v, 2) for v in vals]))
            want = ABL_PRINTED.get((label, c))
            if want is not None and (avg is None or abs(avg - want) > 0.051):
                print(f"Ablation     {c} {label}: {avg}  !! paper: {want}")
            print(f"Ablation     {c} {label:14s} " + ("  --" if avg is None else f"avg {avg:5.1f}")
                  + "  " + " ".join("  -- " if v is None else f"{v:5.1f}" for v in vals))
    return out


def taskmap_cells():
    """Task map (Figure 1, appendix grid): every (m, n) cell and native n/n per benchmark, each mean over the items that
    all 20 configurations share; best[n] is the best m at n, chosen before rounding (ties: the smaller m)."""
    out = []
    for model, bench, key, nkey, _ in taskmap_specs():
        keys = {f"S{m}{n}": key.format(c=f"S{m}{n}") for n in range(1, 5) for m in range(1, 5)}
        keys.update({f"native{n}": nkey.format(n=n) for n in range(1, 5)})
        sets = {c: load(k) for c, k in keys.items()}
        if any(v is None for v in sets.values()):
            continue
        ids = set.intersection(*(set(v) for v in sets.values()))
        mean = {c: 100 * float(np.mean([v[i] for i in ids])) for c, v in sets.items()}
        best = {n: max(range(1, 5), key=lambda m: (mean[f"S{m}{n}"], -m)) for n in range(1, 5)}
        out.append(dict(model=model, bench=bench, items=len(ids), best=best, **{c: round(v, 3) for c, v in mean.items()}))
    return out


def select_aware(cells, nat, ids, half, reps=5000):
    """Bootstrap that repeats the cross-fitted choice of m in every resample of the items, so the interval includes
    the variability of the choice (the intervals of Table 1; 5,000 draws, seed 0)."""
    rng = np.random.default_rng(0)
    ids = sorted(ids, key=str)
    A = np.array([[cells[m][i] for i in ids] for m in range(1, 5)])
    b = np.array([nat[i] for i in ids]); h = np.array([half(i) for i in ids])
    diffs = []
    for _ in range(reps):
        x = rng.integers(0, len(ids), len(ids)); total = 0.0
        for f in (0, 1):
            train, test = x[h[x] == f], x[h[x] == 1 - f]
            m = int(np.argmax(A[:, train].mean(1)))        # ties: the smaller m, as in the main selection
            total += (A[m, test] - b[test]).sum()
        diffs.append(100 * total / len(x))
    return [round(float(np.percentile(diffs, q)), 3) for q in (2.5, 97.5)]



OVERLAP = Path(__file__).resolve().parents[1] / "results" / "overlap" / "mix-data-48m.json"


def overlap_check():
    """Items whose question shares a 13-gram with the adapters' training mixture (overlap_audit.py) are dropped and the
    Ouro-2.6B selection of Table 1 and the 4-shot MATH-500 contrast are recomputed on the remaining items."""
    if not OVERLAP.exists():
        return {}
    drop = {k for r in json.loads(OVERLAP.read_text())["runs"].values() for k in r["overlapping"]}
    out = dict(dropped=len(drop))
    rows = taskmap_select(exclude=drop, quiet=True)
    for n in (1, 2):
        sel = [r for r in rows if r["model"] == "Ouro-2.6B" and r["n"] == n]
        out[f"n{n}"] = (sum(r["diff_select_aware"][1] > 0 for r in sel), len(sel))
        out[f"n{n}_rows"] = {r["bench"]: [float(x) for x in r["diff_select_aware"]] for r in sel}
        flag = "" if out[f"n{n}"] == {1: (13, 15), 2: (9, 15)}[n] else "  !! paper: " + str({1: (13, 15), 2: (9, 15)}[n])
        print(f"Overlap      Ouro-2.6B n={n} without overlapping items: significant with selection-aware intervals "
              f"{out[f'n{n}'][0]} of {out[f'n{n}'][1]}{flag}")
    a, b = load("L:o26x-S14-math:boxed"), load("L:o26x-S44-math:boxed")
    if a and b:
        keep = [i for i in a if i in b and i not in drop]
        r = boot({i: a[i] for i in keep}, {i: b[i] for i in keep})
        out["math_s14_s44"] = [round(r[k], 3) for k in ("diff", "lo", "hi")] + [len(keep)]
        print(f"Overlap      MATH-500 4-shot prefill 1 vs 4 at decode 4 without overlapping items: {r['diff']:+.1f} "
              f"[{r['lo']:+.1f}, {r['hi']:+.1f}] ({r['a']:.1f} vs {r['b']:.1f}, n={len(keep)})")
    return out

def taskmap_select(exclude=frozenset(), quiet=False):
    """Choose the prefill loops m per task without selection bias: for every benchmark and decode loops n, split the
    items in two halves (by a hash of the item id), pick the m with the best mean on one half, score it on the other
    half, and swap; every item is scored once by an m chosen without it. Compared with native n/n on the same items
    (paired bootstrap, 5,000 draws, seed 0)."""
    import hashlib
    half = lambda i: int(hashlib.md5(str(i).encode()).hexdigest(), 16) % 2
    out = []
    for model, bench, key, nkey, dkey in taskmap_specs():
        for n in range(1, 5):
            cells = {m: load(key.format(c=f"S{m}{n}")) for m in range(1, 5)}
            nat = load(nkey.format(n=n))
            if nat is None or any(v is None for v in cells.values()):
                continue
            ids = set(nat).intersection(*(set(v) for v in cells.values())) - exclude
            folds = [[i for i in ids if half(i) == h] for h in (0, 1)]
            chosen, picked = [], {}
            for h in (0, 1):
                train, test = folds[h], folds[1 - h]
                m_best = max(range(1, 5), key=lambda m: (np.mean([cells[m][i] for i in train]), -m))
                chosen.append(m_best)
                for i in test:
                    picked[i] = cells[m_best][i]
            r = boot(picked, {i: nat[i] for i in picked})
            row = dict(model=model, bench=bench, n=n, m_chosen=chosen, items=len(picked), selected=round(r["a"], 1),
                       native=round(r["b"], 1), diff=[round(r[k], 3) for k in ("diff", "lo", "hi")],
                       diff_select_aware=[round(r["diff"], 1), *select_aware(cells, nat, ids, half)])
            ra = boot(picked, {i: cells[n][i] for i in picked})
            # within the adapter: the chosen m against the same adapter's m = n (isolates the prefill loops)
            row.update(adapter_nn=round(ra["b"], 1), diff_adapter=[round(ra[k], 3) for k in ("diff", "lo", "hi")],
                       diff_adapter_select_aware=[round(ra["diff"], 1), *select_aware(cells, cells[n], ids, half)])
            # same-training control: LoRA distillation of native n/n (native 4/4 is the teacher itself)
            dist = None if dkey is None else nat if n == 4 else load(dkey.format(d=DISTILLED[n]))
            if dist is not None and set(picked) <= set(dist):    # only a complete run (partial exports are skipped)
                shared = picked
                r = boot(shared, {i: dist[i] for i in shared})
                row.update(distilled=round(r["b"], 1), distilled_items=len(shared),
                           diff_distilled=[round(r[k], 3) for k in ("diff", "lo", "hi")],
                           diff_distilled_select_aware=[round(r["diff"], 3), *select_aware(cells, dist, ids, half)])
            out.append(row)
            vs_d = (f" | distilled {row['distilled']:5.1f} {row['diff_distilled'][0]:+5.1f} [{row['diff_distilled'][1]:+.1f}, "
                    f"{row['diff_distilled'][2]:+.1f}]") if "distilled" in row else ""
            if quiet:
                continue
            print(f"Select m     {model} {bench:16s} n={n} m={chosen} selected {row['selected']:5.1f} native {row['native']:5.1f} "
                  f"{row['diff'][0]:+5.1f} [{row['diff'][1]:+.1f}, {row['diff'][2]:+.1f}]{vs_d}")
    if quiet:
        return out
    # single values quoted in Section 4.2: (bench, n, field) -> (diff, lo, hi) or point estimate
    quoted = {("GPQA-Diamond", 1, "diff_select_aware"): (12.1, -2.5, 18.7), ("MMLU-Pro", 1, "diff_distilled"): 19.3,
              ("BBH 3-shot", 1, "diff_distilled"): 11.3, ("HumanEval+", 1, "diff_distilled"): 11.6}
    for (bench, n, field), v in quoted.items():
        got = next((r.get(field) for r in out if r["model"] == "Ouro-2.6B" and r["bench"] == bench and r["n"] == n), None)
        want = v if isinstance(v, tuple) else (v,)
        if got is None or any(abs(a - b) > 0.051 for a, b in zip(want, got)):
            print(f"Select m     {bench} n={n} {field}: {got}  !! paper: {v}")
    for model in ("Ouro-2.6B", "Ouro-1.4B"):
        for n in range(1, 5):
            rows = [r for r in out if r["n"] == n and r["model"] == model]
            got = (sum(r["diff_select_aware"][1] > 0 for r in rows), sum(r["diff_adapter_select_aware"][1] > 0 for r in rows),
                   sum(r["diff_adapter_select_aware"][2] < 0 for r in rows), len(rows))
            # quoted in Section 4.2: (selection-aware significant, above adapter n/n, below adapter n/n, benchmarks)
            printed = {("Ouro-2.6B", 1): (14, 9, 0, 15), ("Ouro-2.6B", 2): (9, 3, 0, 15), ("Ouro-1.4B", 1): (12, 6, 0, 15),
                       ("Ouro-1.4B", 2): (2, 1, 0, 15)}.get((model, n))
            flag = f"  !! paper: {printed}" if printed and got != printed else ""
            print(f"Select m     {model} n={n}: significant with selection-aware intervals {got[0]} of {got[3]}; above the "
                  f"adapter's {n}/{n} significantly {got[1]}, below {got[2]}{flag}")
    for model, counts in (("Ouro-2.6B", DISTILLED_COUNTS), ("Ouro-1.4B", DISTILLED_COUNTS_14)):
        for n in (1, 2, 3):
            rows = [r for r in out if r["model"] == model and r["n"] == n and "diff_distilled" in r]
            if rows:   # selection-aware intervals, as Table 1; quoted in Section 4.2: (above, below, settings)
                got = (sum(r["diff_distilled_select_aware"][1] > 0 for r in rows),
                       sum(r["diff_distilled_select_aware"][2] < 0 for r in rows), len(rows))
                printed = counts.get(n)
                flag = f"  !! paper: {printed}" if printed and got != printed else ""
                print(f"Select m     {model} vs distilled {n}/{n}: significantly above on {got[0]}, below on {got[1]} of "
                      f"{got[2]}{flag}")
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
        r = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}   # rounded only when printed
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
    OUT.write_text(json.dumps(dict(comparisons=res, means=means, e2e=e2e_rows, e2e_ratios=ratios, throughput=throughput(), lev=lev(), equal=equal(), sn_equal=sn_equal(), taskmap_select=taskmap_select(), taskmap_cells=taskmap_cells(), main_table=main_table(), ablations=ablations(), overlap=overlap_check()), indent=1))
    print(f"wrote {OUT}; {bad} printed values differ")


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export(sys.argv[2], Path(sys.argv[3]) if len(sys.argv) > 3 else ITEMS)
    else:
        compute()
