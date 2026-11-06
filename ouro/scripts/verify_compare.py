"""Item-level comparison of the from-scratch re-runs (verify_q.sh, run names <run>-rv) with the original runs on this
box: per run, the items both have, the share with the same score, and both means.

    python verify_compare.py RUNS_DIR
"""
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_numbers as PN  # noqa: E402


def bench_scores(runs, run, bench):
    out = {}
    for f in glob.glob(f"{runs}/bench/{run}/s*/{bench}.jsonl"):
        for line in open(f):
            r = json.loads(line)
            out[r["id"]] = float(r["score"])
    return out


def main(runs):
    rows = []
    for d in sorted(glob.glob(f"{runs}/lmeval/out/*-rv")):
        rv = os.path.basename(d)
        orig = rv[:-3]
        task = "gsm8k" if "gsm" in rv else "mmlu"
        metric, filt = ("exact_match", "strict-match") if task == "gsm8k" else ("acc", None)
        a, b = PN.read_lm(runs, orig, metric, filt), PN.read_lm(runs, rv, metric, filt)
        if not a:   # original evaluated on the other box: its exported item scores (names as in paper_numbers.py)
            name = orig.replace("X26-native", "o26n") if "mmlufull" in orig else orig   # native MMLU runs: o26n<n>-mmlufull
            key = f"L:{name}:{metric}" + (f":{filt}" if filt else "")
            a = PN.load(key) or {}
        rows.append((rv, a, b))
    for d in sorted(glob.glob(f"{runs}/bench/*-rv")):
        rv = os.path.basename(d)
        orig = bench_scores(runs, rv[:-3], "ruler16k") or PN.load(f"B:{rv[:-3]}:ruler16k") or {}
        rows.append((rv, orig, bench_scores(runs, rv, "ruler16k")))
    for rv, a, b in rows:
        ids = set(a) & set(b)
        if not ids:
            print(f"{rv:28s} no shared items (original {len(a)}, re-run {len(b)})")
            continue
        same = sum(a[i] == b[i] for i in ids) / len(ids)
        ma, mb = (100 * sum(x[i] for i in ids) / len(ids) for x in (a, b))
        print(f"{rv:28s} items {len(ids):6d}  same score {100 * same:5.1f}%  original {ma:5.1f}  re-run {mb:5.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
