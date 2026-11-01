"""Paired bootstrap over shared example ids: mean score difference A - B with a 95% CI and P(diff <= 0).

usage: paired_bootstrap.py BENCH_DIR NAME_A NAME_B [bench ...]   (reads BENCH_DIR/NAME/s*/BENCH.jsonl)
"""

import glob
import json
import random
import sys


def load(root, name, bench):
    out = {}
    for f in glob.glob(f"{root}/{name}/s*/{bench}.jsonl"):
        for line in open(f):
            r = json.loads(line)
            out[r["id"]] = float(r["score"] if "score" in r else r["correct"])
    return out


def main():
    root, a, b, *benches = sys.argv[1:]
    rng = random.Random(0)
    for bench in benches or ["ruler4k", "ruler8k", "ruler16k", "helmet"]:
        x, y = load(root, a, bench), load(root, b, bench)
        ids = sorted(set(x) & set(y))
        if not ids:
            continue
        d = [x[i] - y[i] for i in ids]
        n = len(d)
        boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
        mean = sum(d) / n
        print(json.dumps(dict(bench=bench, a=a, b=b, n=n, a_mean=round(100 * sum(x[i] for i in ids) / n, 2),
                              b_mean=round(100 * sum(y[i] for i in ids) / n, 2), diff=round(100 * mean, 2),
                              ci95=[round(100 * boots[49], 2), round(100 * boots[1949], 2)],
                              p_le0=round(sum(v <= 0 for v in boots) / len(boots), 4))))


if __name__ == "__main__":
    main()
