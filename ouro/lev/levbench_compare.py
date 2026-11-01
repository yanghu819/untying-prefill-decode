"""LevBench development: read 4 / write k against native depth k, from Lev's scripts/depth_rows.py outputs.

    python ouro/lev/levbench_compare.py ROWS_DIR MODEL     (ROWS_DIR/<MODEL>-read0/*.json, <MODEL>-read4/*.json)
    python ouro/lev/levbench_compare.py ROWS_DIR A B         (any two row directories under ROWS_DIR: A minus B)

Accuracy = argmax of a depth's logits equals the label; paired bootstrap over questions (5,000 resamples, seed 0);
breakdown by state length (tokens) and by LevBench tier.
"""
import glob
import json
import sys

import numpy as np


def load(path):
    rows = json.load(open(glob.glob(f"{path}/*.json")[0]))["rows"]
    return {(r["id"], r.get("question")): r for r in rows}


def correct(r, k):
    return float(int(np.argmax(r["logits"][k - 1])) == r["label"])


def boot(d, rng):
    m = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
    return 100 * d.mean(), 100 * np.percentile(m, 2.5), 100 * np.percentile(m, 97.5)


def main():
    root = sys.argv[1]
    if len(sys.argv) > 3:
        model = f"{sys.argv[2]}-vs-{sys.argv[3]}"
        deep, nat = load(f"{root}/{sys.argv[2]}"), load(f"{root}/{sys.argv[3]}")
    else:
        model = sys.argv[2]
        nat, deep = load(f"{root}/{model}-read0"), load(f"{root}/{model}-read4")
    ids = sorted(set(nat) & set(deep), key=str)
    depths = len(next(iter(nat.values()))["logits"])
    rng = np.random.default_rng(0)
    out = {"model": model, "n": len(ids), "depths": {}}
    print(f"{model}: {len(ids)} questions")
    for k in range(1, depths + 1):
        a = np.array([correct(deep[i], k) for i in ids]); b = np.array([correct(nat[i], k) for i in ids])
        diff, lo, hi = boot(a - b, rng)
        out["depths"][k] = dict(native=100 * b.mean(), read4=100 * a.mean(), diff=diff, lo=lo, hi=hi)
        print(f"  write {k}: native {100 * b.mean():.1f}  read 4 {100 * a.mean():.1f}  {diff:+.1f} [{lo:+.1f}, {hi:+.1f}]")
    for name, key, edges in (("state tokens", "state_tokens", [0, 256, 1024, 10 ** 9]),):
        for lo_e, hi_e in zip(edges[:-1], edges[1:]):
            sel = [i for i in ids if lo_e <= nat[i].get(key, 0) < hi_e]
            if not sel:
                continue
            a = np.array([correct(deep[i], 1) for i in sel]); b = np.array([correct(nat[i], 1) for i in sel])
            d, l, h = boot(a - b, rng)
            print(f"  write 1, {name} {lo_e}-{hi_e}: n={len(sel)} native {100 * b.mean():.1f} read4 {100 * a.mean():.1f} {d:+.1f} [{l:+.1f}, {h:+.1f}]")
    tiers = sorted({nat[i].get("tier") for i in ids} - {None})
    for t in tiers:
        sel = [i for i in ids if nat[i].get("tier") == t]
        a = np.array([correct(deep[i], 1) for i in sel]); b = np.array([correct(nat[i], 1) for i in sel])
        d, l, h = boot(a - b, rng)
        print(f"  write 1, tier {t}: n={len(sel)} native {100 * b.mean():.1f} read4 {100 * a.mean():.1f} {d:+.1f} [{l:+.1f}, {h:+.1f}]")
    json.dump(out, open(f"{root}/{model}-compare.json", "w"), indent=1)


if __name__ == "__main__":
    main()
