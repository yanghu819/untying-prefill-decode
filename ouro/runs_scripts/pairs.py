"""Paired bootstrap for arbitrary run pairs on bench (RULER etc.) and lm-eval outputs.
usage: pairs.py A:B[:label] ...   names: bench run dirs (bench/<A>) or lm-eval output dirs (lmeval/out/<A>);
prints mean A, mean B, A-B and 95% CI over shared items for every bench file / lm-eval task they share."""
import glob, json, os, random, sys
R = "/dev/shm/big-small-ouro/runs"


def bench(run):
    out = {}
    for f in glob.glob(f"{R}/bench/{run}/s*/*.jsonl"):
        b = os.path.basename(f)[:-6]
        for l in open(f):
            r = json.loads(l); out.setdefault(b, {})[r["id"]] = float(r["score"])
    return out


def lm(run):
    out = {}
    for f in glob.glob(f"{R}/lmeval/out/{run}/*/samples_*.jsonl"):
        task = os.path.basename(f).split("samples_")[1].rsplit("_", 1)[0]
        group = "bbh" if task.startswith("bbh") else "mmlu_pro" if task.startswith("mmlu_pro_") else "mmlu" if task.startswith("mmlu") else task
        for l in open(f):
            r = json.loads(l)
            v = next((r[k] for k in ("math_verify", "exact_match", "acc_norm", "acc") if k in r), None)
            if v is None:
                v = next((x for k, x in r.items() if k.startswith("exact_match") and "flexible" in k), None)
            if v is not None:
                out.setdefault(group, {})[(task, r["doc_id"])] = float(v)
    return out


def boot(a, b):
    ids = sorted(set(a) & set(b)); n = len(ids)
    if n < 20:
        return None
    d = [a[i] - b[i] for i in ids]; rng = random.Random(0)
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
    return dict(n=n, a=round(100 * sum(a[i] for i in ids) / n, 1), b=round(100 * sum(b[i] for i in ids) / n, 1),
                diff=round(100 * sum(d) / n, 1), lo=round(100 * bs[49], 1), hi=round(100 * bs[1949], 1))


res = []
for spec in sys.argv[1:]:
    a, b, *lab = spec.split(":")
    A, B = {**bench(a), **lm(a)}, {**bench(b), **lm(b)}
    for k in sorted(set(A) & set(B)):
        r = boot(A[k], B[k])
        if r:
            res.append(dict(run_a=a, run_b=b, label=lab[0] if lab else "", bench=k, **r))
            print(f"{a} - {b} {k}: n={r['n']} {r['a']} vs {r['b']} diff {r['diff']} [{r['lo']}, {r['hi']}]", flush=True)
json.dump(res, open("/tmp/pairs.json", "w"))
