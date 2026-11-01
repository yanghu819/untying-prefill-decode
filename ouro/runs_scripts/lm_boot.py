"""Paired bootstrap over lm-eval log_samples: lm_boot.py OUT_DIR_A OUT_DIR_B METRIC (e.g. acc / exact_match)."""
import glob, json, os, random, sys
def load(d, metric):
    out = {}
    for f in glob.glob(f"{d}/*/samples_*.jsonl"):
        task = os.path.basename(f).split("samples_")[1].rsplit("_", 1)[0]
        for l in open(f):
            r = json.loads(l)
            v = r.get(metric)
            if v is None:
                v = next((x for k, x in r.items() if k.startswith(metric) and "flexible" in k), None)
            if v is not None:
                out[(task, r["doc_id"])] = float(v)
    return out
a, b = load(sys.argv[1], sys.argv[3]), load(sys.argv[2], sys.argv[3])
ids = sorted(set(a) & set(b)); n = len(ids); d = [a[i] - b[i] for i in ids]; rng = random.Random(0)
boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
print(json.dumps(dict(a=os.path.basename(sys.argv[1]), b=os.path.basename(sys.argv[2]), n=n,
    a_mean=round(100*sum(a[i] for i in ids)/n, 2), b_mean=round(100*sum(b[i] for i in ids)/n, 2),
    diff=round(100*sum(d)/n, 2), ci95=[round(100*boots[49], 2), round(100*boots[1949], 2)], p_le0=sum(v <= 0 for v in boots)/2000)))
