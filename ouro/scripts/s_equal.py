"""Equal-decode comparisons for the single-bank universal connector (arm S): read 4 vs read d at write d.
Uses full test sets when both sides have them (lm-eval '-gsmfull'/'-mmlufull', bench '<name>f' with 520 RULER rows),
otherwise the sampled sets; paired bootstrap over shared items. Prints JSON (run on the B300 box)."""
import glob, json, os, random, sys
R = "/dev/shm/big-small-ouro/runs"; Q = f"{R}/lmeval/out"; B = f"{R}/bench"


def lm(name, metric):
    out = {}
    for f in glob.glob(f"{Q}/{name}/*/samples_*.jsonl"):
        task = os.path.basename(f).split("samples_")[1].rsplit("_", 1)[0]
        for l in open(f):
            r = json.loads(l)
            v = r.get(metric)
            if v is None:
                v = next((x for k, x in r.items() if k.startswith(metric) and "flexible" in k), None)
            if v is not None:
                out[(task, r["doc_id"])] = float(v)
    return out


def bench(name):
    out = {}
    for f in glob.glob(f"{B}/{name}/s*/ruler16k.jsonl"):
        for l in open(f):
            r = json.loads(l); out[r["id"]] = float(r["score"])
    return out


def boot(a, b):
    ids = sorted(set(a) & set(b)); n = len(ids); d = [a[i] - b[i] for i in ids]; rng = random.Random(0)
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
    return dict(n=n, a=100 * sum(a[i] for i in ids) / n, b=100 * sum(b[i] for i in ids) / n, diff=100 * sum(d) / n,
                lo=100 * bs[49], hi=100 * bs[1949])


res = []
for d in (1, 2, 3):
    a, b = f"S4{d}", f"S{d}{d}"
    row = {"d": d}
    for key, metric, full, part in (("gsm8k_cot", "exact_match", "-gsmfull", "-gsm"), ("mmlu", "acc", "-mmlufull", "-mmlu")):
        both_full = all(glob.glob(f"{Q}/X26-{x}{full}/*/results_*.json") for x in (a, b))
        suf = full if both_full else part
        row[key] = dict(full=both_full, **boot(lm(f"X26-{a}{suf}", metric), lm(f"X26-{b}{suf}", metric)))
    fa, fb = bench(f"X26-{a}f"), bench(f"X26-{b}f")
    full = len(fa) >= 520 and len(fb) >= 520
    row["ruler16k"] = dict(full=full, **boot(*(fa, fb) if full else (bench(f"X26-{a}"), bench(f"X26-{b}"))))
    res.append(row)
print(json.dumps(res))
