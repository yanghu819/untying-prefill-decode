"""Paired bootstrap that resamples whole clusters of rows (HELMET: the same source question at several context lengths,
id = helmet/<task>/<length>/<index> -> cluster helmet/<task>/<index>; RULER rows are independent). Answers the review's
point that row-level resampling understates the CI when items share a question.
usage: cluster_boot.py RUNS_DIR BENCH A:B [A:B ...]"""
import glob, json, random, sys


def load(root, run, bench):
    return {json.loads(l)["id"]: float(json.loads(l)["score"]) for f in glob.glob(f"{root}/bench/{run}/s*/{bench}.jsonl") for l in open(f)}


def cluster(i):
    p = i.split("/")
    return "/".join(p[:2] + p[3:]) if p[0] == "helmet" and len(p) == 4 else i


def boot(a, b, reps=2000):
    ids = sorted(set(a) & set(b))
    groups = {}
    for i in ids:
        groups.setdefault(cluster(i), []).append(a[i] - b[i])
    keys = sorted(groups); rng = random.Random(0)
    def mean(sample):
        tot = sum(sum(groups[k]) for k in sample); cnt = sum(len(groups[k]) for k in sample)
        return tot / cnt
    bs = sorted(mean([keys[rng.randrange(len(keys))] for _ in keys]) for _ in range(reps))
    return dict(n=len(ids), clusters=len(keys), a=round(100 * sum(a[i] for i in ids) / len(ids), 1),
                b=round(100 * sum(b[i] for i in ids) / len(ids), 1), diff=round(100 * mean(keys), 1),
                lo=round(100 * bs[int(0.025 * reps)], 1), hi=round(100 * bs[int(0.975 * reps) - 1], 1))


root, bench = sys.argv[1], sys.argv[2]
for spec in sys.argv[3:]:
    x, y = spec.split(":")
    print(f"{x} - {y} {bench}:", boot(load(root, x, bench), load(root, y, bench)))
