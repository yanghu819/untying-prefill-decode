"""BBH / MATH-500 / MMLU scores per run under runs/bench (rows deduplicated by id).
BBH and MMLU: mean over tasks/subjects of per-task accuracy (lm-eval group averages); MATH-500: accuracy."""
import glob, json, os, sys
R = "/dev/shm/big-small-ouro/runs/bench/"
N = dict(bbh=6511, math500=500, mmlu=14042)
def rows(name, bench):
    d = {}
    for p in glob.glob(f"{R}{name}/**/{bench}.jsonl", recursive=True):
        for line in open(p):
            try: r = json.loads(line)
            except json.JSONDecodeError: continue
            d[r["id"]] = r
    return list(d.values())
def score(rs, macro):
    if not macro: return 100 * sum(r["correct"] for r in rs) / len(rs)
    by = {}
    for r in rs: by.setdefault(r["task"], []).append(r["correct"])
    return 100 * sum(sum(v) / len(v) for v in by.values()) / len(by)
names = sys.argv[1:] or sorted(d for d in os.listdir(R) if os.path.isdir(R + d))
print("setting".ljust(20) + "".join(b.rjust(14) for b in N))
for n in names:
    cells = []
    for b in N:
        rs = rows(n, b)
        cells.append(f"{score(rs, b != 'math500'):.1f}({len(rs)})" if rs else "-")
    print(n.ljust(20) + "".join(c.rjust(14) for c in cells))
