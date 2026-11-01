"""RULER at 4K / 8K / 16K for native d/d, the single-bank connector (read 4 and read d at write d) and the re-read
specialist (read 4). Uses the full 520-row sets ('<name>f' runs) when present, else bench-mini (260 rows; field 'n').
Paired bootstraps: read 4 minus read d inside the single-bank connector ('diff'), and read 4 write d minus native d/d
('vs_native'). Prints JSON (run on the B300 box)."""
import glob, json, random
B = "/dev/shm/big-small-ouro/runs/bench"
REREAD = {1: "X26-F-delta", 2: "X26-G-delta2", 3: "X26-I-p4d3delta"}


def load(name, bench):
    for run in (f"{name}f", name):  # full set first
        out = {}
        for f in glob.glob(f"{B}/{run}/s*/{bench}.jsonl"):
            for l in open(f):
                r = json.loads(l); out[r["id"]] = float(r["score"])
        if len(out) >= 520 or (run == name and len(out) >= 260):
            return out
    return {}


def mean(x):
    return round(100 * sum(x.values()) / len(x), 1) if x else None


def boot(a, b):
    ids = sorted(set(a) & set(b))
    if len(ids) < 260:
        return None
    d = [a[i] - b[i] for i in ids]; n = len(d); rng = random.Random(0)
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
    return [round(100 * sum(d) / n, 1), round(100 * bs[49], 1), round(100 * bs[1949], 1), n]


res = []
for d in (1, 2, 3, 4):
    row = {"d": d}
    for bench in ("ruler4k", "ruler8k", "ruler16k"):
        s4, sd, nat = load(f"X26-S4{d}", bench), load(f"X26-S{d}{d}", bench), load(f"X26-n{d}", bench)
        row[bench] = dict(native=mean(nat), ours=mean(s4), sym=mean(sd), n=len(s4),
                          reread=mean(load(REREAD[d], bench)) if d in REREAD else None,
                          diff=boot(s4, sd) if d < 4 else None, vs_native=boot(s4, nat))
    res.append(row)
print(json.dumps(res))
