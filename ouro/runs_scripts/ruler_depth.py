"""Read-depth curves on the full RULER sets (520 rows per length): single-bank connector S<p><d> for p = 1..4 at each
write depth d with full-set runs ('X26-S<p><d>f'), plus native d/d. Prints JSON (run on the B300 box)."""
import glob, json
B = "/dev/shm/big-small-ouro/runs/bench"


def mean(run, bench):
    v = [float(json.loads(l)["score"]) for f in glob.glob(f"{B}/{run}/s*/{bench}.jsonl") for l in open(f)]
    return round(100 * sum(v) / len(v), 1) if len(v) >= 520 else None


out = {}
for d in (1, 2, 3):
    out[d] = {b: dict(read=[mean(f"X26-S{p}{d}f", b) for p in (1, 2, 3, 4)], native=mean(f"X26-n{d}f", b))
              for b in ("ruler4k", "ruler8k", "ruler16k")}
print(json.dumps(out))
