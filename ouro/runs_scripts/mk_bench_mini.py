"""Fixed evaluation subsets for fast iteration (ids are a subset of the full files, so paired comparisons with
full-set results stay valid): RULER every 2nd row (260/length), HELMET every 6th row (~1.1K, file order = task order)."""
import json
from pathlib import Path
src, dst = Path("/dev/shm/big-small-ouro/bench"), Path("/dev/shm/big-small-ouro/bench-mini")
dst.mkdir(exist_ok=True)
for name, step in (("ruler4k", 2), ("ruler8k", 2), ("ruler16k", 2), ("helmet", 6)):
    rows = open(src / f"{name}.jsonl").read().splitlines()
    (dst / f"{name}.jsonl").write_text("".join(r + "\n" for r in rows[::step]))
    print(name, len(rows), "->", len(rows[::step]))
