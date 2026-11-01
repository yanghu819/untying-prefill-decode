"""One row per run under runs/bench: every benchmark from evaluate_bench.py (rows deduplicated by id).
RULER: mean over tasks of per-task mean recall; LongBench: HotpotQA / 2Wiki F1; HELMET: category means
(RAG SubEM over NQ/TQA/HotpotQA/PopQA, JSON-KV recall, MS MARCO NDCG@10, ICL over 5 datasets) at 8K and 16K;
BBH/MMLU: macro over tasks; MATH-500, GSM8K: accuracy. Cell = score (n) while incomplete."""
import glob, json, os, sys
R = "/dev/shm/big-small-ouro/runs/bench/"
FULL = dict(ruler4k=520, ruler8k=520, ruler16k=520, longbench=400, bbh=6511, math500=500, mmlu=14042)
HEL = dict(RAG=["rag_nq", "rag_triviaqa", "rag_hotpotqa", "rag_popqa"], KV=["recall_jsonkv"], Rerank=["rerank_msmarco"],
           ICL=["icl_trec_coarse", "icl_trec_fine", "icl_banking77", "icl_clinic150", "icl_nlu"])
def rows(name, bench):
    d = {}
    for p in glob.glob(f"{R}{name}/**/{bench}.jsonl", recursive=True):
        for line in open(p):
            try: r = json.loads(line)
            except json.JSONDecodeError: continue
            d[r["id"]] = r
    return list(d.values())
def val(r): return r["score"] if "score" in r else float(r["correct"])
def macro(rs):
    by = {}
    for r in rs: by.setdefault(r["task"], []).append(val(r))
    return 100 * sum(sum(v) / len(v) for v in by.values()) / len(by)
def cell(x, n, full): return f"{x:.1f}" + ("" if n >= full else f"({n})")
names = sys.argv[1:] or sorted(d for d in os.listdir(R) if os.path.isdir(R + d))
cols = ["R4K", "R8K", "R16K", "Hotpot", "2Wiki"] + [f"{c}{l}" for l in ("8K", "16K") for c in HEL] + ["BBH", "MATH", "MMLU"]
print("setting".ljust(18) + "".join(c.rjust(11) for c in cols))
for n in names:
    out = []
    for b in ("ruler4k", "ruler8k", "ruler16k"):
        rs = rows(n, b); out.append(cell(macro(rs), len(rs), 520) if rs else "-")
    lb = rows(n, "longbench")
    for t in ("hotpotqa", "2wikimqa"):
        rs = [r for r in lb if r["task"] == t]; out.append(cell(100 * sum(val(r) for r in rs) / len(rs), len(rs), 200) if rs else "-")
    h = rows(n, "helmet")
    for l in (8192, 16384):
        for c, tasks in HEL.items():
            per = [[val(r) for r in h if r["task"] == t and r["length"] == l] for t in tasks]
            out.append(f"{100 * sum(sum(p) / len(p) for p in per) / len(per):.1f}" if all(per) else "-")
    for b in ("bbh", "math500", "mmlu"):
        rs = rows(n, b)
        out.append(cell(macro(rs) if b != "math500" else 100 * sum(val(r) for r in rs) / len(rs), len(rs), FULL[b]) if rs else "-")
    print(n.ljust(18) + "".join(c.rjust(11) for c in out))
