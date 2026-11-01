"""HELMET subset scores per run under runs/bench (rows deduplicated by id): category = mean over its datasets
(RAG: NQ/TriviaQA/HotpotQA/PopQA SubEM; Recall: JSON KV SubEM; Re-rank: MS MARCO NDCG@10; ICL: 5 datasets EM)."""
import glob, json, os, sys
R = "/dev/shm/big-small-ouro/runs/bench/"
CATS = dict(RAG=["rag_nq", "rag_triviaqa", "rag_hotpotqa", "rag_popqa"], Recall=["recall_jsonkv"], Rerank=["rerank_msmarco"],
            ICL=["icl_trec_coarse", "icl_trec_fine", "icl_banking77", "icl_clinic150", "icl_nlu"])
def rows(name):
    d = {}
    for p in glob.glob(f"{R}{name}/**/helmet.jsonl", recursive=True):
        for line in open(p):
            try: r = json.loads(line)
            except json.JSONDecodeError: continue
            d[r["id"]] = r
    return list(d.values())
names = sys.argv[1:] or sorted(d for d in os.listdir(R) if os.path.isdir(R + d))
print("setting".ljust(20) + "".join(f"{c}{l // 1024}K".rjust(11) for l in (8192, 16384) for c in CATS) + "      n")
for n in names:
    rs = rows(n)
    if not rs: continue
    cells = []
    for l in (8192, 16384):
        for c, tasks in CATS.items():
            per = [[r["score"] for r in rs if r["task"] == t and r["length"] == l] for t in tasks]
            per = [p for p in per if p]
            cells.append(f"{100 * sum(sum(p) / len(p) for p in per) / len(per):.1f}" if len(per) == len(tasks) else "-")
    print(n.ljust(20) + "".join(c.rjust(11) for c in cells) + f"{len(rs):7d}")
