"""msfetch.py REPO OUT : download a ModelScope repo with 8-way ranged parallel downloads; skips complete files."""
import json, os, subprocess, sys, urllib.request, concurrent.futures as cf
repo, out = sys.argv[1], sys.argv[2]
api = f"https://modelscope.cn/api/v1/models/{repo}/repo/files?Recursive=true"
files = [f for f in json.load(urllib.request.urlopen(api))["Data"]["Files"] if f["Type"] == "blob"
         and not f["Path"].endswith((".gif", ".jpg", ".jpeg", ".png", ".pdf", ".gitattributes"))]
def url(p): return f"https://modelscope.cn/models/{repo}/resolve/master/{p}"
def rng(job):
    p, a, b, u = job
    for _ in range(60):
        have = os.path.getsize(p) if os.path.exists(p) else 0
        if have >= b - a + 1: return
        subprocess.run(["curl", "-sL", "-m", "1800", "-r", f"{a + have}-{b}", u, "-o", "-"], stdout=open(p, "ab"))
    raise RuntimeError(p)
jobs, big = [], []
for f in files:
    dst = os.path.join(out, f["Path"]); os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) == f["Size"]: continue
    step = -(-f["Size"] // 4) if f["Size"] > 2e8 else f["Size"]
    n = -(-f["Size"] // step) if f["Size"] else 1
    big.append((dst, n))
    jobs += [(f"{dst}.part{i}", i * step, min(f["Size"], (i + 1) * step) - 1, url(f["Path"])) for i in range(n)]
with cf.ThreadPoolExecutor(16) as ex: list(ex.map(rng, jobs))
for dst, n in big:
    with open(dst, "wb") as w:
        for i in range(n):
            with open(f"{dst}.part{i}", "rb") as g:
                while (c := g.read(1 << 24)): w.write(c)
            os.remove(f"{dst}.part{i}")
bad = [f["Path"] for f in files if os.path.getsize(os.path.join(out, f["Path"])) != f["Size"]]
print(repo, "OK" if not bad else f"BAD {bad}", flush=True)
