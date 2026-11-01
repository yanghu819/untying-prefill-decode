"""One-shot status for the ACL push: GPUs, dispatcher, trainings, bench progress, identical-id comparisons."""
import glob, json, os, subprocess, time

R = "/dev/shm/big-small-ouro/runs"; B = f"{R}/bench"
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
print(time.strftime("%H:%M"), sh("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\\n' ' '"))
idle = [l.split(",")[0] for l in sh("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits").splitlines()
        if int(l.split(",")[1]) < 2000]
qlen = len(open(f"{R}/jobs.txt").read().splitlines())
if idle and qlen == 0:
    print(f"!!! IDLE ALERT: GPUs {idle} idle and the dispatcher queue is empty — queue the next jobs now")
print("queue:", [l.split()[2] if len(l.split()) > 2 else l[:40] for l in open(f"{R}/jobs.txt").read().splitlines()])
print("dispatched:", [l.split(": ")[0][11:] + " " + l.split()[4] for l in open(f"{R}/jobs.txt.log").read().splitlines()][-6:])
now = time.time()
for d in sorted(glob.glob(f"{R}/NB-*-s2") + glob.glob(f"{R}/X26-*-s2") + glob.glob(f"{R}/X26-[LIH]-*") + glob.glob(f"{R}/X26-X??") + glob.glob(f"{R}/X26-[VS]-universal") + glob.glob(f"{R}/X26-[ZY]?")):
    f = f"{d}/train.jsonl"
    if os.path.exists(f) and os.path.getsize(f):
        t = json.loads(open(f).read().splitlines()[-1])
        if "kl_tokens" not in t: continue
        age = int((now - os.path.getmtime(f)) / 60)
        print(f"train {os.path.basename(d):18s} {t['kl_tokens']/1e6:5.1f}M kl={t['kl']:.3f} done={os.path.exists(d + '/TRAIN_DONE')} age={age}m")
running = sh("ps -eo args | grep -E '[b]ench[23]?.sh' | grep -oE 'bench[23]?.sh [A-Za-z0-9-]+' | sort -u").split("\n")
for r in running:
    if not r: continue
    n = r.split()[1]
    files = glob.glob(f"{B}/{n}/s0/*.jsonl")
    age = int((now - max([os.path.getmtime(f) for f in files] or [now])) / 60)
    print(f"bench {n:14s}", {os.path.basename(f)[:-6]: sum(1 for _ in open(f)) for f in files}, f"age={age}m")


def load(n, b):
    out = {}
    for f in glob.glob(f"{B}/{n}/s*/{b}.jsonl"):
        for l in open(f):
            r = json.loads(l); out[r["id"]] = r["score"]
    return out


def compare(title, names, benches):
    for b in benches:
        d = {n: load(n, b) for n in names}
        have = [n for n in names if d[n]]
        if len(have) < 2: continue
        common = set.intersection(*[set(d[n]) for n in have])
        if not common: continue
        print(f"{title:10s} {b:8s} n={len(common):5d}", {n: round(100 * sum(d[n][i] for i in common) / len(common), 1) for n in have})


compare("NB", ["NB-splice", "NB-n1", "NB-E", "NB-F", "NB-Fz", "NB-O-overfill", "NBff-E-p1d1", "NBff-F-p2d1delta",
               "NB-E-s2", "NB-F-s2", "NB-F-L0-10", "NB-F-L11-21", "NB-n2"], ["ruler4k", "ruler8k", "ruler16k"])
compare("NB-H", ["NBH-n1", "NBH-E", "NBH-F", "NB-O-overfillH", "NBff-E-p1d1H", "NBff-F-p2d1deltaH", "NB-E-s2H", "NB-F-s2H", "NBH-n2"], ["helmet"])
compare("IQ-H", ["IQH-n1", "IQH-E", "IQH-F", "IQH-n2"], ["helmet"])
compare("Ouro", ["X26-zero41", "X26-zero42", "X26-zero44", "X26-E-p1d1", "X26-F-delta", "X26-G-delta2"], ["ruler4k", "ruler8k", "ruler16k"])
# depth-allocation grid on the bench-mini ids: natives on the diagonal, specialists, universal connector cells
compare("grid", ["X26-n1", "X26-E-p1d1", "X26-J", "X26-K", "X26-F-delta", "X26-n2", "X26-D-p2d2", "X26-L-p3d2delta",
                 "X26-G-delta2", "X26-n3", "X26-H-p3d3", "X26-I-p4d3delta", "X26-n4"], ["ruler4k", "ruler8k", "ruler16k", "helmet"])
compare("U", [f"X26-U{p}{d}" for p in range(1, 5) for d in range(1, 4) if p >= d], ["ruler4k", "ruler8k", "ruler16k", "helmet"])
compare("U-520", [f"X26-U{c}f" for c in ("11", "41", "22", "42", "33", "43")], ["ruler16k"])
compare("V", [f"X26-V{p}{d}" for d in range(1, 5) for p in range(1, 5)], ["ruler16k"])
compare("S", [f"X26-S{p}{d}" for d in range(1, 5) for p in range(1, 5)], ["ruler16k"])
compare("Z", ["X26-zero41", "X26-zero42", "X26-zero44"] + [f"X26-Z{d}" for d in range(1, 5)], ["ruler16k"])
