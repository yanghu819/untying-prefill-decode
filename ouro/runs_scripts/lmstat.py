"""lmstat.py: lm-eval queue status (in-flight progress) and a results table of finished jobs."""
import glob, json, os, re, subprocess
Q = "/dev/shm/big-small-ouro/runs/lmeval"
print(subprocess.run("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader", shell=True, capture_output=True, text=True).stdout)
done = {l.split()[-1] for l in open(f"{Q}/done.txt")} if os.path.exists(f"{Q}/done.txt") else set()
started = [l.split() for l in open(f"{Q}/running.txt")]
last = {}
for d, t, g, _, name in started:
    last[name] = (g, f"{d} {t}")
print("queue left:", sum(1 for _ in open(f"{Q}/queue.txt")))
for name, (g, t) in last.items():
    log = f"{Q}/logs/{name}.log"
    bars = re.findall(r"(\w+):\s+(\d+)%\|[^|]*\| (\d+)/(\d+) \[([0-9:]+)<([0-9:?]+)", open(log, errors="ignore").read()[-20000:]) if os.path.exists(log) else []
    has = glob.glob(f"{Q}/out/{name}/*/results_*.json")
    if not has:
        print(f"RUN {g} {name:16s} since {t[11:]}", bars[-1][2:] if bars else "", "(stale? no bar)" if not bars else "")
rows = []
for f in sorted(glob.glob(f"{Q}/out/*/*/results_*.json")):
    name = f.split("/")[-3]
    d = json.load(open(f))
    for task, v in d["results"].items():
        if task.startswith(("mmlu_", "bbh_cot_fewshot_")) and task not in ("mmlu_pro",):
            continue
        m = {k.split(",")[0] + ("/" + k.split(",")[1][:5] if "," in k else ""): round(100 * x, 2) for k, x in v.items()
             if isinstance(x, float) and "stderr" not in k}
        rows.append(f"{name:16s} {task:18s} {m}")
print("\n".join(rows))
