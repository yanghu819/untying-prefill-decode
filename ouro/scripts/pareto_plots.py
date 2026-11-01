"""Pareto plots for the Ouro-2.6B depth-allocation grid: every (p, d) cell, several cost axes x several benchmarks.

Inputs: results/pareto_data.json (scripts/pareto_data.py) and results/throughput-grid/*.jsonl|*.json.
Cost axes (lower is better):
  compute@N  loop-token FLOP units for an 8K-token prompt + N output tokens, relative to native 1 loop:
             prefill = d (native / symmetric LoRA), p + d (asymmetric delta: prefill model + decode model's own prompt
             pass), p (KV sharing, loop-map connector); decode = d per output token.
  kv         decode-phase KV per sequence in units of one loop's KV (= d; KV sharing keeps 1).
  latency@N  measured B=1 seconds at 8K: prefill + N / decode tok/s (symmetric LoRA = native d with LoRA merged;
             asymmetric cells use their merged-LoRA measurement when present, else native-d decode speed).
  ms/token   measured decode GPU-ms per output token at batch 8 (same sources).
  gpu-s/req@N  measured batch-8 GPU seconds per request: (row-prefill seconds of the batch) / 8 + N / batch tok/s.
A cell is Pareto-optimal on a panel if no other cell is at most as costly and at least as good (one strictly).
Outputs: results/figures/pareto_<metric>.png and results/figures/pareto_summary.md
"""
import glob, json, os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent / "results"
data = {k: v for k, v in json.load(open(ROOT / "pareto_data.json")).items() if not k.startswith("_")}
N_IN = 8192
STYLE = {"native": ("tab:gray", "s"), "sym-LoRA": ("tab:blue", "o"), "asym-delta": ("tab:red", "^"),
         "kv-share": ("tab:green", "x"), "universal": ("tab:purple", "D"), "loopmap": ("tab:orange", "v"), "universal-loopmap": ("tab:brown", "P")}


def measured():
    rows = {}
    for f in glob.glob(str(ROOT / "throughput-grid" / "*.json*")):
        name = os.path.basename(f).split(".")[0]
        text = open(f).read().strip()
        recs = json.loads(text) if text.startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
        rows[name] = {(r["context"], r["batch"]): r for r in recs if not r.get("oom")}
    return rows


M = measured()
SRC = {1: "n1", 2: "n2", 3: "n3", 4: "n4"}
ASYM_SRC = {"p4d1": "F41", "p4d2": "G42", "p4d3": "I43"}


def timing(cell, v):
    """(prefill_s, decode tok/s at B=1, decode tok/s at B=8) or None."""
    d, kind = v["d"], v["kind"]
    if kind in ("native", "sym-LoRA"):
        r = M.get(SRC[d], {})
        if (8192, 1) in r and (8192, 8) in r:
            return (r[(8192, 1)]["prefill_s"], r[(8192, 1)]["tokens_per_s"], r[(8192, 8)]["tokens_per_s"],
                    r[(8192, 8)]["prefill_s"] / 8)
    if cell in ASYM_SRC:
        unmerged, merged = M.get(ASYM_SRC[cell], {}), M.get(ASYM_SRC[cell] + "-merged", {})
        dec = merged if (8192, 8) in merged else M.get(SRC[d], {})
        if (8192, 1) in unmerged and (8192, 1) in dec and (8192, 8) in dec:
            pre8 = (merged if (8192, 8) in merged else unmerged)[(8192, 8)]["prefill_s"] / 8
            return unmerged[(8192, 1)]["prefill_s"], dec[(8192, 1)]["tokens_per_s"], dec[(8192, 8)]["tokens_per_s"], pre8
    return None


def cost(axis, cell, v):
    p, d, kind = v["p"], v["d"], v["kind"]
    if axis.startswith("compute@"):
        n = int(axis.split("@")[1])
        pre = {"native": d, "sym-LoRA": d, "kv-share": p, "loopmap": p, "universal-loopmap": p}.get(kind, p + d)
        dec = d
        return (pre * N_IN + dec * n) / (N_IN + n)
    if axis == "kv":
        return 1 if kind == "kv-share" else d
    t = timing(cell, v)
    if t is None:
        return None
    if axis.startswith("latency@"):
        return t[0] + int(axis.split("@")[1]) / t[1]
    if axis == "ms/token":
        return 1000 / t[2]
    if axis.startswith("gpu-s/req@"):  # batch-8 GPU seconds per request: prefill share + decode of N tokens
        return t[3] + int(axis.split("@")[1]) / t[2] * 1.0


def pareto(points):
    front = set()
    for a, (ca, qa) in points.items():
        if not any(cb <= ca and qb >= qa and (cb < ca or qb > qa) for b, (cb, qb) in points.items() if b != a):
            front.add(a)
    return front


AXES = ["compute@64", "compute@1024", "kv", "latency@256", "ms/token", "gpu-s/req@256"]
METRICS = ["ruler4k", "ruler8k", "ruler16k", "helmet", "longbench", "gsm8k_cot", "bbh_cot_fewshot",
           "minerva_math500", "mmlu", "gpqa_main_n_shot_local"]
out = ROOT / "figures"; out.mkdir(exist_ok=True)
summary = ["# Pareto summary (Ouro-2.6B depth-allocation grid)\n",
           "Asymmetric cells (pXdY, p > d) on the front are marked with *; untrained natives are included.\n"]
for metric in METRICS:
    cells = {c: v for c, v in data.items() if metric in v}
    if len(cells) < 3:
        continue
    fig, axes = plt.subplots(1, len(AXES), figsize=(4.2 * len(AXES), 3.8))
    summary.append(f"\n## {metric}\n")
    for ax, axis in zip(axes, AXES):
        pts = {c: (cost(axis, c, v), v[metric]) for c, v in cells.items()}
        pts = {c: x for c, x in pts.items() if x[0] is not None}
        front = pareto(pts)
        for c, (x, y) in pts.items():
            color, marker = STYLE[cells[c]["kind"]]
            ax.scatter(x, y, c=color, marker=marker, s=60, edgecolors="black" if c in front else "none", linewidths=1.6)
            ax.annotate(c, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        # one line per method family (asymmetric: the p=4 row, i.e. the deepest prefill at each decode depth)
        for kind in ("native", "sym-LoRA", "asym-delta", "kv-share"):
            fam = [c for c in pts if cells[c]["kind"] == kind and (kind != "asym-delta" or cells[c]["p"] == 4)]
            if len(fam) >= 2:
                xy = sorted(pts[c] for c in fam)
                ax.plot([x for x, _ in xy], [y for _, y in xy], "-", color=STYLE[kind][0], lw=1.6 if kind == "asym-delta" else 1.0)
        ax.set_xlabel(axis + (" (rel. native-1)" if axis.startswith("compute") else ""))
        ax.set_ylabel(metric); ax.grid(alpha=0.3)
        asym = sorted(c for c in pts if cells[c]["kind"] == "asym-delta")
        summary.append(f"- {axis}: front = {', '.join(sorted(front))} | asym on front: "
                       f"{', '.join(c + ('*' if c in front else '') for c in asym)}\n")
    handles = [plt.Line2D([], [], color=s[0], marker=s[1], ls="", label=k) for k, s in STYLE.items()]
    fig.legend(handles=handles, loc="upper center", ncol=len(STYLE), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / f"pareto_{metric}.png", dpi=130)
    plt.close(fig)
(out / "pareto_summary.md").write_text("".join(summary))
print("".join(summary))
