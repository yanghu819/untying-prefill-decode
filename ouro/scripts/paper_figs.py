"""Paper figures (PDF for LaTeX, PNG for README) from results/latest.json and results/paper_numbers.json (Fig. 1 grid).

usage: python scripts/paper_figs.py  ->  paper/figs/fig{0..4}_*.{pdf,png}
One palette everywhere: our connector in red, native models in greys; no in-plot titles (captions carry the message).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figs"
D = json.loads((ROOT / "results" / "latest.json").read_text())
OURS, OURS_LIGHT = "#D1495B", "#EDA4AE"
NAT, NAT_LIGHT, INK, GRID = "#4D4D4D", "#B3B3B3", "#1A1A1A", "#E6E6E6"
plt.rcParams.update({
    "font.size": 8, "font.family": "DejaVu Sans", "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "axes.edgecolor": NAT, "xtick.color": NAT, "ytick.color": NAT, "axes.labelcolor": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42})

# Fig. 1 grid: one universal connector on Ouro-2.6B; rows = write loops 1..4, columns = read loops 1..4
PN = json.loads((ROOT / "results" / "paper_numbers.json").read_text())
_M = {m["label"]: m["mean"] for m in PN["means"]}
CELLS = {name: [[_M[f"S{p}{d} {name} ({sub})"] for p in range(1, 5)] for d in range(1, 5)]
         for name, sub in (("MMLU", "20/subject"), ("RULER-16K", "260"), ("GSM8K", "400"))}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig1():
    """The division of labour: MMLU and RULER change along the read axis, GSM8K along the write axis."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    cmap = LinearSegmentedColormap.from_list("ours", ["#FFFFFF", "#F6C9CF", OURS, "#7A1F2B"])
    norm = Normalize(40, 90)
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.55))
    for ax, (name, g) in zip(axes, CELLS.items()):
        v = np.array(g)
        im = ax.imshow(v, cmap=cmap, norm=norm, origin="lower", aspect="equal")
        for w in range(4):
            for r in range(4):
                ax.text(r, w, f"{v[w, r]:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v[w, r] > 70 else INK)
        ax.set_xticks(range(4), [1, 2, 3, 4]); ax.set_yticks(range(4), [1, 2, 3, 4] if ax is axes[0] else [])
        ax.set_title(name, fontsize=9, color=INK, pad=5)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("write loops")
    fig.subplots_adjust(left=.07, right=.99, bottom=.18, top=.90, wspace=.10)
    fig.supxlabel("read loops", fontsize=9, y=.03)
    save(fig, "fig1_grid")


def fig0():
    """Method: read p loops once and keep one prompt KV cache; every writing loop attends to it through its own key
    scale (on the query) and value map, and to the answer's own KV, merged by log-sum-exp."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(6.9, 2.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
    LIGHT = "#FBE3E6"

    def box(x, y, w, h, text, fc="white", ec=NAT_LIGHT, lw=0.7, size=8.5, color=INK, weight="normal", sub=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.9", fc=fc, ec=ec, lw=lw))
        ty = y + h / 2 + (1.1 if sub else 0)
        ax.text(x + w / 2, ty, text, ha="center", va="center", fontsize=size, color=color, fontweight=weight)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 1.5, sub, ha="center", va="center", fontsize=7.5, color=NAT)

    def arrow(x0, y0, x1, y1, c=NAT):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color=c, mutation_scale=8, shrinkA=1, shrinkB=1))

    ax.text(11, 27.8, "Read: p loops, once", ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.text(72, 27.8, "Write: repeat d times per token", ha="center", fontsize=9, color=INK, fontweight="bold")
    box(2, 2.5, 18, 4.6, "loop 1", fc="#F2F2F2")
    ax.text(11, 11.6, "$\\vdots$", ha="center", va="center", fontsize=11, color=NAT)
    box(2, 16.0, 18, 4.6, "loop p", fc=LIGHT, ec=OURS)
    arrow(11, 7.4, 11, 9.6); arrow(11, 13.4, 11, 15.6)
    box(24, 15.4, 21, 6.4, "prompt KV cache", fc=LIGHT, ec=OURS, lw=1.0, color=OURS, weight="bold", size=8,
        sub="last loop, stored once")
    box(24, 2.5, 21, 6.2, "answer KV cache", size=8, sub="generated tokens")
    arrow(20.4, 18.3, 23.6, 18.3, OURS)
    box(53, 15.6, 15, 4.4, r"query $\times\,\bar c_k$", fc=LIGHT, ec=OURS)
    box(72, 15.6, 15, 4.4, r"attend $\to W_k$", fc=LIGHT, ec=OURS)
    box(53, 3.4, 34, 4.4, "attend (ordinary)")
    box(90.5, 9.6, 8.5, 4.8, "LSE\nmerge", size=8)
    ax.text(49, 11.8, "query", ha="center", va="center", fontsize=8, color=INK)
    arrow(49, 13.0, 52.6, 17.6, OURS); arrow(49, 10.6, 52.6, 5.6)
    arrow(68.4, 17.8, 71.6, 17.8, OURS)
    ax.plot([45.4, 79.5], [23.6, 23.6], color=OURS, lw=0.8, solid_capstyle="butt")
    ax.plot([45.4, 45.4], [21.9, 23.6], color=OURS, lw=0.8)
    arrow(79.5, 23.6, 79.5, 20.4, OURS)
    arrow(45.4, 5.6, 52.6, 5.6)
    arrow(87.4, 17.6, 92.4, 14.8, OURS); arrow(87.4, 5.6, 92.4, 9.4)
    save(fig, "fig0_method")


def fig2():
    """End to end: accuracy vs seconds per answer (batch 1)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.65), gridspec_kw=dict(width_ratios=[1.2, 1]))
    names = {"native1st": "native 1/1", "native2st": "native 2/2", "native4st": "native 4/4", "S41sb": "read 4 / write 1",
             "S22sb": "read 2 / write 2", "S42sb": "read 4 / write 2", "native2": "native 2/2", "C11": "read 1 / write 1",
             "C21": "read 2 / write 1"}
    E = {names[r["cond"]]: (r["acc"], r["seconds"]) for r in PN["e2e"] if r["model"] == "think" and r["task"] == "gsm" and r["cond"] in names}
    S = {names[r["cond"]]: (r["acc"], r["seconds"]) for r in PN["e2e"] if r["model"] == "sn" and r["task"] == "gsm" and r["cond"] in names}
    ratio = {r["label"]: r["ratio"] for r in PN["e2e_ratios"]}

    def point(ax, P, n, ours, dx, dy, ha="left", hollow=False):
        c = OURS if ours else NAT
        ax.scatter(P[n][1], P[n][0], s=30 if ours else 24, color="white" if hollow else c, edgecolor=c if hollow else "white",
                   lw=1.0 if hollow else 0.5, zorder=4 if hollow else 3)
        if dx or dy:  # (0, 0): labelled by a separate callout
            ax.annotate(n, (P[n][1], P[n][0]), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                        color=OURS if ours else INK, ha=ha, va="center")

    def speed(ax, x0, x1, y, txt):
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", color=OURS, lw=0.9))
        ax.text((x0 + x1) / 2, y + 0.9 if y > 90 else y + 2.2, txt, color=OURS, fontsize=7, ha="center")

    ax = axes[0]
    for n, dx, dy, ha in (("native 1/1", 6, 0, "left"), ("native 2/2", 6, 0, "left"), ("native 4/4", 0, -9, "center")):
        point(ax, E, n, False, dx, dy, ha)
    point(ax, E, "read 4 / write 1", True, 6, 6, "left")
    point(ax, E, "read 4 / write 2", True, 0, 0, "left")
    point(ax, E, "read 2 / write 2", True, 0, 0, "left", hollow=True)
    r = ratio['2.6B-T GSM8K read 4/write 2 vs native 4/4']
    ax.annotate(f"read 2 or 4 / write 2\n{r:.2f}× the time of native 4/4", (E["read 4 / write 2"][1], E["read 4 / write 2"][0]),
                xytext=(-2, -12), textcoords="offset points", fontsize=8, color=OURS, ha="left", va="top")
    ax.set_xlabel("seconds per answer"); ax.set_ylabel("GSM8K accuracy (%)")
    ax.set_xlim(40, 125); ax.set_ylim(55, 104)
    ax.set_title("Ouro-2.6B-Thinking", fontsize=9, color=INK, pad=3)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax = axes[1]
    point(ax, S, "native 2/2", False, -6, 0, "right")
    point(ax, S, "read 2 / write 1", True, 6, -6, "left")
    point(ax, S, "read 1 / write 1", True, 0, 0, "left")
    r = ratio['SenseNova GSM8K read 1/write 1 vs native 2/2']
    ax.annotate(f"read 1 / write 1\n{r:.2f}× the time of native 2/2", (S["read 1 / write 1"][1], S["read 1 / write 1"][0]),
                xytext=(8, 0), textcoords="offset points", fontsize=8, color=OURS, ha="left", va="center")
    ax.text(0.97, 0.04, "y: 90–100%", transform=ax.transAxes, fontsize=7.5, color=NAT, ha="right")
    ax.set_xlabel("seconds per answer"); ax.set_xlim(20, 55); ax.set_ylim(90, 100.5)
    ax.set_title("SenseNova-Loop-9B", fontsize=9, color=INK, pad=3)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    fig.subplots_adjust(wspace=0.22)
    save(fig, "fig2_frontier")


def fig3():
    """Deep reading helps a shallow writer: (a) Ouro-2.6B-Thinking by difficulty; (b) SenseNova without training."""
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.3), gridspec_kw=dict(width_ratios=[1, 1.15]))
    W = PN["wander"]
    buckets = ["≤2", "3–4", "5+"]
    ax = axes[0]
    for name, c, ls in (("native 1/1", NAT_LIGHT, "--"), ("read 1 / write 1", OURS_LIGHT, "-"), ("read 4 / write 1", OURS, "-")):
        v = W[name]["no_answer_by_steps"]
        ax.plot(range(3), v, ls, color=c, marker="o", ms=3.5, lw=1.4)
        ax.text(2.12, v[-1], name, fontsize=7, color=OURS if name.startswith("read") else NAT, va="center")
    ax.set_xticks(range(3), buckets); ax.set_xlim(-0.2, 3.1)
    ax.set_ylabel("no final answer (%)"); ax.set_xlabel("reference solution steps (GSM8K)")
    ax.set_ylim(0, 60)
    ax.set_title("Ouro-2.6B-Thinking, one writing loop", fontsize=8, color=INK, pad=3)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax = axes[1]
    metrics = ["GSM8K:\ngives an answer", "RULER-4K\naccuracy", "MMLU-Pro\naccuracy"]
    mean = lambda c, what: _M[f"SenseNova {c} {what}"]  # noqa: E731
    vals = {lab: [100 - mean(c, "GSM8K no final answer"), mean(c, "RULER-4K"), mean(c, "MMLU-Pro")]
            for lab, c in (("native 1/1", "native1"), ("read 2 / write 1", "splice21"), ("native 2/2", "native2"))}
    cols = {"native 1/1": NAT_LIGHT, "read 2 / write 1": OURS, "native 2/2": NAT}
    x = np.arange(3); wbar = 0.26
    for j, (n, v) in enumerate(vals.items()):
        bars = ax.bar(x + (j - 1) * wbar, v, wbar * 0.92, color=cols[n], label=n)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 1.5, f"{val:.0f}" if val < 99.5 else f"{val:.1f}", ha="center", fontsize=6.8, color=INK)
    ax.set_xticks(x, metrics, fontsize=7); ax.set_ylim(0, 128); ax.set_yticks(range(0, 101, 20)); ax.set_ylabel("%")
    ax.legend(fontsize=6.8, loc="upper center", ncol=3, handlelength=1.0, columnspacing=0.8, bbox_to_anchor=(0.5, 1.01))
    ax.set_title("SenseNova-Loop-9B, no training", fontsize=8, color=INK, pad=3)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "fig3_wandering")


def fig4():
    """Linear probes on the stored value cache, same weights and layer: the literal token fades, the next token emerges."""
    P = D["probe"]
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.0), sharey=True)
    series = (("next", OURS, "-", "next token (i+1)"), ("tok-1", NAT, "-", "previous token (i−1)"),
              ("tok-2", NAT_LIGHT, "--", "token i−2"))
    panels = [("2.6b", "Ouro-2.6B"), ("2.6b-think", "Ouro-2.6B-Thinking"), ("1.4b", "Ouro-1.4B")]
    for ax, (m, title) in zip(axes, panels):
        for t, c, ls, lab in series:
            v = np.array(P[m]["mean"][t])
            ax.plot([1, 2, 3, 4], v - v[0], ls, color=c, marker="o", ms=2.8, lw=1.2, label=lab)
        ax.set_title(title, fontsize=7.5, color=INK, pad=3)
    sn = D["probe_sn"]
    for t, c, ls, lab in series:
        v = np.array(sn[t]); axes[3].plot([1, 2], v - v[0], ls, color=c, marker="o", ms=2.8, lw=1.2)
    axes[3].set_title("SenseNova-Loop-9B", fontsize=7.5, color=INK, pad=3)
    for ax in axes:
        ax.set_xticks([1, 2, 3, 4]); ax.set_xlim(0.7, 4.3)
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    axes[0].set_ylabel("Δ probe accuracy vs\nloop 1 (points)")
    fig.supxlabel("loop (visit) whose value cache is probed", fontsize=8, y=-0.09)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 1.1))
    fig.subplots_adjust(wspace=0.12)
    save(fig, "fig4_probe")


def fig_equal():
    """The headline: reading 4 loops and writing 2 has the decode depth of native 2/2 and nearly the quality of 4/4."""
    E = {e["bench"]: e for e in PN["equal"]}   # all three models on the same items per benchmark
    benches = ["GSM8K", "MATH-500", "MMLU-Pro", "ARC-C", "BBH", "HumanEval", "MBPP+", "HELMET"]
    rows = np.array([(E[b]["native2"], E[b]["ours"], E[b]["native4"]) for b in benches])
    fig, ax = plt.subplots(figsize=(6.9, 2.8))
    y = np.arange(len(benches))[::-1]
    for yi, (n2, ours, n4) in zip(y, rows):
        ax.plot([n2, ours], [yi, yi], color="#D4D4D4", lw=1.0, zorder=1)
    ax.scatter(rows[:, 0], y, s=22, color="#A6A6A6", zorder=2, label="native 2/2")
    ax.scatter(rows[:, 2], y, s=65, marker="|", color=NAT, linewidths=1.6, zorder=3, label="native 4/4")
    ax.scatter(rows[:, 1], y, s=32, color=OURS, zorder=4, label="read 4 / write 2")
    for yi, v in zip(y, rows[:, 1]):
        ax.annotate(f"{v:.0f}", (v, yi), xytext=(5, 5), textcoords="offset points", fontsize=8, color=OURS)
    ax.set_yticks(y, benches, fontsize=8.5)
    ax.set_xlim(40, 102); ax.set_xticks([40, 60, 80, 100]); ax.set_xlabel("score (%)")
    ax.grid(axis="x", color="#E8E8E8", lw=0.5, zorder=0); ax.set_axisbelow(True)
    for sp in ("left", "top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    h, l = ax.get_legend_handles_labels()
    order = [l.index("native 2/2"), l.index("read 4 / write 2"), l.index("native 4/4")]
    ax.legend([h[k] for k in order], [l[k] for k in order], loc="lower center", ncol=3, fontsize=8,
              bbox_to_anchor=(0.45, 1.0), columnspacing=1.2, handletextpad=0.4)
    save(fig, "fig_equal")

def _tput(cond):
    """Best decode tokens/s over tested batch sizes at 8K context, one B300 (results/throughput-sb4)."""
    rows = json.loads((ROOT / "results" / "throughput-sb4" / f"{cond}.json").read_text())
    return max(r["tokens_per_s"] for r in rows if "tokens_per_s" in r and r["context"] == 8192)


def fig_pareto():
    """Accuracy vs decode throughput: native d/d against read 4 / write d, d = 1..4."""
    C = {c["label"]: c for c in PN["comparisons"] if c["section"] == "Pareto"}
    panels = [("MMLU (20/subject)", "MMLU"), ("RULER-16K (260)", "RULER-16K"), ("GSM8K (400)", "GSM8K"),
              ("GSM8K Thinking (full)", "GSM8K (Thinking)")]
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.1))
    for ax, (key, title) in zip(axes, panels):
        nat, ours = [], []
        for d in range(1, 5):
            c = C.get(f"read 4/write {d} vs native {d}/{d} {key}")
            if c is not None and c.get("n", 0):
                nat.append((_tput(f"native{d}st"), c["b"], d)); ours.append((_tput(f"S4{d}sb"), c["a"], d))
            elif key.startswith("GSM8K Thinking"):  # read 4 / write 4 not evaluated on the Thinking model
                nat.append((_tput(f"native{d}st"), _M[f"GSM8K full think-native{d}"], d))
        for pts, col, lab, mk in ((nat, NAT_LIGHT, "native d/d", "o"), (ours, OURS, "read 4 / write d", "o")):
            x, y = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(x, y, "-", color=col, lw=1.4, zorder=2)
            ax.scatter(x, y, s=18, color=col, zorder=3, label=lab)
            for xi, yi, d in pts:
                ax.annotate(f"{d}", (xi, yi), xytext=(0, 5 if col == OURS else -9), textcoords="offset points",
                            fontsize=6.8, color=OURS if col == OURS else NAT, ha="center")
        ax.set_xscale("log"); ax.set_xticks([125, 250, 500, 1000], ["125", "250", "500", "1k"])
        ax.minorticks_off(); ax.set_xlim(100, 1100)
        ax.set_title(title, fontsize=8.5, color=INK, pad=4)
        ax.grid(color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("accuracy (%)")
    fig.supxlabel("decode throughput (tokens/s, log scale)", fontsize=8.5, y=-0.04)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.1), handlelength=1.2)
    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig_pareto")


def fig_lev():
    """Decision models (Lev): accuracy vs decision depth; reading the context with 4 loops after training."""
    L = PN.get("lev", [])
    def series(label, which):
        pts = sorted((r["depth"], r[which]) for r in L if r["label"] == label)
        return [p[0] for p in pts], [p[1] for p in pts]
    models = [m for m in ("Lev-1.4B", "Lev-2.6B", "Lev-3B") if any(r["label"] == f"{m} read-deep trained vs same-training control" for r in L)]
    fig, axes = plt.subplots(1, len(models), figsize=(6.9, 2.2), squeeze=False, sharey=False)
    for ax, m in zip(axes[0], models):
        d, ours = series(f"{m} read-deep trained vs same-training control", "a")
        _, ctl = series(f"{m} read-deep trained vs same-training control", "b")
        _, orig = series(f"{m} control vs original", "b") if any(r["label"] == f"{m} control vs original" for r in L) else (d, None)
        if orig:
            ax.plot(d, orig, "--", color=NAT_LIGHT, marker="o", ms=3.5, lw=1.2, label="Lev (released)")
        ax.plot(d, ctl, "-", color=NAT, marker="o", ms=3.5, lw=1.4, label="Lev, same continued training")
        ax.plot(d, ours, "-", color=OURS, marker="o", ms=4, lw=1.6, label="read deep, decide with d loops")
        ax.set_xticks(d); ax.set_xlabel("decision loops d"); ax.set_title(m + (" (Nanbeige, read 2)" if m == "Lev-3B" else " (Ouro, read 4)"), fontsize=8.5, color=INK, pad=3)
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0][0].set_ylabel("accuracy (%)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.1), handlelength=1.6)
    fig.subplots_adjust(wspace=0.28)
    save(fig, "fig_lev")


if __name__ == "__main__":
    fig0(); fig1(); fig2(); fig3(); fig4(); fig_equal(); fig_pareto(); fig_lev()
    print("\n".join(sorted(p.name for p in OUT.iterdir())))
