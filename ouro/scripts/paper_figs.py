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
    "font.size": 8, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
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
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.2))
    for ax, (name, g) in zip(axes, CELLS.items()):
        v = np.array(g)
        im = ax.imshow(v, cmap=cmap, norm=norm, origin="lower", aspect="equal")
        for w in range(4):
            for r in range(4):
                ax.text(r, w, f"{v[w, r]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v[w, r] > 70 else INK)
        ax.set_xticks(range(4), [1, 2, 3, 4]); ax.set_yticks(range(4), [1, 2, 3, 4] if ax is axes[0] else [])
        ax.set_title(name, fontsize=8.5, color=INK, pad=4)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("write loops (per token)")
    fig.supxlabel("read loops (prompt)", fontsize=8, y=-0.03)
    cb = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02)
    cb.set_label("accuracy (%)", fontsize=7.5); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7, length=0)
    save(fig, "fig1_grid")


def fig0():
    """Method: read p loops once, keep one prompt bank, write d loops per token through per-loop maps."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(6.9, 1.85))
    ax.set_xlim(0, 103); ax.set_ylim(0, 27); ax.axis("off")

    def box(x, y, w, h, fc, ec, lw=0.7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec=ec, lw=lw))

    def arrow(x0, y0, x1, y1, c=NAT):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=c, shrinkA=1, shrinkB=1))

    ax.text(12, 25.6, "Read: p loops, once", ha="center", fontsize=8.5, color=INK, fontweight="bold")
    for y, lab, last in ((2.0, "loop 1", False), (16.0, "loop p", True)):
        box(3, y, 18, 4.6, "#FBE3E6" if last else "#F2F2F2", OURS if last else NAT_LIGHT)
        ax.text(12, y + 2.3, lab, ha="center", va="center", fontsize=8, color=INK)
    ax.text(12, 11.4, "$\\vdots$", ha="center", va="center", fontsize=10, color=NAT)
    arrow(12, 6.9, 12, 9.4); arrow(12, 13.2, 12, 15.7)
    box(26, 15.2, 21, 6.2, "#FBE3E6", OURS, 1.0)
    ax.text(36.5, 19.3, "prompt KV bank", ha="center", fontsize=8, color=OURS, fontweight="bold")
    ax.text(36.5, 16.4, "last loop, stored once", ha="center", fontsize=7, color=INK)
    arrow(21.6, 18.3, 25.6, 18.3, OURS)
    box(26, 2.0, 21, 6.2, "white", NAT_LIGHT)
    ax.text(36.5, 6.1, "answer KV", ha="center", fontsize=8, color=INK, fontweight="bold")
    ax.text(36.5, 3.2, "generated tokens", ha="center", fontsize=7, color=INK)
    ax.text(76, 25.6, "Write: d loops per token", ha="center", fontsize=8.5, color=INK, fontweight="bold")
    box(53, 2.0, 49, 19.4, "white", OURS, 0.9)
    ax.text(55, 18.6, "writing loop k  (k = 1, …, d)", fontsize=8, color=INK, fontweight="bold")
    ax.text(55, 13.8, r"prompt bank: query scaled by $\bar c_k$, then $W_k$", fontsize=7.5, color=INK)
    ax.text(55, 9.2, "answer KV: ordinary attention", fontsize=7.5, color=INK)
    ax.text(55, 4.6, "the two merged by log-sum-exp", fontsize=7.5, color=INK)
    arrow(47.6, 18.3, 52.6, 15.3, OURS); arrow(47.6, 5.1, 52.6, 8.1)
    save(fig, "fig0_method")


def fig2():
    """End to end: accuracy vs seconds per answer (batch 1)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.35), gridspec_kw=dict(width_ratios=[1.35, 1]))
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
        ax.annotate(n, (P[n][1], P[n][0]), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                    color=OURS if ours else INK, ha=ha, va="center")

    def speed(ax, x0, x1, y, txt):
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", color=OURS, lw=0.9))
        ax.text((x0 + x1) / 2, y + 0.9 if y > 90 else y + 2.2, txt, color=OURS, fontsize=7, ha="center")

    ax = axes[0]
    for n, dx, dy, ha in (("native 1/1", 6, 0, "left"), ("native 2/2", 0, -9, "center"), ("native 4/4", 0, -9, "center")):
        point(ax, E, n, False, dx, dy, ha)
    point(ax, E, "read 4 / write 1", True, -6, 0, "right")
    point(ax, E, "read 4 / write 2", True, 6, 6, "left")
    point(ax, E, "read 2 / write 2", True, -8, -8, "right", hollow=True)
    speed(ax, E["native 4/4"][1], E["read 4 / write 2"][1] + 1.5, 78,
          f"read 4 / write 2: {ratio['2.6B-T GSM8K read 4/write 2 vs native 4/4']:.2f}× the time of native 4/4")
    ax.set_xlabel("seconds per answer"); ax.set_ylabel("GSM8K accuracy (%)")
    ax.set_xlim(40, 125); ax.set_ylim(55, 104)
    ax.set_title("Ouro-2.6B-Thinking", fontsize=8, color=INK, pad=3)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax = axes[1]
    point(ax, S, "native 2/2", False, -6, 0, "right")
    point(ax, S, "read 1 / write 1", True, 6, 3, "left")
    point(ax, S, "read 2 / write 1", True, 6, -3, "left")
    speed(ax, S["native 2/2"][1], S["read 1 / write 1"][1] + 0.8, 92.2,
          f"read 1 / write 1: {ratio['SenseNova GSM8K read 1/write 1 vs native 2/2']:.2f}× the time")
    ax.set_xlabel("seconds per answer"); ax.set_xlim(20, 55); ax.set_ylim(90, 100.5)
    ax.set_title("SenseNova-Loop-9B (y axis zoomed)", fontsize=8, color=INK, pad=3)
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


if __name__ == "__main__":
    fig0(); fig1(); fig2(); fig3(); fig4()
    print("\n".join(sorted(p.name for p in OUT.iterdir())))
