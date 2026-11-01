"""Paper figures (PDF for LaTeX, PNG for README / project page) from results/latest.json and the §4.2 grid.

usage: python scripts/paper_figs.py  ->  paper/figs/fig{1..4}_*.{pdf,png}
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
OURS, NAT, INK, MUTED = "#c2410c", "#57534e", "#1c1917", "#a8a29e"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 300})

# §4.2: one universal connector on Ouro-2.6B; rows = write loops 1..4, columns = read loops 1..4; GSM8K / MMLU / RULER-16K
GRID = [[(46.0, 52.4, 53.2), (53.5, 64.0, 60.8), (52.0, 70.4, 62.1), (55.0, 70.4, 61.9)],
        [(71.3, 53.4, 54.5), (82.0, 68.9, 65.6), (82.0, 75.2, 71.2), (81.8, 74.5, 72.3)],
        [(79.8, 53.3, 55.8), (87.5, 68.3, 67.7), (84.8, 75.0, 73.0), (84.3, 75.1, 73.2)],
        [(77.8, 53.0, 55.2), (86.0, 68.9, 67.6), (85.0, 74.8, 74.4), (85.3, 75.4, 72.5)]]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig0():
    """Method sketch: read p loops once, store one prompt bank, write d loops through per-loop RoPE-commuting maps."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(7.0, 2.25))
    ax.set_xlim(0, 100); ax.set_ylim(0, 32); ax.axis("off")

    def box(x, y, w, h, text, fc="#f5f5f4", ec=NAT, color=INK, size=7, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color,
                fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, color=NAT):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=color))

    ax.text(14, 30.5, "READ (prefill, once)", ha="center", fontsize=7.5, color=NAT, fontweight="bold")
    for i in range(4):
        box(3, 3 + 6.2 * i, 22, 4.6, f"loop {i + 1}: shared block" if i < 3 else "loop p: shared block",
            fc="#e7e5e4" if i < 3 else "#fed7aa")
        if i < 3:
            arrow(14, 7.8 + 6.2 * i, 14, 9.0 + 6.2 * i)
    ax.text(14, 0.6, "prompt (n tokens), p loops", ha="center", fontsize=6.5, color=NAT)
    box(33, 19.4, 17, 8, "prompt KV bank\nlast loop only\nstored ONCE", fc="#fff7ed", ec=OURS, color=OURS, bold=True)
    arrow(25.5, 23.4, 32.6, 23.4, OURS)
    ax.text(41.5, 15.6, "memory: b(n + d·t)\nnative: b·d(n + t)", ha="center", fontsize=6.3, color=NAT)

    ax.text(77.75, 30.5, "WRITE (decode, per token)", ha="center", fontsize=7.5, color=NAT, fontweight="bold")
    for k in range(2):
        y = 15.5 - 11 * k
        box(56.5, y, 42.5, 9.5, "", fc="#fafaf9")
        ax.text(77.75, y + 8.0, f"decode loop {k + 1} of d   (LoRA on)", ha="center", fontsize=6.5, color=NAT)
        ax.text(77.75, y + 5.2, r"$q' = \bar c_k \odot q$: one complex scale per RoPE pair", ha="center", fontsize=5.9, color=INK)
        ax.text(77.75, y + 3.1, "(commutes with RoPE, so the bank is never rewritten)", ha="center", fontsize=5.6, color=NAT)
        ax.text(77.75, y + 1.0, r"attention over bank $\to W_k(\cdot)$; LSE-merge with own KV", ha="center", fontsize=5.9, color=INK)
        arrow(50.6, 23.4 - 0.4 * k, 56.1, y + 4.8, OURS)
    arrow(77.75, 15.3, 77.75, 14.4)
    ax.text(57.5, 1.0, "trained by KL to the native model; one connector serves every (p, d)", ha="center",
            fontsize=6.5, color=NAT, style="italic")
    save(fig, "fig0_method")


def fig1():
    g = np.array(GRID)  # (write, read, metric)
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.3))
    for m, (ax, title) in enumerate(zip(axes, ("MMLU", "RULER 16K", "GSM8K"))):
        idx = {0: 1, 1: 2, 2: 0}[m]
        v = g[:, :, idx]
        ax.imshow(v, cmap="Oranges", origin="lower", vmin=v.min() - 8, vmax=v.max() + 2)
        for w in range(4):
            for r in range(4):
                ax.text(r, w, f"{v[w, r]:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v[w, r] > v.min() + 0.75 * (v.max() - v.min()) else INK)
        ax.set_xticks(range(4), [1, 2, 3, 4]); ax.set_yticks(range(4), [1, 2, 3, 4])
        ax.set_xlabel("read loops (prefill)"); ax.set_title(title, fontsize=9)
        ax.spines[:].set_visible(False); ax.tick_params(length=0)
        if m == 0:
            ax.set_ylabel("write loops (decode)")
    fig.suptitle("Reading depth moves knowledge and long context; writing depth moves derivation (Ouro-2.6B)",
                 fontsize=8.5, y=1.04)
    save(fig, "fig1_grid")


def fig2():
    E = D["e2e"]["rows"]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    nat = [n for n in E if n.startswith("native")]
    ax.plot([E[n][5] for n in nat], [E[n][0] for n in nat], "--", color=MUTED, lw=0.8, zorder=1)
    for n, r in E.items():
        ours = n.startswith("read")
        ax.scatter(r[5], r[0], s=28, color=OURS if ours else NAT, zorder=2)
        if n == "read 2 / write 2":
            continue  # same point as read 4 / write 2 (time ratio 1.01): one label for both
        label = "read 2 or 4 / write 2" if n == "read 4 / write 2" else n
        dy = {"native 2/2": -10.0}.get(n, 2.2)
        ax.annotate(label, (r[5], r[0]), xytext=(4, dy), textcoords="offset points", fontsize=7, color=OURS if ours else INK)
    ax.set_xlabel("seconds per answer (batch 1, prefill included)"); ax.set_ylabel("GSM8K accuracy (%)")
    ax.set_xlim(40, 125); ax.set_ylim(55, 101)
    ax.set_title("Ouro-2.6B-Thinking, end to end", fontsize=8.5)
    save(fig, "fig2_frontier")


def fig3():
    W = D["wander"]
    names = {"原生 1/1": "native 1/1", "读 1 写 1": "read 1 / write 1", "读 4 写 1": "read 4 / write 1",
             "原生 2/2": "native 2/2", "读 4 写 2": "read 4 / write 2", "原生 4/4": "native 4/4"}
    buckets = ["≤2 steps", "3–4 steps", "5+ steps"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))
    style = {"native 1/1": (NAT, "--"), "read 1 / write 1": (MUTED, ":"), "read 4 / write 1": (OURS, "--"),
             "native 2/2": (NAT, "-"), "read 4 / write 2": (OURS, "-"), "native 4/4": (INK, "-.")}
    for row in W["rows"]:
        n = names[row[0]]; c, ls = style[n]
        axes[0].plot(buckets, row[2], ls, color=c, marker="o", ms=3, lw=1.1, label=n)
        axes[1].plot(buckets, row[1], ls, color=c, marker="o", ms=3, lw=1.1, label=n)
    axes[0].set_ylabel("answers hitting the 4,096-token budget (%)"); axes[1].set_ylabel("accuracy (%)")
    axes[0].set_title("Shallow writers run out of budget; deep reading helps them finish", fontsize=8.5)
    axes[1].set_title("…and the gain grows with difficulty", fontsize=8.5)
    for ax in axes:
        ax.set_xlabel("GSM8K problem difficulty (reference solution steps)")
    axes[1].legend(fontsize=6.5, frameon=False, loc="lower left", ncol=2)
    save(fig, "fig3_wandering")


def fig4():
    P = D["probe"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)
    lab = {"tok-1": "token i−1 (what was said)", "tok-2": "token i−2 (what was said)", "next": "token i+1 (what comes next)"}
    for ax, m in zip(axes, ("2.6b", "2.6b-think", "1.4b")):
        for t in ("tok-1", "tok-2", "next"):
            v = np.array(P[m]["mean"][t])
            ax.plot([1, 2, 3, 4], v - v[0], marker="o", ms=3, lw=1.2, color=OURS if t == "next" else NAT,
                    ls="--" if t == "tok-2" else "-", label=lab[t])
            ax.annotate(f"{v[0]:.0f}→{v[3]:.0f}%", (4, v[3] - v[0]), xytext=(4, 0), textcoords="offset points",
                        fontsize=6.5, va="center", color=OURS if t == "next" else NAT)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xticks([1, 2, 3, 4]); ax.set_xlim(0.8, 4.9)
        ax.set_title("Ouro-" + m.upper().replace("-THINK", "-Thinking"), fontsize=8.5)
    axes[0].set_ylabel("probe accuracy vs loop 1 (points)")
    axes[1].set_xlabel("loop whose value cache is probed")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.2))
    fig.suptitle("Linear probes on the stored value cache (same weights, same layer): the literal tokens fade, the next token emerges",
                 fontsize=8.5, y=1.04)
    save(fig, "fig4_probe")


if __name__ == "__main__":
    fig0(); fig1(); fig2(); fig3(); fig4()
    print("\n".join(sorted(p.name for p in OUT.iterdir())))
