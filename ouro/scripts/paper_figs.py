"""Paper figures (PDF for LaTeX, PNG for README) and generated LaTeX tables, all from results/paper_numbers.json.

usage: python scripts/paper_figs.py  ->  paper/figs/fig*.{pdf,png}, paper/figs/tab_*.tex
One palette everywhere: our loop adapter in red, native models in greys; no in-plot titles (captions carry the message).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]                 # ouro/
OUT = ROOT.parent / "paper" / "figs"                       # paper/figs at the repository root
OURS, OURS_LIGHT = "#D1495B", "#EDA4AE"
NAT, NAT_LIGHT, INK, GRID = "#4D4D4D", "#B3B3B3", "#1A1A1A", "#E6E6E6"
plt.rcParams.update({
    "font.size": 8, "font.family": "DejaVu Sans", "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "axes.edgecolor": NAT, "xtick.color": NAT, "ytick.color": NAT, "axes.labelcolor": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42})

# Fig. 1 grid: one loop adapter on Ouro-2.6B; rows = decode loops 1..4, columns = prefill loops 1..4
PN = json.loads((ROOT / "results" / "paper_numbers.json").read_text())
_M = {m["label"]: m["mean"] for m in PN["means"]}
CELLS = {name: [[_M[f"S{p}{d} {name} ({sub})"] for p in range(1, 5)] for d in range(1, 5)]
         for name, sub in (("MMLU", "20/subject"), ("RULER-16K", "260"), ("GSM8K", "400"))}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)



def ci(lo, hi):
    """95% interval in points: one decimal, two when one would round a nonzero endpoint to 0.0 (significance is decided
    on the unrounded values)."""
    f = lambda x: f"{x:.2f}" if x != 0 and abs(x) < 0.05 else f"{x + 0.0:.1f}"
    return rf"\ci{{{f(lo)}}}{{{f(hi)}}}"


def frontier(ax, nat, ours, title, xlog=False, lines=True):
    """The one figure style of the paper: native (grey) and loop adapter (red) as lines of (x, y, label) points."""
    for pts, col, dy, mk in ((nat, NAT_LIGHT, -10, "o"), (ours, OURS, 5, "s")):
        pts = [p for p in pts if np.isfinite(p[1])]
        if not pts:
            continue
        x, y = [p[0] for p in pts], [p[1] for p in pts]
        if lines:
            ax.plot(x, y, "-", color=col, lw=1.4, zorder=2)
        ax.scatter(x, y, s=18, color=col, marker=mk, zorder=3)
        for xi, yi, lab, *off in pts:                 # optional 4th element: this label's vertical offset
            if lab:
                ax.annotate(lab, (xi, yi), xytext=(0, off[0] if off else dy), textcoords="offset points", fontsize=8,
                            color=OURS if col == OURS else NAT, ha="center")
    ys = [p[1] for p in nat + ours if np.isfinite(p[1])]
    lo, hi = min(ys), max(ys); pad = 0.14 * (hi - lo + 1)
    ax.set_ylim(lo - pad, hi + pad)
    if xlog:
        ax.set_xscale("log"); ax.minorticks_off()
    ax.set_title(title, fontsize=8.5, color=INK, pad=4)
    ax.grid(color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def legend(fig, ours, labels, y=1.1, native="native n/n", ncol=3):
    h = [plt.Line2D([], [], color=NAT_LIGHT, marker="o", ms=4, lw=1.4), plt.Line2D([], [], color=OURS, marker="s", ms=4, lw=1.4),
         plt.Line2D([], [], ls="none")]
    fig.legend(h, [native, ours, labels], loc="upper center", ncol=ncol, fontsize=8, bbox_to_anchor=(0.5, y),
               handlelength=1.6, frameon=False)


def fig0():
    """Method: prefill m loops once and keep one prompt KV cache; every decode loop attends to it through its own key
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

    ax.text(11, 27.8, "Prefill: m loops, once", ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.text(72, 27.8, "Decode loop k, repeated n times per token", ha="center", fontsize=9, color=INK, fontweight="bold")
    box(2, 2.5, 18, 4.6, "loop 1", fc="#F2F2F2")
    ax.text(11, 11.6, "$\\vdots$", ha="center", va="center", fontsize=11, color=NAT)
    box(2, 16.0, 18, 4.6, "loop m", fc=LIGHT, ec=OURS)
    arrow(11, 7.4, 11, 9.6); arrow(11, 13.4, 11, 15.6)
    box(24, 15.4, 21, 6.4, "prompt KV cache", fc=LIGHT, ec=OURS, lw=1.0, color=OURS, weight="bold", size=8,
        sub="prefill loop m-n+k")
    box(24, 2.5, 21, 6.2, "answer KV", size=8, sub="of decode loop k")
    arrow(20.4, 18.3, 23.6, 18.3, OURS)
    box(53, 15.6, 15, 4.4, r"query $\times\,\bar c_k$", fc=LIGHT, ec=OURS)
    box(72, 15.6, 15, 4.4, r"attention $\to W_k$", fc=LIGHT, ec=OURS, size=7.5)
    box(53, 3.4, 34, 4.4, "answer attention (causal)")
    box(90.5, 9.6, 8.5, 4.8, "joint\nsoftmax", size=8)
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
    """Ouro-2.6B-Thinking: average accuracy over the eight benchmarks of Figure 4 against decode throughput
    (paper_numbers.json, think_frontier)."""
    F = {r["config"]: (r["tokens_per_s"], r["avg"]) for r in PN.get("think_frontier", [])}
    if len(F) < 5:
        return
    nat = [(*F["native 4/4"], "4/4"), (*F["native 2/2"], "2/2", -11)]
    ours = [(*F["adapter 2/2"], "2/2", -11), (*F["adapter 4/2"], "4/2", 6)]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    frontier(ax, nat, ours, "")
    x, y = F["distilled 2/2"]
    ax.scatter([x], [y], s=18, color=NAT_LIGHT, marker="D", zorder=3)
    ax.annotate("2/2 distilled", (x, y), xytext=(-4, 5), textcoords="offset points", fontsize=8, color=NAT, ha="right")
    ax.set_xlim(80, 460)
    ax.set_xlabel("decode throughput (tokens/s, 8K prompt)", fontsize=8)
    ax.set_ylabel("average accuracy (%)", fontsize=8)
    legend(fig, "loop adapter", "labels: prefill/decode loops", y=1.02, ncol=2)
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.18, top=0.8)
    save(fig, "fig2_frontier")


def fig_equal():
    """Ouro-2.6B-Thinking: native 2/2 and 4/4 against the loop adapter at prefill 4 / decode 2."""
    E = {e["bench"]: e for e in PN["equal"] if e.get("model", "2.6B-T") == "2.6B-T"}
    benches = ["GSM8K", "MATH-500", "MMLU-Pro", "ARC-C", "BBH", "HumanEval+", "MBPP+", "HELMET"]
    fig, axes = plt.subplots(2, 4, figsize=(6.9, 3.4))
    for ax, b in zip(axes.flat, benches):
        e = E[b]
        frontier(ax, [(2, e["native2"], "2/2"), (4, e["native4"], "4/4")], [(2, e["ours"], "4/2")], b)
        ax.set_xlim(1.5, 4.5); ax.set_xticks([2, 3, 4])
    fig.supxlabel("decode loops n", fontsize=8.5, y=0.01); fig.supylabel("score (%)", fontsize=8.5, x=0.005)
    legend(fig, "loop adapter, prefill 4 / decode 2", "labels: prefill/decode loops", y=1.02)
    fig.subplots_adjust(wspace=0.35, hspace=0.45, left=0.07, right=0.99, bottom=0.1, top=0.9)
    save(fig, "fig_equal")


def _tput(cond):
    """Best decode tokens/s over tested batch sizes at 8K context, one B300 (results/throughput-sb4)."""
    rows = json.loads((ROOT / "results" / "throughput-sb4" / f"{cond}.json").read_text())
    return max(r["tokens_per_s"] for r in rows if "tokens_per_s" in r and r["context"] == 8192)


def fig_pareto():
    """Accuracy vs decode throughput: native d/d against prefill 4 / decode d, d = 1..4."""
    C = {c["label"]: c for c in PN["comparisons"] if c["section"] == "Pareto"}
    panels = [("MMLU (20/subject)", "MMLU"), ("RULER-16K (260)", "RULER-16K"), ("GSM8K (400)", "GSM8K"),
              ("GSM8K Thinking (full)", "GSM8K (Thinking)")]
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.1))
    for ax, (key, title) in zip(axes, panels):
        nat, ours = [], []
        for d in range(1, 5):
            c = C.get(f"prefill 4/decode {d} vs native {d}/{d} {key}")
            if c is not None and c.get("n", 0):
                nat.append((_tput(f"native{d}st"), c["b"], d)); ours.append((_tput(f"S4{d}sb"), c["a"], d))
            elif key.startswith("GSM8K Thinking"):  # prefill 4 / decode 4 not evaluated on the Thinking model
                nat.append((_tput(f"native{d}st"), _M[f"GSM8K full think-native{d}"], d))
        for pts, col, lab, mk in ((nat, NAT_LIGHT, "native n/n", "o"), (ours, OURS, "prefill 4 / decode n", "s")):
            x, y = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(x, y, "-", color=col, lw=1.4, zorder=2)
            ax.scatter(x, y, s=18, color=col, marker=mk, zorder=3, label=lab)
            for xi, yi, d in pts:   # label the n = 1 end only; each line runs from n = 1 (right) to n = 4 (left)
                if d == 1:
                    ax.annotate(f"n={d}", (xi, yi), xytext=(0, 6 if col == OURS else -11), textcoords="offset points",
                                fontsize=8, color=OURS if col == OURS else NAT, ha="center")
        ys = [p[1] for p in nat + ours]
        lo, hi = min(ys), max(ys); pad = 0.12 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)   # room for the point labels above and below
        ax.set_xscale("log"); ax.set_xticks([125, 250, 500, 1000], ["125", "250", "500", "1k"])
        ax.minorticks_off(); ax.set_xlim(100, 1100)
        ax.set_title(title, fontsize=8.5, color=INK, pad=4)
        ax.grid(color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("accuracy (%)")
    fig.supxlabel("8K decode microbenchmark throughput (tokens/s, log scale)", fontsize=8.5, y=-0.04)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.1), handlelength=1.2)
    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig_pareto")


def fig_lev():
    """Decision models (Lev): accuracy against the number of decision loops n. Grey: native n/n continued with the same
    data and updates as ours; red: the best prefill count m at each n (labels), one continued training per m."""
    L = PN.get("lev", [])
    def series(label, which):
        return {r["depth"]: r[which] for r in L if r["label"] == label}
    models = [m for m in ("Lev-1.4B", "Lev-2.6B", "Lev-3B") if series(f"{m} untied vs same-training control", "a")]
    fig, axes = plt.subplots(1, len(models), figsize=(6.9, 1.75), squeeze=False)
    for ax, m in zip(axes[0], models):
        top = 2 if m == "Lev-3B" else 4
        by_m = {top: series(f"{m} untied vs same-training control", "a")}
        by_m.update({k: series(f"{m} prefill {k} vs same-training control", "a") for k in range(1, top)})
        by_m = {k: v for k, v in by_m.items() if v}
        nat = series(f"{m} untied vs same-training control", "b")
        ns = sorted(nat)
        ours = []
        for n in ns:
            best = max((v[n], k) for k, v in by_m.items() if n in v)
            ours.append((n, best[0], f"m={best[1]}"))
        frontier(ax, [(n, nat[n], "") for n in ns], ours, m)
        ax.set_xticks(ns); ax.set_xlim(ns[0] - 0.4, ns[-1] + 0.4)
    axes[0][0].set_ylabel("accuracy (%)", fontsize=8.5)
    fig.supxlabel("decision loops n", fontsize=8.5, y=0.0)
    legend(fig, "trained with prefill m (best m)", "labels: prefill loops m", y=1.0, native="native n/n, continued (control)")
    fig.subplots_adjust(wspace=0.28, bottom=0.2, top=0.8)
    save(fig, "fig_lev")


def tab_equal(model="1.4B-T"):
    """Appendix table: the headline comparison on another Thinking checkpoint (LaTeX rows from paper_numbers.json)."""
    pm = lambda v: (r"\up{%.1f}" if v[0] >= 0 else r"\dn{%.1f}") % abs(v[0]) + ci(v[1], v[2])
    rows = [e for e in PN["equal"] if e.get("model") == model]
    lines = [rf"{e['bench']} & {e['native2']:.1f} & {e.get('same22', float('nan')):.1f} & {e['ours']:.1f} & {e['native4']:.1f} & "
             rf"{pm(e['vs2'])} & {pm(e['vs22']) if 'vs22' in e else '--'} & {pm(e['vs4'])} \\" for e in rows]
    (OUT / f"tab_equal_{model.replace('.', '')}.tex").write_text("\n".join(lines) + "\n")


def tab_hybrid():
    """Appendix table: SenseNova-Loop-9B, the second prefill visit's state in attention KV only, Gated-DeltaNet state only, or both
    (difference to prefill 1; LaTeX rows from paper_numbers.json)."""
    C = {c["label"]: c for c in PN["comparisons"] if c["section"] == "Hybrid state"}
    pm = lambda c: "--" if c is None else ((r"\up{%.1f}" if c["diff"] >= 0 else r"\dn{%.1f}") % abs(c["diff"])
                                           + ci(c["lo"], c["hi"]))
    lines = []
    for bench in ("RULER-4K", "RULER-8K", "RULER-16K", "MMLU-Pro", "ARC-C", "HELMET"):
        cells = [pm(C.get(f"SenseNova {k} prefill-2 state {part} vs prefill 1 {bench}"))
                 for k in ("untrained", "trained") for part in ("attention KV", "GDN state", "both")]
        lines.append(f"{bench} & " + " & ".join(cells) + r" \\")
    (OUT / "tab_hybrid.tex").write_text("\n".join(lines) + "\n")


def tab_sn():
    """Appendix table: SenseNova-Loop-9B at one writing visit (LaTeX rows from paper_numbers.json, sn_equal)."""
    pm = lambda v: (r"\up{%.1f}" if v[0] >= 0 else r"\dn{%.1f}") % abs(v[0]) + ci(v[1], v[2])
    lines = [rf"{e['bench']} & {e['native1']:.1f} & {e['C11']:.1f} & {e['C21']:.1f} & {e['native2']:.1f} & "
             rf"{pm(e['vs_C11'])} & {pm(e['vs_native2'])} \\" for e in PN.get("sn_equal", [])]
    (OUT / "tab_sn.tex").write_text("\n".join(lines) + "\n")


def fig_taskmap(only=None, name="fig_taskmap_full", ncols=4):
    """Appendix task map: Ouro-2.6B, one-cache loop adapter. Per benchmark, native n/n against the adapter's best prefill
    count m at each number of decode loops n (point labels: that m); all points of a panel on the same items."""
    T = {r["bench"]: r for r in PN["taskmap_cells"] if r["model"] == "Ouro-2.6B"}
    title = {"BBH 3-shot": "BBH"}
    panels = [b for b in ("MMLU", "MMLU-Pro", "ARC-C", "GPQA-Diamond", "HellaSwag", "Winogrande", "BBH 3-shot", "IFEval",
                          "GSM8K", "MATH-500 4-shot", "HumanEval+", "MBPP+", "RULER-4K", "RULER-8K", "RULER-16K") if b in T]
    if only:
        panels = [b for t in only for b in panels if title.get(b, b) == t]
    nrows = -(-len(panels) // ncols)
    H = 1.6 * nrows + 0.3
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.9, H), squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax, b in zip(axes.flat, panels):
        r = T[b]
        ours = [(n, r[f"S{r['best'][str(n)]}{n}"], f"m={r['best'][str(n)]}") for n in range(1, 5)]
        nat = [(n, r[f"native{n}"], "") for n in range(1, 5)]
        frontier(ax, nat, ours, title.get(b, b))
        ax.set_xticks([1, 2, 3, 4]); ax.set_xlim(0.6, 4.4)
    fig.supxlabel("decode loops n", fontsize=8.5, y=0.005); fig.supylabel("score (%)", fontsize=8.5, x=0.005)
    legend(fig, "loop adapter (one cache), best prefill m", "labels: prefill loops m", y=1.0)
    fig.subplots_adjust(wspace=0.32, hspace=0.5, left=0.07, right=0.99, top=1 - 0.38 / H, bottom=0.32 / H)
    save(fig, name)


def fig_aligned(panels, name="fig_taskmap", ncols=4):
    """Figure 1: Ouro-2.6B, native n/n (grey) against the aligned loop adapter at prefill 4 / decode n (red), n = 1..4,
    each panel on the items all configurations share (paper_numbers.json, aligned_cells)."""
    T = {r["bench"]: r for r in PN.get("aligned_cells", [])}
    title = {"BBH 3-shot": "BBH", "MATH-500 4-shot": "MATH-500"}
    panels = [b for b in panels if b in T and all(T[b].get(f"a4{n}") is not None for n in range(1, 5))]
    if not panels:
        return
    nrows = -(-len(panels) // ncols)
    H = 1.6 * nrows + 0.3
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.9, H), squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax, b in zip(axes.flat, panels):
        r = T[b]
        frontier(ax, [(n, r[f"native{n}"], "") for n in range(1, 5)], [(n, r[f"a4{n}"], "") for n in range(1, 5)],
                 title.get(b, b))
        ax.set_xticks([1, 2, 3, 4]); ax.set_xlim(0.6, 4.4)
    fig.supxlabel("decode loops n", fontsize=8.5, y=0.005); fig.supylabel("score (%)", fontsize=8.5, x=0.005)
    legend(fig, "loop adapter, prefill 4 / decode n", "", y=1.0)
    fig.subplots_adjust(wspace=0.32, hspace=0.5, left=0.07, right=0.99, top=1 - 0.38 / H, bottom=0.32 / H)
    save(fig, name)


def tab_select(model="Ouro-2.6B", name="tab_select"):
    """Main table: per benchmark and decode loops n, prefill loops m chosen on one half of the items and scored on the
    other (cross-fitted), minus native n/n on the same items (LaTeX rows from paper_numbers.json, taskmap_select)."""
    rows = [r for r in PN.get("taskmap_select", []) if r.get("model", "Ouro-2.6B") == model]
    R = {(r["bench"], r["n"]): r for r in rows}
    benches = list(dict.fromkeys(r["bench"] for r in rows))
    def cell(r):
        if r is None:
            return "--"
        d, lo, hi = r["diff_select_aware"]           # intervals repeat the choice of m in every resample
        m = r["m_chosen"]
        mm = str(m[0]) if m[0] == m[1] else f"{m[0]},{m[1]}"
        v = (r"\up{%.1f}" if d >= 0 else r"\dn{%.1f}") % abs(d)
        sig = "$^*$" if lo > 0 or hi < 0 else ""          # paired 95% interval excludes zero
        return rf"{v}{sig}{ci(lo, hi)}\,{{\scriptsize ({mm})}}"
    lines = [f"{b} & " + " & ".join(cell(R.get((b, n))) for n in range(1, 5)) + r" \\" for b in benches]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


def tab_distill():
    """Appendix table: Ouro-2.6B-Thinking, LoRA distillation of native 2/2 (same data, teacher and tokens) against native
    2/2 and against prefill 4 / decode 2 (LaTeX rows from paper_numbers.json)."""
    C = {c["label"]: c for c in PN["comparisons"] if c["section"] == "Distillation baseline"}
    pm = lambda c: (r"\up{%.1f}" if c["diff"] >= 0 else r"\dn{%.1f}") % abs(c["diff"]) + ci(c["lo"], c["hi"])
    lines = []
    for b in ("GSM8K", "MATH-500", "MMLU-Pro", "ARC-C", "BBH", "HumanEval+", "MBPP+", "HELMET"):
        ours = C.get(f"2.6B-T prefill 4/decode 2 vs LoRA-distilled native 2/2 {b}")
        dist = C.get(f"2.6B-T LoRA-distilled native 2/2 vs native 2/2 {b}")
        if not ours or not dist:
            continue
        lines.append(rf"{b} & {dist['b']:.1f} & {dist['a']:.1f} & {ours['a']:.1f} & {pm(dist)} & {pm(ours)} \\")
    (OUT / "tab_distill.tex").write_text("\n".join(lines) + "\n")



def tab_lev():
    """Appendix table: Lev decision models, every trained prefill count m minus the same-training control (native n/n)
    at every number of decision loops n, both directions (LaTeX rows from paper_numbers.json, lev)."""
    both = next((r for r in PN.get("lev", []) if r["label"] == "both directions"), None)
    if not both:
        return
    G = {(g["model"], g["m"], g["n"]): g["diff"] for g in both["grid"]}
    def cell(v):
        if v is None:
            return "--"
        d, lo, hi = v
        sig = "$^*$" if lo > 0 or hi < 0 else ""
        return (r"\up{%.1f}" if d >= 0 else r"\dn{%.1f}") % abs(d) + sig + ci(lo, hi)
    lines = []
    for model in dict.fromkeys(g["model"] for g in both["grid"]):
        ms = sorted({m for (mo, m, _) in G if mo == model})
        for m in ms:
            lines.append(rf"{model} & {m} & " + " & ".join(cell(G.get((model, m, n))) for n in range(1, 5)) + r" \\")
    (OUT / "tab_lev.tex").write_text("\n".join(lines) + "\n")


def tab_taskmap(model="Ouro-2.6B"):
    """Appendix table: the full task map, every prefill m / decode n cell and native n/n on every benchmark; the best m
    at each n in bold (LaTeX header and rows from paper_numbers.json, taskmap_cells)."""
    rows = [r for r in PN.get("taskmap_cells", []) if r["model"] == model]
    short = {"MATH-500 4-shot": "MATH", "BBH 3-shot": "BBH", "MMLU-Pro": "MMLU-P", "HumanEval+": "HE+", "GPQA-Diamond": "GPQA",
             "HellaSwag": "HSwag", "Winogrande": "Wino",
             "IFEval": "IFEv", "RULER-4K": "R-4K", "RULER-8K": "R-8K", "RULER-16K": "R-16K"}
    lines = [" & " + " & ".join(short.get(r["bench"], r["bench"]) for r in rows) + r" \\", r"\midrule"]
    for n in range(1, 5):
        for c, name in [(f"native{n}", rf"native {n}/{n}")] + [(f"S{m}{n}", rf"{m}\,/\,{n}") for m in range(1, 5)]:
            best = lambda r: c == f"S{r['best'][str(n)]}{n}"          # chosen before rounding (taskmap_cells)
            cell = lambda r: rf"\textbf{{{r[c]:.1f}}}" if best(r) else f"{r[c]:.1f}"
            lines.append(f"{name} & " + " & ".join(cell(r) for r in rows) + r" \\")
        if n < 4:
            lines.append(r"\midrule")
    (OUT / "tab_taskmap.tex").write_text("\n".join(lines) + "\n")


def tab_main(key="main_table", name="tab_main"):
    """Table 1: main results, fixed configurations grouped by decode loops (accuracy on the items all rows share; bold:
    best of each group; LaTeX rows from paper_numbers.json, main_table)."""
    T = {r["bench"]: r for r in PN.get(key, [])}
    benches = list(T)
    groups = [(1, ["native 1/1", "training-free 4/1", "LoRA-distilled 1/1", "loop adapter 4/1", "aligned adapter 4/1"]),
              (2, ["native 2/2", "training-free 4/2", "LoRA-distilled 2/2", "loop adapter 4/2", "aligned adapter 4/2"]),
              (4, ["native 4/4"])]
    show = {"native 1/1": "Original 1/1", "native 2/2": "Original 2/2", "native 4/4": "Original 4/4",
            "training-free 4/1": "Training-free 4/1", "training-free 4/2": "Training-free 4/2",
            "LoRA-distilled 1/1": "Distilled 1/1", "LoRA-distilled 2/2": "Distilled 2/2",
            "loop adapter 4/1": "Ours, one cache 4/1", "loop adapter 4/2": "Ours, one cache 4/2",
            "aligned adapter 4/1": r"\textbf{Ours, aligned 4/1}", "aligned adapter 4/2": r"\textbf{Ours, aligned 4/2}"}
    lines = []
    for n, labels in groups:
        vals = {l: [T[b].get(l) for b in benches] for l in labels}
        avg = {l: (sum(v) / len(v) if None not in v else None) for l, v in vals.items()}
        cols = [[vals[l][j] for l in labels] for j in range(len(benches))] + [[avg[l] for l in labels]]
        best = [max((x for x in c if x is not None), default=None) for c in cols]
        for i, l in enumerate(labels):
            cells = [c[i] for c in cols]
            fmt = lambda x, b: "--" if x is None else (rf"\textbf{{{x:.1f}}}" if len(labels) > 1 and abs(x - b) < 1e-9 else f"{x:.1f}")
            lines.append(f"{n} & {show[l]} & " + " & ".join(fmt(x, b) for x, b in zip(cells, best)) + r" \\")
        if n < 4:
            lines.append(r"\midrule")
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


def tab_ablation():
    """Ablation table (Section 4.2): main-table average at prefill 4 / decode 1 and 2 for each training-token budget
    (rank 16) and LoRA rank (12M tokens); LaTeX rows from paper_numbers.json, ablations."""
    A = {(r["label"], r["cell"]): r["avg"] for r in PN.get("ablations", [])}
    rows = [("6M tokens", "6M", "16"), ("12M tokens", "12M", "16"), ("24M tokens", "24M", "16"),
            ("48M tokens", r"48M (default)", "16"), ("rank 4, 12M", "12M", "4"), ("rank 64, 12M", "12M", "64")]
    f = lambda x: "--" if x is None else f"{x:.1f}"
    lines = []
    for i, (label, tok, rank) in enumerate(rows):
        if i == 4:
            lines.append(r"\midrule")
        lines.append(f"{tok} & {rank} & {f(A.get((label, 'S41')))} & {f(A.get((label, 'S42')))} \\\\")
    (OUT / "tab_ablation.tex").write_text("\n".join(lines) + "\n")


def tab_e2e():
    """Table 9: end-to-end time per answer at batch 1 (paper_numbers.json, e2e; the 192 GSM8K problems of Figure 6)."""
    E = {(r["model"], r["task"], r["cond"]): r for r in PN["e2e"] if r["group"] == "main"}
    blocks = [(r"\shortstack[l]{GSM8K\\Ouro-2.6B-T}", "think", "gsm",
               [("native2st", r"\nat{2}"), ("native4st", r"\nat{4}"), ("Df2st", "distilled 2/2"), ("S22sb", r"\rw{2}{2}"),
                ("S42sb", r"\rw{4}{2}")]),
              (r"\shortstack[l]{MATH-500\\Ouro-2.6B-T}", "think", "math",
               [("native2st", r"\nat{2}"), ("native4st", r"\nat{4}"), ("S42sb", r"\rw{4}{2}")]),
              (r"\shortstack[l]{GSM8K\\SenseNova}", "sn", "gsm",
               [("native2", r"\nat{2}"), ("C11", r"\rw{1}{1}"), ("C21", r"\rw{2}{1}")])]
    lines = []
    for i, (name, model, task, conds) in enumerate(blocks):
        if i:
            lines.append(r"\midrule")
        for j, (c, label) in enumerate(conds):
            r = E[(model, task, c)]
            head = rf"\multirow{{{len(conds)}}}{{*}}{{{name}}}" if j == 0 else ""
            sec = f"{r['seconds']:.0f}" if r["seconds"] >= 100 else f"{r['seconds']:.1f}"
            lines.append(f"{head} & {label} & {r['acc']:.1f} & {sec} & {r['tokens']} & {r['n']} \\\\")
    (OUT / "tab_e2e.tex").write_text("\n".join(lines) + "\n")


def tab_repro():
    """Appendix B: native Ouro-2.6B 4/4 under our evaluation and as reported by Ouro (paper_numbers.json, repro)."""
    f = lambda x: "--" if x is None else f"{x:.1f}"
    rows = [f"{r['bench']} & {f(r['ours'])} & {f(r['reported'])} & {r['items']:,} \\\\" for r in PN.get("repro", [])]
    (OUT / "tab_repro.tex").write_text("\n".join(rows) + "\n")


def tab_aligned():
    """Appendix: the aligned loop adapter's full map on the main-table benchmarks, every prefill m / decode n cell and
    native n/n (each column on the items all available configurations share); bold: the best m at each n
    (paper_numbers.json, aligned_cells)."""
    rows = PN.get("aligned_cells", [])
    short = {"MATH-500 4-shot": "MATH", "BBH 3-shot": "BBH", "MMLU-Pro": "MMLU-P", "HumanEval+": "HE+",
             "HellaSwag": "HSwag", "Winogrande": "Wino", "RULER-16K": "R-16K"}
    lines = [" & " + " & ".join(short.get(r["bench"], r["bench"]) for r in rows) + r" \\", r"\midrule"]
    f = lambda x, b: "--" if x is None else (rf"\textbf{{{x:.1f}}}" if b else f"{x:.1f}")
    for n in range(1, 5):
        best = {r["bench"]: max((r[f"a{m}{n}"] for m in range(1, 5) if r[f"a{m}{n}"] is not None), default=None)
                for r in rows}
        lines.append(rf"native {n}/{n} & " + " & ".join(f(r[f"native{n}"], False) for r in rows) + r" \\")
        for m in range(1, 5):
            cells = [f(r[f"a{m}{n}"], r[f"a{m}{n}"] is not None and r[f"a{m}{n}"] == best[r["bench"]]) for r in rows]
            lines.append(rf"{m}\,/\,{n} & " + " & ".join(cells) + r" \\")
        if n < 4:
            lines.append(r"\midrule")
    (OUT / "tab_aligned.tex").write_text("\n".join(lines) + "\n")


def tab_select_distilled():
    """Appendix table: the cross-fitted prefill count (Table 1) against the LoRA-distilled native n/n at n = 1, 2, 3;
    selection-aware intervals (LaTeX rows from paper_numbers.json, taskmap_select)."""
    rows = [r for r in PN.get("taskmap_select", []) if r["model"] == "Ouro-2.6B" and r["n"] <= 3]
    R = {(r["bench"], r["n"]): r for r in rows}
    def cell(r):
        if r is None or "diff_distilled_select_aware" not in r:
            return "--"
        d, lo, hi = r["diff_distilled_select_aware"]
        sig = "$^*$" if lo > 0 or hi < 0 else ""
        return (r"\up{%.1f}" if d >= 0 else r"\dn{%.1f}") % abs(d) + sig + ci(lo, hi)
    benches = list(dict.fromkeys(r["bench"] for r in rows))
    lines = [f"{b} & " + " & ".join(cell(R.get((b, n))) for n in (1, 2, 3)) + r" \\" for b in benches]
    (OUT / "tab_select_distilled.tex").write_text("\n".join(lines) + "\n")


TIMED = ["GSM8K", "MATH-500", "BBH", "MMLU-Pro"]
SWEEP_NAME = {"GSM8K": "GSM8K", "MATH-500": "MATH-500 4-shot", "BBH": "BBH 3-shot", "MMLU-Pro": "MMLU-Pro"}


def drop(name):
    """No data yet: remove a stale figure so that the LaTeX source shows its placeholder instead."""
    for ext in ("pdf", "png"):
        (OUT / f"{name}.{ext}").unlink(missing_ok=True)


def _time(task, cond):
    return next((r for r in PN.get("time_breakdown", []) if r["task"] == task and r["cond"] == cond), None)


def fig_time():
    """Figure 1: time per request split into prefill and decode, at the sweep's starting point (Ours 1/1 and 1/2) and
    Original 4/4, for every timed task; label: prefill share."""
    confs = [("a1x1sb", "Ours 1/1"), ("a1x2sb", "Ours 1/2"), ("native4st", "Original 4/4")]
    tasks = [t for t in TIMED + ["HELMET"] if any(_time(t, c) for c, _ in confs)
             or (t == "HELMET" and _time(t, "native4st"))]
    if not tasks:
        return drop("fig_time")
    fig, ax = plt.subplots(figsize=(3.3, 0.32 * len(tasks) * len(confs) + 0.4))
    y, ticks, labs = 0, [], []
    for t in tasks:
        for c, lab in (confs if t != "HELMET" else [("native2st", "Original 2/2"), ("native4st", "Original 4/4")]):
            r = _time(t, c)
            if r is None:
                continue
            ax.barh(y, r["prefill_s"], color=OURS, height=0.7)
            ax.barh(y, r["decode_s"], left=r["prefill_s"], color=NAT_LIGHT, height=0.7)
            ax.annotate(f"{r['prefill_share']:.1f}%  ({r['in_tokens']:.0f} in / {r['out_tokens']:.0f} out)",
                        (r["total_s"], y), xytext=(3, 0), textcoords="offset points", va="center", fontsize=6.5, color=NAT)
            ticks.append(y); labs.append(f"{t}, {lab}"); y += 1
        y += 0.6
    ax.set_yticks(ticks); ax.set_yticklabels(labs, fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("seconds per request (batch 1)")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=OURS), plt.Rectangle((0, 0), 1, 1, color=NAT_LIGHT)], ["prefill", "decode"],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
    ax.set_xlim(0, 1.75 * max(r["total_s"] for t in tasks for c, _ in confs + [("native2st", "")] if (r := _time(t, c))))
    ax.grid(axis="x", color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
    save(fig, "fig_time")


def fig_depth():
    """Figure 2: per timed task, quality (full evaluation set) and time per request against prefill loops m at decode
    n = 1 (dark) and 2 (light), and quality against time (same replayed requests)."""
    rows = PN.get("depth_sweep", [])
    tasks = [t for t in TIMED if any(r["bench"] == SWEEP_NAME[t] for r in rows)]
    if not tasks:
        return drop("fig_depth")
    timed = any(_time(t, f"a{m}x{n}sb") for t in tasks for m in (1, 2, 3, 4, 6, 8, 12, 16) for n in (1, 2))
    nrow = 3 if timed else 1   # time rows only once per-request timings exist
    fig, axes = plt.subplots(3, len(tasks), figsize=(1.75 * len(tasks), 4.6), squeeze=False) if timed else \
        plt.subplots(1, len(tasks), figsize=(1.75 * len(tasks), 1.7), squeeze=False)
    for j, t in enumerate(tasks):
        for n, col in ((1, OURS), (2, OURS_LIGHT)):
            for wide, ls in ((False, "-"), (True, "--")):
                r = next((r for r in rows if r["bench"] == SWEEP_NAME[t] and r["n"] == n and r["adapter"] == ("wide" if wide else "m<=4")), None)
                if r:
                    ms = sorted(int(m) for m in r["quality"])
                    axes[0, j].plot(ms, [r["quality"][str(m)] if str(m) in r["quality"] else r["quality"][m] for m in ms],
                                    ls, color=col, marker="s", ms=2.5, lw=1.2)
            pts = [(m, _time(t, f"a{m}x{n}sb")) for m in (1, 2, 3, 4, 6, 8, 12, 16)]
            pts = [(m, r) for m, r in pts if r]
            if pts and timed:
                axes[1, j].plot([m for m, _ in pts], [r["total_s"] for _, r in pts], "-", color=col, marker="s", ms=2.5, lw=1.2)
                axes[1, j].plot([m for m, _ in pts], [r["prefill_s"] for _, r in pts], ":", color=col, lw=1)
                axes[2, j].plot([r["total_s"] for _, r in pts], [r["quality"] for _, r in pts], "-", color=col, marker="s", ms=2.5, lw=1.2)
                for m, r in pts:
                    axes[2, j].annotate(str(m), (r["total_s"], r["quality"]), xytext=(0, 3), textcoords="offset points", fontsize=6, color=col, ha="center")
        for d in (1, 2, 4):
            r = _time(t, f"native{d}st")
            if r and timed:
                axes[2, j].scatter([r["total_s"]], [r["quality"]], color=NAT, s=10, marker="o", zorder=3)
                axes[2, j].annotate(f"{d}/{d}", (r["total_s"], r["quality"]), xytext=(0, -8), textcoords="offset points", fontsize=6, color=NAT, ha="center")
        axes[0, j].set_title(t, fontsize=8.5, color=INK, pad=4)
        for i in range(min(2, nrow)):
            axes[i, j].set_xscale("log", base=2); axes[i, j].set_xticks([1, 2, 4, 8, 16]); axes[i, j].set_xticklabels(["1", "2", "4", "8", "16"])
            axes[i, j].axvline(4, color=GRID, lw=0.8, zorder=0)
        for i in range(nrow):
            axes[i, j].grid(color="#EDEDED", lw=0.5); axes[i, j].set_axisbelow(True)
    axes[0, 0].set_ylabel("accuracy (%)")
    if timed:
        axes[1, 0].set_ylabel("s / request"); axes[2, 0].set_ylabel("accuracy (%)")
    for j in range(len(tasks)):
        if timed:
            axes[1, j].set_xlabel("prefill loops m", fontsize=7.5); axes[2, j].set_xlabel("s / request", fontsize=7.5)
        else:
            axes[0, j].set_xlabel("prefill loops m", fontsize=7.5)
    h = [plt.Line2D([], [], color=OURS, lw=1.2), plt.Line2D([], [], color=OURS_LIGHT, lw=1.2),
         plt.Line2D([], [], color=NAT, ls="--", lw=1.2), plt.Line2D([], [], color=NAT, marker="o", ls="none", ms=3)]
    fig.legend(h, ["decode n = 1", "decode n = 2", "trained on m ≤ 16", "Original n/n"], loc="upper center", ncol=4,
               fontsize=7.5, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    save(fig, "fig_depth")


def fig_budget():
    """Figure 3: mean accuracy over the timed tasks against mean time per request (same replayed requests): Original
    n/n, Distilled 1/1 and 2/2, Ours m/1 and m/2."""
    tasks = [t for t in TIMED if _time(t, "native4st")]
    if not tasks:
        return drop("fig_budget")
    def point(cond):
        rs = [_time(t, cond) for t in tasks]
        return None if None in rs else (float(np.mean([r["total_s"] for r in rs])), float(np.mean([r["quality"] for r in rs])))
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    nat = [(*point(f"native{d}st"), f"{d}/{d}") for d in (1, 2, 4) if point(f"native{d}st")]
    for n, col in ((1, OURS), (2, OURS_LIGHT)):
        ours = [(*point(f"a{m}x{n}sb"), str(m)) for m in (1, 2, 3, 4, 6, 8, 12, 16) if point(f"a{m}x{n}sb")]
        if ours:
            ax.plot([p[0] for p in ours], [p[1] for p in ours], "-", color=col, marker="s", ms=3, lw=1.3, label=f"Ours m/{n}")
            for x, y, lab in ours:
                ax.annotate(lab, (x, y), xytext=(0, 3), textcoords="offset points", fontsize=6.5, color=col, ha="center")
    if nat:
        ax.plot([p[0] for p in nat], [p[1] for p in nat], "-", color=NAT_LIGHT, marker="o", ms=3, lw=1.3, label="Original n/n")
        for x, y, lab in nat:
            ax.annotate(lab, (x, y), xytext=(0, -9), textcoords="offset points", fontsize=6.5, color=NAT, ha="center")
    for d in (1, 2):
        p = point(f"D{d}st")
        if p:
            ax.scatter(*p, color=NAT, marker="^", s=16, zorder=3)
            ax.annotate(f"Distilled {d}/{d}", p, xytext=(4, -3), textcoords="offset points", fontsize=6.5, color=NAT)
    ax.set_xlabel("seconds per request (batch 1)"); ax.set_ylabel("mean accuracy (%)")
    ax.grid(color="#EDEDED", lw=0.5); ax.set_axisbelow(True); ax.legend(fontsize=7, loc="lower right")
    save(fig, "fig_budget")


if __name__ == "__main__":
    fig_time(); fig_depth(); fig_budget(); fig0(); fig2(); fig_equal(); fig_pareto(); fig_lev(); fig_taskmap(ncols=5); fig_aligned(["MMLU", "BBH 3-shot", "ARC-C", "Winogrande", "GSM8K", "HumanEval+", "MBPP+", "RULER-16K"]); tab_equal("2.6B-T"); tab_equal("1.4B-T"); tab_hybrid(); tab_sn(); tab_select(); tab_select("Ouro-1.4B", "tab_select_14"); tab_distill(); tab_lev(); tab_taskmap(); tab_select_distilled(); tab_main(); tab_main("main_table_14", "tab_main14"); tab_ablation(); tab_e2e(); tab_repro(); tab_aligned()
    print("\n".join(sorted(p.name for p in OUT.iterdir())))
