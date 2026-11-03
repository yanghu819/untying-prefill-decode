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



def frontier(ax, nat, ours, title, xlog=False, lines=True):
    """The one figure style of the paper: native (grey) and loop adapter (red) as lines of (x, y, label) points."""
    for pts, col, dy in ((nat, NAT_LIGHT, -9), (ours, OURS, 5)):
        pts = [p for p in pts if np.isfinite(p[1])]
        if not pts:
            continue
        x, y = [p[0] for p in pts], [p[1] for p in pts]
        if lines:
            ax.plot(x, y, "-", color=col, lw=1.4, zorder=2)
        ax.scatter(x, y, s=18, color=col, zorder=3)
        for xi, yi, lab in pts:
            if lab:
                ax.annotate(lab, (xi, yi), xytext=(0, dy), textcoords="offset points", fontsize=6.8,
                            color=OURS if col == OURS else NAT, ha="center")
    ys = [p[1] for p in nat + ours if np.isfinite(p[1])]
    lo, hi = min(ys), max(ys); pad = 0.14 * (hi - lo + 1)
    ax.set_ylim(lo - pad, hi + pad)
    if xlog:
        ax.set_xscale("log"); ax.minorticks_off()
    ax.set_title(title, fontsize=8.5, color=INK, pad=4)
    ax.grid(color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)


def legend(fig, ours, labels, y=1.1):
    h = [plt.Line2D([], [], color=NAT_LIGHT, marker="o", ms=4, lw=1.4), plt.Line2D([], [], color=OURS, marker="o", ms=4, lw=1.4),
         plt.Line2D([], [], ls="none")]
    fig.legend(h, ["native n/n", ours, labels], loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, y),
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
        sub="last loop, stored once")
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
    """End to end (Ouro-2.6B-Thinking, GSM8K, batch 1): accuracy against seconds per answer."""
    E = {r["cond"]: (r["seconds"], r["acc"]) for r in PN["e2e"] if r["model"] == "think" and r["task"] == "gsm"}
    nat = [(*E[f"native{n}st"], f"{n}/{n}") for n in (1, 2, 4) if f"native{n}st" in E]
    ours = [(*E[c], lab) for c, lab in (("S41sb", "4/1"), ("S42sb", "4/2")) if c in E]
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    frontier(ax, nat, ours, "", lines=False)   # 1/1 answers run longer than 2/2: a line would fold back
    ax.set_xlabel("seconds per answer (batch 1)", fontsize=8); ax.set_ylabel("GSM8K accuracy (%)", fontsize=8)
    legend(fig, "loop adapter", "labels: prefill/decode loops", y=1.04)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.2, top=0.86)
    save(fig, "fig2_frontier")


def fig_equal():
    """Ouro-2.6B-Thinking: native 2/2 and 4/4 against the loop adapter at prefill 4 / decode 2."""
    E = {e["bench"]: e for e in PN["equal"] if e.get("model", "2.6B-T") == "2.6B-T"}
    benches = ["GSM8K", "MATH-500", "MMLU-Pro", "ARC-C", "BBH", "HumanEval", "MBPP+", "HELMET"]
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
        for pts, col, lab, mk in ((nat, NAT_LIGHT, "native n/n", "o"), (ours, OURS, "prefill 4 / decode n", "o")):
            x, y = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(x, y, "-", color=col, lw=1.4, zorder=2)
            ax.scatter(x, y, s=18, color=col, zorder=3, label=lab)
            for xi, yi, d in pts:
                dx = {4: -4, 3: 4}.get(d, 0) if col == OURS else 0   # prefill 4 / decode 3 and 4 sit close: split labels
                ax.annotate(f"{d}", (xi, yi), xytext=(dx, 5 if col == OURS else -9), textcoords="offset points",
                            fontsize=6.8, color=OURS if col == OURS else NAT,
                            ha="center" if dx == 0 else ("right" if dx < 0 else "left"))
        ys = [p[1] for p in nat + ours]
        lo, hi = min(ys), max(ys); pad = 0.12 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)   # room for the point labels above and below
        ax.set_xscale("log"); ax.set_xticks([125, 250, 500, 1000], ["125", "250", "500", "1k"])
        ax.minorticks_off(); ax.set_xlim(100, 1100)
        ax.set_title(title, fontsize=8.5, color=INK, pad=4)
        ax.grid(color="#EDEDED", lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("accuracy (%)")
    fig.supxlabel("decode throughput (tokens/s, log scale)", fontsize=8.5, y=-0.04)
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], ls="none")); l.append("point labels: decode loops n")
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.1), handlelength=1.2)
    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig_pareto")


def fig_lev():
    """Decision models (Lev): accuracy against the number of decision loops n. Grey: native n/n continued with the same
    data and updates as ours; red: the best prefill count m at each n (labels), one continued training per m."""
    L = PN.get("lev", [])
    def series(label, which):
        return {r["depth"]: r[which] for r in L if r["label"] == label}
    models = [m for m in ("Lev-1.4B", "Lev-2.6B", "Lev-3B") if series(f"{m} untied vs same-training control", "a")]
    fig, axes = plt.subplots(1, len(models), figsize=(6.9, 2.0), squeeze=False)
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
    legend(fig, "loop adapter, best prefill m", "labels: prefill loops m", y=1.0)
    fig.subplots_adjust(wspace=0.28, bottom=0.2, top=0.8)
    save(fig, "fig_lev")


def tab_equal(model="1.4B-T"):
    """Appendix table: the headline comparison on another Thinking checkpoint (LaTeX rows from paper_numbers.json)."""
    pm = lambda v: (r"\up{%.1f}" if v[0] >= 0 else r"\dn{%.1f}") % abs(v[0]) + r"\ci{%.1f}{%.1f}" % (v[1], v[2])
    rows = [e for e in PN["equal"] if e.get("model") == model]
    lines = [rf"{e['bench']} & {e['native2']:.1f} & {e.get('same22', float('nan')):.1f} & {e['ours']:.1f} & {e['native4']:.1f} & "
             rf"{pm(e['vs2'])} & {pm(e['vs22']) if 'vs22' in e else '--'} & {pm(e['vs4'])} \\" for e in rows]
    (OUT / f"tab_equal_{model.replace('.', '')}.tex").write_text("\n".join(lines) + "\n")


def tab_hybrid():
    """Appendix table: SenseNova-Loop-9B, the second prefill visit's state in attention KV only, Gated-DeltaNet state only, or both
    (difference to prefill 1; LaTeX rows from paper_numbers.json)."""
    C = {c["label"]: c for c in PN["comparisons"] if c["section"] == "Hybrid state"}
    pm = lambda c: "--" if c is None else ((r"\up{%.1f}" if c["diff"] >= 0 else r"\dn{%.1f}") % abs(c["diff"])
                                           + r"\ci{%.1f}{%.1f}" % (c["lo"], c["hi"]))
    lines = []
    for bench in ("RULER-4K", "RULER-8K", "RULER-16K", "MMLU-Pro", "ARC-C", "HELMET"):
        cells = [pm(C.get(f"SenseNova {k} prefill-2 state {part} vs prefill 1 {bench}"))
                 for k in ("untrained", "trained") for part in ("attention KV", "GDN state", "both")]
        lines.append(f"{bench} & " + " & ".join(cells) + r" \\")
    (OUT / "tab_hybrid.tex").write_text("\n".join(lines) + "\n")


def tab_sn():
    """Appendix table: SenseNova-Loop-9B at one writing visit (LaTeX rows from paper_numbers.json, sn_equal)."""
    pm = lambda v: (r"\up{%.1f}" if v[0] >= 0 else r"\dn{%.1f}") % abs(v[0]) + r"\ci{%.1f}{%.1f}" % (v[1], v[2])
    lines = [rf"{e['bench']} & {e['native1']:.1f} & {e['C11']:.1f} & {e['C21']:.1f} & {e['native2']:.1f} & "
             rf"{pm(e['vs_C11'])} & {pm(e['vs_native2'])} \\" for e in PN.get("sn_equal", [])]
    (OUT / "tab_sn.tex").write_text("\n".join(lines) + "\n")


def fig_taskmap(only=None, name="fig_taskmap_full", ncols=4):
    """Task map: Ouro-2.6B, one loop adapter. Per benchmark, native n/n against the adapter's best prefill count m
    at each number of decode loops n (point labels: that m)."""
    M = {x["label"]: x["mean"] for x in PN["means"] if x.get("section") in ("Task map", "Figure 1")}
    PAR = {c["label"]: c for c in PN["comparisons"] if c["section"] == "Pareto"}
    panels = [("MMLU", "{c} MMLU (20/subject)"), ("MMLU-Pro", "{c} MMLU-Pro"), ("ARC-C", "{c} ARC-C"),
              ("GPQA-Diamond", "{c} GPQA-Diamond"), ("BBH", "{c} BBH 3-shot"), ("IFEval", "{c} IFEval"),
              ("GSM8K", "{c} GSM8K (400)"), ("MATH-500 4-shot", "{c} MATH-500 4-shot"), ("HumanEval", "{c} HumanEval"),
              ("MBPP+", "{c} MBPP+"), ("RULER-4K", "{c} RULER-4K"), ("RULER-8K", "{c} RULER-8K"), ("RULER-16K", "{c} RULER-16K")]
    if only:
        panels = [p for t in only for p in panels if p[0] == t]
    def val(fmt, c):
        k = fmt.format(c=c)
        if k in M:
            return M[k]
        if c.startswith("native"):   # grid-subset natives are stored with the Pareto comparisons
            n, sub = c[6:], fmt.split(" ", 1)[1]
            r = PAR.get(f"prefill 4/decode {n} vs native {n}/{n} {sub}")
            return r["b"] if r else np.nan
        return np.nan
    nrows = -(-len(panels) // ncols)
    H = 1.6 * nrows + 0.3
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.9, H), squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax, (title, fmt) in zip(axes.flat, panels):
        ours = []
        for n in range(1, 5):
            v = [val(fmt, f"S{m}{n}") for m in range(1, 5)]
            b = int(np.nanargmax(v)); ours.append((n, v[b], f"m={b + 1}"))
        nat = [(n, val(fmt, f"native{n}"), "") for n in range(1, 5)]
        frontier(ax, nat, ours, title)
        ax.set_xticks([1, 2, 3, 4]); ax.set_xlim(0.6, 4.4)
    fig.supxlabel("decode loops n", fontsize=8.5, y=0.005); fig.supylabel("score (%)", fontsize=8.5, x=0.005)
    legend(fig, "loop adapter, best prefill m", "labels: prefill loops m", y=1.0)
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
        d, lo, hi = r["diff"]
        m = r["m_chosen"]
        mm = str(m[0]) if m[0] == m[1] else f"{m[0]},{m[1]}"
        v = (r"\up{%.1f}" if d >= 0 else r"\dn{%.1f}") % abs(d)
        sig = "$^*$" if lo > 0 or hi < 0 else ""          # paired 95% interval excludes zero
        return rf"{v}{sig}\ci{{{lo + 0.0:.1f}}}{{{hi + 0.0:.1f}}}\,{{\scriptsize ({mm})}}"
    lines = [f"{b} & " + " & ".join(cell(R.get((b, n))) for n in range(1, 5)) + r" \\" for b in benches]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


def tab_distill():
    """Appendix table: Ouro-2.6B-Thinking, LoRA distillation of native 2/2 (same data, teacher and tokens) against native
    2/2 and against prefill 4 / decode 2 (LaTeX rows from paper_numbers.json)."""
    C = {c["label"]: c for c in PN["comparisons"] if c["section"] == "Distillation baseline"}
    pm = lambda c: (r"\up{%.1f}" if c["diff"] >= 0 else r"\dn{%.1f}") % abs(c["diff"]) + r"\ci{%.1f}{%.1f}" % (c["lo"], c["hi"])
    lines = []
    for b in ("GSM8K", "MATH-500", "MMLU-Pro", "ARC-C", "BBH", "HumanEval", "MBPP+", "HELMET"):
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
        return (r"\up{%.1f}" if d >= 0 else r"\dn{%.1f}") % abs(d) + sig + r"\ci{%.1f}{%.1f}" % (lo + 0.0, hi + 0.0)
    lines = []
    for model in dict.fromkeys(g["model"] for g in both["grid"]):
        ms = sorted({m for (mo, m, _) in G if mo == model})
        for m in ms:
            lines.append(rf"{model} & {m} & " + " & ".join(cell(G.get((model, m, n))) for n in range(1, 5)) + r" \\")
    (OUT / "tab_lev.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    fig0(); fig2(); fig_equal(); fig_pareto(); fig_lev(); fig_taskmap(); fig_taskmap(["MMLU", "GPQA-Diamond", "RULER-16K", "BBH", "GSM8K", "MATH-500 4-shot", "HumanEval", "MBPP+"], "fig_taskmap"); tab_equal("2.6B-T"); tab_equal("1.4B-T"); tab_hybrid(); tab_sn(); tab_select(); tab_select("Ouro-1.4B", "tab_select_14"); tab_distill(); tab_lev()
    print("\n".join(sorted(p.name for p in OUT.iterdir())))
