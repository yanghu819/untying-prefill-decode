# Untying Prefill and Decode Loops in Looped Language Models

**Paper:** [`paper/untying-prefill-decode-loops.pdf`](paper/untying-prefill-decode-loops.pdf) (arXiv link coming) ·
**Author:** Yang Hu · **Code license:** MIT · **Keywords:** looped language model, recurrent depth, universal
transformer, Ouro, SenseNova-Loop, Lev, test-time compute, prefill vs decode, loop adapter, KV cache, RoPE

Looped (recurrent-depth) language models run one block of layers several times per token and, at inference, run the
same number of loops when they prefill the prompt and when they decode each answer token. This repository unties the
two. A small **loop adapter** (LoRA plus per-loop key–value maps on a frozen model) lets a looped model prefill with *m*
loops and decode with *n*; one set of its parameters serves all 16 pairs up to four loops, and one stored prompt KV
cache serves every decode loop.

**The paper's statistics are recomputed from released per-item scores by one script** and checked against the printed
values: `python ouro/scripts/paper_numbers.py compute` (numpy only). See [`REPRODUCE.md`](REPRODUCE.md) for what it covers.

## Key findings

All numbers use the Ouro models' own evaluation protocol (strict-match GSM8K / MATH-500, evalplus code; native
Ouro-2.6B 4/4 reproduces the published scores within 0-3 points). 95% paired bootstrap intervals in the paper.

1. **Prefill with more loops than you decode.** One loop adapter on Ouro-2.6B, evaluated at all 16 (prefill *m*,
   decode *n*) pairs on 15 benchmark settings (the *task map*, Figure 1). With one decode loop the best prefill count is
   two to four on every setting, never one: within the same adapter, three or four prefill loops instead of one raise
   MMLU from 51.4 to 66.6 and HumanEval+ from 51.2 to 65.9.
2. **It beats native, the adapter's own n/n, and a distillation baseline at one decode loop.** Choosing *m* per
   benchmark on held-out halves beats native Ouro-2.6B 1/1 significantly on **14 of 15** settings (+3.9 to +21.3), the
   same adapter at prefill 1 on **10**, and native 1/1 distilled with LoRA on the same data and teacher on **9**
   (intervals that repeat the choice of *m*). With two decode loops distillation matches it; with three or four the
   native model is as good. Ouro-1.4B: 13 and 3 of 15.
3. **Reasoning model.** On Ouro-2.6B-Thinking, prefill 4 / decode 2 scores above native 2/2 on all eight benchmarks
   (significantly on four) and above a distilled 2/2 on MMLU-Pro (+6.0) and ARC-C (+3.2).
4. **End-to-end time.** On 192 timed GSM8K problems (batch 1, prefill included) prefill 4 / decode 2 answers in
   **0.59x** [0.52, 0.66] the time of native 4/4; the saving comes from decoding with two loops (a distilled 2/2 is as
   fast), the extra prefill loops buy accuracy.
5. **Looped decision models (Lev), both directions.** With more prefill than decision loops, Lev models are
   significantly above a matched-data and matched-update control on 12 of 13 (*m*, *n*) pairs; with fewer, they are
   below it on all 13 (significantly on 11), in each of four training seeds.
6. **One KV cache serves every loop exactly.** A per-frequency complex key scale commutes with RoPE, so it can be
   applied to the query; with a post-attention value map, one stored prompt cache serves every decode loop.
7. **Evaluation protocol matters.** Under a non-standard protocol, one prefill loop looked best on 4-shot MATH-500 and
   MBPP+; that was answer format (Appendix B of the paper).

![Task map on Ouro-2.6B](paper/figs/fig_taskmap.png)

![The loop adapter](paper/figs/fig0_method.png)

![Prefill 4 / decode 2 on eight benchmarks](paper/figs/fig_equal.png)

![End-to-end time per answer](paper/figs/fig2_frontier.png)

![Looped decision models](paper/figs/fig_lev.png)

## When to use it, and when not

- Choose the prefill count per benchmark (or task family) on a small labeled set; it matters most with one or two
  decode loops. It is not a per-prompt router and the choice need not transfer to unseen tasks.
- The best count depends on the prompt format (few-shot vs zero-shot) and the checkpoint.
- Comparisons with native models include training the adapter; within-adapter comparisons isolate the prefill loops.
- More prefill loops raise time to first token; equal decode loops do not mean equal latency.

## Repository map

| path | what |
|---|---|
| `REPRODUCE.md` | every table and figure → script → result file; training and evaluation commands |
| `ouro/scripts/paper_numbers.py`, `ouro/results/items/` | per-item scores of every run the paper uses; recomputes every number |
| `ouro/model.py`, `ouro/adapter_model.py`, `ouro/batched.py`, `ouro/train_batched.py` | frozen Ouro backbone, loop adapter, batched KL distillation from the native model |
| `ouro/single_bank.py`, `ouro/bench_throughput.py` | exact single-bank decoding (two flash-attention calls merged by log-sum-exp), native baselines, throughput |
| `ouro/evaluate_bench.py`, `ouro/lmeval_ouro.py`, `ouro/run_lmeval.py`, `ouro/lmeval_tasks/` | evaluation (RULER, HELMET; lm-evaluation-harness wrapper and zero-shot thinking tasks) |
| `ouro/sensenova_runner.py`, `ouro/sensenova_bridge.py` | SenseNova-Loop-9B (hybrid Gated-DeltaNet) runner, state splicing and loop adapter |
| `ouro/scripts/` | paper numbers, figures, end-to-end latency |
| `ouro/runs_scripts/` | the launchers used for every run |
| `third_party/ouro/` | Ouro modeling code (Apache-2.0, see its LICENSE) |
| `paper/` | PDF, LaTeX source, figures, arXiv build script |

## Citation

```bibtex
@misc{hu2026untying,
  title  = {Untying Prefill and Decode Loops in Looped Language Models},
  author = {Hu, Yang},
  year   = {2026},
  note   = {https://github.com/yanghu819/untying-prefill-decode}
}
```
