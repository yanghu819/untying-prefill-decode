# Untying Prefill and Decode Loops in Looped Language Models

**Paper:** [`paper/untying-prefill-decode-loops.pdf`](paper/untying-prefill-decode-loops.pdf) (arXiv link coming) ·
**Author:** Yang Hu · **Code license:** MIT · **Keywords:** looped language model, recurrent depth, universal
transformer, Ouro, SenseNova-Loop, Lev, test-time compute, prefill vs decode, loop adapter, KV cache, RoPE

Looped (recurrent-depth) language models run one block of layers several times per token and, at inference, run the
same number of loops when they prefill the prompt and when they decode each answer token. This repository unties the
two. A small **loop adapter** (LoRA plus per-loop key–value maps on a frozen model) lets a looped model prefill with *m*
loops and decode with *n*; one set of its parameters serves all 16 pairs up to four loops. Decode loop *k* reads the
prompt KV of prefill loop *m−n+k* (**aligned caches**, the default), or every decode loop reads one stored prompt cache
(**one cache**, the memory-saving variant).

**The paper's statistics are recomputed from released per-item scores by one script** and checked against the printed
values: `python ouro/scripts/paper_numbers.py compute` (numpy only). See [`REPRODUCE.md`](REPRODUCE.md) for what it covers.

## Key findings

Ouro base-model evaluation settings where available (strict-match GSM8K, final-answer MATH-500, evalplus code; native
Ouro-2.6B 4/4 reproduces the published scores within one point on MMLU, MMLU-Pro, BBH, ARC-C and HellaSwag). 95%
paired bootstrap intervals in the paper; every number is recomputed and checked by one script.

1. **Decode cheaply, prefill deeply.** On Ouro-2.6B with one decode loop, prefilling with four loops raises the average
   over 11 benchmarks from 44.2 (native 1/1) to **56.8**, above native 1/1 distilled with LoRA on the same data (49.8);
   with two decode loops the adapter averages **67.2**, within 3 points of native 4/4 (70.2). Two to four prefill loops
   are best on 14 of 15 settings at one decode loop (4-shot MATH-500 is the exception).
2. **Aligned caches match the native model at equal depth** (e.g. RULER-16K at 4/3: 79.8 vs 78.7) and decode at about
   the cost of native *n*/*n*; one cache gives up under one point of average accuracy for the smallest KV cache.
3. **The adapter is what makes untying work.** Without it the same untied configurations average 26.6 and 30.9.
   Training is cheap: 6M answer tokens already reach most of the 48M result. Three training seeds: ±0.2.
4. **Speed.** On Ouro-2.6B-Thinking, prefill 4 / decode 2 matches native 4/4's average accuracy over eight benchmarks
   at **3.2×** its decode throughput (one cache) and answers 192 timed GSM8K problems in **0.59×** the time at batch 1.
5. **Looped decision models (Lev), both directions.** More prefill than decision loops is significantly above a
   matched-training control on 12 of 13 (*m*, *n*) pairs; fewer is below it on all 13 (significantly on 11).
6. **Everything is traceable.** `ouro/results/provenance/` records every run's command, configuration, checkpoint
   hash, log and outputs; from-scratch re-runs reproduce the main-table cells item for item
   (`ouro/results/verify_rerun.txt`).

![Task map on Ouro-2.6B](paper/figs/fig_taskmap.png)

![The loop adapter](paper/figs/fig0_method.png)

![Prefill 4 / decode 2 on eight benchmarks](paper/figs/fig_equal.png)

![End-to-end time per answer](paper/figs/fig2_frontier.png)

![Looped decision models](paper/figs/fig_lev.png)

## When to use it, and when not

- Use it at low decode depth (one or two decode loops), with prefill 4 as the default; tune the prefill count per task
  family on a small labeled set if needed. It is not a per-prompt router.
- The best count depends on the prompt format: on few-shot MATH-500 one prefill loop is best.
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
