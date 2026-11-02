# Untying Prefill and Decode Loops in Looped Language Models

**Paper:** [`paper/untying-prefill-decode-loops.pdf`](paper/untying-prefill-decode-loops.pdf) (arXiv link coming) ·
**Author:** Yang Hu · **Code license:** MIT · **Keywords:** looped language model, recurrent depth, universal
transformer, Ouro, SenseNova-Loop, Lev, test-time compute, prefill vs decode, loop adapter, KV cache, RoPE

Looped (recurrent-depth) language models run one block of layers several times per token and, at inference, run the
same number of loops when they prefill the prompt and when they decode each answer token. This repository unties the
two. A small **loop adapter** (LoRA plus per-loop key–value maps on a frozen model) lets a looped model prefill with *m*
loops and decode with *n*; one set of its parameters serves all 16 pairs up to four loops, and one stored prompt KV
cache serves every decode loop.

**Every number in the paper is recomputed from released per-item scores by one script:**
`python ouro/scripts/paper_numbers.py compute` (numpy only, seconds on a laptop). See [`REPRODUCE.md`](REPRODUCE.md).

## Key findings

1. **Prefill loops and decode loops buy different things.** On Ouro-2.6B, with one loop adapter evaluated at all 16
   (m, n) pairs, knowledge (MMLU) and long-context retrieval (RULER-16K) change mainly with the prefill loops *m* and
   multi-step math (GSM8K) mainly with the decode loops *n*. At one decode loop, prefilling with 4 loops instead of 1
   adds **+15.2 MMLU** [14.3, 16.1] and **+8.6 RULER-16K** [4.3, 12.8] (full test sets, paired bootstrap 95% intervals).
   The effect depends on the task and its prompt (it hurts few-shot MATH and 3-shot BBH on the 1.4B models).
2. **Prefill 4 / decode 2 on Ouro-2.6B-Thinking** scores at least as high as native 2/2 on all eight standard
   benchmarks we test (significantly higher on five: MMLU-Pro, BBH, ARC-C, MBPP+, HELMET), at the same decode cost, and
   is within two points of native 4/4 on six. With the same adapter at prefill 2 / decode 2 as control, the prefill
   loops account for the MMLU-Pro (+8.1) and HELMET (+7.5) gains, not for GSM8K, MATH-500 or MBPP+.
3. **Fewer decode loops are faster end to end.** On 64 timed GSM8K problems (batch 1, prefill included) prefill 4 /
   decode 2 answers in **0.55×** [0.45, 0.67] the time of native 4/4; on all 1,319 problems it is 1.0 point below it
   (93.3 vs 94.2). On SenseNova-Loop-9B, prefill 1 / decode 1 scores 95.6 vs 95.8 for native 2/2 at **0.59×** the time.
4. **Looped decision models (Lev).** Prefilling the context with more loops than the decision uses makes Lev models
   4.4–7.5 points more accurate at one decision loop on three backbones (LevBench public development set), across four
   training seeds per backbone and against a control trained for twice as many updates.
5. **One KV cache serves every loop exactly.** A per-frequency complex key scale commutes with RoPE, so it can be
   applied to the query; with a post-attention value map, one stored prompt cache serves every decode loop (512/512
   greedy tokens identical to materialised per-loop caches).

![Prefill x decode grid on Ouro-2.6B](paper/figs/fig1_grid.png)

![The loop adapter](paper/figs/fig0_method.png)

![Prefill 4 / decode 2 on eight benchmarks](paper/figs/fig_equal.png)

![End-to-end time per answer](paper/figs/fig2_frontier.png)

![Looped decision models](paper/figs/fig_lev.png)

## When to use it, and when not

- More prefill loops for knowledge and long-context tasks; fewer decode loops when answers are long and latency matters.
- Gains against native models include training the adapter; prefill-loop effects are measured within one adapter
  (prefill m / decode n vs prefill n / decode n).
- For short verbatim retrieval or few-shot prompts that must be imitated, use fewer prefill loops.
- At three or more decode loops, one prompt KV cache from the fourth loop trails the native model on retrieval.
- The 1.4B adapters lose 3-shot CoT BBH.

## Repository map

| path | what |
|---|---|
| `REPRODUCE.md` | every table and figure → script → result file; training and evaluation commands |
| `ouro/scripts/paper_numbers.py`, `ouro/results/items/` | per-item scores of every run the paper uses; recomputes every number |
| `ouro/model.py`, `ouro/adapter_model.py`, `ouro/batched.py`, `ouro/train_batched.py` | frozen Ouro backbone, loop adapter, batched KL distillation from the native model |
| `ouro/single_bank.py`, `ouro/bench_throughput.py` | exact single-bank decoding (two flash-attention calls merged by log-sum-exp), native baselines, throughput |
| `ouro/evaluate_bench.py`, `ouro/lmeval_ouro.py`, `ouro/run_lmeval.py`, `ouro/lmeval_tasks/` | evaluation (RULER, HELMET; lm-evaluation-harness wrapper and zero-shot thinking tasks) |
| `ouro/sensenova_runner.py`, `ouro/sensenova_bridge.py` | SenseNova-Loop-9B (hybrid Gated-DeltaNet) runner, state splicing and loop adapter |
| `ouro/scripts/` | end-to-end latency, probes, figures |
| `ouro/runs_scripts/` | the launchers used for every run |
| `third_party/ouro/` | Ouro modeling code (Apache-2.0, see its LICENSE) |
| `paper/` | PDF, LaTeX source, figures, arXiv build script |

## Citation

```bibtex
@misc{hu2026untying,
  title  = {Untying Prefill and Decode Loops in Looped Language Models},
  author = {Hu, Yang},
  year   = {2026},
  note   = {https://github.com/yanghu819/read-deep-write-shallow}
}
```
