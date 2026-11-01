# Read Deep, Write Shallow: How Looped Language Models Should Spend Their Depth

**Paper:** [`paper/read-deep-write-shallow.pdf`](paper/read-deep-write-shallow.pdf) (arXiv link coming) · **Author:** Yang Hu ·
**Code license:** MIT · **Keywords:** looped language model, recurrent depth, universal transformer, Ouro, SenseNova-Loop,
test-time compute, prefill vs decode depth, KV cache, KV sharing, RoPE, depth allocation, efficient reasoning, overthinking

Looped (recurrent-depth) language models run one block of layers several times per token, and they spend the same number
of loops reading the prompt (prefill) and writing the answer (decode). This repository shows how that depth should be
split. A small **depth-split connector** (LoRA plus per-loop key–value maps on a frozen model) lets a looped model read
the prompt with *p* loops and write the answer with *d* loops, for any pair up to four.

## Key findings (quotable)

1. **In looped language models, depth spent reading the prompt buys understanding, while depth spent writing the answer
   buys multi-step derivation.** On Ouro-2.6B, at one writing loop, reading 4 loops instead of 1 adds **+15.2 MMLU**
   [14.3, 16.1] and **+8.6 RULER-16K** [4.5, 12.5] within the same connector; GSM8K moves mainly with writing depth.
   The read-depth effect replicates on Ouro-1.4B, Nanbeige4.2-3B, Ouro-2.6B-Thinking, Ouro-1.4B-Thinking and
   SenseNova-Loop-9B, and under a second training seed (MMLU +15.2 / +15.2, RULER-16K +8.6 / +8.6).
2. **Reading deeper rescues a collapsed shallow writer, even without training.** SenseNova-Loop-9B's one-loop writer
   solves 2.2% of GSM8K; letting the *same* writer read a two-loop prompt state (no parameter changed) cuts unanswered
   problems from 97% to 44% and lifts RULER-4K from 21.6 to 83.5.
3. **Writing shallow is where time is saved.** On Ouro-2.6B-Thinking, a connector that writes with 2 loops comes within
   one point of the 4-loop model's GSM8K accuracy (93.3 vs 94.2 on 1,319) at **0.55×** its time per answer (batch 1);
   on SenseNova-Loop-9B a one-loop writer matches the two-loop model at **0.59×**. Reading 2 or 4 loops before writing
   gives the same end-to-end time: the saving comes from shallow writing, enabled by the connector.
4. **One KV cache serves every loop exactly.** A per-frequency complex scale commutes with RoPE, so it can be folded into
   the query; with a post-attention value map, one stored prompt cache serves every decode loop (512/512 greedy tokens
   identical). Storing the answer once as well (trained in the inference-time "self-live" view) gives **2.17×** the
   decode throughput of native 4/4 on long answers (1K prompt, 2,048 generated tokens; batch 192 vs 32).
5. **Shallow KV keeps what was said, deep KV keeps what comes next.** Linear probes on the value cache (same weights,
   same layer): from loop 1 to 4, the previous token becomes less linearly decodable (57.7 → 45.5%) and the next token
   more (26.0 → 36.1%). Behaviourally, reading deep costs short-context retrieval (RULER-4K −3.8) and 4-shot MATH
   (−9.8), but the MATH loss turns into a **+7.0** gain when the exemplars are removed.

![The depth-split connector](paper/figs/fig0_method.png)

![Read x write map on Ouro-2.6B](paper/figs/fig1_grid.png)

| ![end-to-end frontier](paper/figs/fig2_frontier.png) | ![probes](paper/figs/fig4_probe.png) |
|---|---|

![Shallow-writer wandering by difficulty](paper/figs/fig3_wandering.png)

## Main results

| finding | evidence (paired bootstrap 95% CI) |
|---|---|
| Reading buys understanding (Ouro-2.6B, same connector, write 1) | read 4 − read 1: MMLU (14,042) +15.2 [14.3, 16.1]; RULER-16K (520) +8.6 [4.5, 12.5]; GSM8K (1,319) +8.0 [5.5, 10.3] |
| Reasoning models | Ouro-2.6B-Thinking HELMET: read 4 / write 2 +26.0 [22.9, 29.1] over native 2/2, +4.3 [2.5, 6.2] over native 4/4; Ouro-1.4B-Thinking (write 1): GSM8K +13.6 [10.4, 16.5], MMLU-Pro +17.1 [12.4, 22.1], HELMET +11.9 [8.9, 14.6] |
| Third family (hybrid Gated-DeltaNet / attention) | SenseNova-Loop-9B, same one-loop decoder, read 2 − read 1: HELMET +6.7 [5.8, 7.7], MMLU-Pro +5.5 [3.0, 8.1], BBH +7.0 [5.9, 8.0] |
| Shallow writers run out of budget | Ouro-2.6B-Thinking GSM8K: answers hitting the 4,096-token budget 43.5% (native 1/1) → 19.8% (connector, read 1) → 13.0% (read 4); the gain grows with problem difficulty |
| Writing shallow saves time | write-2 connector: 0.55× [0.46, 0.64] the time of native 4/4 per GSM8K answer, within one point of its accuracy |
| Serving | single prompt bank: 0.97 / 1.19 / 1.36 / 1.62× the fastest native implementation at write 1–4 (8K); prompt + answer stored once: 2.17× native 4/4 on long answers |

All numbers, sample sizes and intervals are in the paper; per-item outputs for the end-to-end, probe and throughput
results are in [`results/`](results/).

## When to use it, and when not

- Use read-deep / write-shallow for knowledge, long-context reading and long reasoning answers.
- Shallow writing needs the connector: plain KV sharing loses 17.5 GSM8K points at one writing loop.
- For short verbatim retrieval or few-shot prompts that must be imitated, read fewer loops (deep KV keeps less of the
  literal text).
- At three or more writing loops, a single deep prompt bank trails the native model on retrieval.
- 1.4B models collapse at read 4 / write 1 on few-shot CoT BBH; read 1–2, or write 2, does not.
- A shared answer bank is free at write 2–3 and costs 2.6 GSM8K points at write 4.

## Repository map

| path | what |
|---|---|
| `ouro/model.py`, `ouro/adapter_model.py`, `ouro/batched.py`, `ouro/train_batched.py` | frozen Ouro backbone, depth-split connector (LoRA + RoPE-commuting per-loop KV maps), batched KL distillation from the native model |
| `ouro/single_bank.py`, `ouro/bench_throughput.py` | exact single-bank decoding (two flash-attention calls merged by log-sum-exp), shared answer bank, native baselines, throughput |
| `ouro/evaluate_bench.py`, `ouro/lmeval_ouro.py`, `ouro/run_lmeval.py`, `ouro/lmeval_tasks/` | evaluation (RULER, HELMET, BBH, MATH; lm-evaluation-harness wrapper and zero-shot thinking tasks) |
| `ouro/looped_bridge.py` | Nanbeige4.2 and other Hugging Face looped families |
| `ouro/sensenova_runner.py`, `ouro/sensenova_bridge.py` | SenseNova-Loop-9B (hybrid Gated-DeltaNet) runner, cache splicing and connector |
| `ouro/scripts/` | analysis: end-to-end latency, said-vs-meant probes, wandering decomposition, bootstrap, figures |
| `ouro/runs_scripts/` | the launchers used for every run (GPU dispatcher, training and evaluation queues) |
| `third_party/ouro/` | Ouro modeling code (Apache-2.0, see its LICENSE) |
| `paper/` | PDF, LaTeX source and figures |

## Reproduce (outline)

```bash
# environment (uv)
cd environments/ouro && uv sync && cd ../..
# connector on Ouro-2.6B: one (p, d) cell sampled per micro-batch, 48M KL tokens, 4 GPUs
DATA=<mix dir> ./ouro/runs_scripts/run_mix2.sh X26-S-universal S 0,1,2,3 29751 48000000 --micro 1
# single-bank decoding check (exactness) and throughput
python ouro/scripts/single_bank_check.py <Ouro-2.6B> <checkpoint> <prompts.jsonl>
python -m ouro.bench_throughput --model-path <Ouro-2.6B> --condition S41sb --checkpoint <ckpt> --merge-lora ...
# evaluation: lm-evaluation-harness with our runner (conditions native<d>, S<p><d>, O<p><d>, ...)
python ouro/run_lmeval.py --model ouro --model_args path=<model>,condition=S41,checkpoint=<ckpt> --tasks gsm8k_cot --num_fewshot 3
```

Data preparation scripts are `ouro/prepare_adapter_data.py`, `ouro/prepare_mix.py`, `ouro/scripts/prepare_think_mix.py`
and `ouro/scripts/sn_prepare.py`; benchmark preparation is in `ouro/prepare_full_qa.py` (RULER) and
`ouro/prepare_helmet.py` (HELMET).

## Honest notes

Gains against native models include the adaptation of the connector; reading-depth effects are isolated by comparing
read *p* / write *d* with read *d* / write *d* of the same connector, or without training (SenseNova). Thinking
checkpoints on raw HELMET prompts can emit an immediate end-of-sequence; we decode them so that an answer cannot be
empty, for every configuration. Linear probes measure decodability, not causal use.

## Citation

```bibtex
@misc{hu2026readdeep,
  title  = {Read Deep, Write Shallow: How Looped Language Models Should Spend Their Depth},
  author = {Hu, Yang},
  year   = {2026},
  note   = {https://github.com/yanghu819/read-deep-write-shallow}
}
```
