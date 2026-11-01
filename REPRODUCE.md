# Reproducing the paper

Every number in the paper comes from one of three places, all in this commit:

1. **Per-item scores** (`ouro/results/items/`, 1 MB): one file per evaluation run used by the paper (item id → score).
   `python ouro/scripts/paper_numbers.py compute` recomputes every paired comparison, interval and mean in the tables
   and text (paired bootstrap, 5,000 resamples, seed 0) and writes `ouro/results/paper_numbers.json`. This takes seconds
   on a laptop and needs only numpy.
2. **Per-answer timing records** (`ouro/results/e2e/`): Table 3 and the time ratios in Section 4.2, also recomputed by
   `paper_numbers.py compute`.
3. **Throughput and probe outputs** (`ouro/results/throughput-*`, `ouro/results/probe/`), summarised in
   `ouro/results/latest.json`; `ouro/scripts/paper_figs.py` draws Figures 1–3 from `paper_numbers.json` and Figure 5 from
   `latest.json`.

To regenerate the per-item scores themselves you need GPUs: train the connectors, run the evaluations, then
`python ouro/scripts/paper_numbers.py export RUNS_DIR` collects exactly the items the paper uses.

## Environment, models, data

```bash
cd environments/ouro && uv sync && cd ../..          # torch, transformers 4.x (Ouro), lm-eval, flash-attn
```

- Models (Hugging Face): `ByteDance/Ouro-2.6B`, `ByteDance/Ouro-1.4B`, `ByteDance/Ouro-2.6B-Thinking`,
  `ByteDance/Ouro-1.4B-Thinking`, `nanbeige/Nanbeige4.2-3B-Base`, `sensenova/SenseNova-Loop-9B-Preview` (needs
  transformers 5.18 on `PYTHONPATH` for the SenseNova scripts only).
- Training mix (48M KL tokens): `ouro/prepare_mix.py` (OpenThoughts, Tulu SFT, LongAlpaca, FineWeb-Edu; base models),
  `ouro/scripts/prepare_think_mix.py` (same samples in the thinking chat template), `ouro/scripts/sn_prepare.py`
  (re-tokenised for SenseNova).
- Benchmarks: `ouro/prepare_full_qa.py` (RULER 4K/8K/16K, 13 tasks x 40 rows), `ouro/prepare_helmet.py` (HELMET, 1,108
  rows), `ouro/runs_scripts/mk_bench_mini.py` (the 260-row RULER-16K subset of Figure 1). GSM8K, MMLU, MMLU-Pro,
  ARC-Challenge, BBH and MATH-500 come from lm-evaluation-harness (`ouro/lmeval_tasks/` adds the zero-shot thinking tasks
  and MATH-500).

The launchers assume the layout `M=/dev/shm/big-small-ouro` (models in `$M/model-2.6b`, …, data in `$M/mix-data-48m`,
benchmarks in `$M/bench`, outputs in `$M/runs`); change `M` at the top of a launcher to move it.

## Training (one B300 node, 3–4 GPUs per run)

| checkpoint | used in | command |
|---|---|---|
| Ouro-2.6B connector `X26-S-universal` | Fig. 1, Tables 1, 2, 4–7 | `ouro/runs_scripts/launch_s.sh` (= `run_mix2.sh X26-S-universal S <gpus> <port> 48000000 --micro 1`) |
| second seed `X26-S-seed2` | Table 7 | `ouro/runs_scripts/launch_seed2.sh` (`--seed 2`) |
| Ouro-1.4B connector `O14-S-universal` | Table 2 | `ouro/runs_scripts/launch_scale.sh` |
| Ouro-2.6B-Thinking `T26-S-universal` | Tables 2, 3, Figs. 2, 4 | `ouro/runs_scripts/launch_think.sh` |
| Ouro-1.4B-Thinking `T14-S-universal` | Table 2 | `ouro/runs_scripts/launch_think14.sh` |
| SenseNova-Loop-9B connector | Table 2, Fig. 2 | `ouro/runs_scripts/launch_sn.sh` (`ouro/sensenova_bridge.py`) |
| answer once `X26-O-universal`, first-loop branch `X26-M-universal`, bank per loop `X26-V-universal` | Table 6 | `launch_o.sh`, `launch_m.sh`, `launch_v.sh` |
| no-map control `X26-Sfz-universal` (training; not yet in the paper) | — | `launch_fz.sh` (arm S with `--freeze-maps`) |
| direct sharing per cell `X26-Y1`, `X26-Y2` | Table 6 | `run_mix.sh X26-Y<d> Y<d> <gpu> <port> 24000000 --micro 1` |
| Nanbeige per-cell `NBX-p<p>d<d>` | Appendix B | `ouro/runs_scripts/launch_nbs21.sh`, `nbs_cell.sh` |

## Evaluation

A configuration is a *condition* string: `native<d>` (unmodified model, d loops), `S<p><d>` (connector, read p, write d),
`O<p><d>` / `M<p><d>` / `V<p><d>` (ablation arms), `Y<d>` (direct sharing). Typical commands (from the launchers):

```bash
# lm-evaluation-harness through our runner (GSM8K 3-shot CoT, MMLU 5-shot, ARC 25-shot, BBH 3-shot CoT, MMLU-Pro, MATH-500)
./ouro/runs_scripts/lm1.sh GPU X26-S41-gsmfull - "path=$M/model-2.6b,condition=S41,checkpoint=$CK,max_length=16384" \
    "--tasks gsm8k_cot --num_fewshot 3"
# thinking models: zero-shot chat template, four GSM8K shards
./ouro/runs_scripts/lm1.sh GPU T26-S41-gsm8k_think_s0 - "path=$M/model-2.6b-think,condition=S41,checkpoint=$CK" \
    "--tasks gsm8k_think_s0"
# RULER / HELMET with our scripts (evaluate_bench.py)
MODEL=$M/model-2.6b DATA=$M/bench BENCHES='ruler4k ruler8k ruler16k' ./ouro/runs_scripts/bench3.sh X26-S41f S41 $CK GPU 1
```

The run names are the ones `paper_numbers.py` reads (`CMP` and `MEANS` at the top of the script list every run used for
every number in the paper).

## Paper element → source

| element | script | result file |
|---|---|---|
| Figure 1 (grid) | `paper_numbers.py` (means) → `paper_figs.py` | `results/items/*X26-S<p><d>*` (subsets) |
| Tables 1, 2, 5, 6, 7, Appendix B and Limitations numbers | `paper_numbers.py compute` | `results/items/`, `results/paper_numbers.json` |
| Table 3, Figure 2, time ratios in Section 4.2 (first record of every timing file is a warm-up and is dropped) | `ouro/scripts/e2e_latency.py` (batch 1, one idle GPU) → `paper_numbers.py` | `results/e2e/*.jsonl` |
| Table 4 (throughput) | `ouro/bench_throughput.py` via `ouro/runs_scripts/tput_sb.sh` | `results/throughput-*` |
| single-bank exactness (512/512 tokens) | `ouro/scripts/single_bank_check.py` | — |
| Figure 3 / Section 4.3 (answers without a boxed answer by difficulty, length, hesitation markers; SenseNova splice) | `paper_numbers.py compute` (`wander`, means) → `paper_figs.py` | `results/items/T__*` |
| Figure 5 / Section 4.4 (probes) | `ouro/scripts/said_meant_probe.py`, `said_meant_probe_sn.py` | `results/probe/` |
| Figures | `ouro/scripts/paper_figs.py` | `paper/figs/` |
| arXiv build | `paper/build_arxiv.py` (from `paper/latex/main.tex`) | `paper/arxiv/` |

## What is not here

Raw lm-evaluation-harness sample files (16 GB) and checkpoints are not in the repository; the per-item scores are
sufficient to verify every statistic, and the commands above regenerate the rest. Timing depends on the GPU (one NVIDIA
B300 for every timing in the paper).
