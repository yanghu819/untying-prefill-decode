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

To regenerate the per-item scores themselves you need GPUs: train the loop adapters, run the evaluations, then
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
benchmarks in `$M/bench`, outputs in `$M/runs`); change `M` at the top of a launcher to move it. GPQA-Diamond is read
from a local CSV (`$M/gpqa-data/`, not redistributed) and IFEval from `$M/ifeval-data/` (task definitions in
`ouro/lmeval_tasks/gpqa_n_shot_local`, `ifeval_local`).

## Training (one B300 node, 3–4 GPUs per run)

| checkpoint | used for | launcher |
|---|---|---|
| Ouro-2.6B loop adapter `X26-S-universal` | task map, cross-fitted table, Pareto, design variants | `launch_s.sh` (= `run_mix2.sh X26-S-universal S <gpus> <port> 48000000 --micro 1`) |
| second seed `X26-S-seed2` | training-seed appendix | `launch_seed2.sh` |
| Ouro-1.4B loop adapter `O14-S-universal` | Ouro-1.4B task map, other checkpoints | `launch_scale.sh` |
| Ouro-2.6B-Thinking `T26-S-universal` | Figure 4, timing, Thinking appendix | `launch_think.sh` |
| Ouro-1.4B-Thinking `T14-S-universal` | Ouro-1.4B-Thinking appendix | `launch_think14.sh` |
| LoRA distillation of native 2/2 `T26-D-native22` (arm D) | distillation table | `launch_t26d.sh` |
| SenseNova-Loop-9B loop adapter `SN-C-universal`; GDN-state variant `SNG-C-universal` | SenseNova and hybrid appendices | `launch_sn.sh`; `launch_sng.sh` (`sensenova_bridge.py --mix G`) |
| variants `X26-O`, `X26-M`, `X26-V`, `X26-Sfz` (maps frozen) | design-variant appendix | `launch_o.sh`, `launch_m.sh`, `launch_v.sh`, `launch_fz.sh` |
| Nanbeige per-pair `NBX-p<m>d<n>` | evaluation appendix | `launch_nbs21.sh`, `nbs_cell.sh` |
| Lev continuations (read-deep arm and control, four seeds, 2x control) | decision models | `ouro/lev/train_rd.sh`, `after_train.sh` |

## Evaluation

A configuration is a *condition* string: `native<n>` (unmodified model, n loops for prefill and decode) or `S<m><n>`
(loop adapter, prefill m, decode n); `D` is the distillation baseline and `O/M/V<m><n>` are the design variants. On
SenseNova, `C<m>1[A|G]` and `splice<m>1[A|G]` select which part of the prompt state comes from the second prefill visit.

```bash
# lm-evaluation-harness through our runner
./ouro/runs_scripts/lm1.sh GPU o26x-S13-math - "path=$M/model-2.6b,condition=S13,checkpoint=$CK,max_length=16384" \
    "--tasks minerva_math500 --num_fewshot 4"
# RULER / HELMET with our scripts (evaluate_bench.py)
MODEL=$M/model-2.6b DATA=$M/bench BENCHES='ruler4k ruler8k ruler16k' ./ouro/runs_scripts/bench3.sh X26-S41f S41 $CK GPU 1
```

The task-map grids were run from job lists (`grid_q*_a1.txt`, `grid_q*_a2.txt`, `grid14_q_*.txt`, one shell line per
job) by `grid_worker.sh`, which starts the next job whenever a GPU is idle. Run names are the ones `paper_numbers.py`
reads (`CMP`, `MEANS`, `TASK_MAP`, `TASK_MAP_14`, `EQUAL`, `SN_EQUAL`, `SN_STATE` at the top of the script).

## Paper element → source

| element | runs (launcher) | computed by |
|---|---|---|
| Figure 1 / full task map (Ouro-2.6B, 13 benchmarks x 16 pairs + native) | `grid_q*_a{1,2}.txt`, `mn_low_prefill.sh`, `mn_math_grid.sh` | `paper_numbers.py` (`Task map` means) → `paper_figs.py fig_taskmap` |
| Table 1, cross-fitted choice of prefill loops (and the Ouro-1.4B table) | as above; `grid14_q_*.txt` | `paper_numbers.py` `taskmap_select` → `paper_figs.py tab_select` |
| Figure 4 and its numbers table (Ouro-2.6B-Thinking, eight benchmarks) | `launch_think.sh` evaluations, `grid_q3_a2.txt` (prefill 1 / decode 2) | `paper_numbers.py` `equal` → `fig_equal`, `tab_equal` |
| distillation table | `eval_t26d.sh` | `paper_numbers.py` (`Distillation baseline`) → `tab_distill` |
| Pareto figure, throughput table | `tput_sb.sh` (`ouro/bench_throughput.py`) | `paper_numbers.py` `throughput`, `paper_figs.py fig_pareto` |
| end-to-end time (Figure 7, timing table) | `ouro/scripts/e2e_latency.py` (batch 1, one idle GPU), `sn_e2e_after.sh` | `paper_numbers.py` (`RATIOS`) |
| decision models (Lev) | `ouro/lev/` | `paper_numbers.py` `lev` → `fig_lev` |
| Ouro-1.4B-Thinking appendix | `t14_head.sh` | `paper_numbers.py` `equal` → `tab_equal` |
| SenseNova and hybrid-state appendices | `sn_head.sh`, `sn_math_rest.sh`, `sn_rebalance*.sh`, `sn_arc_mix.sh`, `launch_sng.sh` | `paper_numbers.py` (`Hybrid state`, `sn_equal`) → `tab_hybrid`, `tab_sn` |
| single-cache exactness (512/512 tokens) | `ouro/scripts/single_bank_check.py` | — |
| answers without a boxed answer; probes | `ouro/scripts/wander_detail.py`; `said_meant_probe.py`, `said_meant_probe_sn.py` | `paper_numbers.py` (`wander`); `results/probe/` |
| paper PDF and arXiv package | — | in order: `paper_numbers.py compute`, `paper_figs.py`, compile `paper/latex/main.tex`, `paper/build_arxiv.py` |

`paper_numbers.py compute` also checks every number quoted in the text against hard-coded expectations (including the
12-of-13 / 10-of-13 counts of Table 1) and reports any mismatch.

## What is not here

Raw lm-evaluation-harness sample files (16 GB) and checkpoints are not in the repository; the per-item scores are
sufficient to verify every statistic, and the commands above regenerate the rest. Timing depends on the GPU (one NVIDIA
B300 for every timing in the paper).
