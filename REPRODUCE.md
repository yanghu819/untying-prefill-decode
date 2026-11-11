# Reproducing the paper

The paper's numbers come from these places, all in this commit:

1. **Per-item scores** (`ouro/results/items/`): one file per evaluation run used by the paper (item id → score).
   `python ouro/scripts/paper_numbers.py compute` recomputes every paired comparison, interval and mean in the tables
   and text (paired bootstrap, 5,000 resamples, seed 0) and writes `ouro/results/paper_numbers.json`. This takes seconds
   on a laptop and needs only numpy.
2. **Timing records** (`ouro/results/e2e/`, `ouro/results/throughput-sb4/`): the end-to-end timing table, time ratios
   and decode throughput, also recomputed by `paper_numbers.py compute`.
3. **Decision-model records** (`ouro/results/lev/`): per-depth LevBench development outputs for the Lev appendix.

`ouro/scripts/paper_figs.py` draws every figure and generated table from `paper_numbers.json`.

## Main-text results and where they come from

| Paper item | `paper_numbers.py` function | Runs (launcher in `ouro/runs_scripts/`) |
|---|---|---|
| Figure 1, Section 3 (time split) | `time_breakdown` | `time_controls.sh`, `time_sweep.sh` (passes `clean`, `clean3`) |
| Figure 2, Section 4 (prefill depth, m = 1..16) | `depth_sweep`, `same_weights`, `same_weights_seeds`, `plateau` | `sweep_depth.sh`, `launch_variant.sh wide`, `seed_main.sh ARM=a CELLS="11 12"` |
| Ouro-2.6B-Thinking and Ouro-1.4B sweeps | `think_sweep`, `depth_sweep_14` | `sweep_think.sh`, `sweep_depth.sh MP=model-1.4b` |
| Figure 3, Section 5 (equal measured time) | `time_breakdown`, `budget_pairs`, `batch_time` | `time_controls.sh` (folded maps `a<m>x<n>pm`), `tput_batch.sh` |
| Table 1 and its seeds | `main_table` | `launch_aligned.sh`, `seed_main.sh`, `launch_x26_distill.sh` (`RANK`, `SEED`), `launch_variant.sh fixed4 / lora` |
| Baseline learning rate (Appendix) | `lr_check` | `LRS=0.3/3 launch_x26_distill.sh`, `LRS=... launch_variant.sh lr` |
| Maps by context length (Section 6) | `CMP` rows "Maps by length" | `launch_variant.sh lora`, RULER via `bench3.sh` |
| MATH-500 exemplar check | `CMP` rows "Aligned MATH" | `math0_q.sh ARM=a` |

Every number quoted in the text has a check in `paper_numbers.py`; `compute` prints `!!` next to any that disagrees.

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
  rows), `ouro/runs_scripts/mk_bench_mini.py` (the 260-row RULER-16K subset of the appendix Pareto figure). GSM8K, MMLU, MMLU-Pro,
  ARC-Challenge, BBH and MATH-500 come from lm-evaluation-harness (`ouro/lmeval_tasks/` adds the zero-shot thinking tasks
  and MATH-500). MATH-500 is scored on the final answer (last `\boxed{}`, else the Minerva closing sentence) with math_verify (`paper_numbers.py boxed`), not with lm-eval's strict Minerva sentence match.

The launchers assume the layout `M=/dev/shm/big-small-ouro` (models in `$M/model-2.6b`, …, data in `$M/mix-data-48m`,
benchmarks in `$M/bench`, outputs in `$M/runs`); change `M` at the top of a launcher to move it. GPQA-Diamond is read
from a local CSV (`$M/gpqa-data/`, not redistributed) and IFEval from `$M/ifeval-data/` (task definitions in
`ouro/lmeval_tasks/gpqa_n_shot_local`, `ifeval_local`).

## Provenance

`ouro/results/provenance/` traces every result to its run: the launching command, evaluation configuration and
software versions, times, checkpoint sha256 and training log, the cleaned evaluation log and the model's outputs
(see its README). Regenerate with `python ouro/scripts/provenance.py RUNS_DIR OUT_DIR` on each evaluation machine.

## Training (one B300 node, 3–4 GPUs per run)

| checkpoint | used for | launcher |
|---|---|---|
| Ouro-2.6B aligned loop adapter `X26-a-universal` (arm a, the main method) | Table 1, Figures 2-3, aligned map, throughput | `launch_aligned.sh` (`SEED=2/3`: seeds; `TOK=6/12`: token ablation) |
| Ouro-1.4B and Thinking aligned adapters `O14-a-`, `T26-a-`, `T14-a-universal` | Ouro-1.4B table, Thinking figures and appendix | `launch_aligned_more.sh 14 / think / think14`, then `eval_think_aligned.sh` |
| Ouro-2.6B one-cache loop adapter `X26-S-universal` (arm S, the memory-saving variant) | one-cache rows of Table 1, task map, cross-fitted table, Pareto, design variants | `launch_s.sh` (= `run_mix2.sh X26-S-universal S <gpus> <port> 48000000 --micro 1`) |
| second seed `X26-S-seed2` | training-seed appendix | `launch_seed2.sh` |
| Ouro-1.4B loop adapter `O14-S-universal` | Ouro-1.4B task map, other checkpoints | `launch_scale.sh` |
| Ouro-2.6B-Thinking `T26-S-universal` | Figure 4, timing, Thinking appendix | `launch_think.sh` |
| Ouro-1.4B-Thinking `T14-S-universal` | Ouro-1.4B-Thinking appendix | `launch_think14.sh` |
| LoRA distillation of native 2/2 `T26-Df-native22` (arm D, every answer token supervised; `T26-D-native22` is the earlier control without the first answer token) | distillation table, timing | `launch_t26df.sh` |
| LoRA distillation of native n/n `X26-{E,D,H}48f-p{1,2,3}d{1,2,3}`, `X14-*48f-*` (Ouro-2.6B / 1.4B) | distilled comparisons of the cross-fitted choice | `launch_x26_distill.sh ARM GPUS PORT first` (`SIZE=14`), evaluated by `after_x26_distill.sh` |
| SenseNova-Loop-9B loop adapter `SN-C-universal`; GDN-state variant `SNG-C-universal` | SenseNova and hybrid appendices | `launch_sn.sh`; `launch_sng.sh` (`sensenova_bridge.py --mix G`) |
| variants `X26-O`, `X26-M`, `X26-V`, `X26-Sfz` (maps frozen) | design-variant appendix | `launch_o.sh`, `launch_m.sh`, `launch_v.sh`, `launch_fz.sh` |
| Nanbeige per-pair `NBX-p<m>d<n>` | evaluation appendix | `launch_nbs21.sh`, `nbs_cell.sh` |
| Lev continuations (untied arm and control, four seeds, 2x control) | decision models | `ouro/lev/train_rd.sh`, `after_train.sh` |

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

## Appendix elements → source

The main text (Figures 1-3, Table 1) is mapped in the table at the top of this file; this table covers the appendix.

| element | runs (launcher) | computed by |
|---|---|---|
| per-benchmark results table (Appendix, Extended Results; Table 1 shows its averages) | aligned: `launch_aligned.sh`; one cache: task-map runs; training-free rows: `raw_q.sh`; distilled rows: `launch_x26_distill.sh` + `after_x26_distill.sh`, `controls_first_chain.sh`, `distill_fullsets.sh`; seeds: `seed_main.sh`, `seed2_q.sh` | `paper_numbers.py` `main_table` → `paper_figs.py tab_main` |
| ablation table (training tokens, LoRA rank) | `ablate_tokens.sh` | `paper_numbers.py` `ablations` → `tab_ablation` |
| aligned map (appendix) | `launch_aligned.sh` (`CELLS=...`) | `paper_numbers.py` `aligned_cells` → `fig_aligned`, `tab_aligned` |
| smaller model at equal time; corrected distilled timing | `time_small.sh`; `time_t26d.sh` | `paper_numbers.py` `smaller_model`, `e2e_table` |
| aligned-cache throughput | `tput_aligned.sh` (decoder checked by `ouro/scripts/check_aligned_runner.py`) | `paper_numbers.py` `appendix_checks` |
| from-scratch re-runs | `verify_q.sh` | `ouro/scripts/verify_compare.py` → `ouro/results/verify_rerun.txt` |
| full task map (appendix; Ouro-2.6B, 15 settings x 16 pairs + native; Ouro's protocol) | `queues/grid_q*_a{1,2}.txt`, `mn_low_prefill.sh`, `mn_math_grid.sh`, `queues/fullsets_q.txt`, `protocol_q.sh` (HellaSwag, Winogrande), `evalplus_q.sh` + `ouro/scripts/score_evalplus.py` (code) | `paper_numbers.py` (`Task map` means) → `paper_figs.py fig_taskmap` |
| cross-fitted choice of prefill loops per benchmark (appendix, and the Ouro-1.4B table) | as above; `grid14_q_*.txt` | `paper_numbers.py` `taskmap_select` → `paper_figs.py tab_select` |
| prefill 4 / decode 2 on Ouro-2.6B-Thinking, eight benchmarks (appendix figure and table) | `launch_think.sh` evaluations, `queues/grid_q3_a2.txt` (prefill 1 / decode 2) | `paper_numbers.py` `equal` → `fig_equal`, `tab_equal` |
| distillation table | `eval_t26d.sh` | `paper_numbers.py` (`Distillation baseline`) → `tab_distill` |
| Pareto figure, throughput table | `tput_sb.sh` (`ouro/bench_throughput.py`) | `paper_numbers.py` `throughput`, `paper_figs.py fig_pareto` |
| end-to-end time (appendix timing table) | `ouro/scripts/e2e_latency.py` (batch 1, one idle GPU), `sn_e2e_after.sh` | `paper_numbers.py` (`RATIOS`) |
| decision models (Lev), both directions | `ouro/lev/`, `ouro/runs_scripts/lev_mn.sh` | `paper_numbers.py` `lev` → `fig_lev`, `tab_lev` |
| Ouro-1.4B-Thinking appendix | `t14_head.sh` | `paper_numbers.py` `equal` → `tab_equal` |
| SenseNova and hybrid-state appendices | `sn_head.sh`, `sn_math_rest.sh`, `sn_rebalance*.sh`, `sn_followup.sh`, `sn_arc_mix.sh`, `launch_sng.sh` | `paper_numbers.py` (`Hybrid state`, `sn_equal`) → `tab_hybrid`, `tab_sn` |
| single-cache exactness (512/512 tokens) | `ouro/scripts/single_bank_check.py` | — |
| paper PDF and arXiv package | — | in order: `paper_numbers.py compute`, `paper_figs.py`, compile `paper/latex/main.tex`, `paper/build_arxiv.py` |

`paper_numbers.py compute` also checks every number quoted in the text against hard-coded expectations (including the
11-of-13 / 9-of-13 counts of the cross-fitted table and 10 / 2 of 13 on Ouro-1.4B) and reports any mismatch with `!!`. Not covered: the long-answer throughput
and batch-size measurements, the parser diagnostics and the MBPP+ format counts (reported as measured).

## Helpers

`grid_worker.sh` (queue runner, two jobs per GPU), `wait_gpus.sh` (start a job when K GPUs are free), `lm1.sh` (one
lm-evaluation-harness job), `bench3.sh` (RULER / HELMET shards), `run_mix2.sh` (training), `mk_code_local.py` and
`mk_gpqa_local.py` (local task files), `ouro/lev/levbench_compare.py` (Lev comparisons), `launch_think_o.sh` (answer
stored once, Thinking), `mbpp3_q.sh` (the earlier MBPP+ prompt of the evaluation appendix). Queue files with the exact commands of
the grid runs are in `ouro/runs_scripts/queues/`; superseded launchers are in `ouro/runs_scripts/deprecated/`.

## What is not here

Raw lm-evaluation-harness sample files (16 GB) and checkpoints are not in the repository; the per-item scores are
sufficient to verify every statistic, and the commands above regenerate the rest. Timing depends on the GPU (one NVIDIA
B300 for every timing in the paper).
