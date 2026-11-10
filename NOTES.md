# Lab notes

Running log of experiments behind *Untying Prefill and Decode Loops in Looped Language Models*. Every number quoted in
the paper is recomputed by `ouro/scripts/paper_numbers.py` from `ouro/results/items/`; these notes record what was run,
why, and what changed in the paper. Launchers for every run are in `ouro/runs_scripts/`.

## State on 2026-11-03

- **Method.** A loop adapter (rank-16 LoRA plus per-loop RoPE-commuting key scales and value maps on a frozen looped
  model) lets the model prefill with *m* loops and decode with *n*; one set of parameters is trained for all 16 pairs
  (KL distillation from native 4/4, one pair sampled per micro-batch). One stored prompt KV cache serves every decode loop.
- **Framing.** The best (m, n) depends on the benchmark, prompt and decode budget. No claim that more prefill loops are
  always better; comparisons are made at the same number of decode loops n.
- **Task map (complete).** Ouro-2.6B, 13 benchmarks x 16 pairs + native 1/1-4/4 (220/220 cells). Choosing m per
  benchmark on held-out halves beats native n/n significantly on 12/13 benchmarks at n = 1 and 10/13 at n = 2.
  Within one adapter, 4-shot MATH-500 at decode 4: prefill 1 vs prefill 4 +16.6 [12.4, 21.0].
  Ouro-1.4B replication (5 benchmarks): 5/5 and 3/5.
- **Ouro-2.6B-Thinking at decode 2.** Prefill 4 beats prefill 1 on all 8 benchmarks (7 significant); against a LoRA
  distillation baseline at native 2/2 the adapter adds MMLU-Pro +5.2 and ARC-C +2.9.
- **Timing, hybrid model, decision models**: see the paper (Sections 4.4-4.5, appendices).
- **Nothing running.** Paper package builds (`paper/build_arxiv.py`); remaining work is submission formalities.

## Log

### 2026-11-07 (restructure for the 4-page arXiv version, advisor review)
- Story: in decode-dominated tasks, spend loops on prefill. Three experiment groups, each with its launcher:
  1. Time breakdown per request (prefill / decode / total, in/out tokens), batch 1, dedicated GPUs:
     `time_sweep.sh` (ouro/scripts/time_requests.py replays the exact lm-eval prompts and stop rules of GSM8K, MATH-500,
     BBH, MMLU-Pro; 100 requests per task, same items for every condition; replayed answers stored and scored) and
     `time_helmet.sh` (Thinking, HELMET, 1,108 long prompts, e2e_latency.py `helmet`).
  2. Prefill-depth sweep m = 1..16 at n = 1, 2 on the same weights: `sweep_depth.sh` (extrapolation of X26-a-universal,
     conditions a<m>x<n>) and `launch_variant.sh wide` (X26-a-wide, trained on m in {1,2,3,4,6,8,12,16}; OURO_CELLS).
  3. Same-budget controls: rank-80 LoRA-distilled 1/1 and 2/2 (`RANK=80 launch_x26_distill.sh`, 62.9M parameters vs
     the adapter's 63.3M), fixed-prefill-4 LoRA-only (`launch_variant.sh fixed4`, identity maps), training-free aligned
     (`CELLS="ali41 ali42" raw_q.sh`).
- Time breakdown (batch 1, 100 requests per task): prefill is 0.3-2.5% of the request time at Ours 1/1 and 1/2, as
  at Original n/n. But the adapter's decode step costs more than the original's at the same loops: GSM8K 44 vs 36
  ms/token at 1/1, 106 vs 69 at 1/2 (BBH 47 vs 30, 97 vs 62). At batch 1 the extra per-loop work (key/value maps, two
  attention calls merged by log-sum-exp, an aligned bank per decode loop) is launch-bound; at large batch (Table 10)
  it is hidden. Section 5's comparison on measured time has to include it.
- Deeper prefill changes the answer length, and that, not prefill, drives the time per request (same 100 requests,
  decode 1): MMLU-Pro 118 -> 386 -> 461 output tokens at m = 1, 4, 8 (5.5 -> 18.1 -> 20.6 s; accuracy 22 -> 46 -> 44),
  GSM8K and BBH nearly unchanged in length, MATH-500 shorter (633 -> 490 tokens; accuracy 37 -> 22). Prefill stays
  1-6% of the time. The paper must report both: the direct prefill cost and the length effect.
- Red-team round 2 (four Codex reviewers incl. an advisor persona; all Overall 2/5). Fixed in text: SwiftKV
  direction (it shortens prefill), overlap wording (benchmark items removed, training data unchanged), seed claim (three
  seeds only for the one-cache adapter), exact-recall scope (HELMET recall, 34 items), plateau wording (a stopping
  heuristic; the wide adapter shifts it on some settings; per-setting table in Appendix), wide training depths
  {1,2,3,4,6,8,12,16}, first-pass timing interference disclosed, Figure 3 caption, 4-page body (compact Table 1, full
  table, method figure and the batch-32 panel moved to the appendix), provenance regenerated (181 new runs). Open:
  clean timing pass (a2), timing of the fixed-prefill-4 and rank-80 controls and Original 3/3, maps folded into the
  aligned banks at prefill (no per-step map cost), a universal aligned LoRA-only control.
- Clean pass vs first pass (same 100 requests, identical answers in every cell): times differ in both directions by
  up to about 29%, median about 9%, for Original and Ours alike (Original 2/2 MMLU-Pro +20%, Ours 4/2 MMLU-Pro -22%).
  Batch-1 replay is launch-bound and varies run to run even on an idle box. Plan: figures and Section 5 from the clean
  pass (all configurations under the same conditions), the spread reported, time differences below about 20% not
  claimed.
- Controls on the measured-time plane (clean pass 2, four CoT tasks, mean accuracy / s per request): at one decode
  loop Distilled 1/1 34.2 / ~10 s and rank 80 33.5 / 10.2 s, Ours 4/1 with maps folded 41.3 / 11.6 s, 3/1 41.5 / 11.4,
  prefill-4 LoRA 4/1 42.0 / 11.0; at two decode loops Original 2/2 56.8 / 15.7, Distilled 2/2 58.5 / 16.4, rank 80
  60.5 / 15.8, Ours 3/2 folded 59.5 / 16.3, 4/2 59.0 / 16.4, prefill-4 LoRA 59.5 / 15.7. Folding the maps makes deep
  prefill at one decode loop a point between n=1 and n=2 that nothing else reaches (+7-8 points over distilled 1/1 for
  ~10% more time); at two decode loops all are within noise. The prefill-4 LoRA matches the adapter here as well.
  Pending: pass 3 (reverse order) to average with pass 2.
- Maps folded into the aligned banks at prefill (AlignedRunner premap, condition a<m>x<n>pm): decode then runs the native
  attention path. On 10 GSM8K requests (shared GPU) 4/2 drops from 88.0 to 61.2 ms/token, 4/1 from 47.8 to 36.3;
  6/10 and 8/10 greedy answers are identical (bf16 rounding of the folded banks), so its quality is measured on the
  replayed requests themselves. time_controls.sh times it, the fixed-prefill-4 LoRA, rank-80 distilled and
  Original 3/3 on the same requests (a2, after the clean pass).
- Timing noise: a second pass of the same requests (identical answers) differs by up to 25% in time (rep-*, a1).
  Some first-pass jobs shared their GPU with evaluations. 2026-11-08 14:00: a clean full pass of time_sweep.sh
  (OUT=clean) on all eight a2 GPUs with nothing else on the box; keep a2 free of other jobs until it ends.
- Training seeds (adapter and distilled controls) change the LoRA initialisation and the (m, n) pair drawn per
  micro-batch; the data order is the same for every seed. Distilled 2/2 seed 2 (`SEED=2 launch_x26_distill.sh D`) tests
  the small decode-2 gap (Ours 67.2 vs Distilled 65.3) against training noise.
- Quality vs time per request on the four chain-of-thought tasks (same 100 requests; Figure 3): batch 1 measured and
  batch 32 estimated (measured step cost x measured output tokens) agree. At batch 32: Original 2/2 56.8 at 15.0 s,
  Distilled 2/2 58.5 at 16.5 s, Ours 4/1 43.0 at 13.5 s, Ours 6/2 60.8 at 20.5 s, Original 4/4 64.8 at 29.3 s. On these
  tasks a second decode loop buys more accuracy per second than deeper prefill; short prompts keep decode launch-bound
  even at batch 32, so the adapter's per-step overhead is not amortised.
- Table 1 briefly moved (RULER column) when an unfinished seed-2 RULER run shrank the shared item set; seed runs below
  95% of the items now wait, as in the depth sweep.
- Hung evaluations (100% GPU, no progress, seen about seven times on aligned conditions): `kill -USR1` on the lm-eval
  process (faulthandler) showed the Python thread waiting in lmeval_ouro.score_batch on a GPU->CPU read, i.e. a CUDA
  kernel that never returns. Requeued hung jobs now run with OURO_NO_CUDNN_SDP=1 (cuDNN SDPA off, as evaluate_bench
  already allows), honoured by run_lmeval.py; the default kernel choice is unchanged for comparability.
  X26-ali42-hellaswag finished on its cuDNN-off rerun; X26s2-a41-hellaswag hung (stack: Python blocked launching an
  RMSNorm kernel, i.e. the GPU queue full behind a stuck kernel) and was requeued the same way.
  2026-11-08 07:55: two rank-80 distilled (native runner, not aligned) log-likelihood jobs hung as well, both on GPUs
  shared by two jobs; requeued with OURO_NO_CUDNN_SDP=1 and their OURO_LORA_RANK=80 (requeue must keep a job's env).
- Backups: `ouro/scripts/backup_ms.py` runs on both boxes every 2 h (first run at 19:50) and uploads new or changed
  lm-eval / bench outputs (one tar.gz per run), timings, training logs and final checkpoints, plus the prepared data
  (a1, once), to the private ModelScope model repo torushy/untying-prefill-decode-backup (folders a1/, a2/). The token
  lives only in ~/.b300-handoff/ms.token and /root/.ms.token on the boxes.
- Incident 16:25: a cleanup kill reached PID 1 of the a2 container; every process there restarted (/dev/shm intact).
  Lost: aligned seed 3 at 46.8M/48M (no intermediate checkpoint; to be retrained), rank-80 distilled 1/1 (restarted),
  running evals (requeued; lm-eval resumes from its cache). HELMET timing (e2e_latency.py) now follows the evaluation's
  decoding exactly (NO_EMPTY=1, no stop inside leading whitespace): replayed answers equal the evaluated ones.
- Thinking with aligned caches: Ours 4/2 70.6 vs Original 4/4 71.5 over eight benchmarks (HELMET 58.9 vs 57.0 with one
  cache), at 314 vs 125 decode tokens/s; Distilled 2/2 67.5.
- Red-team round 1 (Codex gpt-6-astra, three ARR personas, all 2-2.5/5): HELMET was item-averaged in Figure 6; corrected
  (adapter 4/2 70.1 vs native 4/4 71.5, +2.6 over distilled 2/2). GSM8K strict/flexible and overlap-excluded checks added.

### 2026-11-07 (aligned loop adapter)
- Diagnosis: on its own diagonal the one-cache adapter fell below native on retrieval (RULER-16K 4/4: 73.4 vs 81.3; Lev-3B
  m=n: -1.8 in all seeds). Unrestricted per-loop maps did not help, training-free aligned caches helped GSM8K but not
  retrieval. Aligned adapter (arm a, ouro/runs_scripts/launch_aligned.sh): decode loop k reads prefill loop m-n+k.
- X26-a-universal (48M): main-table avg 56.8 (4/1) and 67.2 (4/2) vs one cache 56.4 / 66.4; matches native at 4/4
  (MMLU 74.5/74.6, GSM8K 68.8/69.4); RULER-16K 4/3 79.8 vs native 78.7. Paper now leads with aligned caches; one cache
  is the memory-saving variant (Thinking figures still one cache until T26-a lands).
- Running: O14-a evals, T26-a training (then eval_think_aligned.sh), aligned seeds 2/3 (wait_gpus.sh), full aligned map,
  verify_q.sh re-runs. Related work: 30 verified citations added.
- Six aligned lm-eval jobs hung (100% GPU, no progress for 1.5-12 h; the same configs finished elsewhere): killed by
  PID and requeued (lm-eval resumes from its cache). Aligned seed 2: `ARM=a seed_main.sh 2` on a1 (trains, then queues
  the a41/a42 main-table benchmarks); seed 3 (trained directly with run_mix2.sh --seed 3 on a2) gets its benchmarks
  with `ARM=a seed_main.sh 3`, which skips training when the checkpoint exists.
- launch_aligned.sh skips benchmarks already on the box and skips training when the checkpoint exists (a2 holds a
  copied X26-a-universal without TRAIN_DONE; one accidental restart there overwrote its logs for a minute, killed, logs
  restored from a1, checkpoint md5 unchanged). Its SEED branch is removed (seeds go through seed_main.sh). Queued the
  missing a34 benchmarks for the full aligned map. PLatour bib entry: authors from the paper PDF.

### 2026-11-06 (mainstream restructure)
- Main text follows comparable papers: Table 1 (fixed 4/n configurations vs native, training-free untying and
  same-data LoRA distillation; avg 44.2 / 26.6 / 49.8 / 56.4 at n=1, 63.0 / 30.9 / 65.3 / 66.4 at n=2, native 4/4
  70.2), Figure 1 with the fixed prefill-4 adapter, ablation table (6M-48M tokens, LoRA rank 4/16/64), three training
  seeds for the main cells (avg 56.4 +- 0.0 and 66.6 +- 0.2). Niche analyses (per-benchmark choice, full map,
  overlap, MATH exemplar effect, hybrid states) are in the appendix. Throughput at 2K/15K prompts and 1.4B-Thinking
  timing are collected but not yet in the paper.

### 2026-11-05 (provenance)
- `ouro/results/provenance/`: 1,059 evaluation runs (commands, configs, versions, times, sample sha256, checkpoint
  sha256), 26 training records, cleaned logs, generative outputs; host names and IPs scrubbed. All paper keys covered
  except the stopped Ouro-1.4B H48f control (not used).

### 2026-11-05 (MATH-500 rescored on the final answer)
- lm-eval's strict Minerva match is a closing-sentence test: native Ouro-1.4B 4/4 scores 4.4 strict vs 70.6 on the final
  answer (last \boxed{}, math_verify; paper_numbers.py `boxed`). All MATH-500 keys now use `boxed`.
- Consequence: on 4-shot MATH-500 one prefill loop is best at every decode depth on Ouro-2.6B (S12 65.0 vs native 4/4
  53.6; S14 68.2 vs S44 51.0; replicated by seed 2); 0-shot reverses it (S41-S11 +8.0, S42-S22 +7.4). The old
  "format artifact" reading of MATH was wrong; the MBPP+ indentation artifact stands.
- New counts: 2.6B n=1 14/15 vs native, 9 within adapter, distilled 8 up / 1 down; n=2 9/15, adapter 3, distilled
  3 up / 2 down of 15; n=3-4 only MATH. 1.4B 12/15, 2/15. Zero-shot MATH grid (math0_q.sh) queued on a1 to test the
  exemplar explanation at every cell.

### 2026-11-05 (Codex round 7 fixes)
- Overlap audit rerun with question+choices fields (ctx, sentence) and code keys remapped to humaneval_chat/mbpp_chat;
  HellaSwag/Winogrande now audited; items shorter than 13 tokens are reported as unassessed. 187/542 code prompts
  overlap (was 126). Without overlapping items the chosen m wins 13/15 at n=1 (MBPP+ inconclusive, +2.5 [-1.1, 10.2])
  and 8/15 at n=2.
- Distilled 2/2 comparison: 1 significant gain, 2 losses, 11 inconclusive of 14 (D48f gsmfull landed; mmlufull running).
- Text: low-decode-budget scoping, seed claims limited to seed 0 plus direction across seeds, Ouro-1.4B-Thinking
  corrections, two-seed table guards, related work (Ouro 5.4.2 KV reuse, FlashLoop).
- Queued: 1.4B E48f/D48f evals (after_x26_distill.sh now on Ouro's protocol), S43 HellaSwag rerun (OOM), T26x-S12 evalplus.

### 2026-11-03 (new agent)
- Paper rewritten under Ouro's protocol (2026-11-05). Core story: prefill with more loops than you decode. Ouro-2.6B,
  15 settings: at one decode loop the best prefill count is 2-4 everywhere; chosen m vs native 1/1 14/15, vs the
  adapter's 1/1 10/15, vs distilled 1/1 (corrected controls E48f/D48f/H48f) 9/15 (RULER-4K below); at two decode loops
  8/14 vs native but 0/12 vs distilled 2/2; none at three or four. Ouro-1.4B 13/15, 3/15. Thinking 4/2 > native 2/2 on
  8/8 (4 significant), > distilled 2/2 (T26-Df) on MMLU-Pro +6.0, ARC-C +3.2. Timing on 192 GSM8K problems: 0.59x
  native 4/4; prefill 4/decode 2 takes 1.21x the time of distilled 2/2 (the saving is decoding with two loops). Without
  maps, GSM8K is +2.3 at 4/1 under strict match. Nanbeige GSM8K flips sign with strict match (-8.4). New appendix
  paragraphs on protocol and on what departing from it does. Open: HellaSwag S22/S32/S34/S44 reruns (n=2..4 counts are
  of 14), seed-2 replication, 1.4B controls, Codex round 7.
- Evaluation protocol aligned with Ouro (paper Appendix C.1, Table 16). Our native 4/4 already matched Ouro on MMLU
  (75.4 vs 74.6), MMLU-Pro (55.2 vs 55.7), BBH (80.4 vs 80.5) and ARC-C (66.5 vs 66.4), but not on code or MATH. Found:
  (1) 4-shot MATH-500 "prefill 1 beats its teacher" is answer-format: with one prefill loop the model ignores the
  exemplars' format and writes longer solutions (full-response verify 68.4, strict 18.6; prefill 4: 51.8 / 46.8);
  (2) our "MBPP+" task scored only the first MBPP assertion, and on base 2.6B every prefill >= 2 drops Python
  indentation in 35-63% of answers (prefill 1: <3%). Changes: GSM8K and MATH-500 scored with strict match (as Ouro);
  code with evalplus as Ouro ran it (evalplus default for a tokenizer with a chat template; humanevalplus_chat_local /
  mbppplus_chat_local + score_evalplus.py): native 4/4 HumanEval 75.6/69.5, MBPP 77.5/64.6 vs Ouro's 78.7/70.7 and
  80.4/66.6. BBH and MMLU-Pro on the full sets; HellaSwag and Winogrande added (protocol_q.sh). The evalplus prompt
  needs no trailing newline in direct completion (the reference transformers model emits end-of-text at once).
- Overlap audit (ouro/scripts/overlap_audit.py; 13-gram token matches between every training sequence of mix-data-48m
  and each evaluation question): MATH-500 101/500, code 126/542, MMLU 106/14,042, BBH 75/6,511, MMLU-Pro 24/1,400,
  others <= 3. Without those items Table 1 keeps 11/13 and 9/13 (selection-aware) and the 4-shot MATH contrast is
  +19.0 [14.0, 23.8] (399 items). Appendix A; guarded. The Thinking mixture (mix-think-48m) is still to audit.
- Distilled controls E48/D48/H48 complete (MMLU/GSM8K on full sets). Chosen m vs distilled n/n, selection-aware:
  above on 7 / 5 / 3 of 13 at n = 1 / 2 / 3, below on 1 / 1 / 3 (RULER; distillation alone helps short-context
  retrieval). Section 4.2 sentence + generated appendix table tab_selectdist; guard DISTILLED_COUNTS. These controls omit
  the first answer token; corrected E48f/D48f/H48f training, numbers to be swapped in when they land.
- Extra timing: 128 more GSM8K problems for native 2/2, 4/4, adapter 4/2, 2/2 and distilled 2/2 (time_more.sh, a1 GPU 3).
- Task map: Ouro-2.6B MMLU and GSM8K now on the full sets (14,042 / 1,319) for every cell; Table 1 counts unchanged
  (selection-aware 11/13, 9/13; within the adapter 6/13, 3/13; fixed choice 12, 11). Ouro-1.4B replication extended to
  all 13 settings: 10/13 and 2/13 selection-aware, 4/13 and 1/13 within the adapter, significantly below native 2/2 on
  3-shot BBH (-5.4); stated in 4.2 and Appendix D. Full-set GSM8K/MMLU of the distilled controls running.
- Lev m < n replicated over seeds 1-3 (21 runs, `lev_mn_seeds.sh`): all 13 pairs below the control in every seed
  (52/52 estimates), significantly in every seed on 10 of 13 (guard LEV_SEEDS). Section 4.5 no longer limited to seed 0.
- Distilled controls (preliminary, first-token-uncorrected E48/D48): chosen m vs distilled 1/1 significantly above on 7
  of 13, below on 1 (RULER-4K); vs distilled 2/2 above on 7 of 12, below on 2 (RULER-4K/8K). Gains over distillation
  are in knowledge, code and 4-shot math; distillation alone recovers short-context retrieval. Paper text waits for
  H48 and the corrected E48f/D48f/H48f.
- Codex round 6 (paper/codex_review_untie6.md, 8 objections). Fixed two real analysis bugs: (1) task-map means are now
  over the items all 20 configurations share (`taskmap_cells`; Figures 1/8 and the grid table read it; MMLU-Pro's best
  m at decode 1 is 4, not 3); (2) significance was decided on intervals rounded to one decimal; intervals are now kept
  at three decimals and printed with two where one would hide a nonzero bound (Lev m < n: 11 of 13 significant, not 10;
  two tiny no-map MMLU losses now stated). Found: the native-LoRA control (arms D/E/H) omits the KL on the first answer
  token (about 0.2% of supervised positions); stated in the paper, corrected re-runs to follow. Also: LevBench named as
  the rights-subset-v1 development export; matched-data/update control; hybrid conditions called mixed-state; 64 timed
  GSM8K problems in the abstract; Figure 6 label collision; Figure 8 shows only the five settings not in Figure 1;
  float barriers in the appendix; README/llms.txt/REPRODUCE counts updated.
- Codex round 5 (PDF pages attached, paper/codex_review_untie5.md; 10 objections). Applied: Table 1 intervals now
  repeat the choice of m (5,000 draws): 11/13 and 9/13 (fixed choice: 12, 10); within the adapter 6/13 and 3/13; 1.4B
  3/5 and 1/5; native comparisons described as the complete trained method; Figure 6 adds the same-adapter 2/2 point and
  says the saving comes from the adapted decoder; MELT added and CDB described fully; hybrid conditions described as
  holding post-block states at two visits; Lev scoped to seed-0 runs; map ablation and design-variant wording without
  equivalence claims, unrestricted-key rows printed and guarded (BANK); stale timing guards fixed and now checked;
  8-pt figure labels, square markers for the adapter; pseudocode uses a frozen native teacher and torch.randint;
  reproducibility appendix lists what the script does not recompute. Not applied: heatmap for Lev (paper uses one line
  style). Open: MATH-500 parser adjudication, distilled 2/2 timing, training-data overlap audit.
- Codex fourth review (paper/codex_review_untie4.md): 3/5 borderline; recomputed every statistic, all match
  paper_numbers.json. Applied: "significantly" in every count of wins; GSM8K / HumanEval peaks stated per the data
  (two; three or four); Figure 1 = best m per n chosen on the evaluation items (held-out: Table 1); MBPP+ "ties" ->
  no significant difference; distillation "not on knowledge" -> not significantly; Lev wording limited to the runs
  trained; Figure 4/5 titles narrowed; Figure 5 RULER is the 260-item set. New from existing scores: selection-aware
  bootstrap (choice of m repeated in every resample) keeps 11/13 and 9/13 significant at n = 1, 2; within the adapter
  the chosen m beats the adapter's own n/n on 8/13 and 5/13, never significantly below (isolates the prefill loops
  from training). Full grid of the task map now a generated appendix table (tab_taskmap). Open from the review:
  time the distilled 2/2 (after D48), MATH-500 parser adjudication, Lev seeds for m < n.
- Baseline recomputed on the cluster from a fresh clone: 0 printed values differ.
- Same-training controls for Table 1 launched. Table 1 compares the loop adapter with *untrained* native n/n; the
  only distillation control so far is Ouro-2.6B-Thinking at 2/2. New: LoRA distillation of native 1/1, 2/2, 3/3 on base
  Ouro-2.6B (arms E, D, H; `launch_x26_distill.sh`) with the adapter's data (mix-data-48m), teacher, 48M tokens, micro
  and seed. The older X26-E/D/H runs (24M tokens, different micro) are not matched and are not used. Then all 13
  benchmarks at the distilled n/n, so the chosen m can be compared with distilled n/n as well as native n/n.
- Ouro-1.4B replication extended from 5 to 10 lm-eval benchmarks (GSM8K, MMLU, BBH, GPQA-Diamond, IFEval added;
  `grid14b_q.txt`, 100 jobs on `grid_worker.sh`); RULER on 1.4B to follow.
- Lev m < n done (7 runs). Both directions: with more prefill than decision loops the model is significantly above the
  same-training control on 12 of 13 (m, n) pairs; with fewer it is below on all 13 (significantly on 10), down to
  -13.0 [-14.7, -11.3] (Lev-2.6B prefill 1 at four decision loops). Lev moved back into the main text (Section 4.5,
  user's request) with a generated both-directions table in the appendix (`tab_lev`); counts and quoted values are
  guarded in `paper_numbers.py` (`LEV_COUNTS`, `LEV_PRINTED`).

### 2026-11-03
- Handoff to a new agent: HANDOFF.md (state, rules, pipeline, pitfalls) and DEPRECATED.md (code kept only for
  imports). Deleted 44 per-item files the paper no longer uses, stale figure copies and superseded Codex reviews.
- All figures now one style (grey native n/n line, red loop-adapter line, point labels); Lev moved to the appendix;
  Lev m < n runs launched (`lev_mn.sh`).
- Cleanup: removed the probe and "answers without a final answer" analyses (scripts, results, figures, appendix
  sections), the unused SenseNova training-free appendix, the stale `latest.json` aggregate and long-context throughput
  records, and the 1.4B BBH cells the paper does not use. Labels in `paper_numbers.py` / `paper_figs.py` and launcher
  comments now use prefill m / decode n and loop adapter throughout. Paper numbers unchanged (0 printed values differ).

### 2026-11-02
- Repository consolidated: this repository is now the only maintained one; the earlier private research repository
  (with unrelated exploratory lines) is archived. Removed scripts and modules the paper does not use.
- Added GPQA-Diamond (0-shot multiple choice, normalized accuracy) and IFEval (prompt-level strict accuracy) to the task
  map, with local task definitions (`ouro/lmeval_tasks/ifeval_local`, `gpqa_n_shot_local`). GPQA questions are not
  redistributed; only per-item scores keyed by index are released.
- Reframed the paper as untying prefill and decode loops (title, notation prefill m / decode n, "loop adapter").
- Distillation baseline (arm D: native 2/2 layout, LoRA only, same data, teacher and 48M tokens as the adapter), on
  Ouro-2.6B-Thinking. Prefill 4 / decode 2 minus distilled 2/2: MMLU-Pro +5.2 [1.2, 9.3], ARC-C +2.9 [1.4, 4.4],
  BBH +2.2 [0.0, 4.5], HELMET +2.5 (n.s.). Distilled 2/2 minus native 2/2: HELMET +10.5, BBH +4.7, MMLU-Pro +1.7 (n.s.),
  ARC-C +0.7 (n.s.). So the untied adapter adds knowledge accuracy that plain distillation does not; most of the HELMET
  gain over native 2/2 comes from distillation. GSM8K, MATH-500 and code still running.
- Distillation baseline, remaining benchmarks (prefill 4 / decode 2 minus distilled 2/2): MATH-500 +0.8 (n.s.),
  HumanEval +5.5 (n.s.), MBPP+ +3.2 [-0.5, 6.9], GSM8K +0.8 [-0.4, 2.0] (all 1,319). The adapter is at or above
  plain distillation on all eight benchmarks, significantly on MMLU-Pro and ARC-C.
- Task map at 147/220 cells. Pattern: generating code and math prefers prefill 1 (MBPP+ prefill 1 / decode 2: 75 vs
  best native 68; 4-shot MATH prefill 1 / decode 3: 69 vs 55); knowledge and retrieval prefer more prefill loops (MMLU and
  MMLU-Pro m >= 3; GPQA-Diamond best at prefill 4 / decode 1, 38 vs best native 34; RULER best at m = 2-3, n = 4).
  HumanEval best at prefill 4 / decode 2 (44 vs best native 38). BBH: native 4/4 (81) slightly above the best cell so far.
- Cross-fitted choice of m per task (`taskmap_select` in paper_numbers.py): pick m on one half of the items, score it on
  the other half, swap; compare with native n/n on the same items. Interim (some cells still running): at n = 1 the chosen
  m beats native 1/1 on every benchmark so far (+5 to +21 points); at n = 2 it still wins clearly (MATH 4-shot +13.8,
  BBH +6.5, RULER-4K +5.7, ARC-C +2.7); at n >= 3 it is roughly on par (BBH -2.3, RULER-4K -2.9), except 4-shot MATH
  (+13.6/+14.4). Untying helps most when decoding is cheap.
- Paper restructured around the task map: Figure 1 = task map (8 panels; all 14 in the appendix), Table 1 = cross-fitted
  choice of m per benchmark and n vs native n/n, then Ouro-2.6B-Thinking at decode 2 (with the distillation control),
  end-to-end time, decision models. Old cross-model table moved to the appendix (Other Checkpoints); the fixed-n table
  removed. Abstract, introduction and conclusion rewritten. Bibliography script added (GPQA, IFEval).
- Codex review of the restructured draft (paper/codex_review_untie.md): weak reject as written; the headline generalized
  too far. Fixed: "property of the task" -> depends on benchmark, prompt and decode budget (HumanEval best at m = 4, GSM8K
  at m = 2); lead with the within-adapter 4-shot MATH contrast (+16.6 [12.4, 21.0]); "same decode cost" -> same number of
  decode loops, with the latency caveat; cross-fitting described as per-benchmark calibration (not per-prompt routing,
  intervals exclude selection variability); full 14-panel task map added to the appendix. Coverage claims ("nearly every
  benchmark", ranges) to be re-checked when the last cells land.
- Task map on Ouro-2.6B complete except native 4/4 on HumanEval / MBPP+ (queued). Final counts: cross-fitted m beats native
  n/n significantly on 12 of 13 benchmarks at n = 1 (+5.2 to +20.9; MBPP+ ties) and 10 of 13 at n = 2. Ouro-2.6B-Thinking at
  decode 2: prefill 1 below prefill 4 on all eight benchmarks (significantly on seven) - zero-shot chat prompts prefer more
  prefill loops, unlike the few-shot base model. One Thinking MATH shard hung for 8 h (third such hang) and was restarted.
- Replication queued: Ouro-1.4B task map (4-shot MATH, MBPP+/HumanEval, MMLU-Pro, ARC-C; BBH partly existing), 76 jobs.
- Codex second review of the untied draft (paper/codex_review_untie2.md): close to weak accept. Fixed: stale compiled PDF
  (always regenerate figures/tables, then compile); "decode cost" wording removed; HELMET deficit no longer attributed to
  prefill loops; Thinking claim narrowed (four beats one, not "optimal"; not all its benchmarks are zero-shot); conclusion
  without the code-and-math slogan; Appendix A formula b(l+t); MBPP+ at four decode loops marked pending.
- Ouro-1.4B replication (interim): MMLU-Pro cross-fitted m vs native n/n +17.9 / +6.2 / +2.9 / +0.5 (n = 1..4), ARC-C
  +17.2 / +4.2 / +1.1 / -0.1 - same shape as Ouro-2.6B. Code differs: MBPP+ best at prefill 2 (73-74) rather than 1.
  4-shot MATH grid pending.
- Task maps complete. Ouro-2.6B: 220/220 cells; MBPP+ at four decode loops +18.5. Ouro-1.4B (own adapter; MATH 4-shot,
  MBPP+, HumanEval, MMLU-Pro, ARC-C): chosen m beats native 1/1 significantly on 5 of 5 and native 2/2 on 3 of 5; the large
  one-prefill-loop gains of 2.6B on math and code do not carry over (chosen m mostly 2-3). Appendix table tab:select14.
- Codex third review (paper/codex_review_untie3.md): weak accept; "would defend acceptance". Applied: last decode-cost
  phrase removed; conclusion rewritten to what the figures and tables show; distillation control now has its own table
  (tab:distill, generated); Section 4.2 and the Pareto paragraph tightened (body ends on page 7).

### 2026-11-08 aligned seeds, Thinking depth sweep, distilled 1/1 seed 2
- Aligned adapter (Ours) over three training seeds (X26-a, X26s2-a, X26s3-a), 11-benchmark average on shared items:
  4/1 56.8 / 56.7 / 56.7 (mean 56.7 s.d. 0.1), 4/2 67.2 / 66.9 / 67.2 (mean 67.1 s.d. 0.2). In the Table 1 caption.
- Ouro-2.6B-Thinking, aligned adapter, decode 2, prefill m = 1,2,3,4,6(,8): GSM8K 90.7 92.3 92.9 94.1 93.9; MMLU-Pro
  36.4 41.4 49.3 51.0 48.1 48.3; BBH 60.1 .. 74.9 (m4) .. 71.8 (m8); HELMET peaks at m3 (60.7). Same saturation as the
  base model; one sentence in Section 4 (guard THINK_PRINTED).
- Started Distilled 1/1 seed 2 (X26-E48f-p1d1-s2, a1 GPUs 2,7; evals auto-queued as o26x-E48fs2-*) so the n=1
  same-budget comparison (Ours 4/1 vs Distilled 1/1) has a seed check like Distilled 2/2.
- Aligned adapter, MATH-500 prefill 4 vs 1 (same weights, paired, 500 problems): 4-shot -9.4 [-13.6,-5.4] (n=1),
  -18.0 [-22.2,-14.0] (n=2); 0-shot +8.0 [+4.0,+11.8] (n=1), +10.4 [+6.6,+14.2] (n=2). The exemplar reversal seen with
  the one-cache adapter holds for Ours; Section 6 now states it for Ours (math0_q.sh ARM=a).
- Started Distilled 1/1 seed 3 (X26-E48f-p1d1-s3, a1 GPUs 1,4); DIST_SEEDS = s2, s3 for both 1/1 and 2/2.

### 2026-11-09 timing pass 3 done: Section 5 on folded maps, two-pass means
- All times are the mean of clean + clean3 (opposite orders, idle a2). clean3 vs clean: |diff| median 5.9%, max 22.5%
  over 128 configuration-task cells; replayed answers identical in both passes.
- Per-step maps (sb) cost +19..+40% decode time per token vs Original (was quoted 31-51% from the contaminated pass);
  folded maps (pm) are within 10% (-10, -9, -2, -1, +5%), i.e. no overhead.
- Four CoT tasks, quality / s per request (pm for Ours): Ours 1/1 31.2/9.7, 4/1 41.2/11.2, Original 1/1 27.0/11.1,
  Distilled 1/1 34.2/10.2 (r80 33.5/10.6); Original 2/2 56.8/15.7, Ours 3/2 59.5/16.1, 4/2 59.0/17.2, Distilled 2/2
  58.5/16.1 (r80 60.5/15.5), prefill-4 LoRA 4/2 59.5/15.5; Original 3/3 63.5/21.7, 4/4 64.8/29.4.
  At n=1 Ours beats Original and Distilled at equal time; a second decode loop buys more; at n=2 Ours, Distilled and
  the prefill-4 LoRA are within noise. Prefill 1->4 at n=1 (pm): +16% time, all from longer answers (276 -> 319 tokens).
- Paper: abstract, Setup, Sections 3-6, Limitations and the timing appendix rewritten on these numbers; Figure 3 shows
  Ours with folded maps only (the per-step sweep to m=16 is in Figure 2); body still 4 pages.
- Red team round 3 (Codex gpt-6-astra ultra, 2/5): fixed Thinking GSM8K time (63 s at 2/2, 119 s at 4/4 -> "one to
  two minutes"), held-out comparison in Section 4 (32.3 not 31.9), timing scope (100 requests / two passes = base model),
  small-p approximation; frozen prefill loops added to Limitations; Figure 2 bottom row (quality vs time, unreadable,
  per-step maps) dropped, Figure 3 is the only quality-time figure. Open: the maps' value under the aligned layout
  (X26-a-lora on a1 is exactly the requested control); per-task frontier under fixed budgets (possible appendix).
- Distilled 2/2 over three seeds (D48f, s2, s3): 65.3 / 65.0 / 65.0, mean 65.1 s.d. 0.2, vs Ours 4/2 67.1 s.d. 0.2: the 2-point gap is ten seed s.d. (Table 1 caption).
- Wide adapter HumanEval+/MBPP+ sweep cells were unscored until 11-09 (score_evalplus_all.sh); with them the plateau
  rule gives 11 of 18 settings at m*=2-3, three at 4-5 (MMLU-Pro n=1 5, HumanEval+ n=1 5, MBPP+ n=2 4), MATH-500 at 1,
  no stop on MMLU-Pro / ARC-C n=2. Section 4, Appendix C and the guards updated; abstract says "on most benchmarks".
- Ouro-1.4B depth sweep (O14-a<m>x<n>, adapter trained on m<=4), partial: n=1 peaks at m=3-4 and falls after (MMLU
  42.8 -> 63.7 at m=3 -> 31.9 at m=16; ARC-C 45.4 -> 56.3 at m=4 -> 35.6); BBH collapses at m>=2 (known 1.4B early stop
  under arm a); code is 0 at m=16 (format breaks). n=2 still running.

### 2026-11-09 baseline audit (user: "没有压低baseline")
- Protocol: every row scored identically; native 4/4 within 1 point of Ouro's published MMLU, MMLU-Pro, BBH, ARC-C,
  HellaSwag (Table repro). GSM8K strict vs flexible and MATH-500 4-shot vs Ouro's unreleased prompt are documented;
  flexible GSM8K shrinks Ours 4/2 - Original 2/2 to +1.2 [-0.8, +3.3] (appendix states it).
- Timing: Original n/n uses the fastest native implementation (static per-loop banks + the same fused two-call flash
  attention as Ours; appendix lists the slower dynamic / contiguous caches). Native and folded-map decode run the same
  code path; per-token times overlap (n=1: 32-43 ms per task-pass for every config), the -9% for Ours at n=1 is pass
  noise (native1st clean3 MATH 43.4, BBH 38.1), so the paper only claims "within 10%".
- Distilled: same data, batch 24, 48M KL tokens, LR schedule and micro 1 as the adapter, plus first-token supervision;
  rank 16 and rank 80. Neither side's LR was tuned -> 3x-LR Distilled 1/1 (a2 GPUs 4-7) and 2/2 (a1 GPUs 2,3)
  launched (LRS=3, runs X26-E48f-p1d1-lr3, X26-D48f-p2d2-lr3); if either beats the 1x control, it becomes the baseline.
- Section 5 now quotes Ours 4/2 (the Table 1 configuration) first, 3/2 (best of Ours) in parentheses.
- Table 1 Ours rows are the pre-set m=4, not a per-task best m.
- Also 0.3x LR: X26-D48f-p2d2-lr0.3 (a1 GPUs 5,6), X26-E48f-p1d1-lr0.3 (a2 GPUs 1,3): Distilled gets an LR sweep {0.3, 1, 3}; the adapter keeps its single untuned LR.
- Strongest baseline also tuned: rank-80 Distilled 2/2 at 3x LR (X26-D48fr80-p2d2-lr3, a1 GPUs 0,7).
- Aligned LoRA-only control (X26lo: aligned layout, all (m,n) pairs, maps frozen at identity): avg 55.0 (4/1), 66.8
  (4/2) vs Ours 56.8 / 67.2; at 4/1 the ten non-RULER benchmarks move by <= 2.8 points, RULER-16K 49.8 vs 64.7
  (-14.9). The maps are what keep one all-pairs adapter good on long prompts (prefill-4-only LoRA: 58.3). Method's
  ablation sentence now uses this aligned control; Table 1 has the row "Ours 4/n, maps frozen".
- Ouro-1.4B sweep complete: best m is 3 or 4 on 13 of 18 settings, lower beyond; BBH n=1 collapses at m>=2 (known
  1.4B early stop), code 0 at m=16. Section 4 sentence + Appendix table (tab_depth14).
- Body trimmed to 4 pages (intro roadmap sentence removed, HELMET sentence and Conclusion shortened).
- Distilled 1/1 over three seeds: 49.8 / 49.4 / 50.0, mean 49.7 s.d. 0.3 (Ours 4/1 56.7 s.d. 0.1). Table 1 caption.
- Red team round 4 (Codex): 3/5 borderline (was 2/5). Fixed: rank-80 'about as many' parameters (62.9M vs 63.3M), both MATH-500 settings at m*=1, three sentences rewritten. Queued RULER-4K/8K for the maps-frozen control (maps' benefit beyond RULER-16K). Open: LR-tuned Distilled (running); training prefill across depths (Limitations).
- Ours also gets the LR check: X26-a-lr3 (main aligned adapter, LR x3; a2 GPUs 4,6; evals X26alr3-a4<n>-*). Early Distilled 1/1 x3 vs x1: GSM8K 39.1 vs 40.4, Winogrande 65.9 vs 66.5, MMLU 53.7 vs 52.7 (no clear gain).
- Ours at 0.3x LR: X26-a-lr0.3 (a2 GPUs 7,5). Ours and Distilled both get LR in {0.3, 1, 3}.
- Maps by RULER length (Ours vs maps frozen, aligned): 4/1 4K -0.1, 8K +1.4 [-0.6,+3.2], 16K +14.9 [+11.8,+18.1];
  4/2 4K +1.1, 8K +0.8, 16K +2.0 [+0.4,+3.6]. Vs prefill-4 LoRA: 4/1 16K +6.4, 4K/8K within 0.5. The maps matter only
  for the longest prompts at one decode loop; Method says so (guard 1.4).
- LR check so far: Distilled 1/1 x3 = 50.1 vs x1 49.8 (rank 80: 50.2); others pending.
- Prefill-4 LoRA seed 2 (SEED=2 launch_variant.sh fixed4, X26-a-fixed4-lora-s2, a2 GPUs 1,3; evals X26f4s2-a4<n>-*).
- LR check, Distilled 1/1 (11-bench avg): x0.3 49.2, x1 49.8, x3 50.1, rank 80 50.2, rank 80 x3 50.0 -> range 49.2-50.2, Ours 4/1 56.8. The n=1 baseline is not under-tuned. 2/2 and Ours variants pending. Also queued aligned seeds 2/3 at a11/a12 (CELLS="11 12" seed_main.sh, a2) for the same-weights depth gain over seeds.
- Maps-frozen control seed 3 (SEED=3 launch_variant.sh lora, X26-a-lora-s3, a1 GPUs 4,5).
- Prefill-4 LoRA seed 3 (SEED=3 launch_variant.sh fixed4, X26-a-fixed4-lora-s3, a1 GPUs 1,7).
- Wide adapter seed 2 (SEED=2 launch_variant.sh wide, X26-a-wide-s2, 32 cells m<=16, a2 GPUs 0,2); its depth sweep (X26ws2-*) runs after training via sweep_depth.sh.
- Wide adapter seed 3 (X26-a-wide-s3, a1 GPUs 0,3,6).
- LR check complete for Distilled: 1/1 49.2 / 49.8 / 50.1 (x0.3 / x1 / x3), rank 80 50.2 / 50.0 (x1 / x3); 2/2 64.7 /
  65.3 / 65.2, rank 80 65.4 / 65.6. Max change from scaling the LR 0.6. Setup sentence + Appendix "Baseline learning
  rate". Ours x3: 4/1 57.4 (x1 56.8); 4/2 and x0.3 pending.
- Same-weights depth gain over three seeds: 4/1 - 1/1 = 9.6 / 8.5 / 8.8 (9.0 +- 0.6), 4/2 - 1/2 = 9.3 / 9.0 / 9.0
  (9.1 +- 0.2). Section 4 (replaces the MMLU / HumanEval+ examples).
- Maps-frozen seed 2: RULER-16K Ours vs frozen +22.3 [+18.5,+26.0] at 4/1, +6.9 at 4/2 (seed 0: +14.9, +2.0).
- Ours at x3 LR: 4/1 57.4, 4/2 67.8 (+0.6 each over x1 56.8 / 67.2): Ours moves as much as Distilled; both are reported at the default LR, so the comparison does not favour Ours. x0.3 pending.
- Rank-80 Distilled 2/2 (closest baseline) seeds 2 and 3: X26-D48fr80-p2d2-s{2,3}, a2 GPUs 1,3 and 4,5 (62.9M trainable params checked).
- Rank-80 Distilled 1/1 seed 2: X26-E48fr80-p1d1-s2 (a2 GPUs 6,7).
- LR check complete: Ours x0.3 / x1 / x3 = 55.5 / 56.8 / 57.4 (4/1), 66.8 / 67.2 / 67.8 (4/2); Ours's weakest LR beats the best Distilled (50.2, 65.6). Appendix 'Baseline learning rate' says so; both reported at the default LR.

### 2026-11-10 wrap-up step 1
- budget_pairs (same 400 timed requests, paired bootstrap): Ours 4/1 - Original 1/1 +14.2 [9.0, 19.8] at x1.01 time;
  - Distilled 1/1 +7.0 [1.8, 12.2] at x1.10; Original 2/2 - Ours 4/1 +15.5 [10.2, 20.8] at x1.40; Ours 4/2 vs
  Distilled 2/2 / rank 80 / prefill-4 LoRA: +0.5 / -1.5 / -0.5, all intervals contain 0. Section 5 states these.
- Appendix guide paragraph (which appendices support the body; native = Original, LoRA-distilled = Distilled, loop
  adapter = Ours); REPRODUCE.md maps each main-text result to its function and launcher.
