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

### 2026-11-03 (new agent)
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
