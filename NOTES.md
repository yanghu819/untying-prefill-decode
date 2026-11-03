# Lab notes

Running log of experiments behind *Untying Prefill and Decode Loops in Looped Language Models*. Every number quoted in
the paper is recomputed by `ouro/scripts/paper_numbers.py` from `ouro/results/items/`; these notes record what was run,
why, and what changed in the paper. Launchers for every run are in `ouro/runs_scripts/`.

## State on 2026-11-02

- **Method.** A loop adapter (rank-16 LoRA plus per-loop RoPE-commuting key scales and value maps on a frozen looped
  model) lets the model prefill with *m* loops and decode with *n*; one set of parameters is trained for all 16 pairs
  (KL distillation from native 4/4, one pair sampled per micro-batch). One stored prompt KV cache serves every decode loop.
- **Framing.** The best (m, n) depends on the task. No claim that more prefill loops are always better; comparisons
  are made at matched decode cost n.
- **Task map (main result, in progress).** Ouro-2.6B, one adapter, all 16 cells plus native 1/1-4/4 on MMLU, MMLU-Pro,
  ARC-C, GPQA-Diamond, BBH, IFEval, GSM8K, 4-shot MATH-500, HumanEval, MBPP+ and RULER 4K/8K/16K.
  Interim: 4-shot MATH-500 is best at prefill 1 (prefill 1 / decode 3: 68.6 vs native 4/4 54.0); RULER-4K is best at
  prefill 2 (prefill 2 / decode 4: 92.1); MMLU needs prefill >= 3; with one decode loop the adapter's best prefill beats
  native 1/1 on every benchmark measured so far.
- **Ouro-2.6B-Thinking.** Prefill 4 / decode 2 is at least as high as native 2/2 on eight benchmarks (significantly
  higher on five) and answers GSM8K in 0.55x the time of native 4/4. Prefill 1 / decode 2 cells are running.
- **Ouro-1.4B-Thinking.** The 3-shot BBH drop seen earlier is a prefill-loop effect: prefill 1 / decode 2 scores 46.4
  against 24.0 at prefill 2 / decode 2.
- **SenseNova-Loop-9B (hybrid Gated-DeltaNet / attention).** With a trained adapter, giving the decoder the second
  prefill visit's state only in the Gated-DeltaNet layers raises HELMET and RULER over giving it in both kinds of layer
  and lowers MMLU-Pro; an adapter retrained on that state shows the same trade-off.
- **Lev looped decision models.** More prefill loops for the context: +4.4 to +7.5 points at one decision loop on three
  backbones (LevBench public development set), across four seeds per backbone and against a control trained for twice
  as many updates.
- **Running.** Remaining task-map cells, native full-set RULER, a LoRA-only distillation baseline at native 2/2 on
  Ouro-2.6B-Thinking (arm D), and the remaining m < n cells on 1.4B BBH.

## Log

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
