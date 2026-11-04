**Score: 2/5 — reject in its current form.**

I audited all 22 pages. Independent recomputation reproduces **the entire committed JSON exactly**, and all **10 generated table files regenerate byte for byte**. However, computation finishes successfully despite **five missing-input warnings**, and another required comparison is silently omitted. Numerical reproducibility is improved; completeness and interpretation remain inadequate.

Sources: :codex-file-citation{path="/Users/mac2/untied/untying-prefill-decode/paper/latex/main.pdf" purpose="source"} :codex-file-citation{path="/Users/mac2/untied/untying-prefill-decode/paper/latex/main.tex" purpose="source"} :codex-file-citation{path="/Users/mac2/untied/untying-prefill-decode/ouro/results/paper_numbers.json" purpose="source"} :codex-file-citation{path="/Users/mac2/untied/untying-prefill-decode/ouro/scripts/paper_numbers.py" purpose="source"}

I count **nine substantive objections**, ranked below. The subsequent figure, table, and layout audit specifies additional fixes without counting the same underlying problem twice.

1. **The timed distilled baseline is still the wrong checkpoint.**  
   Figure 6, §4.4, Table 8; pp7 and 18.

   Accuracy comparisons use corrected **T26-Df-native22**. Timing still uses **D2st**, associated with the earlier **T26-D-native22** checkpoint. No corrected `Df2st` timing records are present. Making the timing launcher configurable has not repaired the experiment.

   The existing 192-item timing records give:

   | Configuration | Accuracy (%) | Mean seconds/answer |
   |---|---:|---:|
   | Native 2/2 | 92.7 | 63.3 |
   | Native 4/4 | 92.7 | 118.5 |
   | Earlier distilled 2/2 | 93.8 | 57.7 |
   | Adapter 2/2 | 95.8 | 68.3 |
   | Adapter 4/2 | 94.3 | 69.6 |

   Consequently, Figure 6’s “about as fast” hides a **1.21× [1.06, 1.38]** answer-time cost against that distilled control. “The prefill loops buy accuracy” is also not demonstrated by these plotted points: adapter 4/2 scores below adapter 2/2.

   Sampling remains consecutive IDs **1–192**, following removal of item 0 as warm-up. The corresponding generation-cap counts are **11/8/13/5/6** in the table’s order. MATH uses IDs **1–40**, with **7/6/10** capped answers for native 2/2, native 4/4, and adapter 4/2. These details matter when shorter generated answers contribute to speed.

   **Exact fix:** time the corrected distilled checkpoint on common items, record its hash, and regenerate Figure 6 and the ratios. Replace Table 9’s earlier 64-item Ouro cohort with the main cohort, or explicitly label it a pilot. Report selection, TTFT, generated tokens, cap counts, pooled decode time/token, and timing intervals. Qualify conclusions to the consecutive-ID sample or use a fixed random/stratified sample.

   Until retiming, label the point as an **earlier distillation control**. Replace the disputed caption explanation with:  
   > “Extra prefill does not improve accuracy on this timed subset; its gains on other benchmarks are reported in §4.3.”

2. **The validation pipeline still cannot establish the promised numerical completeness.**  
   Appendix L, pp22; reproducibility instructions.

   Four unique required inputs are missing:

   - HellaSwag `S43`.
   - Distilled decode-two MMLU, `D48f`.
   - Thinking `S12` HumanEval+.
   - Thinking `S12` MBPP+.

   The two code inputs each generate two warnings; HellaSwag generates one. Distilled MMLU is silently skipped. Missing records do not fail validation, and even the numerical-mismatch counter does not determine the exit status.

   The JSON also omits these printed empirical measurements:

   - Long-answer throughput **586/219/570/253**.
   - Shared-answer-cache batch/throughput **32→192**, **219→476**, **759 versus 586**.
   - Alternative-cache throughput **500/245/105** and **267/130/57**.
   - The **512-token** cache-equivalence result.
   - Parser diagnostics and MBPP formatting counts.
   - The pooled **1.27×** decode-time/token statistic.
   - Manifests establishing actual parameter counts, consumed training targets, and **6.75-hour** training duration.

   Appendix L now admits some omissions. That is better disclosure, but it does not satisfy “every number.” The reproducibility instructions still name the old distilled checkpoint and obsolete **11/13, 9/13, 10/13, 2/13** counts.

   **Exact fix:** add required-input and measurement manifests; serialize every empirical aggregate with its source, configuration, and checkpoint identity; assert expected comparison counts; and make missing inputs or mismatches exit nonzero. Generate numerical tables and headline counts from those records. Update the experiment mapping in :codex-file-citation{path="/Users/mac2/untied/untying-prefill-decode/REPRODUCE.md" purpose="source"}.

3. **The advertised complete 15-setting task map remains incomplete.**  
   Figure 1 caption, Figure 8, Table 3; pp2 and 14.

   There are **14 complete maps**, because Ouro-2.6B HellaSwag `S43` is missing. Therefore:

   - Figures 1 and 8 together show **14 settings**.
   - Figure 8 contains **six panels**, while its caption says **five**.
   - Table 3 omits HellaSwag.
   - Table 1’s decode-three column contains **14 completed settings**.

   **Exact fix:** complete HellaSwag `S43` and regenerate everything. Figure 8 should then say **“The seven task-map settings not shown in Figure 1.”** Until then, explicitly identify 14 complete maps and the missing configuration; explain the table dash.

   **This does not invalidate the one-decode-loop universal claim.** All four prefill choices exist for all 15 settings at decode one, and every optimum is indeed between two and four.

4. **“About as good” still turns inconclusive comparisons into equivalence.**  
   Abstract, Limitations, Table 4; pp1, 8, and 15.

   The revised counts are correct: at decode two there are **1 significant gain, 2 significant losses, and 11 inconclusive differences across 14 comparisons**. GSM8K is now present; MMLU remains missing. These results do not establish equivalence.

   The same problem appears in §3 and Appendix H’s “no worse on GSM8K” description of frozen maps. At 1/1 the difference is **−0.7 [−2.3, 0.9]**. That is an inconclusive loss, not demonstrated noninferiority.

   **Exact fix:** remove “about as good” and state the counts directly. Complete MMLU and explain every dash. If equivalence is intended, declare a margin and perform the appropriate analysis. Change the frozen-map claim to:  
   > “No tested GSM8K difference is significantly negative; at 4/1 the frozen-map control scores 2.3 points higher [0.2, 4.5].”

5. **The evaluation-protocol claim remains broader than the reproduction evidence.**  
   Introduction, contribution statement, Conclusion, Limitations, Appendix B.

   The abstract’s substitution caveat is improved, but “the models’ own evaluation protocol” remains elsewhere. Ouro specifies five-shot MATH and different Thinking decoding/scoring settings. More seriously, this paper’s native Ouro-2.6B 4/4 scores are **69.4 GSM8K and 48.2 MATH**, versus published **81.58 and 90.85**—gaps of approximately **12.2 and 42.7 points**. These discrepancies do not prove an implementation bug, but they require explicit diagnosis alongside the selectively reported near-matching benchmarks. [Ouro evaluation settings and results](https://arxiv.org/html/2510.25741v4#S5.SS1)

   Appendix B also says two results reversed “under final-answer scoring,” although the code evaluation changes prompts and evaluation machinery too. Limitations still calls this “an answer-format effect,” stronger than the demonstrated diagnosis.

   **Exact fix:** consistently say **“Ouro base-model settings where available, with documented substitutions.”** Add an original-versus-reproduced baseline table covering all relevant tasks, including math. Explain the large discrepancies or narrow the claims to this paper’s evaluation protocol.

   Replace the categorical format claim with:  
   > “These effects reversed under the revised evaluation protocols; the diagnostics indicate sensitivity to formatting and answer extraction.”

   Separate prompt, parser, sanitization, and test-suite changes before attributing a reversal to one cause.

6. **Figure 1 still does not display the causal effect advertised by its bold caption.**  
   Figure 1 and related introductory claims; pp1–2.

   The red–grey difference includes adaptation, maps, and cache changes. It is not the effect of increasing prefill within one trained adapter. The added explanatory sentence acknowledges this, but readers still encounter the causal headline first.

   Nor does the gap monotonically “close” as decode loops increase: individual benchmarks can reverse or widen again.

   **Exact fix:** add the same adapter’s \(m=n\) curve, or change the bold caption to:  
   > “At one decode loop, the trained adapter outperforms the native model on these benchmarks.”

   Use within-adapter contrasts to demonstrate what prefill buys. Replace the monotonic wording with **“The advantage generally diminishes at higher decode depths.”** Restrict Figure 5’s associated accuracy statement to its four displayed settings.

7. **The printed training objective still differs from the implementation.**  
   Figure 3, §3, Appendix A; pp4 and 11.

   Figure 3 divides by `answer.sum()`. Actual training supervises all suffix targets but divides by the global sum of **\(|a|-1\)**. The new budget caveat discloses scheduling accounting, but does not correct the objective or the “48M answer tokens” description.

   For existing checkpoints, the implemented objective is:

   \[
   L=\frac{\sum_i\sum_{j=1}^{|a_i|}\mathrm{KL}_{ij}}
           {\sum_i(|a_i|-1)}.
   \]

   **Exact fix:** print this denominator in Figure 3 and Appendix A. Report nominal scheduling tokens separately from actual supervised targets; the latter equals the former plus the number of consumed examples. Use the supervised mask consistently in future training.

   This is **not** a remaining unequal-first-target accuracy control: the corrected adapter and distilled control both supervise the first target.

8. **Appendix K attaches a significance qualification to the wrong comparison.**  
   P21, lines 1108–1110.

   Adapter 4/2 versus native 4/4 has ICL **+6.6 [3.0, 10.5]**, which is significant. The text incorrectly says it is not significant.

   The nonsignificant result is a different contrast: prefill 4 versus 1 at decode one, **+5.1 [−0.4, 10.5]**.

   **Exact fix:** write:  
   > “Against native 4/4, adapter 4/2 improves RAG by 6.1 [4.1, 8.4] and ICL by 6.6 [3.0, 10.5].”

   Move the nonsignificance qualification to the within-adapter prefill-4-versus-1 comparison. Print Table 15’s intervals rather than relegating them to JSON.

9. **Hybrid component attribution still exceeds the controlled contrasts.**  
   Appendix J and Table 13; pp20–21.

   The revised table title and disclosure of exterior states are improvements. Nevertheless, “either second-visit part recovers most of the gain” and “attention KV raises MMLU-Pro” still invite attribution to that component. Comparisons against one-visit prefill also change post-block states.

   **Exact fix:** describe these as performance differences between **mixed configurations**. Base component interpretation on comparisons holding exterior states fixed, and print their paired intervals. For example, GDN-mixed versus both yields:

   - HELMET **+5.6 [3.8, 7.5]**.
   - RULER-16K **+3.7 [1.6, 5.9]**.
   - MMLU-Pro **−3.5 [−5.8, −1.3]**.

   Add the corresponding attention-mixed comparisons. If claiming that either component alone recovers the gain, include a matched-exterior condition with both internal states from visit one.

Every figure has the following disposition. Font sizes below are measured in the compiled PDF; plotting source: :codex-file-citation{path="/Users/mac2/untied/untying-prefill-decode/ouro/scripts/paper_figs.py" purpose="source"}.

| Figure | Objection and exact fix |
|---|---|
| **1, p2** | The low-decode advantage is visible, but its attribution is wrong under objection 6. Labels are approximately **7.37 pt**; enlarge to at least 8 pt after scaling. Current PDF best-\(m\) labels **do match JSON**; there is no verified stale-label error. |
| **2, p2** | The single prompt cache is clear. However, attention outputs entering “joint softmax” depict the wrong operation. Label branches \((o_P,\lambda_P)\), \((o_A,\lambda_A)\), and the node **“LSE-weighted merge”**, with \(g=\sigma(\lambda_P-\lambda_A)\). Mark decoder LoRA. Change the header to **“Decode: \(k=1,\ldots,n\) per token.”** |
| **3, p4** | Teacher and sampling range are clear. Fix objection 7’s denominator. Code is approximately **7.47 pt**, comments are pale, and the two listings are badly unbalanced. Enlarge and darken; put training below inference or rebalance the panels. |
| **4, p6** | The bold **point-estimate** claim is correct: six of eight are within two points. Independent y-scales hide that criterion. Use differences from native 4/4, paired CIs, and a ±2-point band; include adapter 2/2. Enlarge **6.93-pt** labels. |
| **5, p7** | The bold statement is supported on the four shown settings. The microbenchmark qualification is appropriate. Use red squares consistently; enlarge and separate crowded **6.93-pt** depth labels. Add a precise pointer to paired uncertainty. |
| **6, p7** | The speed contrast against native 4/4 is clear; the previous label collision is resolved. Fix the checkpoint and explanatory caption under objection 1. Add timing and accuracy uncertainty. |
| **7, p8** | The best-prefill envelope hides every harmful \(m<n\) condition, so it cannot demonstrate the section’s “both directions” result. Replace it with model-wise difference heatmaps, marked diagonals, zero-centered colors, and significance marks. Keep the useful seed and selection disclosures. |
| **8, p14** | Six panels, not five; HellaSwag missing. Apply objection 3, Figure 1’s attribution fix, and its typography fix. The explicit in-sample-selection caveat is appropriate. |

Every table is accounted for below. Generated entries agree with the JSON; that does not remove the missing-data and provenance objections.

| Table | Objection and exact fix |
|---|---|
| **1, p6** | Explain the missing HellaSwag decode-three cell. Selected-depth tuples are approximately **5.98 pt**, CIs **6.97 pt**, and visually run together. Separate depths from intervals and enlarge them. Show the within-adapter control alongside the native comparison. |
| **2, p13** | The four stale GSM8K values are fixed: **+8.5/+8.9 and −0.1/−1.1**. Add paired CIs and generate the table from records. The narrowed “selected comparisons” heading is appropriate. |
| **3, p14** | “Every cell” is undermined by missing HellaSwag. Split the approximately **6.97-pt** table into readable panels. Define **“HE+: HumanEval+”**, matching the actual header; remove the unused “HEval” definition. Expand MATH/BBH shot settings. |
| **4, p15** | The bold claim is now correctly restricted to decode one. Explain missing MMLU decode two and HellaSwag decode three. Remove equivalence language elsewhere; complete the missing evaluations. |
| **5, p17** | Same tiny selected-depth tuples and intervals as Table 1. Separate the notation and enlarge. Subset sizes are disclosed appropriately. |
| **6, p17** | Selected-example and counterexample disclosures are good. Enlarge intervals; keep each benchmark’s sample size easy to locate. |
| **7, p18** | Scores and paired differences agree; smaller-Thinking prose now describes code gains correctly. Add per-row sample sizes and enlarge CIs. |
| **8, p18** | **MATH uses 400 items**, versus Table 7’s 500. Thus native **84.2 versus 84.8** and adapter **83.8 versus 85.0** are legitimate cohort differences, but unexplained. Print common-item counts; preferably complete the remaining 100 distilled items. |
| **9, p18** | This is the earlier **64-item** GSM8K cohort, not Figure 6’s 192. Replace it or label it a pilot. Rename “time” to **mean seconds/answer** and add timing diagnostics and intervals. |
| **10, p18** | Adapter rows still say **decode \(d\)** under \(n\) headers. Replace both `\rw{4}{d}` with `\rw{4}{n}`. Print winning tested batch sizes: adapter **64/64/64/64**, native **80/40/24/20**. Say **largest successful tested batch**, and disclose row-wise prefill and repeated-token replay. |
| **11, p19** | Separate variant groups with a rule. Add the significant MMLU 1/1 and 2/2 frozen-map losses referenced in §3 but omitted here. Enlarge CIs. |
| **12, p20** | Seed 0 is now explicit. Separate model groups and mark diagonal cells. Explain structural dashes for Lev-3B. This table does support both directions. |
| **13, p21** | Revised title and exterior-state disclosure are better. Add direct mixed-versus-both contrasts with CIs under objection 9. |
| **14, p22** | Negative results are retained appropriately. Move the long MBPP formatting explanation into setup text, release its records, and add per-row sample sizes. |
| **15, p22** | Print CIs, especially with only **41 reranking and 34 recall items**. Change “ICL helps … on Ouro” to **“ICL point estimates rise on both, significantly on 1.4B.”** Correct objection 8’s misplaced qualification. |
| **16, p22** | Numerical entries and explicit full-answer-scoring disclosure are appropriate. Keep the interpretation as interaction with prompt setup, since removing exemplars also changes length and format. Enlarge CIs. |

The remaining layout and wording defects are concrete:

- **P12:** Appendix C’s heading is stranded at the bottom; its content begins on p13.
- **P13:** Appendix D is an orphan heading, followed by almost an entire blank page.
- **P14:** excessive space separates Figure 8 and Table 3.
- **P15:** one modest table occupies an otherwise nearly empty page.
- **P16:** Appendix E’s short text is separated from its tables by the page break and substantial whitespace.
- **P22:** the short reproducibility paragraph splits bottom-left to top-right, leaving most of the right column empty.

**Exact layout fix:** retain `\raggedbottom`, restructure the top-only double-column floats and repeated `\FloatBarrier`s, keep headings with substantive content, and place reproducibility together after its tables. Re-render all pages. I found **no unresolved `??` references, gross margin overflow, or obvious clipped plot content**.

Appendix A also misdescribes the overlap audit as testing “question and choices.” The script takes the first available string field and does not append choices; missing text can also become unassessed. Change this to **“extracted question text shorter than 13 tokens, or unavailable”**, or actually audit the documented combined text and recompute.

Keep **nominal, per-benchmark** beside headline significance counts. Add a formal citation and pinned revision for SenseNova; its official model card supports the stated 32-layer/56-execution architecture. [SenseNova model card](https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview)

Relative to the previous review, the disposition is:

:codex-file-citation{path="/Users/mac2/untied/untying-prefill-decode/paper/codex_review_untie7.md" purpose="source"}

| Previous substantive point | Current status |
|---|---|
| **1. Broken contamination exclusion** | **Core resolved.** Current IDs really exclude **94 HumanEval and 93 MBPP** items; **13/15 and 8/15** sensitivity counts reproduce. Residual audit-description issue noted above. |
| **2. Wrong timed distilled checkpoint** | **Unresolved.** |
| **3. Incomplete comparisons/equivalence claim** | **Partial.** GSM8K added and 14-comparison counts corrected; MMLU missing and “about as good” retained. |
| **4. False four-seed significance claim** | **Resolved.** Revised text distinguishes significance from persistent point-estimate direction; all four seed grids support it. |
| **5. Incomplete task map** | **Unresolved.** |
| **6. Smaller-Thinking code gains misstated** | **Resolved.** Six within-adapter gains and three significant gains against native 2/2 are now described correctly. |
| **7. Overbroad recommendation/Figure 1 attribution** | **Partial.** Low-decode scope improved; Figure 1 still does not isolate its advertised cause. |
| **8. Hybrid attribution** | **Partial.** Title and conditional setup improved; controlled contrast presentation remains insufficient. |
| **9. Protocol/format overclaiming** | **Partial.** Substitutions and extraction sensitivity are better disclosed, but inconsistent wording and unexplained baseline gaps remain. |
| **10. Incomplete provenance/validation** | **Unresolved.** Some omissions are now admitted. |
| **11. Training normalization/token accounting** | **Partial.** Budget caveat added; objective and actual supervised-target totals remain incorrect or unspecified. |
| **12. Stale Table 2 numbers** | **Resolved numerically.** |

Other improvements deserve credit: Table 4’s false first-target statement is removed; Table 12 identifies seed 0; some appendix stretching is fixed; and the shared-answer-cache description now acknowledges its training/inference approximation. Ouro’s cache-sharing experiment and FlashLoop are now acknowledged in related work. [Ouro §5.4.2](https://arxiv.org/html/2510.25741v4#S5.SS4.SSS2), [FlashLoop](https://arxiv.org/abs/2609.29812)

**Score: 2/5.**

REMAINING SUBSTANTIVE OBJECTIONS: 9