**Score: 2/5 — reject in its current form.** The revision addresses several previous objections, but this audit found new errors in the task-map comparisons, training control, and significance calculations.

I reviewed all 18 rendered pages. Recomputing the statistics entirely in memory reproduces **every field of the committed JSON exactly**; all nine generated table files also regenerate **byte-for-byte**. The errors below are therefore reproducible errors in the analysis, rather than stale LaTeX.

Sources: :codex-file-citation{path="paper/latex/main.pdf" purpose="source"} :codex-file-citation{path="paper/latex/main.tex" purpose="source"} :codex-file-citation{path="ouro/results/paper_numbers.json" purpose="source"} :codex-file-citation{path="ouro/scripts/paper_numbers.py" purpose="source"}

I count **eight substantive objections**, ranked below. Presentation and wording corrections follow separately.

1. **The task map compares configurations evaluated on different items—and this changes a reported optimum.**  
   Figures 1/8, Table 3, §4.1.

   The means are calculated independently over each file rather than over common item IDs. Four Ouro-2.6B settings are affected:

   | Setting | Inconsistent evaluation sizes |
   |---|---|
   | MMLU | 1,140 versus 14,042 |
   | GSM8K | 400 versus 1,319 |
   | BBH | 1,620 versus 6,511 |
   | MMLU-Pro | 420, 980, or 1,400 |

   On the common **420 MMLU-Pro items**, the decode-one optimum changes from **prefill 3 to prefill 4**: prefill 3 scores 46.2, while prefill 4 scores **47.4**, not the printed 45.8. Other consequential corrections include adapter 4/2 **53.4→51.2**, native 1/1 **24.4→26.9**, and native 4/4 **56.1→55.2**.

   Table 3’s native MMLU scores at decode 1/2/4 should become **51.9/68.6/75.4**, replacing **51.5/67.7/74.6**. Corresponding GSM8K scores should become **41.8/78.5/87.0**, replacing **41.5/77.9/86.4**. Figures 1/8 already use matched-subset native scores for these two settings, so Table 3 currently contradicts its claimed correspondence with those figures. BBH 4/1 changes **51.5→51.7**, and native 4/4 changes **80.7→80.4**.

   **Exact fix:** intersect IDs across all 20 displayed configurations within each benchmark before calculating means, maxima, labels, or boldface. Store the common sample size and an ID-set identifier in the JSON. Generate figures and Table 3 from that same structure. **Table 1 already intersects configurations within each decode depth; its 11/13 and 9/13 counts survive this correction.**

2. **The released distillation-control code supervises different answer positions from the loop adapter.**  
   §4.3, Table 7, Appendix A.

   `loopmap_forward` includes the first answer-token prediction. `native_lora_forward`, used for the native LoRA control, discards it and supervises `smask[:,1:]`. Meanwhile, the shared trainer budgets and normalizes using `len(suffix)-1`. Consequently, the loop adapter receives **one additional supervised target per example**, and its stated normalization by supervised answer tokens is inaccurate.

   This makes the advertised matched training control non-equivalent. It does **not** establish that this discrepancy explains the performance gap.

   **Exact fix:** verify the historical training code against checkpoint provenance; include first-answer KL in the native control, normalize and budget using actual supervised masks, and rerun the affected control if these were the training paths used. Report actual supervised-token and trainable-parameter counts for both arms. Until then, replace “Untying is more than distillation” with **“The complete loop-adapter system outperforms the evaluated native-2/2 LoRA baseline.”**

   Evidence: :codex-file-citation{path="ouro/batched.py" purpose="source"} — lines 79–106 versus 179–189; :codex-file-citation{path="ouro/train_batched.py" purpose="source"} — lines 163–184.

3. **The flagship MATH result still lacks semantic validation against extraction effects.**  
   Abstract, §4.1, Appendix B.

   **51.8→68.4** and **+16.6 [12.4,21.0]** reproduce correctly. However, Appendix B acknowledges that strict extraction reverses the analogous decode-one comparison. A full-response `math_verify` score does not by itself establish that the extracted expression is the intended final answer.

   **Exact fix:** blindly adjudicate parser-discordant responses for **S14 versus S44**, preferably also S11 versus S41. Report intended-final-answer correctness and format compliance separately, with paired intervals. Release the discordant responses and adjudications. Pending this check, call the headline metric **“full-response math-verify score.”** This remains a validation gap, not a demonstrated parser bug.

4. **The speed evidence still lacks the strongest trained baseline and a representative timing sample.**  
   Abstract, §4.4, Figure 6, Table 8.

   The added adapter 2/2 control correctly shows that deeper prefill does not explain the saving. The distilled native-2/2 checkpoint is still untimed.

   The timing samples are consecutive IDs **1–64 for GSM8K** and **1–40 for MATH**, after warm-up. Truncation is material: native 1/1 hits the GSM8K cap on **31/64** items; adapter 4/2 hits the MATH cap on **10/40**, versus **7/40** for native 2/2 and **6/40** for native 4/4.

   **Exact fix:** time the corrected distilled baseline under the same implementation; use a larger fixed random or stratified sample; report mean TTFT, generated tokens, cap rates, decode time/token, and total answer time with intervals. Change the abstract to **“takes 0.55× the time of native 4/4 on 64 timed GSM8K problems.”** Preserve the existing distinction between timing-subset accuracy and full-set accuracy.

5. **Rounding confidence bounds before testing significance produces incorrect stars and counts.**  
   Abstract, §3, §4.5, Tables 1/11.

   These are actual classification errors:

   | Comparison | Unrounded interval | Required correction |
   |---|---|---|
   | Lev-1.4B, prefill 3 / decision 4 | **[−2.35517, −0.03046]** | Add Table 11 star; significant losses for \(m<n\) become **11/13**, not 10/13 |
   | RULER-4K, Table 1, decode 3 | **[−6.05481, −0.00954]** | Add star; its upper bound excludes zero |
   | No-map MMLU, 1/1 | **[−0.72639, −0.03561]** | Small but significant loss |
   | No-map MMLU, 2/2 | **[−0.52699, −0.00712]** | Small but significant loss |

   The last two contradict the broad statement that tested MMLU differences are not significant. The displayed 4/1 comparison remains nonsignificant.

   **Exact fix:** retain full precision and explicit significance flags in the JSON; round only when formatting. Display extra decimals when rounding would turn a nonzero endpoint into zero. Update the abstract and §4.5 to **11 significant losses**. Restrict the no-map statement to the explicitly listed conditions or report the additional small significant losses.

   The **11/13 and 9/13 selected-prefill wins**, **6/13 and 3/13 within-adapter wins**, and **12/13 positive-direction Lev significance count** remain correct.

6. **The numerical record still does not cover every printed empirical measurement.**  
   Appendices A/B/G/L; Tables 8/9/13.

   Appendix L now discloses omissions, but disclosure does not satisfy the requested complete audit. Missing from the JSON/recomputation path are:

   - Long-answer throughput: **586, 219, 570, 253**.
   - Shared-answer-cache measurements: **32→192**, **219→476**, **759 versus 586**.
   - Dynamic/contiguous-cache throughput: **500/245/105**, **267/130/57**.
   - Parser diagnostics: **−9.8**, **19.6 versus 9.8**.
   - MBPP+ format counts: **294/236/372/375 of 378**.
   - The **512-token** cache-equivalence check.
   - The pooled **1.27×** per-token timing statistic, which is recoverable from supplied records but is not serialized by the script.

   **Exact fix:** release the missing records and add named aggregations and validation for each measurement. Source training quantities—48M/20M tokens, parameter counts, training duration—to manifests or logs. Make missing records and mismatches terminate validation unsuccessfully; the current script can print problems and still finish successfully.

7. **The adaptation mixture remains unaudited for benchmark overlap.**  
   Appendix A.

   The disclosure is clear, but gains against unadapted baselines remain potentially affected by exposure during adaptation. There is no evidence here establishing that contamination occurred.

   **Exact fix:** audit the actual sampled mixture for normalized exact and approximate evaluation-prompt matches, publish overlap counts and IDs, and recompute results on unmatched items. Retrain if the overlap materially compromises the conclusions.

8. **The hybrid section still makes component-level causal statements after admitting the intervention changes exterior states.**  
   Appendix J, Table 12.

   The new qualification correctly states that comparisons with one-visit prefill also change states downstream of the repeated block. The subsequent phrases “either part alone,” “Gated-DeltaNet state raises,” and “what improves is the recurrent state” still attribute composite differences to individual components.

   **Exact fix:** call these **mixed-state conditions**. Base component claims on direct G-versus-both or A-versus-both comparisons, which hold exterior states fixed, and print their paired intervals. Alternatively add the missing baseline with two-visit exterior states and first-visit states for both components inside the repeated block. The **+9.1 HELMET** comparison against one-visit prefill cannot be presented as an isolated GDN effect.

Every figure needs the following disposition. Numerical repairs in objection 1 take precedence over cosmetic changes.

| Figure | Objection and exact fix |
|---|---|
| **1, p2** | Task dependence is visible, but the advertised **51.8→68.4** within-adapter contrast is absent: the grey endpoint is native **54.0**. Add the adapter diagonal or a decode-4 MATH prefill-sweep inset. Correct the BBH item-set mismatch. Use “13 benchmark settings,” not “thirteen benchmarks.” |
| **2, p2** | “Joint softmax” after attention outputs suggests the wrong operation. Label branches \((o_P,\lambda_P)\) and \((o_A,\lambda_A)\); rename the box **“LSE-weighted merge”** and show \(g=\sigma(\lambda_P-\lambda_A)\). Mark decoder LoRA. Change the header to **“Decode: \(k=1,\ldots,n\) per token.”** |
| **3, p4** | The native teacher, sampling range, and retained-KV wording are fixed. Code remains small, comments pale, and the right column largely empty. Reflow the snippets and increase code size and comment contrast. |
| **4, p5** | The six-of-eight point-estimate statement is correct, but independently scaled axes conceal its common two-point criterion. Plot differences from native 4/4 with paired CIs and a shaded **±2-point band**, or annotate every difference. Include adapter 2/2 to show the prefill contribution. |
| **5, p6** | Rename the x-axis **“8K decode microbenchmark throughput”**: the plotted accuracy and throughput use different workloads. The Thinking panel lacks adapter 4/4, so “cutting … costs less accuracy” is ambiguous. Use **“At one or two decode loops, the adapter is more accurate than native n/n.”** Use red squares consistently; its numerical annotations remain 6.8-point in the source and need enlargement. |
| **6, p7** | **Definite collision:** adapter 2/2 and 4/2 labels overlap at approximately 63 seconds and 96.9%. Place callouts on opposite sides with leader lines: **“2/2: 62.6 s”**, **“4/2: 63.1 s.”** Add the distilled baseline and timing uncertainty. The narrowed bold caption is now appropriate. |
| **7, p8** | The envelope hides every \(m<n\) condition, so it cannot convey the section’s “both directions” result. Replace it with model-wise difference heatmaps from Table 11, marking diagonals and significance. State explicitly that each \(m\) uses separately trained weights and that the displayed maximum is selected on development items. |
| **8, p13** | Fix the MMLU-Pro optimum and inconsistent item sets. Repeat the evaluation-set-selection caveat directly; “as Figure 1” is insufficient. The single occupied fourth-row panel wastes substantial space. Replace the redundant envelope with a compact full-grid/difference display, or show only the five settings omitted from Figure 1 and change the caption accordingly. |

Across the figures, measure font size **after PDF scaling**. Raising source fonts to eight points does not guarantee eight-point final text. I found no gross clipping; Figure 6 has the definite annotation collision.

Every table is accounted for below. “Verified” means agreement with the numerical records, subject to the substantive defects above.

| Table | Objection and exact fix |
|---|---|
| **1, p5** | Selection-aware treatment is now appropriate. Fix the missing RULER-4K star. Separate selected-\(m\) tuples from tiny interval text. Put the **within-adapter comparator** beside the native comparison so readers can assess attribution directly. |
| **2, p12** | Two rounded estimates per selected comparison do not establish general training robustness. Add CIs and rename the section **“Two-seed checks on selected comparisons.”** State that this check does not cover the headline MATH optimum. |
| **3, p14** | Repair common-item evaluation first. Also remove bold from MMLU 4/1: prefill 3 is **803/1,140**, prefill 4 **802/1,140**; both round to 70.4 but are not tied maxima. Determine maxima before rounding. Enlarge the table or split benchmark columns into two panels. |
| **4, p14** | Selection-aware values and notation are verified. Apply Table 1’s separation of CIs and selected-\(m\) tuples. |
| **5, p14** | Numerical entries verified; the caption appropriately identifies selected examples and points to counterexamples. Enlarge CI text. |
| **6, p15** | Numerical entries verified; retain the negative smaller-model results and same-adapter control. Enlarge CIs and expose adapter 2/2 in Figure 4. |
| **7, p15** | Arithmetic verified. Its scientific interpretation depends on correcting objection 2. Preserve the distinction between **eight positive estimates** and **two significant gains**. |
| **8, p16** | Rename “time” to **“mean time per answer.”** Add timing intervals, token counts, cap rates, TTFT, and the distilled baseline. Serialize the pooled 1.27× statistic. |
| **9, p16** | Replace unexplained \(d\) with \(n\). Add winning tested batch sizes: adapter **64/64/64/64**, native **80/40/24/20** at decode 1/2/3/4. Say **“largest successful tested batch”**, since intermediate sizes were not exhaustively tested. Supply long-answer provenance. |
| **10, p16** | The five unrestricted-key rows are now present. Replace \(p/d\) with \(m/n\), separate variant groups visually, and address the omitted small significant no-map MMLU losses. |
| **11, p17** | Fix the missing star and significant-loss count. Add model-group rules, distinguish diagonal cells, and put **“seed 0”** in the caption. |
| **12, p17** | Arithmetic verified. Exterior-state disclosure is now present, but the bold title and interpretation need objection 8’s correction. Include direct mixed-condition comparisons and intervals. |
| **13, p17** | Scores and paired differences verified. Retain the negative results. Release the records behind the MBPP+ format counts. |
| **14, p18** | Print the CIs. The blanket ICL-benefit wording exceeds the evidence on 2.6B: **+5.1 [−0.4,10.5]**, versus **+9.3 [3.6,15.0]** on 1.4B. Write: **“ICL point estimates rise on both, significantly on 1.4B.”** The small recall/re-ranking samples make uncertainty especially relevant. |
| **15, p18** | Numerical entries verified. The revised “prompt setup” caption and acknowledgement that removing exemplars changes several factors are appropriate. No additional substantive objection. |

Additional wording, provenance, and layout corrections:

- **Lev dataset identity:** specify **“3,270 answerable questions from LevBench’s rights-subset-v1 development export”** and pin its revision. The count is correct for that hosted subset; it is not the full original benchmark. [Official dataset card](https://huggingface.co/datasets/songdj/LevBench).
- **Lev control terminology:** use **“matched-data/update control”** instead of “same-training control.” The body now properly discloses the compute difference and doubled-update check.
- **MELT description:** add **“plus a small per-loop sliding-window cache.”** Its shared cache is accompanied by these local caches. [MELT paper](https://arxiv.org/html/2605.07721v3).
- **Benchmark counting:** consistently use **13 settings from 11 benchmark families**; the three RULER lengths are not independent benchmark families.
- **Nominal significance:** the labeling is now explicit. Add multiplicity sensitivity if counts of significant settings remain the principal summary; avoid presenting those counts as independent replications.
- **Appendix D remains an empty heading** on p12, immediately followed by E. Place explanatory text and its floats before E, with a targeted float barrier.
- **Appendix G’s Tables 8–9 appear on p16**, several sections after G begins. Move declarations or add a barrier at G’s end.
- **Appendix L is fragmented across pp17–18** by queued tables, leaving much of the last page empty. Flush Tables 14–15 before starting L and keep its short text together.
- **REPRODUCE.md remains stale:** “1 MB,” blanket coverage language, and **12/13, 10/13** as Table 1’s headline counts need updating. Its figure-number corrections are now correct. :codex-file-citation{path="REPRODUCE.md" purpose="source"}
- **Cross-references:** I found no unresolved `\ref`/`\eqref` targets. Figure 7 now precedes References; that previous placement problem is resolved.

Relative to the previous review, the disposition is:

:codex-file-citation{path="paper/codex_review_untie5.md" purpose="source"}

| Previous substantive objection | Current disposition |
|---|---|
| **1. Selection uncertainty** | **Core resolved:** selection-aware 11/9 and within-adapter 6/3 are primary; 5,000 draws, fixed partition, and nominal significance are disclosed. The newly discovered rounding defect requires a separate correction. |
| **2. MATH semantic adjudication** | **Unresolved.** |
| **3. Timing baseline/sample** | **Partly resolved:** adapter 2/2 is plotted and attribution corrected; distilled timing, representative sampling, and truncation reporting remain. |
| **4. Training versus untying** | **Interpretation substantially resolved:** native comparisons are explicitly the complete trained method, and within-adapter comparisons are supplied. The new control-mask discrepancy is a separate problem. |
| **5. Numerical coverage** | **Partly resolved:** omissions are acknowledged and stale timing expectations corrected; complete provenance and validation remain missing. |
| **6. MELT/CDB related work** | **Resolved**, apart from the small MELT cache-description correction. |
| **7. Hybrid intervention confound** | **Partly resolved:** the confound is disclosed, but causal wording persists. |
| **8. Lev generalization** | **Largely resolved in prose:** seed-0/development scope is stated. Caption terminology, dataset version, and full-grid replication scope still need precision. |
| **9. Training contamination audit** | **Unresolved.** |
| **10. Map-ablation interpretation** | **Mostly resolved:** narrowed RULER interpretation and unrestricted-map CIs are present. The newly discovered small significant MMLU losses require narrowing the remaining universal statement. |

Other resolved requests include Figure 3’s explicit native teacher and sampling range, Figure 4’s “point estimates” wording, Figure 5’s missing Thinking-depth caveat, Figure 6’s narrowed caption and added control, Table 3’s abbreviation definitions, Table 15’s prompt-setup wording, checkpoint-specific evaluation coverage, pooled-timing aggregation wording, and the previously wrong appendix cross-references.

**Score: 2/5.**

REMAINING SUBSTANTIVE OBJECTIONS: 8