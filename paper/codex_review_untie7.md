**Score: 2/5 — reject in its current form.**

I reviewed all 21 pages. Independent recomputation reproduces **the entire committed JSON exactly**, and all **10 generated table files regenerate byte for byte**. The remaining problems include incorrect prose, a broken overlap exclusion, incomplete comparisons, mismatched checkpoint provenance, and misleading presentation.

The recomputation nevertheless reports **five missing inputs and exits successfully**. Agreement with this JSON does not establish completeness.

Audit sources: :codex-file-citation{path="/Users/mac2/untied/untying-prefill-decode/paper/latex/main.pdf" purpose="source"} :codex-file-citation{path="/Users/mac2/untied/untying-prefill-decode/paper/latex/main.tex" purpose="source"} :codex-file-citation{path="/Users/mac2/untied/untying-prefill-decode/ouro/results/paper_numbers.json" purpose="source"} :codex-file-citation{path="/Users/mac2/untied/untying-prefill-decode/ouro/scripts/paper_numbers.py" purpose="source"}

I count **12 substantive objections**, ranked below. The subsequent figure, table, and layout checks specify additional presentation fixes without counting the same objection twice.

1. **The contamination sensitivity analysis excludes none of the 126 flagged code prompts.**  
   Appendix A, p11, lines 724–735.

   The overlap report identifies **94 `humaneval_local/*`** and **32 `mbpp_plus_instruct_local/*`** items. Current scores use **`humaneval_chat/*`** and **`mbpp_chat/*`**. The exclusion code compares these strings directly; their intersection is **zero**. Consequently, “Without those items … 14 of 15 and 8 of 15” does not describe the claimed exclusion.

   **Exact fix:** map both records to stable dataset task IDs, verify the mapping, assert the expected exclusion counts, and recompute selection and intervals. Publish audited/excluded/remaining counts per benchmark. HellaSwag and Winogrande remain unaudited. Questions shorter than 13 tokens or lacking extracted text must be classified as **unassessed**, rather than clean; add normalized exact matching for these cases.

2. **The timed distilled baseline is not the corrected accuracy baseline.**  
   Figure 6, §4.4, Table 8.

   Accuracy comparisons use the retrained **T26-Df** control, including the first answer target. Timing launchers still select **T26-D-native22**, the earlier control omitting that target. Therefore the plotted distilled point and **1.21× [1.06, 1.38]** ratio do not measure the corrected baseline in Table 8.

   The timing sample is also still consecutive GSM8K IDs **1–192**, with MATH IDs **1–40** and SenseNova GSM8K IDs **1–64**. Length-cap counts on the 192-item sample are **11/8/13/5/6** for native 2/2, native 4/4, old distilled 2/2, adapter 2/2, and adapter 4/2. MATH cap counts are **7/6/10 of 40** for native 2/2, native 4/4, and adapter 4/2.

   **Exact fix:** time the corrected checkpoint on common items and record its hash. Report sample selection, TTFT, generated tokens, cap rates, pooled decode time/token, and answer-time intervals. Use a fixed random/stratified sample for a representative benchmark claim; otherwise explicitly qualify the consecutive-ID sample.

3. **“Distillation matches it” asserts equivalence from incomplete, mostly inconclusive comparisons.**  
   Abstract, introduction, §4.2, Limitations; Table 4.

   At decode two, distilled MMLU and GSM8K results are missing. Among the **13 completed comparisons**, the adapter has **one significant win**, **two significant losses**, and **ten inconclusive differences**. Those findings do not establish equivalence.

   **Exact fix:** complete MMLU and GSM8K, explain every dash, and replace “distillation matches it” with:

   > “Across 13 completed comparisons at two decode loops, the adapter has one significant gain, two significant losses, and ten inconclusive differences.”

   An equivalence claim requires a declared margin and an appropriate analysis. Restrict Table 4’s bold caption to **one decode loop**, where its stated story is supported.

4. **The abstract’s four-seed significance claim is false.**  
   Abstract, p1, lines 28–34; introduction and §4.5.

   Actual significant-count summaries are:

   | Training seed | Significant gains, \(m>n\) | Significant losses, \(m<n\) |
   |---|---:|---:|
   | 0 | 12/13 | 11/13 |
   | 1 | 12/13 | 10/13 |
   | 2 | 11/13 | 12/13 |
   | 3 | 10/13 | 12/13 |

   All point estimates retain the claimed direction, but **12 gains and 11 losses do not hold in every seed**.

   **Exact fix:** replace the abstract ending with:

   > “At seed 0, gains are significant on 12 of 13 higher-prefill pairs and losses on 11 of 13 lower-prefill pairs. All point estimates retain their direction across four seeds; ten lower-prefill pairs are significantly worse in every seed.”

   Put **seed 0** directly in Table 12’s caption.

5. **The claimed complete 15-setting task map is incomplete.**  
   Figures 1/8, Table 3, evaluation-coverage statements.

   The missing configuration is **Ouro-2.6B HellaSwag S43**. Consequently:

   - Figure 8 has **six panels**, while its caption says **five**.
   - Figures 1 and 8 together cover **14 settings**.
   - Table 3 has **14 benchmark columns**, omitting HellaSwag.
   - Table 1’s decode-three column has **14 completed settings**, not 15.

   **Exact fix:** complete S43 and regenerate everything. Then Figure 8 should say **“The seven task-map settings not shown in Figure 1.”** Until completion, explicitly identify 14 complete maps and the missing cell. Make generation fail when an expected benchmark/configuration is absent.

6. **The smaller Thinking model’s significant code improvements are incorrectly called undetectable.**  
   §4.3, p6; Appendix F, p16; Table 7.

   For Ouro-1.4B-Thinking, adapter 4/2 minus adapter 2/2 is:

   - **HumanEval+: +7.9 [1.8, 14.0]**
   - **MBPP+: +6.3 [2.4, 10.3]**

   Both are significant. “The same holds” and Appendix F’s “no detectable difference … HumanEval and MBPP+” are wrong.

   Against native 2/2, HumanEval+ is also significantly higher: **+10.4 [4.3, 17.1]**. Appendix F incorrectly lists only GSM8K and ARC-Challenge as significant gains.

   **Exact fix:** state **six significant within-adapter gains**, with only GSM8K and MATH inconclusive. Against native 2/2, state **three significant gains, one significant loss, and four inconclusive differences**.

7. **The general recommendation exceeds the demonstrated generative result, and Figure 1 does not isolate its advertised cause.**  
   Abstract, introduction, conclusion; Figure 1.

   “Prefill with more loops than you decode” is broader than the evidence. At larger decode depths, optimal prefill can equal or fall below decode depth—for example, MATH’s best prefill is two at decode three and four. The strongest consistent result is at **one decode loop**.

   Figure 1’s red–grey gap additionally includes adaptation and cache changes. It is not a displayed within-adapter prefill effect.

   **Exact fix:** scope the recommendation to low decode depth:

   > “At one decode loop, deeper prefill improves the evaluated Ouro base-model settings; gains shrink at higher decode depths.”

   Add the same adapter’s diagonal to Figure 1, or change its bold caption to describe the complete trained-system comparison. Retain the within-adapter comparisons as the evidence isolating prefill.

8. **Hybrid component attribution still exceeds the controlled contrasts.**  
   Appendix J; Table 13.

   Exterior-state changes are now disclosed, and direct mixed-condition comparisons have been added. However, “Which part … benefits,” “attention KV raises,” and “state alone” still imply isolated component effects. Table 13’s comparisons against one-visit prefill also change states outside the repeated block.

   **Exact fix:** rename the table **“Performance of mixed prompt-state configurations after two-visit prefill.”** Describe A-mixed and G-mixed conditions, removing “alone.” Print paired intervals for A-mixed versus both and G-mixed versus both, which hold exterior states fixed. Base component claims on those contrasts.

9. **The protocol and answer-format claims remain too categorical.**  
   Abstract, introduction, conclusion; Appendix B.

   “The models’ own evaluation protocol” is not literally accurate: MATH uses four-shot Minerva instead of Ouro’s five-shot setup, and Thinking evaluation changes sampling and scoring. Ouro’s original settings confirm these differences. [Ouro, Appendix C.1](https://arxiv.org/html/2510.25741v4#A3.SS1)

   Likewise, “Two results … were answer format, not the prefill loops” asserts a causal resolution that parser comparisons alone cannot establish. Prefill can itself change answer formatting.

   **Exact fix:** say **“Ouro base-model settings where available, with documented substitutions”**, and describe Thinking separately. Replace the format diagnosis with:

   > “These earlier effects reversed under the documented final-answer scoring; the diagnostics indicate extraction sensitivity.”

   Identify the decode depth for **68.4/51.8 versus 18.6/46.8**. Semantic correctness claims still require adjudicating parser-discordant answers.

10. **The numerical provenance remains incomplete, and validation can succeed despite missing data.**  
    Appendices A/B/G/L; Tables 9/10/14.

    Still outside the promised complete recomputation path are long-answer throughput **586/219/570/253**, shared-cache batch/throughput measurements **32→192, 219→476, 759 versus 586**, alternative-cache throughput **500/245/105 and 267/130/57**, format diagnostics, MBPP counts **294/236/372/375**, and the **512-token** equivalence record. The pooled **1.27×** statistic is recoverable but not serialized.

    The current computation emits missing-input warnings and exits zero. Hard-coded expectations also omit Table 2’s stale numbers, so its clean mismatch report does not validate them.

    **Exact fix:** serialize every empirical measurement with its source/configuration; attach training manifests for parameter counts, token totals, and duration. Make missing required records, unexpected comparison counts, and mismatches fail validation. Generate hand-written numerical tables from the same records. Update stale coverage and count claims in `REPRODUCE.md`.

11. **The printed training normalization and token budget do not describe the implementation.**  
    Figure 3, §3, Appendix A.

    Corrected controls and the adapter supervise all suffix targets, but normalization, scheduling, and budget accounting still use **`len(suffix) − 1`**. Actual supervised positions therefore exceed the nominal budget by one per consumed example.

    **Exact fix:** report nominal scheduling tokens and actual supervised targets separately for existing checkpoints, and describe the denominator actually used. Future normalization and budgeting should use the supervised mask consistently.

    This is now a bookkeeping defect, **not an unequal first-target accuracy control**. Remove Table 4’s false statement that its control “omits the first answer token”: the current numerical pipeline uses corrected **E48f/D48f/H48f** controls.

12. **Table 2 contains four obsolete GSM8K numbers.**  
    p12, two-seed checks.

    | Comparison | Printed | Correct from JSON |
    |---|---:|---:|
    | Prefill 4−1, decode 1, seed 1 | +8.0 | **+8.5** |
    | Prefill 4−1, decode 1, seed 2 | +7.2 | **+8.9** |
    | Prefill 4−2, decode 2, seed 1 | −1.7 | **−0.1** |
    | Prefill 4−2, decode 2, seed 2 | −2.9 | **−1.1** |

    **Exact fix:** generate the table from the current strict-match comparisons, add paired intervals, and title it **“Two-seed checks on selected comparisons.”**

Every figure needs the following disposition. Typography measurements account for scaling in the compiled PDF.

| Figure | Objection and exact fix |
|---|---|
| **1, p2** | Address objection 7’s attribution problem. The curves communicate a low-depth advantage, but not its within-adapter magnitude. Increase approximately **7.37-point** labels to at least 8 points after scaling. |
| **2, p2** | “Joint softmax” after attention outputs suggests the wrong operation. Label branches \((o_P,\lambda_P)\), \((o_A,\lambda_A)\); use **“LSE-weighted merge”** and show the weighting equation. Mark decoder LoRA. Change the header to **“Decode: \(k=1,\ldots,n\) per token.”** |
| **3, p4** | Approximately **7.5-point** code and pale comments are unnecessarily difficult to read. Balance the two snippets, enlarge code, and darken comments. The teacher and sampling range are now explicit and correct. |
| **4, p5** | The six-of-eight point-estimate claim is correct, but independent y-scales hide the common two-point criterion. Show differences from native 4/4 with paired CIs and a **±2-point band**, or annotate every difference. Include adapter 2/2. Current labels shrink to approximately **6.93 points**. |
| **5, p6** | The microbenchmark axis and caption are now appropriately qualified. Increase approximately **6.93-point** depth labels; use red squares consistently. Add uncertainty or a precise pointer to the corresponding paired comparisons. |
| **6, p7** | “About as fast” conceals the reported significant **21%** disadvantage against the old distilled baseline. Extra prefill also does not improve the plotted subset: adapter 2/2 is **95.8%**, adapter 4/2 **94.3%**. State that distinction and point to other-task gains separately. Retiming under objection 2 takes priority; add timing intervals. The previous label collision is resolved. |
| **7, p8** | The best-prefill envelope hides every harmful \(m<n\) condition, so it cannot show “both directions.” Replace it with model-wise difference heatmaps, marked diagonals and significance. Keep the now-correct development-selection, separate-training, and seed disclosures. |
| **8, p13** | Correct the missing-setting/count problem in objection 5. Its compact layout and explicit selection caveat are improved. Apply Figure 1’s attribution and typography fixes. |

The plot-generation source supporting these checks is :codex-file-citation{path="/Users/mac2/untied/untying-prefill-decode/ouro/scripts/paper_figs.py" purpose="source"}.

Every table is accounted for below. Agreement with the JSON is qualified by the substantive objections above.

| Table | Remaining objection and exact fix |
|---|---|
| **1, p5** | Selected-\(m\) tuples reach approximately **5.98 points** and merge visually with CIs. Separate estimate, interval, and selected depths. Explain the missing HellaSwag decode-three cell. Expose the same-adapter comparison beside the native comparison. |
| **2, p12** | Correct objection 12’s four numbers; add CIs and narrow the robustness scope. |
| **3, p13** | Missing HellaSwag; approximately **6.97-point** body. Split benchmark columns into two readable panels. Define **HE+: HumanEval+**, rather than the unused “HEval”; expand MATH and BBH setup abbreviations. Pre-rounding maximum selection is repaired. |
| **4, p14** | Explain missing comparisons, restrict the bold claim to decode one, and remove the now-false first-target omission statement. |
| **5, p16** | Numerical entries agree. Apply Table 1’s typography and notation separation. |
| **6, p16** | Numerical entries agree. Enlarge CI text; retain the disclosure that these are selected benchmark examples, with counterexamples elsewhere. |
| **7, p17** | Entries agree, but prose contradicts its significant code gains: fix objection 6. Add sample sizes and enlarge CIs. |
| **8, p17** | **MATH uses 400 items**, versus Table 7’s 500: hence native 2/2 **84.2 versus 84.8**, and adapter 4/2 **83.8 versus 85.0**. State per-row sample sizes and common-item intersections so this does not look like inconsistent arithmetic. |
| **9, p17** | Values are correct for the old **64-item** cohort, but Figure 6 and §4.4 use **192**. Replace it with the main cohort, including the corrected distilled checkpoint once timed, or explicitly label it a pilot. Rename “time” to **“mean seconds per answer”** and add timing diagnostics. |
| **10, p17** | Replace row-label \(d\) with \(n\). Print tested winning batch sizes and say **“largest successful tested batch”**. Supply long-answer measurement provenance. |
| **11, p18** | Separate variant groups with a rule. Add the significant MMLU 1/1 and 2/2 losses discussed in §3 but omitted here. |
| **12, p19** | State **seed 0**, separate model groups, and mark diagonals. It supports both directions; the abstract’s seed generalization does not. |
| **13, p20** | Apply objection 8’s conditional interpretation and direct-comparison intervals. |
| **14, p21** | Entries agree. Retain negative results. Move the lengthy MBPP format diagnostics into setup text and release their underlying records. |
| **15, p21** | Print CIs instead of requiring readers to inspect JSON, especially with **41 re-ranking and 34 recall items**. ICL is **+5.1 [−0.4, 10.5]** on 2.6B and **+9.3 [3.6, 15.0]** on 1.4B. Write **“ICL point estimates rise on both, significantly on 1.4B.”** |
| **16, p21** | Arithmetic and explicit full-response scoring disclosure are appropriate. Retain the prompt-setup qualification; address objection 9’s causal wording in the surrounding text. |

For reference, the current JSON’s 192-item timing cohort is **92.7%/63.3 s**, **92.7%/118.5 s**, **93.8%/57.7 s**, **95.8%/68.3 s**, and **94.3%/69.6 s** for native 2/2, native 4/4, old distilled 2/2, adapter 2/2, and adapter 4/2, respectively. These must not silently replace the corrected-baseline experiment.

The appendix layout is visibly broken:

- **p12:** Appendix D is an orphan heading.
- **p14:** one modest table occupies an otherwise nearly empty page.
- **p15:** Appendix E’s heading and paragraphs are separated by enormous blank regions.
- **p16:** Appendix F’s heading is excessively separated from its text.
- **p18:** Appendix H repeats the severe vertical stretching.
- **p21:** the short reproducibility paragraph splits between bottom-left and top-right, leaving substantial empty space.

**Exact fix:** use `\raggedbottom` in the appendix, restructure the top-only double-column floats and repeated barriers, keep headings with substantive content, and place the short reproducibility paragraph together after its tables. Re-render every page. I found **no unresolved cross-reference targets, gross margin overflow, or clipped figure content**.

Two further wording fixes: keep **nominal** significance beside the headline counts, rather than implying independent benchmark replications; and distinguish the single-cache contribution from Ouro’s own last-step cache-sharing experiment. [Ouro §5.4.2](https://arxiv.org/html/2510.25741v4#S5.SS4.SSS2) The related-work discussion should also acknowledge the directly relevant training-free efficiency approach in [FlashLoop](https://arxiv.org/abs/2609.29812), without implying that its measurements are directly comparable.

Relative to the eight substantive objections in the previous review, the disposition is:

:codex-file-citation{path="/Users/mac2/untied/untying-prefill-decode/paper/codex_review_untie6.md" purpose="source"}

| Previous objection | Current disposition |
|---|---|
| **1. Inconsistent task-map item sets** | **Core resolved.** Means now use common items; corrected optima are present. HellaSwag completeness is a separate remaining defect. |
| **2. First-answer supervision mismatch** | **Resolved for accuracy controls.** Corrected base and Thinking controls are used. The omission caption, token accounting, and old timed checkpoint remain wrong. |
| **3. Flagship MATH semantic validation** | **Resolved as a headline problem:** the claim was removed and primary scoring changed. Semantic adjudication was not performed; categorical format attribution remains unsupported. |
| **4. Timing baseline/sample** | **Partly resolved:** a distilled point and 192-item sample were added. Checkpoint identity, representative sampling, and truncation reporting remain unresolved. |
| **5. Rounded-bound significance errors** | **The identified errors are resolved:** stars, the 11-loss count, and small significant MMLU losses are repaired. Preserve explicit significance flags/full precision going forward. |
| **6. Complete numerical provenance** | **Unresolved.** Disclosing omissions does not provide the missing records or failing validation. |
| **7. Contamination audit** | **Partly addressed but substantively broken:** an audit exists; the code-item exclusion mapping is invalid. |
| **8. Hybrid causal confound** | **Substantially improved, partly unresolved:** direct contrasts and retraining were added, but titles and component wording still overattribute. |

Other resolved requests include Figure 6’s annotation collision, Figure 5’s microbenchmark labeling, Figure 7’s development-selection and dataset-subset disclosures, Figure 8’s repeated selection caveat, Table 3’s pre-rounding maximum choice, ablation \(m/n\) notation, and MELT’s local-cache description. The main appendix pagination has deteriorated.

**Score: 2/5.**

REMAINING SUBSTANTIVE OBJECTIONS: 12