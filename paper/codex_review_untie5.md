**Score: 3/5 — borderline.** The paper has substantial evidence, but several central interpretations remain stronger than the controls justify.

I inspected all 17 rendered pages, the [source](paper/latex/main.tex), and the [previous review](paper/codex_review_untie4.md). Recomputing [the statistics script](ouro/scripts/paper_numbers.py) entirely in memory produces an **exact match with every field of the checked-in [JSON](ouro/results/paper_numbers.json)**. All nine generated table files also regenerate byte-for-byte.

That establishes consistency for the covered results. It does **not** establish the paper’s claim that every reported measurement is covered.

The following **ten substantive objections** are ranked by severity.

1. **The headline significance counts exclude uncertainty in choosing the prefill count.**  
   Abstract, §4.2, Table 1, Conclusion. The paper acknowledges selection-aware intervals, yet headlines the conditional **12/13 and 10/13**. More seriously, the newly added within-adapter control headlines **8/13 and 5/13**, also conditional on the selected depths.

   I reran the existing selection-aware algorithm with **5,000 resamples, seed 0**:

   | Comparison | Decode 1 | Decode 2 |
   |---|---:|---:|
   | Selected prefill versus native | **11/13** significant wins | **9/13** |
   | Selected prefill versus same adapter’s diagonal | **6/13** | **3/13** |

   There are no significant losses. These are fresh audit calculations; the within-adapter selection-aware results are not currently in the JSON.

   **Exact fix:** make these intervals primary, store them in the JSON, and update Table 1, abstract, introduction, §4.2 and conclusion accordingly. Retain the old counts explicitly as conditional results. Document that the partition remains fixed; this is not repeated-split validation. Label significance as nominal and report sensitivity to multiplicity.

2. **The flagship MATH improvement still lacks validation against answer-format effects.**  
   Abstract, §4.1, Appendix B. The arithmetic **51.8→68.4**, or **+16.6 [12.4,21.0]**, is correct. However, Appendix B admits that strict Minerva extraction reverses the analogous decode-one comparison. Scoring the full response with `math_verify` may be appropriate, but it does not establish that the extracted expression is the model’s intended final answer.

   **Exact fix:** blindly adjudicate parser-discordant responses for **S14 versus S44**, and preferably S11 versus S41. Report semantic final-answer correctness separately from format compliance, with paired intervals; release the discordant responses and adjudication. Pending that check, call the headline metric **“full-response math-verify score.”** This is a validation gap, not a demonstrated scoring bug.

3. **The speed claim still lacks the strongest trained baseline and representative timing evidence.**  
   §4.4, Figure 6, Table 8. The **0.55× [0.45,0.67]** result is correct, but it compares the complete adapted system against unadapted native 4/4. The existing same-adapter 2/2 already achieves **62.6 s and 96.9%**, versus **63.1 s and 96.9%** for 4/2. Deeper prefill therefore does not explain this timing advantage.

   The timing records also use consecutive items **1–64**, or **1–40** for MATH. Budget truncation is substantial in some conditions: native 1/1 reaches the GSM8K cap on **31/64** items; adapter 4/2 reaches the MATH cap on **10/40**.

   **Exact fix:** time the existing distilled-native 2/2 checkpoint under the same implementation and protocol; add it and adapter 2/2 to Figure 6. Report TTFT, generated tokens, cap rates, decode time/token and total time. Use a fixed random or stratified timing sample, preferably larger. Scope the abstract’s speed claim to **64 timed GSM8K items** and to the complete adapted system.

4. **The large base-model gains still combine training and untying.**  
   Abstract, Table 1, §4.2. The new same-adapter diagonal comparison is valuable and resolves the previous review’s cheapest requested control. It does not make the much larger native comparisons estimates of the effect of untying alone. Completed base-model native 1/1 and 2/2 distillation results are absent; loaders and launchers are not experimental results.

   **Exact fix:** put the selection-aware within-adapter results beside the native results and describe the latter as gains from the **complete trained method**. Complete the matched base-model distillation controls if retaining claims that exceed adaptation alone.

5. **The numerical audit does not cover several printed measurements, and its mismatch report is incomplete.**  
   Appendices A, B, G and L; Tables 9 and 13. These empirical quantities are absent from the JSON/recomputation path:

   - Long-answer throughput: **586, 219, 570, 253**.
   - Shared-answer-cache batch/throughput results: **32→192**, **219→476**, **759 versus 586**.
   - Dynamic/contiguous-cache throughput: **500/245/105** and **267/130/57**.
   - Parser diagnostics: **−9.8, 19.6, 9.8**.
   - MBPP+ malformed-output counts: **294/236/372/375 of 378**.
   - The **512/512** token-agreement check.

   Moreover, the script’s expected timing intervals remain **[0.46,0.64]** and **[0.91,1.11]**, while the actual and correctly printed intervals are **[0.45,0.67]** and **[0.90,1.11]**. These expectations are printed but never validated.

   **Exact fix:** release the missing measurement/diagnostic records, aggregate them into the JSON, and validate every section with failures producing a nonzero exit status. Correct the stale expectations. Give parameter counts and training time configuration/log provenance. Remove the blanket “every number” claim until coverage is complete.

6. **Related Work omits a particularly close predecessor.**  
   **MELT** adapts pretrained Ouro models through post-training/distillation to use shared KV memory. It is directly relevant to the paper’s adaptation and memory contributions, yet absent from the bibliography. [MELT paper](https://arxiv.org/html/2605.07721)

   CDB also explicitly uses full-depth prefill with early-exit decoding and studies shared caches; merely saying it “exits early during decode” understates the overlap. [CDB, Appendix A and §4.2](https://arxiv.org/html/2608.09444v2)

   **Exact fix:** add MELT and distinguish the contributions explicitly: independent configurable \(m,n\), the empirical task map, and the constrained prompt-cache maps. Expand the CDB comparison. If claiming a superior memory–quality trade-off, provide an appropriate comparison or narrow the claim to the evaluated baselines.

7. **The hybrid “component” intervention changes downstream states too.**  
   Table 12 and Appendix J. The implementation retains two-visit **post-loop cache states** for the attention-only, GDN-only and both conditions. The C11 reference has one-visit post-loop states. Therefore “GDN state only versus C11” changes GDN state **and downstream states**. The reported **+9.1 HELMET** cannot be attributed exclusively to the named component. See [cache selection](ouro/sensenova_runner.py:76).

   **Exact fix:** add a baseline retaining the two-visit exterior states but first-visit attention and GDN states inside the repeated block. Compare each component intervention against that baseline. Alternatively describe the existing interventions as composite and limit component attribution to direct **G versus both** and **A versus both** comparisons, which hold exterior states fixed. Apply the same qualification to the retrained GDN variant.

8. **Lev’s directional recommendation exceeds its replication.**  
   §4.5, Conclusion, Figure 7. The point-estimate pattern is correct: **13 positive, 12 significant** for \(m>n\); **13 negative, 10 significant** for \(m<n\). But each \(m\) uses separately trained weights, lower-\(m\) results use one seed, and the untied training forward costs roughly twice the matched-update control. The extra-seed and doubled-update checks cover maximum-prefill configurations.

   **Exact fix:** replace “the context should be read with many loops” with an observation restricted to the **seed-0 LevBench development runs**. Call the reference a **matched-data/update control**. State that the full directional pattern is not replicated across seeds; replicate weaker lower-\(m\) conditions before generalizing it.

9. **The adaptation data have not been checked for benchmark overlap.**  
   Appendix A discloses this, which is preferable to hiding it. Disclosure still leaves the generalization interpretation unresolved, especially for gains against unadapted baselines.

   **Exact fix:** audit the actual sampled training mixture for normalized exact and approximate benchmark-prompt matches; publish overlap counts and recompute results on unmatched items. Retraining is necessary only if the overlap materially compromises that analysis. There is currently no basis to assert that contamination did occur.

10. **The map ablation supports narrower conclusions than the text claims.**  
    §3 and Appendix H. RULER-16K versus GSM8K/MMLU does not isolate prompt length, so “the maps matter mainly for long prompts” conflates task and length. Likewise, nonsignificant unrestricted-map comparisons do not establish “no measurable cost.”

    **Exact fix:** write: **“The maps improve RULER-16K in these ablations; differences on the tested GSM8K and MMLU conditions are small and statistically inconclusive.”** Replace “no measurable cost” with **“no significant difference detected in five tested configurations.”** Print those five estimates and intervals in Table 10.

The rendered figures need the following corrections, ordered by consequence.

| Figure | Objection and exact fix |
|---|---|
| **6, p7** | **The missing adapter 2/2 point changes the visual interpretation.** Add its **62.6 s, 96.9%** point and the distilled baseline discussed above. Replace the bold sentence and §4.4 title with **“The adapted two-loop decoder answers faster than native 4/4 on these GSM8K items.”** Native 1/1 itself disproves the current general “fewer loops save time” formulation. Remove the oversized three-column legend; it shrinks the axes to roughly two-thirds of the column. |
| **7, p8** | Grey “native n/n” actually means continued control; red “loop adapter” combines separately trained models. Rename them **“continued control”** and **“untied models, best m.”** State selection on development items. The envelope displays no \(m<n\) cells, so it cannot communicate “both directions.” Replace it with three difference heatmaps, marking the diagonal and significant cells. |
| **4, p5** | The six-of-eight count is correct, but separately scaled axes obscure the common two-point criterion. Say **“Point estimates within two points…”** Use paired-difference plots with CIs and a shaded ±2-point band, or at least annotate differences. Include same-adapter 2/2 to expose attribution. |
| **1, p2; 8, p12** | Task dependence is visible, and evaluation-set selection is now disclosed. But the headline **51.8→68.4** contrast is absent: the grey endpoint is native **54.0**, not adapter **51.8**. Add the adapter diagonal or a MATH prefill-sweep inset at decode 4. State **“smallest maximizing m shown”** because labels hide ties. Figure 8 should repeat the selection caveat directly. |
| **5, p6** | The revised one/two-loop claim and **260-item** RULER label are correct. However, the caption implies red \(n=1,\ldots,4\) in every panel; Thinking adapter 4/4 is absent. Specify **“Thinking adapter: n=1–3.”** Rename the x-axis **“8K decode microbenchmark throughput”** and state that accuracy and throughput come from separate workloads. |
| **3, p4** | `no_grad()` does not disable adapters: calling the same `model` does not specify a native teacher. Use **`native_teacher(...)`** explicitly. For PyTorch semantics use **`torch.randint(1, 5, ...)`**. Replace “cache written once” with **“Only the final prefill loop’s KV is retained for decoding.”** Increase code size and comment contrast; the mostly empty right panel wastes space. |
| **2, p2** | The cache story is readable. However, outputs feeding a box called “joint softmax” can imply softmax over attention outputs. Label the branches **\((o_P,\lambda_P)\)** and **\((o_A,\lambda_A)\)**; rename the box **“LSE-weighted merge”** and show its weight formula. Mark decoder LoRA. Replace “loop k, repeated n times” with **“Decode: k=1,…,n per token.”** |

Across Figures 1, 4, 7 and 8, annotations are **6.8 points before document scaling**, and ticks are 7 points. Target at least 8-point final text. Different marker shapes would help overlapping curves and grayscale printing. I found **no gross clipped labels or actual annotation collisions**.

Every table is accounted for below. Covered numerical entries agree with the JSON; the objections concern inference, coverage or presentation.

| Table | Objection and exact fix |
|---|---|
| **1, p5** | Make selection-aware intervals primary and show the same-adapter comparator. Separate selected-\(m\) tuples from tiny CI text. |
| **2, p11** | Two rounded seed differences show a useful repeat, not broad training robustness. Add their CIs and call this a **two-seed check**. |
| **3, p13** | Correct full grid, including tied maxima, but too dense: 13 columns in scriptsize. Split into two benchmark panels; define `HEval`, `IFEv`, `MMLU-P`, and `R-*`; enlarge scores and emphasize decode blocks. |
| **4, p13** | Apply Table 1’s interval treatment. Repeat the meaning of stars, intervals and selected-\(m\) parentheses rather than requiring the reader to reconstruct it from Table 1. |
| **5, p13** | The caption appropriately admits selected examples and points to counterexamples. No additional correctness objection. Enlarge the CI text. |
| **6, p14** | Strong control structure; the negative smaller-model results are visible. Retain them. Improve CI readability and bring the same-adapter comparator into Figure 4. |
| **7, p14** | The distillation comparison is numerically correct. Keep the distinction between eight positive estimates and only two significant gains. Improve CI readability. |
| **8, p15** | Add timing-ratio intervals, token counts and cap rates. Include the distilled-native baseline. Define whether “time” is the arithmetic mean. |
| **9, p15** | The long-answer block lacks released numerical provenance. Add winning batch sizes: at 8K, adapter **64** throughout versus native **80/40/24/20**. This makes the batch-size explanation checkable. Replace unexplained \(d\) with \(n\). |
| **10, p15** | Replace “all CIs include 0” with five rows: **1/1 +0.1 [−1.3,1.5]; 2/2 +0.6 [−0.8,2.0]; 3/1 −0.9 [−2.9,1.2]; 3/2 +1.2 [−0.4,2.7]; 4/2 −1.0 [−2.6,0.5]**. Change \(p/d\) to \(m/n\). |
| **11, p16** | Correct both-direction evidence. Visually distinguish the diagonal and model groups; promote its structure into Figure 7. Explicitly identify the seed-0 scope. |
| **12, p16** | Grouping is readable, but interpretation requires the exterior-state correction in objection 7. Enlarge CIs. |
| **13, p17** | Useful negative results are retained. Preserve the MBPP+ exclusion explanation, but provide recomputable records for its format-failure counts. Enlarge CIs. |
| **14, p17** | Important category conclusions have no displayed uncertainty, despite only **34 recall** and **41 re-ranking** items for Ouro. Print paired CIs, or supply an adjacent CI table; referring readers to JSON is insufficient here. |
| **15, p17** | “Depends on context length and prompt format” exceeds the displayed experiment: removing exemplars changes several things together. Replace with **“The effect of prefill loops changes with the prompt setup.”** |

Additional factual, wording and layout fixes:

- **Evaluation coverage:** §4 says every model is evaluated at all 16 pairs. SenseNova uses only \(m\in\{1,2\},n=1\), and the supplied Thinking results do not document a complete grid. Specify coverage by checkpoint instead.
- **Benchmark count:** the headline contains **13 settings from 11 benchmark families**, because RULER contributes three lengths. Use that wording consistently.
- **Wrong causal wording:** §4.3 says prefilling with four loops makes the model faster. At decode 4, both systems already prefill four times. Attribute throughput gains to the shared prompt cache and measured batch capacity.
- **Residual equivalence wording:** replace Appendix H’s “leaves GSM8K unchanged” with “no significant difference detected”; replace Appendix A’s “the same” for **82.5 versus 83.1 ms/token** with “similar observed means.”
- **Timing aggregation:** Table 8’s **1.27×** is valid using pooled generation time divided by pooled tokens. The JSON’s mean of per-item ratios instead yields approximately **1.24×**. Name both aggregations explicitly and store the pooled statistic.
- **Bootstrap documentation:** the current selection-aware function uses **1,000**, despite Appendix B’s blanket **5,000** claim. Use 5,000 consistently; the audit above confirms the counts remain unchanged.
- **Wrong cross-reference:** Appendix A says Table 10 compares shared-answer caches, but Table 10 omits those rows. Restore the rows or refer to the actual Appendix H paragraph.
- **Unsupported cross-reference:** Appendix F points to “Limitations” for the smaller Thinking model’s BBH degeneration, but that discussion is absent there. Add it or remove the pointer.
- **Reproduction instructions:** `REPRODUCE.md` calls the 260-row RULER subset Figure 1’s subset; it belongs to Figure 5. It also identifies the end-to-end plot as Figure 7 rather than Figure 6.
- **Archive size:** “1 MB” is stale. The compressed item files total **2,318,016 bytes**; required Lev records add roughly **155 MB**. Remove the size claim or give accurate separate sizes.
- **Float placement:** Figure 7 lands inside References, after the conclusion. Appendix D appears as an empty heading before E; its figure/table arrive later. Appendix G’s timing tables drift to p15 after later sections begin. Move declarations earlier and use targeted float barriers before References and affected appendix transitions.
- **Unused space:** reflow Figure 8’s mostly empty final row, compact Figure 3, and eliminate the two-line reproducibility continuation stranded in p17’s right column.

The previous review’s points are now resolved as follows:

| Previous objection | Current disposition |
|---|---|
| Significant counts confused with positive estimates | **Resolved** in the corrected abstract/results wording. |
| GSM8K/HumanEval optimum wording | **Resolved** when “peak” means the global maximum; HumanEval now allows three or four. |
| Figure 1 falsely described as every cell | **Resolved.** |
| MBPP+ described as a tie | **Resolved:** now “no significant difference.” |
| Distillation described as giving no knowledge gain | **Resolved:** now “not significantly.” |
| Lev loss magnitude paired with a negative interval | **Resolved.** |
| Conclusion omitted “significantly” | **Resolved.** |
| Figures 1/8 omitted evaluation-set selection caveat | **Resolved**, though Figure 8 should be self-contained. |
| Figure 4 vague quality claim | **Resolved numerically:** six of eight within two points. |
| Figure 5 universal depth-cutting claim and wrong RULER size | **Resolved**, with the missing Thinking-4/4 caveat still needed. |
| Figure 6 omitted native-1/1 latency explanation | **Resolved in the caption**, but its bold generalization remains too broad. |
| Same-adapter 2/2 timing control | **Partly resolved:** text/Table 8 include it; Figure 6 does not. |
| Training versus untying | **Partly resolved:** within-adapter comparison added; its uncertainty needs correction. |
| Selection uncertainty | **Partly resolved:** analysis added, but conditional results remain primary. |
| Lev scope and Figure 7 presentation | **Partly resolved:** prose improved; replication limits and misleading graphic remain. |
| MATH adjudication; distilled-native timing | **Unresolved.** |

The checked headline arithmetic—including **51.8→68.4**, Thinking’s **eight positive/five significant** comparisons, **six of eight within two points**, and **0.55×** timing—is sound. The remaining acceptance risks concern what those numbers establish, and whether all printed evidence is actually reproducible.

**Score: 3/5.**  
REMAINING SUBSTANTIVE OBJECTIONS: 10