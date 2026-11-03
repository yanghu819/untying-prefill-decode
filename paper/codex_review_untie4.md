**Score: 3/5 — borderline. Confidence: 4/5.** The evidence is substantial; attribution and claim scope remain acceptance risks.

I recomputed [the statistics](ouro/scripts/paper_numbers.py) in memory: **exact match with [paper_numbers.json](ouro/results/paper_numbers.json)**. All eight generated tables also match. I found no effect-size or confidence-interval transcription errors in the requested sections.

The following wording in [the manuscript](paper/latex/main.tex) needs correction or qualification.

| Location and quotation | Correct value or scope |
|---|---|
| **Abstract, lines 49–50:** “Looped decision models show both directions: with more prefill than decision loops they beat a same-training control on 12 of 13 \((m,n)\) pairs, and with fewer they fall below it on all 13.” **Introduction, lines 106–108 repeats this count.** | For \(m>n\): **13/13 positive estimates, 12/13 significant**. For \(m<n\): **13/13 negative estimates, 10/13 significant**. Insert “significantly” before the first count and distinguish significance from direction. |
| **Introduction, lines 94–96**, and **§4.1, lines 287–288:** “GSM8K is best with two prefill loops and HumanEval with four.” | Not uniformly across decode budgets. For \(n=1,2,3,4\), best \(m\) is **GSM8K: 4, {2,3}, 2, 2; HumanEval: 4, {3,4}, 3, 4**. Counterexamples: GSM8K at \(n=1\), **55.0 versus 53.5** for \(m=4\) versus 2; HumanEval at \(n=3\), **40.2 versus 39.0** for \(m=3\) versus 4. If describing global maxima, say so; HumanEval’s maximum ties \(m=3\) and 4. |
| **§4.1, line 283:** “Figure 1 shows accuracy at every \((m,n)\).” | It shows **four best-\(m\) points per task**, not all 16 cells. Say “the best accuracy over \(m\) at each \(n\).” |
| **§4.2, lines 297–299:** “…ties on MBPP+, where native 1/1 is already strong…” | **−0.3 [−3.4, 2.9]** points. Say “shows no significant difference”; this is not an equivalence test. |
| **§4.3, lines 322–324:** “A LoRA on native 2/2, trained with the same data, teacher and number of tokens as the loop adapter, gains on HELMET and BBH but not on knowledge…” | Knowledge point estimates also improve: **MMLU-Pro +1.7 [−2.1, 5.2]**, **ARC-C +0.7 [−0.4, 1.8]**. Say “no significant gain on knowledge.” |
| **§4.5, line 377:** “The sign of \(m-n\) decides the outcome.” **Conclusion, lines 391–392:** “On looped decision models the direction is clear-cut: more prefill than decision loops helps, fewer hurts.” | Correct **as observed off-diagonal point-estimate signs in these seed-0 LevBench development runs**; too categorical as a general or statistical conclusion. The \(m>n\) exception to significance is Lev-2.6B 4/3: **+0.5 [−0.7, 1.7]**. Three \(m<n\) intervals include zero. Each \(m\) also uses separately trained weights. |
| **§4.5, lines 380–381:** “…by up to \(13.0[-14.7,-11.3]\) points…” | Magnitude and interval use opposite signs. Write **difference −13.0 [−14.7, −11.3]**, or **loss 13.0 [11.3, 14.7]**. |
| **Conclusion, lines 389–391:** “…choosing it per benchmark pays most with one or two decode loops, where it beats native Ouro-2.6B on 12 and 10 of 13 benchmarks.” | Add **“significantly.”** Positive point estimates are **12/13 at both depths**; significant wins are **12/13 and 10/13**. |

The remaining requested numerical claims check out, including **51.8→68.4**, the **5–21-point** range, Thinking’s **eight positive/five significant** comparisons, **six of eight within two points**, and the **0.55×** timing result. Section 4.5’s **+5.3, +7.5, +4.4**, **12/13 significant gains**, and **10/13 significant losses** are correct.

Each figure’s caption tells the following story:

| Figure | Verdict |
|---|---|
| **1: Task map** | Supports task dependence. Label the curves **best \(m\) selected on the evaluation items**; they are not held-out selection results or the complete grid. |
| **2: Method diagram** | Supported: one prompt cache, query/value maps, joint softmax. |
| **3: Pseudocode** | Supported: the log-sum-exp merger implements the stated joint attention. |
| **4: “Close to the quality of four”** | Overbroad. Use **“within two points on six of eight benchmarks.”** MATH-500 loses **4.8 points**, HELMET **8.7**. Confidence intervals do not establish equivalence. |
| **5: “Cutting decode loops costs less accuracy”** | Not universally true: cutting **4→3**, adapter/native losses are **0.3/0.1** on MMLU and **1.1/0.0** on Thinking GSM8K. Qualify “cutting to one or two loops.” **Dataset-label error:** its RULER panel uses **260 items**, versus **520** in Figure 1; native-four accuracy is **79.8 versus 81.3**. |
| **6: End-to-end latency** | Supports adapted 4/2 versus native 4/4: **63.1 versus 114.2 seconds**. “Fewer loops” is not universal: native 1/1 takes **86.5 s**, versus **70.0 s** for 2/2. Add the already measured **same-adapter 2/2: 62.6 s**, versus 4/2’s **63.1 s**, both **96.9%** on these 64 items. |
| **7: Lev** | Supports the selected deep-prefill trend, but **shows no \(m<n\) cells** and no uncertainty. The best-\(m\) envelope spans separately trained models. Rename “native” to **continued control** and “loop adapter” to **untied model**; a full difference heatmap would show both directions. |
| **8: Full task map** | Supports task dependence across all 13 tasks, with the same evaluation-set selection caveat as Figure 1. |

My five most likely acceptance-blocking objections, ranked:

1. **The headline gains confound training with untying.** Base Ouro-2.6B’s 12/13 and 10/13 wins use untrained native baselines; the Thinking distillation control does not resolve that headline. **Cheapest answer:** report selected \(m\) versus the same adapter’s \(m=n\) from existing scores, and call native comparisons gains from the complete method. Stronger: matched base-model native 1/1 and 2/2 distillation.

2. **The flagship MATH effect may partly measure answer formatting.** The appendix acknowledges that strict Minerva extraction reverses the decode-one comparison. **Cheapest experiment:** blind adjudication of parser-discordant responses, especially **S14 versus S44**, reporting semantic final-answer correctness separately from formatting.

3. **The speed claim lacks the strongest trained baseline.** Deeper prefill itself gives **1.01×** answer time; the **0.55×** saving includes shallower decoding and shorter answers. **Cheapest experiment:** time the existing distilled-native 2/2 checkpoint on the same items and add it, alongside same-adapter 2/2, to Figure 6.

4. **Headline significance excludes selection uncertainty.** Bootstrap intervals fix the split and chosen \(m\); the figures additionally select maxima on evaluation items. **Cheapest experiment:** repeated splits and a bootstrap that repeats selection, using existing scores; report sensitivity to multiple-comparison correction.

5. **Lev’s universal framing exceeds its replication.** All-direction results use one seed per \(m\), one development set, and separately trained weights. Existing four-seed and compute controls cover maximum \(m\). **Cheapest edit:** restrict the conclusion to those observed runs and state both point-estimate and significant counts explicitly. Stronger: repeat the weakest lower-\(m\) comparisons.