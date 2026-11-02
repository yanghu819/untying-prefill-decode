1. **Recommendation: Weak reject, as written.** The central finding is compelling, but the headline generalizations exceed the evidence, and the comparisons blur loop allocation, adaptation, and computational cost.

2. **Story: clearer and potentially compelling for NLP.** Independently allocating prompt processing and answer generation is an accessible question. The strongest finding is **4-shot MATH-500: with the same adapter and four decode loops, reducing prefill from four loops to one raises accuracy from 51.8 to 68.4**, a 16.6-point gain [12.4, 21.0]. Figure 1 is correctly positioned; this controlled contrast should lead its interpretation and the abstract. The native-model gain is less diagnostic because it includes training. :codex-file-citation{path="/Users/huyang/Documents/ChatGPT/untying-prefill-decode/ouro/results/paper_numbers.json" purpose="source"}

3. **Fix 1 — Replace the task-category slogan.** GSM8K does not favor one prefill loop; HumanEval’s current best cell uses four. The appendix’s MATH prompting reversal defeats “a property of the task.” Replace that claim throughout with:
   > “Preferred prefill depth depends on the benchmark, prompting protocol, and decode budget. One loop performs best on 4-shot MATH-500 and among available MBPP+ cells; GSM8K and HumanEval favor deeper prefill. MMLU favors three or four loops.”

   Also replace §4.1’s inaccurate ranges:
   > “MMLU reaches about 70 at one decode loop and 74–75 at larger budgets; GSM8K spans 71–88 with two or more decode loops.”

   Delete the premature claim that retrieval wants four decode loops.

4. **Fix 2 — State the available results, including losses.** “Every benchmark” implies coverage the current selection JSON lacks; “about as good except MATH” hides significant regressions. Replace the abstract/introduction/§4.2 summaries with:
   > “Across ten currently available one-loop comparisons, selection improves over native 1/1 by 5.2–20.9 points. At two decode loops, six of seven estimates are positive, five significantly. At higher decode counts, results are mixed: BBH loses 2.3 points at n=3 and provisional RULER-4K loses 2.9 at n=4; MATH-500 gains 13.6–14.4.”

5. **Fix 3 — Explain what cross-fitting establishes.** The code holds the split and chosen counts fixed during bootstrapping. Replace the final selection paragraph with:
   > “For each benchmark and fixed decode count, we select m using half its labeled items, evaluate on the other half, and swap. This estimates benchmark-level calibration, not per-prompt routing or transfer to unseen tasks. Bootstrap intervals resample held-out outcomes with the split and selected counts fixed; they do not measure selection instability.”

   This is useful calibration, but requires labeled examples from the target benchmark. :codex-file-citation{path="/Users/huyang/Documents/ChatGPT/untying-prefill-decode/ouro/scripts/paper_numbers.py" purpose="source"}

6. **Fix 4 — Separate the comparison axes.** Replace “same decode cost” with “same number of decode loops,” including captions. Add:
   > “Native-model contrasts combine adapter training with loop allocation. Within-adapter comparisons isolate changes in prefill depth. Equal decode-loop counts do not imply equal latency: on timed GSM8K, prefill 4 / decode 2 costs 1.27× native 2/2 per generated token at batch 1, while shorter answers reduce total time.”

   Replace “at least as accurate on eight benchmarks” with “higher point estimates on eight benchmarks, with significant gains on five.”

7. **Fix 5 — Make interim coverage visible.** Figure 1 compares, for example, RULER-16K adapter scores over 520 items with a native score over 71. Table 1 hides counts including 42 and 208. Report counts, mark provisional results, and use matched items for comparisons. Caption replacement:
   > “Available results are shown; blank cells are pending. Provisional entries report their item counts. Scores computed on different subsets are not directly comparable.”

   “Fourteen benchmarks at every pair” is not currently demonstrated. :codex-file-citation{path="/Users/huyang/Documents/ChatGPT/untying-prefill-decode/paper/latex/main.pdf" purpose="source"}

8. **Appendix: residual emphasis, not wholesale contradiction.** Fixed-4/2 rescue results and probes still dominate, but their caveats are appropriate. Promote the MATH prompting reversal; retain counterexamples. Rename “Answer stored once” to “Shared-answer-KV variant” to avoid contradicting the default cache layout. Replace “written with one loop” with “decoded with one loop.” Add the promised full task map: its missing label currently produces “Figure ??” in Figure 1’s caption. :codex-file-citation{path="/Users/huyang/Documents/ChatGPT/untying-prefill-decode/paper/latex/main.tex" purpose="source"}