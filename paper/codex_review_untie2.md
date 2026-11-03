1. **Recommendation: Weak reject as delivered; close to weak accept.** The empirical contribution is credible, but stale compiled results and surviving generalizations still misrepresent it.

2. **[Previous review](/Users/huyang/Documents/ChatGPT/untying-prefill-decode/paper/codex_review_untie.md), point by point:**
   - **#2 fixed:** the controlled MATH-500 result now leads.
   - **#3 partial:** benchmark/prompt dependence and numerical ranges are corrected; the conclusion resurrects “one loop suits code and math.”
   - **#4 mostly fixed:** 12/13 and 10/13 are supported; losses are acknowledged. However, “MBPP+ stays ahead” at three/four loops implies an unavailable native-4 comparison.
   - **#5 fixed:** benchmark calibration, no routing/transfer claim, and conditional bootstrap uncertainty are explicit.
   - **#6 partial:** training versus allocation is distinguished, but equal-cost wording survives in Setup, Figure 4, §4.3 and Conclusion.
   - **#7 fixed in current data, not compiled delivery:** coverage is complete apart from the two stated native baselines.
   - **#8 mostly fixed:** full-map reference, decoding terminology and cache naming improved; prompting reversal remains insufficiently integrated into the main takeaway.

3. **Remaining/new contradictions:**
   - **The [PDF](/Users/huyang/Documents/ChatGPT/untying-prefill-decode/paper/latex/main.pdf) is stale.** Figure 1 shows RULER-16K \(m=2,n=3\) as **96**, versus **69.3/520 items** in [ground truth](/Users/huyang/Documents/ChatGPT/untying-prefill-decode/ouro/results/paper_numbers.json). Table 1 reports **−3.1 [−7.3,1.0]**, whereas current results are **−5.6 [−8.7,−2.6]**: a significant loss. Current figure/table sources are corrected. Zero script mismatches do not certify embedded PDF content.
   - **§4.3 overclaims optimality:** four beating one does not establish four as optimal; Thinking MBPP+ actually scores 79.4 at 2/2 versus 79.1 at 4/2. Moreover, Thinking MMLU-Pro/BBH use five/three shots. The base/Thinking reversal is legitimate regime dependence, not itself a contradiction.
   - **HELMET attribution is wrong:** 4/2 versus native 4/4 holds prefill at four; its recall deficit cannot be attributed to “more prefill loops.”
   - Appendix A’s shared-cache formula should be \(b(\ell+t)\), not \(b(n+t)\).

4. **Three highest-value improvements, with exact text for [the manuscript](/Users/huyang/Documents/ChatGPT/untying-prefill-decode/paper/latex/main.tex):**

   **A. Recompile from finalized assets; clarify coverage and §4.2:**
   > “Native 4/4 results for HumanEval and MBPP+ are unavailable. MATH-500 gains 13.6 and 14.4 points at three and four decode loops; MBPP+ gains 18.0 at three.”

   **B. Replace the conclusion’s category slogan and §4.3’s optimality claim:**
   > “Preferred prefill depth depends on the checkpoint, benchmark, prompting protocol, and decode budget. One loop performs best on base-model 4-shot MATH-500 and MBPP+; GSM8K and HumanEval favor deeper prefill.”

   > “At two decode loops, Thinking benefits from four versus one prefill loop on all eight benchmarks, significantly on seven; this does not establish four as optimal. Only GSM8K and MATH-500 use zero-shot prompts.”

   **C. Replace equal-cost language throughout with “the same number of decode loops”; repair causal attribution:**
   > “On timed GSM8K at batch 1, 4/2 costs 1.27× native 2/2 per generated token; shorter answers reduce total time.”

   > “The HELMET deficit is concentrated in exact recall; this native-model contrast does not isolate prefill depth.”

5. **Clean and convincing for ACL?** Substantially more convincing, but not clean yet. Independent allocation with regime-dependent optima is a strong story. Correct the compiled evidence and claim boundaries; another broad experimental campaign is unnecessary.