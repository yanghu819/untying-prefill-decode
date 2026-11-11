# ARR Responsible NLP checklist — draft answers

Target: ACL 2027 via ARR, submission deadline 2027-01-04 (AoE). Fill these into the OpenReview form; sections are named as
in `paper/latex/main.tex` (appendix letters shift, so names are used). Items marked **TODO** need the author's decision.

## A. For every submission
- **A1 Limitations.** Yes: the unnumbered "Limitations" section after the Conclusion.
- **A2 Potential risks.** Yes, briefly: the method changes inference compute of existing open checkpoints and adds no
  new capabilities or data; risks are those of the underlying models. **TODO:** add one sentence to Limitations if
  the form requires it in the paper.

## B. Did you use or create scientific artifacts?
Yes. Models: Ouro-1.4B/2.6B (+Thinking), SenseNova-Loop-9B-Preview, Nanbeige4.2-3B-Base, Lev decision models.
Benchmarks: MMLU, MMLU-Pro, ARC-Challenge, GPQA-Diamond, BBH, IFEval, GSM8K, MATH-500, HumanEval, MBPP+, RULER, HELMET,
LevBench (development set only). Training data: OpenThoughts, Tulu SFT, LongAlpaca, FineWeb-Edu, Lev's S31 subset.
- **B1 Cite creators.** Yes (Section 2 Setup; appendices Training Details and Evaluation Details).
- **B2 Licenses.** **TODO:** list each artifact's license in the Evaluation Details appendix (Ouro code is Apache-2.0, see
  `third_party/ouro/LICENSE`; our code is MIT). Check the model cards for the others.
- **B3 Intended use.** Yes: all artifacts are used for research evaluation, consistent with their terms; GPQA questions
  are not redistributed (only per-item scores keyed by index), and LevBench's locked test split is never evaluated.
- **B4 Personal information / offensive content.** N/A: public benchmarks and corpora; no new data collected.
- **B5 Documentation of artifacts.** Yes: README, REPRODUCE.md, per-item scores.
- **B6 Statistics.** Yes: item counts per benchmark (Evaluation Details appendix; table and figure captions).

## C. Did you run computational experiments?
- **C1 Model size and compute.** Parameters: Section 6 Method / Training Details appendix (rank-16 LoRA, 63M trainable on Ouro-2.6B). GPU type:
  NVIDIA B300. Training time: 6.75 h on 4 GPUs for the Ouro-2.6B adapter (Training Details appendix). **TODO:** total GPU hours of all
  runs (sum the training and evaluation logs; state an estimate in the Training Details appendix).
- **C2 Hyperparameters.** Yes: Training Details appendix (optimiser, schedule, batch, tokens). The learning rate was
  checked at 0.3x and 3x for the distilled controls and for Ours (Method Details appendix, "Baseline learning rate");
  both are reported at the default.
- **C3 Descriptive statistics.** Yes: paired bootstrap 95% intervals (5,000 resamples, seed 0) everywhere; training-seed
  robustness in the Table 1 and Table 5 captions, Section 4 and the Two-Seed Checks appendix; single runs otherwise, stated.
- **C4 Existing packages.** Yes: lm-evaluation-harness (with versions in `environments/ouro`), math-verify,
  FlashAttention, Hugging Face transformers.

## D. Human annotators / participants
- No (D1–D5 N/A).

## E. AI assistants in research or writing
- **E1.** Yes. **TODO (author's wording):** e.g. "We used AI assistants (Claude, Codex) for writing and running
  experiment code, for drafting and editing text, and for internal review of claims. All results are recomputed from
  released per-item scores by the released script; the authors checked every claim."

## Anonymity for review
- The review PDF says "Anonymous ACL submission"; the arXiv build adds the author and repository link.
- **TODO:** the code link for review must be anonymous (e.g. an anonymized mirror of this repository); the public
  repository name and the README author line would de-anonymize. Do not post the arXiv version during the ARR
  anonymity period if the target cycle forbids it (check the cycle's preprint policy).
