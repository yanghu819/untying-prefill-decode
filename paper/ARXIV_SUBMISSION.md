# arXiv submission (fill in at https://arxiv.org/submit)

Upload: paper/arxiv/arxiv.tar.gz (main.tex + main.bbl + acl.sty + acl_natbib.bst + figs/). Compiles with pdflatex.

Title:
Read Deep, Write Shallow: How Looped Language Models Should Spend Their Depth

Authors:
Yang Hu

Primary category: cs.CL (Computation and Language)
Cross-list: cs.LG (Machine Learning)

Comments:
11 pages, 5 figures. Code, results and per-item outputs: https://github.com/yanghu819/read-deep-write-shallow

License: CC BY 4.0 (recommended for reach)

Abstract (plain text):
Looped (recurrent-depth) language models run the same number of loops when reading the prompt and when writing the answer. A small depth-split connector (LoRA plus per-loop key-value maps on a frozen model) lets the prompt be read with p loops and the answer written with d loops. On Ouro-2.6B, the full 4x4 map shows a division of labour: depth spent reading the prompt buys understanding, while depth spent writing the answer buys multi-step derivation. At one writing loop, reading 4 loops instead of 1 adds 15.2 MMLU and 8.6 RULER-16K points, and the effect replicates on five more checkpoints. On SenseNova-Loop-9B, whose one-loop writer collapses, merely reading a two-loop prompt state, without any training, cuts unanswered GSM8K problems from 97% to 44%. Writing shallow is where time is saved: a connector writing two loops comes within one point of the 4-loop Ouro-2.6B-Thinking on GSM8K at 0.55x its time per answer. Through a RoPE-commuting key scale, one prompt KV cache serves every loop exactly; storing the answer once too more than doubles long-answer decode throughput. Reading deep hurts short-context retrieval and few-shot mathematics, consistent with linear probes: deeper loops keep less of what was said and more of what comes next. Code: https://github.com/yanghu819/read-deep-write-shallow
