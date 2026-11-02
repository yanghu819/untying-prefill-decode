#!/bin/bash
# Ouro-2.6B-Thinking distillation baseline (Codex round 3): arm D = native 2/2 layout (per-loop prompt KV), LoRA on the
# frozen model, no maps; same data (mix-think-48m), 48M KL tokens, native 4/4 teacher, micro 1, seed as arm S.
M=/dev/shm/big-small-ouro; R=$M/runs; cd $R
MODEL=$M/model-2.6b-think DATA=$M/mix-think-48m ./run_mix2.sh T26-D-native22 D 0,1,2,3 29791 48000000 --micro 1
