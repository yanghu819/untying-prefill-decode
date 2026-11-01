"""lm-eval-harness CLI with small compatibility patches for remote-code looped models.

- transformers >= 4.56 removed Cache.get_max_length (Nanbeige remote code still calls it).
- transformers 5 has no AutoModelForVision2Seq (SenseNova-Loop needs transformers 5.18).
Importing ouro.lmeval_ouro registers --model ouro (our runners: natives and delta arms).
"""
import faulthandler
import os
import signal
import sys

# ouro/ holds evaluate.py, which would shadow HF `evaluate` (code_eval metric); import ouro.* via PYTHONPATH instead
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

faulthandler.register(signal.SIGUSR1, all_threads=True)  # `kill -USR1 <pid>` dumps every Python stack to stderr

from transformers import cache_utils

for name in ("DynamicCache", "Cache"):
    cls = getattr(cache_utils, name, None)
    if cls is not None and not hasattr(cls, "get_max_length"):
        cls.get_max_length = lambda self: None

import transformers

if not hasattr(transformers, "AutoModelForVision2Seq"):  # transformers 5: lm-eval's VLM wrapper cannot import; unused
    import types
    sys.modules["lm_eval.models.hf_vlms"] = types.ModuleType("lm_eval.models.hf_vlms")

from lm_eval.__main__ import cli_evaluate

import ouro.lmeval_ouro  # noqa: F401  (registers --model ouro)

if __name__ == "__main__":
    sys.exit(cli_evaluate())
