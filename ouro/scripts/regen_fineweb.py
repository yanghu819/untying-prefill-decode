"""Regenerate fineweb_train.bin exactly as prepare_adapter_data.prepare_fineweb did (pinned fineweb-edu sample-10BT
revision, published order, Ouro tokenizer), reading the first parquet shards downloaded through the HF mirror (streaming
file listing does not work through the mirror).  usage: HF_ENDPOINT=https://hf-mirror.com regen_fineweb.py OUT_DIR"""
import json
from pathlib import Path
import sys

import datasets
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from ouro.prepare_adapter_data import FINEWEB_DATASET, FINEWEB_REVISION, prepare_fineweb

files = [hf_hub_download(FINEWEB_DATASET, f"sample/10BT/{i:03d}_00000.parquet", repo_type="dataset",
                         revision=FINEWEB_REVISION, cache_dir="/dev/shm/fwcache") for i in range(2)]
print("downloaded", files, flush=True)


def rows(*args, **kwargs):
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=1000, columns=["id", "text"]):
            yield from batch.to_pylist()


datasets.load_dataset = lambda *a, **k: rows()
tok = AutoTokenizer.from_pretrained("/dev/shm/big-small-ouro/model-2.6b", local_files_only=True)
print(json.dumps(prepare_fineweb(tok, 793554944, Path("/dev/shm/fwcache"), Path(sys.argv[1]), 1000)), flush=True)
print("FINEWEB_DONE", flush=True)
