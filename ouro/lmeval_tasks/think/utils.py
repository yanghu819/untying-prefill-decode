"""Zero-shot reasoning tasks for chat / thinking models: answer in \\boxed{}, scored with math_verify."""
from math_verify import parse, verify


def gsm_target(doc):
    return doc["answer"].split("####")[-1].strip().replace(",", "")


def score(gold, response):
    try:
        return float(verify(parse("$" + gold + "$"), parse(response)))
    except Exception:
        return 0.0


def gsm_results(doc, results):
    return {"math_verify": score(gsm_target(doc), results[0])}


def math_results(doc, results):
    return {"math_verify": score(doc["answer"], results[0])}


def shard(dataset, index, count):
    """Every count-th document starting at index (interleaved, so each shard spans the whole difficulty range)."""
    return dataset.select(range(index, len(dataset), count))
def math_shard0(dataset):
    return shard(dataset, 0, 5)
def math_shard1(dataset):
    return shard(dataset, 1, 5)
def math_shard2(dataset):
    return shard(dataset, 2, 5)
def math_shard3(dataset):
    return shard(dataset, 3, 5)
def math_shard4(dataset):
    return shard(dataset, 4, 5)
def gsm_shard0(dataset):
    return shard(dataset, 0, 4)
def gsm_shard1(dataset):
    return shard(dataset, 1, 4)
def gsm_shard2(dataset):
    return shard(dataset, 2, 4)
def gsm_shard3(dataset):
    return shard(dataset, 3, 4)
