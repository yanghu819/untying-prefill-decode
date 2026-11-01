"""Official RULER retrieval recall and LongBench Qasper token F1.

RULER: NVIDIA/RULER c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a,
scripts/eval/{evaluate.py,synthetic/constants.py}; Apache-2.0.
LongBench: THUDM/LongBench 2e00731f8d0bff23dc4325161044d0ed8af94c1e,
LongBench/{metrics.py,eval.py}; MIT (see licenses/).
Full RULER plus English Qasper, HotpotQA and 2WikiMQA are supported.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import string


RULER_TASKS = ("niah_single_1", "niah_multikey_2", "vt", "cwe")


def normalize_answer(text):
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def qa_f1_score(prediction, reference):
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(reference).split()
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(ref)
    return 2 * precision * recall / (precision + recall)


def score_generation(task, prediction, answers):
    """Return a per-example score in [0, 1], without rounding.

    RULER officially uses case-insensitive substring recall, not exact match,
    and does not penalize extra output. Qasper takes max F1 over references.
    """
    if task in ("qa_1", "qa_2"):
        prediction = re.sub(r"[\x00-\x1f]", "\n", prediction.strip()).strip()
        return float(any(answer.lower() in prediction.lower() for answer in answers))
    if task in RULER_TASKS or task.startswith("niah_") or task == "fwe":
        prediction = re.sub(r"[\x00-\x1f]", "\n", prediction.strip()).strip()
        return sum(answer.lower() in prediction.lower() for answer in answers) / len(answers)
    if task in ("qasper", "hotpotqa", "2wikimqa"):
        return max(qa_f1_score(prediction, answer) for answer in answers)
    raise ValueError(f"Unsupported generation task: {task}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = {row["id"]: row for row in map(json.loads, args.data.read_text().splitlines())}
    predictions = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    groups, scores = defaultdict(list), []
    for prediction in predictions:
        row = data[prediction["id"]]
        value = score_generation(row["task"], prediction["prediction"], row["answers"])
        key = f'{row["task"]}/{row["context_length"]}'
        groups[key].append(value)
        scores.append({"id": row["id"], "task": row["task"], "score": value})
    result = {
        "score_units": "percent",
        "tasks": {key: {"count": len(values), "score": 100 * sum(values) / len(values),
                        "official_rounded": round(100 * sum(values) / len(values), 2)}
                  for key, values in groups.items()},
        "examples": scores,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
