# SPDX-License-Identifier: Apache-2.0
"""Pinned LongBench prompts and paired regression scores, not an official full evaluation."""

import hashlib
import json
import re
import string
import zipfile
from collections import Counter
from difflib import SequenceMatcher
from huggingface_hub import hf_hub_download


def prompt_for(dataset, row):
    if dataset == "qasper":
        return (
            "Read the article below and answer the question with a short phrase. "
            "Answer yes/no for a yes/no question, or unanswerable when the article "
            "gives no answer. Output only your answer.\n\nArticle:\n"
            + row["context"]
            + "\n\nQuestion:\n"
            + row["input"]
            + "\n\nAnswer:"
        )
    if dataset == "hotpotqa":
        return (
            "Answer the question using the passages below. Output only a short "
            "answer without explanation.\n\n"
            + row["context"]
            + "\n\nQuestion:\n"
            + row["input"]
            + "\n\nAnswer:"
        )
    return (
        "Complete the code below by writing only the next line. Do not repeat "
        "the existing code or add explanations or Markdown fences.\n\n" + row["context"]
    )


def load_cases(manifest):
    path = hf_hub_download(
        "zai-org/LongBench",
        "data.zip",
        repo_type="dataset",
        revision=manifest["dataset_revision"],
    )
    with zipfile.ZipFile(path) as archive:
        data = {}
        for dataset in ("qasper", "hotpotqa", "lcc"):
            name = next(
                n for n in archive.namelist() if n.endswith(f"/{dataset}.jsonl")
            )
            data[dataset] = [json.loads(s) for s in archive.read(name).splitlines()]
    for sample in manifest["samples"]:
        dataset, index = sample["id"].split("-")
        row = data[dataset][int(index)]
        prompt = prompt_for(dataset, row)
        assert row["_id"] == sample["source_id"]
        assert hashlib.sha256(prompt.encode()).hexdigest() == sample["prompt_sha256"]
        yield {
            **sample,
            "dataset": dataset,
            "prompt": prompt,
            "answers": row["answers"],
            "max_tokens": 64 if dataset == "lcc" else 128,
        }


def score(text, case):
    if case["dataset"] == "lcc":
        line = next(
            (
                s.strip()
                for s in text.splitlines()
                if s.strip() and not s.strip().startswith("```")
            ),
            "",
        )
        return max(
            SequenceMatcher(None, line, a.strip(), autojunk=False).ratio()
            for a in case["answers"]
        )

    def normalize(text):
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return re.sub(r"\b(a|an|the)\b", " ", text).split()

    predicted = normalize(text)
    scores = []
    for answer in case["answers"]:
        expected = normalize(answer)
        common = sum((Counter(predicted) & Counter(expected)).values())
        scores.append(
            2 * common / (len(predicted) + len(expected))
            if predicted or expected
            else 1.0
        )
    return max(scores)
