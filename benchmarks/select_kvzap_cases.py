# SPDX-License-Identifier: Apache-2.0
"""Select a fresh, length-stratified screen without reading model outputs."""

import argparse
import hashlib
import json
import random
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from kvzap_quality import prompt_for

MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
MODEL_REVISION = "241a666dad6cb93c8ff213d39a7f34a36bf26db4"
BANDS = {
    "qasper": [(2048, 4096, 6), (4096, 8192, 6), (8192, 16384, 4)],
    "hotpotqa": [(8192, 16384, 16)],
    "lcc": [(2048, 4096, 6), (4096, 8192, 6), (8192, 16384, 4)],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()
    previous = json.loads(args.exclude_manifest.read_text())
    excluded = set(previous["excluded_ids"]) | {s["id"] for s in previous["samples"]}
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=MODEL_REVISION, local_files_only=True
    )
    path = hf_hub_download(
        "zai-org/LongBench",
        "data.zip",
        repo_type="dataset",
        revision=previous["dataset_revision"],
        local_files_only=True,
    )
    rng = random.Random(args.seed)
    samples = []
    with zipfile.ZipFile(path) as archive:
        for dataset, bands in BANDS.items():
            name = next(
                n for n in archive.namelist() if n.endswith(f"/{dataset}.jsonl")
            )
            rows = [json.loads(s) for s in archive.read(name).splitlines()]
            used_contexts = {
                row["context"]
                for i, row in enumerate(rows)
                if f"{dataset}-{i}" in excluded
            }
            candidates = [[] for _ in bands]
            for i, row in enumerate(rows):
                case_id = f"{dataset}-{i}"
                if case_id in excluded or row["context"] in used_contexts:
                    continue
                prompt = prompt_for(dataset, row)
                tokens = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                )
                for bucket, (low, high, _) in zip(candidates, bands):
                    if low <= len(tokens) < high:
                        bucket.append(
                            {
                                "id": case_id,
                                "source_id": row["_id"],
                                "tokens": len(tokens),
                                "prompt_sha256": hashlib.sha256(
                                    prompt.encode()
                                ).hexdigest(),
                            }
                        )
            for bucket, (_, _, count) in zip(candidates, bands):
                assert len(bucket) >= count, (dataset, len(bucket), count)
                rng.shuffle(bucket)
                samples.extend(bucket[:count])
    manifest = {
        "dataset_revision": previous["dataset_revision"],
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "selection_seed": args.seed,
        "token_length_bands": BANDS,
        "excluded_ids": sorted(excluded),
        "exclude_contexts_of_prior_examples": True,
        "samples": samples,
        "purpose": "Fresh validation after development; never tune on these outputs.",
    }
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print({"cases": len(samples), "prompt_tokens": sum(s["tokens"] for s in samples)})


if __name__ == "__main__":
    main()
