# SPDX-License-Identifier: Apache-2.0
"""Compare native layerwise KV quantization with full and quantized MLX-LM caches."""

import argparse
import gc
from contextlib import nullcontext
import hashlib
import json
import time
import zipfile
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download
from cache_memory import physical_footprint, system_state
from cache_quality import load_cases, score

from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models import base
from vllm_mlx.quantized_prefill import quantized_prompt_cache

MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
MODEL_REVISION = "241a666dad6cb93c8ff213d39a7f34a36bf26db4"
DATA_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"


_quantized_attention = base.quantized_scaled_dot_product_attention


def fused_quantized_baseline(queries, keys, values, scale, mask, group_size=64, bits=8):
    # Use fused large prefill for the early-quantization baseline.
    if queries.shape[2] >= 512:
        return mx.fast.scaled_dot_product_attention(
            queries * scale,
            mx.dequantize(*keys, group_size=group_size, bits=bits),
            mx.dequantize(*values, group_size=group_size, bits=bits),
            scale=1.0,
            mask=mask,
        )
    return _quantized_attention(
        queries, keys, values, scale=scale, mask=mask, group_size=group_size, bits=bits
    )


def emit(record):
    print(json.dumps(record), flush=True)


def measure(model, tokenizer, tokens, mode, args, max_tokens, stop_at_eos):
    bits = 4 if mode.startswith("quant4") else 8 if mode.startswith("quant8") else None
    context = (
        quantized_prompt_cache(
            model, tokens, bits, prefill_step_size=args.prefill_step_size
        )
        if mode.endswith("_layerwise")
        else nullcontext(make_prompt_cache(model))
    )
    with context as cache:
        result, first = measure_cache(
            model, tokenizer, tokens, mode, args, max_tokens, stop_at_eos, cache
        )
    del cache, context
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    result["active_bytes_after_release"] = mx.get_active_memory()
    return result, first


def measure_cache(model, tokenizer, tokens, mode, args, max_tokens, stop_at_eos, cache):
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    before = system_state()
    started = time.perf_counter()
    first_time = None
    first_logprobs = None
    output = []

    def progress(done, total):
        if done == total or (done > 0 and done % 8192 == 0):
            emit(
                {
                    "event": "progress",
                    "mode": mode,
                    "processed": done,
                    "total": total,
                    "elapsed_s": time.perf_counter() - started,
                }
            )

    generator = generate_step(
        mx.array(tokens),
        model,
        max_tokens=max_tokens,
        prompt_cache=cache,
        prefill_step_size=(
            len(tokens) if mode.endswith("_layerwise") else args.prefill_step_size
        ),
        kv_bits={
            "quant4": 4,
            "quant8": 8,
            "quant4_deferred": 4,
            "quant8_deferred": 8,
        }.get(mode),
        quantized_kv_start=(
            len(tokens) if mode.endswith("_deferred") else args.quantized_kv_start
        ),
        prompt_progress_callback=progress,
    )
    try:
        for token, logprobs in generator:
            now = time.perf_counter()
            if first_time is None:
                first_time = now
                first_logprobs = np.array(logprobs.astype(mx.float32)).astype(
                    np.float64
                )
                first_logprobs -= np.logaddexp.reduce(first_logprobs)
                kv_bytes_first = sum(c.nbytes for c in cache)
                active_bytes_first = mx.get_active_memory()
                footprint_first = physical_footprint()
            if stop_at_eos and token in tokenizer.eos_token_ids:
                break
            output.append(token)
    finally:
        generator.close()
    elapsed = time.perf_counter() - started
    mx.synchronize()
    result = {
        "mode": mode,
        "prompt_tokens": len(tokens),
        "generation_tokens": len(output),
        "ttft_s": first_time - started,
        "decode_tps": (
            (len(output) - 1) / (elapsed - (first_time - started))
            if len(output) > 1
            else None
        ),
        "total_s": elapsed,
        "kv_bytes_first": kv_bytes_first,
        "kv_bytes_end": sum(c.nbytes for c in cache),
        "active_bytes_first": active_bytes_first,
        "physical_footprint_first": footprint_first,
        "physical_footprint_end": physical_footprint(),
        "active_bytes_end": mx.get_active_memory(),
        "peak_bytes": mx.get_peak_memory(),
        "system_before": before,
        "system_after": system_state(),
        "text": tokenizer.decode(output),
        "output_sha256": hashlib.sha256(json.dumps(output).encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        "first_logprobs_sha256": hashlib.sha256(first_logprobs.tobytes()).hexdigest(),
    }
    return result, first_logprobs


def document_tokens(tokenizer, length):
    path = hf_hub_download(
        "zai-org/LongBench",
        "data.zip",
        repo_type="dataset",
        revision=DATA_REVISION,
        local_files_only=True,
    )
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith("/qasper.jsonl"))
        rows = [json.loads(s) for s in archive.read(name).splitlines()][:24]
    prompt = (
        "Read these research articles and summarize their methods and limitations.\n\n"
        + "\n\nNext article:\n\n".join(row["context"] for row in rows)
        + "\n\nWrite a detailed comparison of the methods and their limitations."
    )
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert len(tokens) >= length
    return tokens[: length - 128] + tokens[-128:]


def retrieval_tokens(tokenizer, length):
    path = hf_hub_download(
        "zai-org/LongBench",
        "data.zip",
        repo_type="dataset",
        revision=DATA_REVISION,
        local_files_only=True,
    )
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith("/lcc.jsonl"))
        rows = [json.loads(s) for s in archive.read(name).splitlines()][:48]
    filler = tokenizer.encode(
        "\n\n# Next file\n".join(r["context"] for r in rows), add_special_tokens=False
    )
    assert len(filler) > length
    expected = {
        "alpha": "copper-731",
        "bravo": "violet-286",
        "charlie": "marble-594",
        "delta": "cedar-862",
    }
    sections, previous = [], 0
    for position, (name, value) in zip((0.05, 0.25, 0.50, 0.75), expected.items()):
        stop = int(length * position)
        sections.append(tokenizer.decode(filler[previous:stop]))
        sections.append(
            f'\n\n# Configuration audit record\nAUDIT_LABEL_{name.upper()} = "{value}"\n\n'
        )
        previous = stop
    sections.append(tokenizer.decode(filler[previous:length]))
    prompt = (
        "Review the following source files and configuration records.\n\n"
        + "".join(sections)
        + "\n\nReturn only a JSON object mapping alpha, bravo, charlie, and delta to "
        "the exact values of AUDIT_LABEL_ALPHA, AUDIT_LABEL_BRAVO, "
        "AUDIT_LABEL_CHARLIE, and AUDIT_LABEL_DELTA from the records above."
    )
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert len(tokens) >= length
    return tokens[: length - 256] + tokens[-256:], expected


def retrieval_score(text, expected):
    try:
        answer, _ = json.JSONDecoder().raw_decode(text[text.index("{") :])
    except (ValueError, json.JSONDecodeError):
        return 0.0
    if not isinstance(answer, dict):
        return 0.0
    return sum(answer.get(k) == v for k, v in expected.items()) / len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=(
            "full",
            "quant4",
            "quant8",
            "quant4_deferred",
            "quant8_deferred",
            "quant4_layerwise",
            "quant8_layerwise",
        ),
        default=["quant4_deferred", "quant4_layerwise"],
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--task", choices=("articles", "retrieval"), default="articles")
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--quantized-kv-start", type=int, default=0)
    parser.add_argument("--trials", type=int, default=1)
    args = parser.parse_args()
    revision = args.revision or (MODEL_REVISION if args.model == MODEL else None)
    if revision is None:
        parser.error("Specify --revision for a non-default model")
    base.quantized_scaled_dot_product_attention = fused_quantized_baseline
    source = Path(__file__).resolve().parents[1]
    emit(
        {
            "event": "metadata",
            "arguments": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "model": args.model,
            "revision": revision,
            "mlx": version("mlx"),
            "mlx_lm": version("mlx-lm"),
            "quantized_baseline": "4/8-bit cache with fused large prefill",
            "source_sha256": {
                p: hashlib.sha256((source / p).read_bytes()).hexdigest()
                for p in [
                    "vllm_mlx/quantized_prefill.py",
                    "vllm_mlx/models/llm.py",
                    "benchmarks/bench_quantized_prefill.py",
                    "benchmarks/cache_quality.py",
                    "benchmarks/cache_memory.py",
                ]
            },
            "device": mx.device_info(),
        }
    )
    model_path = snapshot_download(
        args.model,
        revision=revision,
        local_files_only=True,
        allow_patterns=["*.json", "*.safetensors"],
    )
    model, tokenizer = load(model_path)
    emit(
        {
            "event": "loaded",
            "active_bytes": mx.get_active_memory(),
            "system": system_state(),
        }
    )
    warmup = tokenizer.encode("This is a warmup for a long document analysis. " * 32)
    for mode in args.modes:
        result, _ = measure(model, tokenizer, warmup, mode, args, 8, False)
        emit({"event": "warmup", **result})

    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        cases = list(load_cases(manifest))
        if args.case_ids:
            cases = [c for c in cases if c["id"] in args.case_ids]
            assert len(cases) == len(args.case_ids)
        for index, case in enumerate(cases):
            tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": case["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            logprobs = {}
            modes = (
                args.modes[index % len(args.modes) :]
                + args.modes[: index % len(args.modes)]
            )
            for mode in modes:
                result, logprobs[mode] = measure(
                    model,
                    tokenizer,
                    tokens,
                    mode,
                    args,
                    case["max_tokens"],
                    True,
                )
                emit(
                    {
                        "event": "quality",
                        "id": case["id"],
                        "dataset": case["dataset"],
                        "answers": case["answers"],
                        "score": score(result["text"], case),
                        **result,
                    }
                )
            if "full" in logprobs:
                reference = logprobs["full"]
                emit(
                    {
                        "event": "first_token_kl",
                        "id": case["id"],
                        "kl": {
                            mode: float(np.sum(np.exp(reference) * (reference - p)))
                            for mode, p in logprobs.items()
                        },
                    }
                )
    else:
        expected = None
        if args.task == "retrieval":
            tokens, expected = retrieval_tokens(tokenizer, args.prompt_tokens)
        else:
            tokens = document_tokens(tokenizer, args.prompt_tokens)
        for trial in range(args.trials):
            modes = (
                args.modes[trial % len(args.modes) :]
                + args.modes[: trial % len(args.modes)]
            )
            for mode in modes:
                result, _ = measure(
                    model,
                    tokenizer,
                    tokens,
                    mode,
                    args,
                    args.generation_tokens,
                    False,
                )
                emit(
                    {
                        "event": "performance",
                        "trial": trial,
                        "retrieval_expected": expected,
                        "retrieval_score": (
                            retrieval_score(result["text"], expected)
                            if expected
                            else None
                        ),
                        **result,
                    }
                )


if __name__ == "__main__":
    main()
