# SPDX-License-Identifier: Apache-2.0
"""Run the registered code audit through SimpleEngine HTTP or a direct KV control."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from cache_memory import system_state

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks/results/layerwise-kv/model-tier/protocol.json"


def messages(protocol):
    sections = [protocol["instruction"]]
    sources = {}
    for name, start, end in protocol["source_excerpts"]:
        if name not in sources:
            sources[name] = subprocess.check_output(
                ["git", "show", f"{protocol['base_commit']}:{name}"],
                cwd=ROOT,
                text=True,
            ).splitlines()
        code = "\n".join(sources[name][start - 1 : end])
        sections.append(f"{name} (lines {start}-{end})\n```python\n{code}\n```")
    sections.append(protocol["question_text"])
    return [{"role": "user", "content": "\n\n".join(sections)}]


def grade(text, expected):
    try:
        answer = json.loads(text)
        strict = isinstance(answer, dict)
    except ValueError:
        strict = False
        try:
            answer, _ = json.JSONDecoder().raw_decode(text[text.index("{") :])
        except ValueError:
            answer = None
    answer = answer if isinstance(answer, dict) else {}
    correct = [
        key
        for key, value in expected.items()
        if type(answer.get(key)) is type(value) and answer[key] == value
    ]
    return {
        "correct_fields": correct,
        "strict_exact_answer": strict
        and set(answer) == set(expected)
        and len(correct) == len(expected),
    }


def http_measure(url, mode, prompt):
    import httpx

    options = {
        "quant4_layerwise": {"kv_cache_bits": 4},
        "quant8_layerwise": {"kv_cache_bits": 8},
    }.get(mode, {})
    body = dict(
        model="model-tier",
        enable_thinking=False,
        temperature=0,
        top_p=1,
        repetition_penalty=1,
        max_tokens=128,
        **options,
    )
    with httpx.Client(base_url=url, timeout=600, trust_env=False) as client:
        warm = client.post(
            "/v1/chat/completions",
            json={
                **body,
                "messages": [{"role": "user", "content": "Say ready."}],
                "max_tokens": 8,
            },
        )
        warm.raise_for_status()
        before = system_state()
        started = time.perf_counter()
        first = None
        parts = []
        usage = None
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                **body,
                "messages": prompt,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk["choices"]:
                    content = choice["delta"].get("content")
                    if content:
                        if first is None:
                            first = time.perf_counter()
                        parts.append(content)
            else:
                raise RuntimeError("SSE closed before [DONE]")
        done = time.perf_counter()
        assert first is not None and usage is not None
        return {
            "ttft_s": first - started,
            "total_s": done - started,
            "decode_tps": (usage["completion_tokens"] - 1) / (done - first),
            "prompt_tokens": usage["prompt_tokens"],
            "generation_tokens": usage["completion_tokens"],
            "text": "".join(parts),
            "system_before": before,
            "system_after": system_state(),
        }


def direct_measure(repo, revision, prompt, args):
    from huggingface_hub import snapshot_download
    from mlx_lm import load
    from mlx_lm.models import base
    from bench_quantized_prefill import fused_quantized_baseline, measure

    base.quantized_scaled_dot_product_attention = fused_quantized_baseline
    snapshot = snapshot_download(repo, revision=revision, local_files_only=True)
    model, tokenizer = load(snapshot)
    warmup = tokenizer.encode("This is a warmup for a long document analysis. " * 32)
    measure(model, tokenizer, warmup, args.mode, args, 8, False)
    tokens = tokenizer.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    result, _ = measure(model, tokenizer, tokens, args.mode, args, 128, True)
    return result


def main():
    protocol = json.loads(PROTOCOL.read_text())
    addendum = json.loads(PROTOCOL.with_name("precision-addendum.json").read_text())
    models = protocol["models"] | addendum["models"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("size", choices=models)
    parser.add_argument(
        "mode",
        choices=(
            "full",
            "quant4",
            "quant4_deferred",
            "quant4_layerwise",
            "quant8",
            "quant8_deferred",
            "quant8_layerwise",
        ),
    )
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--quantized-kv-start", type=int, default=5000)
    parser.add_argument(
        "--url", help="Base URL of a fresh server with --served-model-name model-tier"
    )
    args = parser.parse_args()
    prompt = messages(protocol)
    repo, revision = models[args.size]
    if args.url:
        if args.mode not in ("full", "quant4_layerwise", "quant8_layerwise"):
            parser.error("Use the direct harness for MLX-LM quantization controls")
        result = http_measure(args.url, args.mode, prompt)
    else:
        result = direct_measure(repo, revision, prompt, args)
    result.update(grade(result["text"], protocol["expected"]))
    result.update(
        event="audit",
        model=repo,
        revision=revision,
        size=args.size,
        mode=args.mode,
        transport="http" if args.url else "direct",
        user_message_sha256=hashlib.sha256(prompt[0]["content"].encode()).hexdigest(),
        answer_text_sha256=hashlib.sha256(result["text"].encode()).hexdigest(),
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
