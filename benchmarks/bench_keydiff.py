# SPDX-License-Identifier: Apache-2.0
"""Compare real MLX generation with full, pruned, and quantized prompt caches.

Example (JSONL on stdout; model download/logging on stderr):
    python benchmarks/bench_keydiff.py --model mlx-community/Qwen3-4B-4bit \
        --lengths 4096 --budget 1024 --task decode --repeats 3 > decode.jsonl

Use --task retrieval to check a synthetic passkey at several context depths.
This is a smoke test, not a substitute for a long-context quality benchmark.
--prompts accepts JSONL with {"id": ..., "prompt": ..., "answers": [...]} for
additional workloads. Prompts are rendered with the model's chat template.
"""

import argparse
import gc
import json
import platform
import random
import re
import string
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.sample_utils import make_sampler

from vllm_mlx.keydiff import KeyDiffKVCache, make_keydiff_cache


def answer_f1(prediction, answers):
    """Maximum normalized English token F1 over accepted reference answers."""

    def normalize(text):
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return re.sub(r"\b(a|an|the)\b", " ", text).split()

    predicted = normalize(prediction)
    scores = []
    for answer in answers:
        expected = normalize(answer)
        common = sum((Counter(predicted) & Counter(expected)).values())
        scores.append(
            2 * common / (len(predicted) + len(expected))
            if predicted or expected
            else 1.0
        )
    return max(scores) if scores else None


class PlainKeyDiffCache(KeyDiffKVCache):
    """Ablation: score the entire prompt without protecting its question tail."""

    recent_tokens = 0


class TailCache(KeyDiffKVCache):
    """Same one-shot byte budget, retaining four sinks plus recent prompt rows."""

    def _compact(self):
        if self.offset != self.prompt_length or self._cache.size() <= self.budget:
            return
        keys, values = self._cache.state
        keep = min(4, self.budget)
        indices = mx.concatenate(
            [mx.arange(keep), mx.arange(self.offset - self.budget + keep, self.offset)]
        )
        self._cache.state = keys[:, :, indices], values[:, :, indices]
        mx.eval(self._cache.state)


def render(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def synthetic_case(tokenizer, length, depth, task):
    rng = random.Random(42)
    lines = []
    while len(tokenizer.encode("\n".join(lines))) < length:
        i = len(lines)
        lines.append(
            f"Archive entry {i}: parcel {rng.randrange(100000, 999999)} was checked "
            f"at the {rng.choice(['harbor', 'warehouse', 'station', 'office'])}. "
            "The inspection found no damage. The record was filed for later review."
        )
    filler = tokenizer.encode("\n".join(lines))[: max(1, length - 100)]
    cut = int(len(filler) * depth)
    needle = "\nThe secret access code for project Meridian is 583921.\n"
    text = tokenizer.decode(filler[:cut]) + needle + tokenizer.decode(filler[cut:])
    if task == "retrieval":
        instruction = "What is the secret access code for project Meridian? Reply with only the code."
    else:
        instruction = "Write a detailed 1000-word report about how these archive entries should be inspected and organized."
    return {
        "id": f"{task}-{length}-{depth}",
        "prompt": text + "\n\n" + instruction,
        "answers": ["583921"] if task == "retrieval" else [],
    }


def run(model, tokenizer, tokens, mode, budget, max_tokens, prefill_step_size):
    if mode == "keydiff":
        caches = make_keydiff_cache(model, budget, len(tokens))
    elif mode == "plain-keydiff":
        caches = [PlainKeyDiffCache(budget, len(tokens)) for _ in model.layers]
    elif mode == "tail":
        caches = [TailCache(budget, len(tokens)) for _ in model.layers]
    elif mode == "rotating":
        caches = [RotatingKVCache(budget, keep=4) for _ in model.layers]
    else:
        caches = [KVCache() for _ in model.layers]
    kwargs = {"kv_bits": int(mode[-1])} if mode.startswith("kv") else {}
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    start = time.perf_counter()
    first_time = None
    text = []
    stream = stream_generate(
        model,
        tokenizer,
        prompt=tokens,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=0),
        prompt_cache=caches,
        prefill_step_size=prefill_step_size,
        **kwargs,
    )
    for response in stream:
        if first_time is None:
            first_time = time.perf_counter()
            first_kv_bytes = sum(c.nbytes for c in caches)
            first_active_bytes = mx.get_active_memory()
        text.append(response.text)
    elapsed = time.perf_counter() - start
    ttft = first_time - start
    count = response.generation_tokens
    result = dict(
        mode=mode,
        kv_budget=(
            budget if mode in {"keydiff", "plain-keydiff", "tail", "rotating"} else None
        ),
        keydiff_recent_tokens={
            "keydiff": KeyDiffKVCache.recent_tokens,
            "plain-keydiff": 0,
        }.get(mode),
        prompt_tokens=len(tokens),
        generated_tokens=count,
        ttft_s=ttft,
        decode_tps=(count - 1) / (elapsed - ttft) if count > 1 else None,
        total_s=elapsed,
        kv_bytes_first=first_kv_bytes,
        kv_bytes_end=sum(c.nbytes for c in caches),
        active_bytes_first=first_active_bytes,
        peak_bytes=mx.get_peak_memory(),
        text="".join(text),
    )
    del caches, stream, response
    gc.collect()
    mx.clear_cache()
    result["active_bytes_after_release"] = mx.get_active_memory()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["full", "keydiff", "plain-keydiff", "tail", "kv4", "kv8", "rotating"],
        default=["full", "keydiff", "tail", "kv4", "kv8"],
    )
    parser.add_argument("--lengths", nargs="+", type=int, default=[4096, 8192])
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--depths", nargs="+", type=float, default=[0.1, 0.5, 0.9])
    parser.add_argument("--task", choices=["decode", "retrieval"], default="retrieval")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--prompts", type=Path)
    args = parser.parse_args()
    model, tokenizer = load(args.model)
    print(
        json.dumps(
            {
                "environment": {
                    "model": args.model,
                    "platform": platform.platform(),
                    "mlx": version("mlx"),
                    "mlx-lm": version("mlx-lm"),
                },
                "arguments": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
            }
        ),
        flush=True,
    )
    # Warm each policy once before measuring model execution and compilation.
    warmup = render(tokenizer, "Describe the purpose of a warehouse.")
    for mode in args.modes:
        run(
            model,
            tokenizer,
            warmup,
            mode,
            min(args.budget, 8),
            8,
            args.prefill_step_size,
        )
    if args.prompts:
        cases = [
            json.loads(line)
            for line in args.prompts.read_text().splitlines()
            if line.strip()
        ]
    else:
        depths = args.depths if args.task == "retrieval" else [0.5]
        cases = [
            synthetic_case(tokenizer, length, depth, args.task)
            for length in args.lengths
            for depth in depths
        ]
    for case in cases:
        tokens = render(tokenizer, case["prompt"])
        for repeat in range(args.repeats):
            # Rotate policy order to reduce systematic warmup/thermal bias.
            modes = (
                args.modes[repeat % len(args.modes) :]
                + args.modes[: repeat % len(args.modes)]
            )
            for mode in modes:
                result = run(
                    model,
                    tokenizer,
                    tokens,
                    mode,
                    args.budget,
                    args.max_tokens,
                    args.prefill_step_size,
                )
                result.update(id=case["id"], repeat=repeat)
                answers = case.get("answers", [])
                result["answer_match"] = (
                    any(a.lower() in result["text"].lower() for a in answers)
                    if answers
                    else None
                )
                result["answer_f1"] = answer_f1(result["text"], answers)
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
