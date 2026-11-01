# KeyDiff on M1 Pro: development results

This prototype reduces resident decode KV memory and retains MLX's normal
attention path. **It does not preserve full-cache answer quality in this
development sample.** Keep it opt-in and experimental; these results do not
justify enabling it by default or presenting it as a quality-preserving
optimization upstream.

## Environment

- November 2, 2026; M1 Pro, 32 GB unified memory, AC power, macOS 26.6.2.
- Python 3.12.15, MLX 0.32.4, mlx-lm 0.31.3; base commit
  [`3af7d7d`](https://github.com/waybarrios/vllm-mlx/commit/3af7d7d).
- Qwen3-4B-4bit revision `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`;
  Llama-3.2-1B-Instruct-4bit revision `08231374eeacb049a0eade7922910865b8fce912`,
  both from `mlx-community`. Weight quantization is unchanged across policies.
- Greedy decoding, thinking disabled, prefill chunks of 2048 tokens. Each
  policy receives a short warmup. Timed runs are sequential; the test suite
  and HTTP test server are stopped during performance measurements.
- [Measurement records](results/keydiff-m1-pro.jsonl) include generated text,
  individual timings, memory counters, configuration, and quality scores.
  Environment records start each experiment; every record has an `experiment`
  field. The earlier unprotected prototype is labeled `plain-keydiff` on
  export; its measured values and generated text are unchanged.

## Decode performance

The Qwen3 prompt has 4048 tokens and requests a long archive report; every
measured run generates 128 tokens. Reported times and throughput are medians
of three runs. The 2048-row variant is measured in a separate three-run sweep.
The 128-row protected tail is included in the KeyDiff budget.

| Policy | Prompt budget | KV MiB | TTFT s | Decode tok/s | Total s | Peak Metal MiB |
|---|---:|---:|---:|---:|---:|---:|
| Full KV | — | 576 | 12.359 | 48.16 | 14.996 | 3479.2 |
| 8-bit KV | — | 306 | 13.802 | 44.35 | 16.858 | 3882.4 |
| 4-bit KV | — | 162 | 13.923 | 41.61 | 16.783 | 4222.3 |
| Four sinks + tail | 1024 | 180 | 12.399 | 55.90 | 14.671 | 3479.2 |
| KeyDiff + recent 128 | 1024 | 180 | 12.595 | 55.19 | 14.966 | 3479.2 |
| KeyDiff + recent 128 | 2048 | 324 | 13.653 | 46.85 | 16.364 | 3479.2 |

KV bytes are allocated key/value buffers at the first yielded token, including
allocation padding and mlx-lm's decode lookahead. Peak/active bytes are Metal
memory counters, not process RSS. The full-prefill peak is unchanged by this
post-prefill scheme. A lower KV allocation does not mean a proportionally
lower total model footprint, TTFT, or end-to-end latency.

Qwen3 active Metal memory returned to **2159.78 MiB after every measured cache release**. The separate Llama 1B decode smoke run used 4065 input tokens and 128 output tokens: full KV allocated 128 MiB and decoded at 124.31 tok/s; KeyDiff allocated 40 MiB and decoded at 165.49 tok/s. That Llama comparison has only one run per policy. Thermal conditions were not controlled; the later 2048-row sweep should not be treated as a tightly paired timing comparison.

## Question answering: eight development cases

We use the first eight Qasper rows from
[LongBench](https://huggingface.co/datasets/zai-org/LongBench), with complete
articles, a concise-answer prompt, and a 128-token output limit. Rendered
inputs contain 2959–6556 tokens. Scores are the mean of normalized English
token F1, taking the best reference answer, on a 0–100 scale.

| Policy | Prompt row budget | Mean F1 |
|---|---:|---:|
| Full KV | Unbounded | 25.20 |
| 8-bit KV, group size 64 | Unbounded | 25.20 |
| 4-bit KV, group size 64 | Unbounded | 13.85 |
| KeyDiff + recent 128 | 2048 | 18.60 |
| KeyDiff + recent 128 | 1024 | 9.23 |
| Four sinks + tail, one-shot | 1024 | 7.56 |
| Plain KeyDiff, no tail protection | 1024 | 1.23 |

The unprotected prototype exposed an integration problem: kvpress's
[QA pipeline](https://github.com/NVIDIA/kvpress/blob/main/kvpress/pipeline.py)
compresses context before appending the question. A combined chat prompt can
lose question/instruction KV if every row is eligible for eviction. Protecting
128 recent rows helps here, and increasing the budget helps further, but
neither setting reaches full KV or 8-bit KV quality on this sample.

These eight examples were used during development, including the decision to
protect a recent window. They are **not a held-out evaluation or an official
LongBench result**. The full-cache model itself scores poorly. This sample
cannot establish general QA, coding, summarization, or multi-turn quality.
Larger held-out evaluations are needed before an upstream feature proposal.

A separate Llama 1B smoke test placed the same six-digit passkey at 10%, 50%, and 90% of an approximately 4K-token prompt. Full KV and KeyDiff (1024 rows) retrieved it in **3/3** cases; the four-sink plus tail baseline succeeded in **1/3**. This is a small synthetic check, not evidence of broad answer-quality parity.

## Validation

The 37 new tests pass, covering PyTorch/MLX score parity, GQA key/value selection, protected tails, original RoPE positions, physical masks, unchanged inactive-budget generation, KV-quantization rejection, request isolation, stream closure, and API forwarding.

The final full offline suite reports **3875 passed, 28 skipped, 27 deselected, 14 failed, and 3 setup errors**. The 14 failing test IDs and 3 error IDs exactly match an untouched checkout of base `3af7d7d` in the same environment (3838 passed there). These include existing MLX stream failures, missing Node/npm commands, tests requiring an uncached model, and a cache-persistence failure. The full suite is not green; this change introduces no additional failing test IDs.

Ruff and Black pass across the repository code/tests and the new benchmark. Mypy reports the same **722 existing errors** as the base, with no new diagnostic messages after normalizing shifted line numbers. No runtime dependency was added.

The real Qwen3 HTTP server was exercised through streaming and non-streaming
chat/completion endpoints. An inactive budget matched the ordinary greedy
answer. Invalid budgets and incompatible SpecPrefill requests failed before
SSE headers. After a client disconnected, one transient ordinary busy response
preceded successful reuse of the server once cleanup released the request.

## Reproduce

See the [feature guide](../guides/keydiff.md) for API use and cache semantics.

```bash
python benchmarks/bench_keydiff.py --model mlx-community/Qwen3-4B-4bit \
  --lengths 4096 --budget 1024 --task decode --repeats 3 --max-tokens 128
python benchmarks/bench_keydiff.py --model mlx-community/Qwen3-4B-4bit \
  --lengths 4096 --budget 2048 --task decode --repeats 3 --max-tokens 128 \
  --modes keydiff
```

To prepare the QA inputs without redistributing article text:

```python
import json
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    "zai-org/LongBench", "data.zip", repo_type="dataset",
    revision="5e628be450b7e67fb7ae6e201bd6d8f7056f7672",
)
with zipfile.ZipFile(path) as archive:
    name = next(n for n in archive.namelist() if n.endswith("qasper.jsonl"))
    rows = [json.loads(line) for line in archive.read(name).decode().splitlines()]
cases = []
for i, row in enumerate(rows[:8]):
    prompt = (
        "Read the article below and answer the question with a short phrase. "
        "Answer yes/no for a yes/no question, or unanswerable when the article "
        "gives no answer. Output only your answer.\n\nArticle:\n"
        + row["context"] + "\n\nQuestion:\n" + row["input"] + "\n\nAnswer:"
    )
    cases.append({"id": f"qasper-{i}", "prompt": prompt, "answers": row["answers"]})
Path("qasper-workload.jsonl").write_text(
    "".join(json.dumps(case) + "\n" for case in cases)
)
```

```bash
python benchmarks/bench_keydiff.py --model mlx-community/Qwen3-4B-4bit \
  --prompts qasper-workload.jsonl --budget 1024 --max-tokens 128 \
  --modes full plain-keydiff keydiff tail kv4 kv8
python benchmarks/bench_keydiff.py --model mlx-community/Qwen3-4B-4bit \
  --prompts qasper-workload.jsonl --budget 2048 --max-tokens 128 --modes keydiff
python benchmarks/bench_keydiff.py --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --lengths 4096 --budget 1024 --task decode --max-tokens 128
python benchmarks/bench_keydiff.py --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --lengths 4096 --budget 1024 --task retrieval --max-tokens 32 \
  --modes full keydiff tail
```

The dataset ZIP SHA-256 is
`cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64`.
