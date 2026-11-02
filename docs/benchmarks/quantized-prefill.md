# Layerwise quantized KV prefill

This opt-in path lowers the memory peak of full-precision dense prefill followed
by KV quantization. It finishes the entire prompt for one layer, converts that
layer to MLX's native `QuantizedKVCache`, and releases its dense cache before
processing the next layer. All prompt tokens remain in the cache.

On an M1 Pro with 32 GiB, Llama 3.1 8B Instruct with 32K input tokens uses
**7.66 GiB peak physical memory instead of 12.83 GiB** for the existing deferred
4-bit policy. Time to first token increases by 3.5%; decode throughput is
essentially unchanged. The 48 paired regression examples produce exactly the
same output tokens as deferred quantization. This is a memory optimization of
that existing policy; quantization itself can still change answers compared
with a full-precision KV cache.

## Usage and scope

```bash
vllm-mlx serve mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --port 8000
```

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    messages=[{
        "role": "user",
        "content": Path("document.txt").read_text() + "\nExplain the main methods.",
    }],
    temperature=0,
    max_tokens=1024,
    stream=True,
    extra_body={"kv_cache_bits": 4},
)
for chunk in response:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
```

`kv_cache_bits` accepts 4 or 8 on both OpenAI chat and completion requests.
Omitting it keeps the existing path. The implementation supports dense Llama
and Qwen3 models with head dimensions divisible by 64. It adds no predictor,
runtime dependency, custom attention kernel, or custom cache type.

Enabled requests require text-only `SimpleEngine`, without continuous batching,
speculative decoding, SpecPrefill, or a rotating cache. They use fresh prompt
caches and bypass prefix reuse. Repeated questions about the same document will
therefore repeat prefill. This restriction matters when comparing interactive
multi-turn workloads. Short prompts can have little memory benefit.

## Isolated Llama 3.1 8B measurements

Hardware: Apple M1 Pro, 10 CPU cores, 32 GiB unified memory. Runtime: MLX 0.32.4,
MLX-LM 0.31.3, macOS 26.6.2. The model weights use 4-bit quantization in every
mode; model-weight precision and KV precision are separate settings. Exact
revisions and asset details are in
[`assets.json`](../../benchmarks/results/layerwise-kv/assets.json).

Each mode processes the same 32,768-token research-article prompt and generates
128 greedy tokens. Performance runs ignore EOS to keep output length fixed.
Model loading and warmup are outside the request timer; TTFT includes cache
conversion. The table contains one isolated run per mode, not confidence
intervals. Quantized baselines use fused large-chunk prefill.

| Measurement | Full KV | Early 4-bit, start=0 | Deferred 4-bit | Layerwise 4-bit |
|---|---:|---:|---:|---:|
| Time to first token | 186.50 s | 207.39 s | 185.67 s | 192.21 s |
| Decode throughput | 17.36 tok/s | 17.34 tok/s | 17.42 tok/s | 17.40 tok/s |
| Whole request, 128 output tokens | 193.82 s | 214.72 s | 192.96 s | 199.51 s |
| Resident KV at first yield | 4.031 GiB | 1.134 GiB | 1.134 GiB | 1.134 GiB |
| MLX peak active allocation | 8.879 GiB | 6.209 GiB | 8.879 GiB | 6.262 GiB |
| Whole-process peak physical footprint | 12.827 GiB | 7.592 GiB | 12.827 GiB | 7.658 GiB |

Relative to deferred 4-bit quantization, peak physical footprint falls by
**40.3%**, TTFT rises by 6.53 seconds, and the short-output request takes 3.4%
longer. Decode changes by -0.14%, within the limits of a single-run measurement.
The 128 generated tokens and first-token log probabilities match exactly.

Early quantization already uses little memory. It is included to show the
tradeoff: the layerwise path is 7.1% faster overall than the measured start=0
configuration, with 0.9% higher physical peak. The start=0 configuration is the
`generate_step` default, **not** the MLX-LM CLI default of 5000. Deferred
quantization uses `quantized_kv_start=prompt length`; layerwise execution keeps
the same dense-prefill computation while converting completed layers earlier.

Physical peak comes from `/usr/bin/time -l` and includes loading and warmup.
MLX active allocation excludes allocator buffers and other process memory.
Existing system swap was not cleared; swap-out deltas are retained in the
[measurement summary](../../benchmarks/results/layerwise-kv/llama-32k-summary.json).
The final standalone benchmark reproduces the earlier native-core measurement's
prompt, output, first-logprob hashes, and KV sizes.

## Quality and numerical agreement

The comparison contains 16 Qasper, 16 HotpotQA, and 16 LCC examples. Qasper and
LCC span 2K-16K input lengths; HotpotQA spans 8K-16K. These examples were reused
to compare implementations after exploring cache policies. They are not new
held-out evidence for a policy selected using earlier results, and they are not
the complete official LongBench benchmark.

Layerwise and ordinary deferred quantization produce identical output-token
hashes and first-logprob hashes in **all 48 cases**. See the
[parity summary](../../benchmarks/results/layerwise-kv/quality-vs-deferred.json).
The underlying 4-bit policy differs from full KV:

| Metric | Full KV | Deferred / layerwise 4-bit | Change |
|---|---:|---:|---:|
| Qasper word F1 | 52.69 | 52.69 | 0.00 pp |
| HotpotQA word F1 | 58.31 | 56.44 | -1.88 pp |
| LCC first-line similarity | 35.96 | 37.95 | +1.98 pp |
| Equal-weight mean of these scores | 48.99 | 49.03 | +0.04 pp |

The overall paired bootstrap 95% interval is [-1.61, +1.34] points. In
`hotpotqa-65`, full KV answers "Vernon L. Smith"; both quantized implementations
add another person, reducing word F1 from 0.8 to 0.5. This remains a real
regression. Code string similarity is not functional correctness, and its gains
must not be interpreted as better working code. All outputs and individual
changes are in the [full-KV comparison](../../benchmarks/results/layerwise-kv/quality-vs-full.json).

The 48 examples contain 47 distinct contexts because two Qasper questions share
a paper. Bootstrap intervals resample questions, not document clusters. Passing
this small screen does not establish universal non-inferiority. Four-bit model
weights are used throughout; these measurements do not establish the same
tradeoff for every checkpoint or KV bit width.

## Implementation and validation

The decoder processes a layer's prompt in 2048-token chunks, with the final
prompt token evaluated separately to match MLX-LM's generation loop. It retains
that layer's hidden outputs for the next layer, then quantizes its completed KV
cache. MLX-LM samples from a separate call for the last prompt token; the adapter
replays its already computed hidden state once. Subsequent decoding uses the
original decoder and native quantized caches. The request-local decoder is
restored on completion, errors, and cancellation.

There are 121 passing focused/lifecycle tests. Thirty-two numerical combinations
cover Llama/Qwen3, FP16/FP32, 4/8-bit KV, single-token and multi-chunk prompts;
they compare every generated logprob vector, token, cache array, offset, and
processor input against ordinary deferred quantization. The full offline suite
has 3881 passes; its 14 failures and three setup errors exactly match the
untouched base. Ruff and Black pass; the same mypy command reports 722 existing
errors on both base and candidate, with no additions.

Real-checkpoint HTTP checks cover both endpoints, streaming and non-streaming,
invalid options, rejection before SSE headers, disconnect, and the following
ordinary request. Both base and candidate briefly return the existing
`text_generation_busy` 503 during disconnect cleanup, then recover in about
0.71 seconds with unchanged ordinary output. See
[`verification.json`](../../benchmarks/results/layerwise-kv/verification.json).

## Reproduction

Download the pinned model and LongBench `data.zip` revisions from `assets.json`.
From this checkout, with its dependencies installed:

```bash
HF_HUB_OFFLINE=1 python benchmarks/bench_quantized_prefill.py \
  --modes full quant4_deferred quant4_layerwise \
  --prompt-tokens 32768 --generation-tokens 128 > performance.jsonl

HF_HUB_OFFLINE=1 python benchmarks/bench_quantized_prefill.py \
  --modes quant4 --quantized-kv-start 5000 \
  --prompt-tokens 32768 --generation-tokens 128 > early5000.jsonl

HF_HUB_OFFLINE=1 python benchmarks/bench_quantized_prefill.py \
  --manifest benchmarks/results/layerwise-kv/validation_manifest.json \
  --modes full quant4_deferred quant4_layerwise > quality.jsonl

python benchmarks/summarize_cache_quality.py quality.jsonl \
  --manifest benchmarks/results/layerwise-kv/validation_manifest.json \
  --output quality-summary.json
```

Run GPU measurements sequentially. Use separate `/usr/bin/time -l` processes
for per-mode physical peaks. The [evidence index](../../benchmarks/results/layerwise-kv/README.md)
records provenance and the separate larger-model evaluation. Earlier KVzap
research, including its failed quality gate, is retained in
[fork PR #2](https://github.com/NageNalock/vllm-mlx/pull/2).
