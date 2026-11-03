# Layerwise quantized KV prefill

The original study is preserved below. The [review rerun](../../benchmarks/results/layerwise-kv/review-rerun.json) checks the final request-local decoder, the existing server chunk-size option, and the current test suites. The current decoder shares weights without replacing the loaded model's decoder.

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

## Serving a larger model in the smaller model's measured memory footprint

On a 32 GiB M1 Pro, we gave 4-bit-weight Qwen3 models the same **22,817 actual
input tokens** from a pinned revision of vllm-mlx. The task was to diagnose a
second request reaching a busy `SimpleEngine`, including its exception class,
HTTP behavior, rejection counter and queue setting. All seven fields have
source-derived answers. We verified the key by exercising the base engine's
lock and HTTP exception handler. The pinned tokenizers produced identical input
IDs across model sizes and all server calls reported the same token count.
Each fresh server handled one warmup followed by one measured streaming chat
request, with thinking disabled, greedy decoding and EOS enabled.

The original target required 8B to stay below the **lower of two default 4B
server peaks**, deliver first content within 300 seconds, generate at least
5 tokens/s and return all seven typed fields in a single correct JSON object.
The first six real HTTP runs used forward and reverse order. After observing
the 4-bit answer regression, we registered the 8-bit precision follow-up and
the next 8B-to-14B screen with the same input and gates. The following table
reports every server configuration. Ranges are the two observed values after
rounding. The process peak includes server startup and model loading. Each
request timer starts after the short warmup.

| Model | KV policy | Runs | Process peak (GiB) | First content (s) | Decode (tok/s) | Exact fields |
|---|---|---:|---:|---:|---:|---:|
| 4B | Default | 2 | 9.08 | 112.4–112.6 | 24.99–25.11 | 2/7 |
| 8B | Default | 2 | 11.22 | 164.9–165.0 | 19.16–19.46 | 4/7 |
| 8B | Layerwise 4-bit | 2 | 7.05 | 170.7–171.0 | 17.59–18.17 | 2/7 |
| 8B | Layerwise 8-bit | 2 | 7.84–7.85 | 171.2 | 16.69–17.13 | 4/7 |
| 14B | Default | 1 | 15.41 | 287.0 | 10.06 | 4/7 |
| 14B | Layerwise 4-bit | 1 | 10.79 | 294.9 | 11.67 | 4/7 |
| 14B | Layerwise 8-bit | 1 | 11.58 | 295.4 | 11.09 | 4/7 |

**4B to 8B.** With 8-bit KV, the larger model's maximum measured server peak
was 7.85 GiB, below the 9.08 GiB default 4B reference. Both runs reproduced
the full-KV 8B answer byte for byte. The default 4B identified the exception
and HTTP status. The 8B answer also identified `text_generation_busy` and
correctly distinguished the busy response from the rate limiter's `Retry-After`
header. It still missed the requested counter, exact environment-variable name
and queue value. The 4-bit KV setting lowered peak further but misdiagnosed the
busy condition as an engine shutdown. The original seven-field gate failed.

**8B to 14B.** The 14B 4-bit KV screen ran below the 11.22 GiB default 8B
reference and met both speed thresholds. It scored four fields, including the
wait setting, but returned the wrong machine error and header. Its counter
included `self.` where the prompt requested just the attribute name. The 14B
8-bit KV run reproduced the default 14B text and exceeded the reference by
0.36 GiB. Neither 14B candidate triggered the registered confirmation runs.
Every default-size server finished on the actual 32 GiB machine. The measured
crossover describes a process footprint available to an application.

**Matched computation.** The direct 8B layerwise 4-bit run peaked at 6.95 GiB.
Ordinary deferred 4-bit quantization on the *smaller* 4B model peaked at 8.94
and 8.84 GiB with chunks of 2048 and 128. The corresponding 8B deferred peaks
were 11.08 and 11.04 GiB. With the matched 2048-token chunks, layerwise and
deferred 8B produced identical output-token and first-logprob hashes and
identical cache sizes at both KV precisions. Deferring quantization also
reproduced the 4-bit diagnosis error. The tested 4-bit early-quantization
alternatives reached 7.78 GiB (start 5000, chunk 2048) and 7.02 GiB (start 0,
chunk 128). Both fit inside the direct 4B default reference of 8.94 GiB.

The 8-bit direct controls isolate the cost of preserving dense prefill. After
seeing the early 2048-token result, we also tested both early thresholds with
chunks of 128.

| 8B direct KV policy | Chunk | Peak (GiB) | First token (s) | Decode (tok/s) | Exact fields |
|---|---:|---:|---:|---:|---:|
| Deferred 8-bit | 2048 | 11.08 | 165.4 | 17.08 | 4/7 |
| Layerwise 8-bit | 2048 | 7.70 | 170.9 | 17.06 | 4/7 |
| Early 8-bit, start 5000 | 2048 | 9.37 | 168.8 | 17.09 | 3/7 |
| Early 8-bit, start 5000 | 128 | 8.58 | 199.0 | 17.03 | 3/7 |
| Early 8-bit, start 0 | 128 | 8.58 | 200.2 | 17.09 | 3/7 |

Both small-chunk early 8-bit controls fit the default 4B direct footprint.
They took 28–29 seconds longer to first token than layerwise 8-bit and each
answered three fields correctly. Layerwise 8-bit reproduced the default 8B
output-token sequence, which contained four correct fields.

The original comparison was published before the main measurements and the
precision follow-up before any 8-bit or 14B audit output.
[Original protocol](../../benchmarks/results/layerwise-kv/model-tier/protocol.json) ·
[Precision addendum](../../benchmarks/results/layerwise-kv/model-tier/precision-addendum.json)

The summary records the decisions and checkpoint hashes. Individual runs retain
the answers, process resource footers and system-pressure observations. The
exact-field scores describe this one incident.
[Summary](../../benchmarks/results/layerwise-kv/model-tier/summary.json) ·
[Individual runs](../../benchmarks/results/layerwise-kv/model-tier/runs.jsonl)

## Qwen3 32B capacity check

A separate, pre-specified check uses Qwen3-32B with 4-bit model weights, a
32,768-token code-context prompt, and 128 fixed output tokens. Thinking is
disabled. The task retrieves four known configuration values; it does not
establish general code-analysis quality. All three modes run sequentially on
the same 32 GiB machine, starting at normal system memory pressure.

| Policy | Result | Physical peak | First token | Decode | Four-fact recovery |
|---|---|---:|---:|---:|---:|
| Early 4-bit, start=5000 | Completed | 22.90 GiB | 1104.24 s | 4.27 tok/s | 4/4 |
| Deferred 4-bit | Stopped at the 20-minute budget | Unavailable | Unavailable | Unavailable | Unavailable |
| Layerwise 4-bit | Completed | 21.91 GiB | 1125.30 s | 3.68 tok/s | 4/4 |

The existing early policy already completes this workload. The layerwise path
therefore does **not** demonstrate a newly enabled model tier. Its 18.8-minute
wait and 3.68 tok/s also fall short of the desired few-minute prefill and fluent
generation. System-wide swap-out increased by 3.86 GiB in the early run and
7.03 GiB in the layerwise run. These are single-run observations with live system
state, not isolated per-process I/O or reliable small latency differences.

The deferred run's last logged progress was 24,576 of 32,768 tokens. The guard
counts loading and warmup, whereas request timings exclude them. It stopped
the benchmark process group after the pre-specified duration; no final
answer or process-peak footer was available. This is censored evidence, not an
OOM result or proof that a longer run could never finish. No memory limits were
changed, swap was not cleared, and unrelated applications were left running.

These are standalone engine measurements. The HTTP smoke uses the Llama
checkpoint. The tokenizer warning retained in the stopped run refers to the
untruncated source pool; inference uses 32,768 tokens, as its progress records
confirm. See the [frozen protocol](../../benchmarks/results/layerwise-kv/large_model_protocol.json),
[raw status and pressure trace](../../benchmarks/results/layerwise-kv/large_model_status.json),
and [measurement summary](../../benchmarks/results/layerwise-kv/qwen32b-32k-summary.json).

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

| Metric | Full KV | Early 4-bit, start=0 | Deferred / layerwise 4-bit | Deferred change vs full |
|---|---:|---:|---:|---:|
| Qasper word F1 | 52.69 | 45.37 | 52.69 | 0.00 pp |
| HotpotQA word F1 | 58.31 | 57.66 | 56.44 | -1.88 pp |
| LCC first-line similarity | 35.96 | 33.63 | 37.95 | +1.98 pp |
| Equal-weight mean of these scores | 48.99 | 45.55 | 49.03 | +0.04 pp |

The early start=0 baseline illustrates why preserving dense prefill can matter:
it has a similarly small cache, but different prefill arithmetic and more answer
changes in this screen. Its comparison is preserved in
[`early0-vs-full.json`](../../benchmarks/results/layerwise-kv/early0-vs-full.json).
This descriptive result does not establish that layerwise quantization beats
every early-quantization threshold or generalizes to other workloads.

For deferred / layerwise 4-bit versus full KV, the overall paired bootstrap 95%
interval is [-1.61, +1.34] points. In
`hotpotqa-65`, full KV answers "Vernon L. Smith"; both quantized implementations
add another person, reducing word F1 from 0.8 to 0.5. This remains a real
regression. Code string similarity is not functional correctness, and its gains
must not be interpreted as better working code. All outputs and individual
changes are in the [full-KV comparison](../../benchmarks/results/layerwise-kv/quality-vs-full.json).

The 48 examples contain 47 distinct contexts because two Qasper questions share
a paper. Bootstrap intervals resample questions, not document clusters. These
observations do not establish universal non-inferiority. Four-bit model
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

There are 137 passing focused/lifecycle tests. Forty-eight numerical combinations
cover Llama/Qwen3, FP16/BF16/FP32, 4/8-bit KV, single-token and multi-chunk prompts;
they compare every generated logprob vector, token, cache array, offset, and
processor input against ordinary deferred quantization. The full offline suite
has 3897 passes; its 14 failures and three setup errors exactly match the
untouched base. Ruff and Black pass; the same mypy command reports 722 existing
errors on both base and candidate, with no additions.

Real-checkpoint HTTP checks cover both endpoints, streaming and non-streaming,
invalid options, rejection before SSE headers, disconnect after the first content
chunk, and the following ordinary request. Both base and candidate briefly return
the existing
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

### Replay the Qwen3 repository audit

The harness reconstructs the user message from Git commit `3af7d7d`. Download
the checkpoint revision specified for the size under test and set
`MODEL_SNAPSHOT` to its local snapshot directory. The 4B and 8B revisions are
in the [original protocol](../../benchmarks/results/layerwise-kv/model-tier/protocol.json).
The [follow-up registration](../../benchmarks/results/layerwise-kv/model-tier/precision-addendum.json)
pins the 14B revision.

Start a fresh server from commit `3af7d7d` for a default run, or from this PR
for a layerwise run. In a second terminal, run the client from this PR using the
matching size and policy. The client performs a short warmup and times the
streaming request through `[DONE]`. Stop the server after the client returns
to obtain the server's lifetime physical-peak footer.

```bash
# Server terminal, in the checkout being measured
/usr/bin/time -l env PYTHONPATH=. HF_HUB_OFFLINE=1 python -m vllm_mlx.cli serve \
  "$MODEL_SNAPSHOT" --served-model-name model-tier --host 127.0.0.1 \
  --port 8100 --timeout 600 --offline

# Client terminal, with an 8B server running the PR
PYTHONPATH=. python benchmarks/bench_model_tier.py 8B quant8_layerwise \
  --url http://127.0.0.1:8100

# Direct control, in a separate process after the server exits
/usr/bin/time -l env PYTHONPATH=. HF_HUB_OFFLINE=1 \
  python benchmarks/bench_model_tier.py 8B quant8 --prefill-step-size 128 \
  --quantized-kv-start 5000
```

Use `full` on the default server and `quant4_layerwise` or `quant8_layerwise`
on the PR server. Without `--url`, the same script measures a direct-engine
control and records output-token and first-logprob hashes. Compare process
footprints within each transport. HTTP peak belongs to the server process.

The measured harness revisions are `3fe6917` for the original comparison and
`eac9abb` for the precision follow-up. The current offline helper limits its
snapshot lookup to JSON and safetensors files. The summary records an initial
setup attempt that exited before model load when ancillary files were absent.
