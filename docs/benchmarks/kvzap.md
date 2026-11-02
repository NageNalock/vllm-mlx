# Experimental KVzap on Apple Silicon

KVzap is an opt-in prompt-cache optimization for a single text request using
Llama 3.1 8B Instruct. It physically removes low-scoring prompt entries from
each KV head, then uses native MLX fused attention for decoding. The candidate
preserves full-precision dense attention during prefill. It finishes the prompt
for one layer before compacting that layer's KV and moving to the next layer.
This lowers prefill peak memory and accelerates decoding, at the cost of some
prefill latency. It does not establish that a larger model fits.

## Scope and usage

The tested checkpoint is
[`mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`](https://huggingface.co/mlx-community/Meta-Llama-3.1-8B-Instruct-4bit).
The predictor is specific to Llama 3.1 8B **Instruct**; matching architecture
alone does not validate base models, other fine-tunes, or other quantizations.
The architecture check rejects incompatible shapes, but does not identify
arbitrary local weights.

```bash
vllm-mlx serve mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --port 8000
```

```python
from openai import OpenAI
from pathlib import Path

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
long_document_and_question = (
    Path("document.txt").read_text() + "\n\nExplain the main methods and limitations."
)
response = client.chat.completions.create(
    model="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    messages=[{"role": "user", "content": long_document_and_question}],
    max_tokens=1024,
    temperature=0,
    stream=True,
    extra_body={"kvzap": True},
)
for chunk in response:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
```

The first enabled request downloads a pinned, approximately 4 MB NVIDIA
predictor. Ordinary requests keep the existing path. Enabled requests require
text-only `SimpleEngine`, without continuous batching, speculative decoding,
SpecPrefill, or a rotating KV cache. They bypass system and prefix-cache reuse.
That tradeoff can be substantial for repeated questions about the same document.

Compression happens once after the entire prompt has been evaluated. Generated
tokens remain unpruned, so their memory use grows with the answer. There is no
automatic threshold tuning, support for cache serialization, or claimed
lossless quality guarantee. Short prompts may save little memory or run slower.

## Measurement method

Hardware: Apple M1 Pro, 10 CPU cores, 32 GiB unified memory; macOS 26.6.2.
Runtime: MLX 0.32.4 and MLX-LM 0.31.3. Model weights use 4-bit quantization.
Predictor threshold is -7, the protected recent window is 128 tokens, and the
prefill chunk size is 2048. Exact revisions and hashes are in
[`model_assets.json`](../../benchmarks/results/kvzap/model_assets.json).

The 4-bit and 8-bit baselines include fused large-chunk prefill, making them
stronger than the stock quantized path in MLX-LM 0.31.3. Each performance run
warms up first, processes 32,768 identical prompt
tokens, and produces exactly 128 tokens with greedy decoding. Fixed-length
timing deliberately ignores EOS; it is separate from the quality evaluation,
which stops at EOS. Tokenization, model loading, and initial predictor download
are outside the request timer. TTFT includes cache compaction.

### Isolated 32K comparison

The prompt contains research-article excerpts and a request to compare their
methods. The source pool is truncated before inference; the actual model input
is 32,768 tokens. These are one isolated run per mode, not confidence intervals.

| Measurement | Full KV | Strong 8-bit KV | Strong 4-bit KV | KVzap |
|---|---:|---:|---:|---:|
| Time to first token | 186.50 s | 210.60 s | 207.39 s | 198.37 s |
| Decode throughput | 17.36 tok/s | 16.03 tok/s | 17.34 tok/s | 21.83 tok/s |
| Whole request, 128 output tokens | 193.82 s | 218.52 s | 214.72 s | 204.18 s |
| Resident KV at first yield | 4.031 GiB | 2.142 GiB | 1.134 GiB | 1.556 GiB |
| MLX peak active allocation | 8.879 GiB | 7.235 GiB | 6.209 GiB | 6.665 GiB |
| Whole-process peak physical footprint | 12.827 GiB | 9.522 GiB | 7.592 GiB | 8.143 GiB |

Relative to full KV, the final integration reduces resident KV by **61.4%** and
physical peak by **36.5%**, and improves decode throughput by **25.7%**. TTFT
increases by 11.86 seconds and the short-output request is 5.3% slower overall.
Relative to 4-bit KV, decoding is 25.9% faster and the request is 4.9% faster,
but physical peak is 7.3% higher and resident KV is about 37% larger. It is a
memory/latency/quality tradeoff, not a claim to beat low-bit KV in every metric.
See the [raw comparison](../../benchmarks/results/kvzap/final_performance_summary.json).

MLX active allocation excludes cached allocator buffers and other process
memory. The physical peak includes loading and warmup and is measured by
`/usr/bin/time -l`; it must not be confused with resident KV bytes. Pre-existing
system swap was not cleared. System swap-out deltas are recorded per run.

Faster decoding does not remove the dense-prefill cost. Longer answers need
separate measurements; multiplying decode speedup by whole-request latency
would be incorrect. Low-bit KV can use less memory than pruning. No larger-model
capacity improvement has been demonstrated.

## Quality and validation

The 12-case development screen compared the revised KVzap with full KV. Eleven
outputs were identical. The mean of the paired QA F1 / code similarity scores
changed by -0.13 percentage points; one code completion changed. This development
set was used during development and is not held-out evidence.

The candidate and threshold were frozen before inspecting a fresh 48-case
validation set: 16 Qasper, 16 HotpotQA, and 16 LCC examples, excluding all
previously selected examples and their document contexts. Qasper and LCC are
stratified across 2K-4K, 4K-8K, and 8K-16K prompt lengths; HotpotQA uses 8K-16K.
Full KV, 4-bit KV, and KVzap use identical prompts and greedy decoding. The
[`protocol`](../../benchmarks/results/kvzap/layerwise_validation_protocol.json) and
[`manifest`](../../benchmarks/results/kvzap/dense_validation_manifest.json) record the
parameters and selection. Results will be inserted after all paired runs finish.
This is a paired regression screen using documented prompts and metrics, not the
full official LongBench benchmark or a proof of universal non-inferiority.

A supplementary 32K code-context retrieval task places four configuration
values at 5%, 25%, 50%, and 75% of the document. It is a small synthetic check,
not evidence that arbitrary code analysis is unaffected. Full KV and the final
integration both recovered all four values. The final run is
[`here`](../../benchmarks/results/kvzap/layerwise-retrieval-32k.jsonl).

### Rejected variants and known failures

An earlier 8-bit-prefill prototype improved 32K decoding by 39.5% and reduced
resident KV by 27.4% relative to the strong 8-bit baseline. However, its complete
16-case Qasper subset lost 6.79 percentage points relative to full KV, exceeding
the pre-specified 3-point gate. That evaluation stopped after 21 paired cases;
it must not be presented as a complete 48-case evaluation. See the
[`stop record`](../../benchmarks/results/kvzap/validation_stop.json) and
[`all completed results`](../../benchmarks/results/kvzap/validation.jsonl).

Attribution matters: for `qasper-131`, both plain 8-bit KV and that prototype
replaced a specific language list with broad language categories. Full-precision
prefill restored the full-KV answer. For `qasper-130`, however, pruning itself
omitted method details and lost 18.17 points of word F1. That failure remains in
the revised candidate. Cache pruning is not lossless, even when the aggregate
score changes little. The earlier variants and their development examples are
not counted as fresh validation for the revised candidate.

An online variant that pruned between prefill chunks was rejected: even with a
protected 2048-token window, its development LCC score fell by 27.39 percentage
points. The shipped candidate retains dense attention throughout prefill.

## Implementation and reproducibility

The method uses NVIDIA's released
[`KVzap linear predictor`](https://huggingface.co/nvidia/KVzap-linear-Llama-3.1-8B-Instruct)
and the scoring convention from
[`kvpress`](https://github.com/NVIDIA/kvpress/blob/2d66e747719a7bf1b9276f2cae6d135b96b92fd7/kvpress/presses/kvzap_press.py).
This is an MLX implementation with physical per-head storage, not a PyTorch
dependency or a claim to have invented the pruning algorithm. The
[KVzap paper](https://arxiv.org/abs/2601.07891) does not validate this combination
of 4-bit model weights, physical MLX caches, and Apple Silicon kernels.

The cache keeps logical RoPE positions separate from retained storage lengths.
Each layer processes all prompt queries densely in chunks, retaining its full
hidden outputs for the next layer. Only then are its retained KV entries gathered
into per-head caches for fused SDPA. This avoids keeping dense prompt KV for all
layers at once. MLX-LM separately samples from the last prompt token; the adapter
replays that token's already computed hidden state once. Request-local decoder
and attention wrappers are restored on normal exit, errors, and cancellation.

Tests cover an independent masked attention reference, exact first-logit parity,
no-pruning continuation, logical positions, processor-input parity, cancellation,
the following ordinary request, prefix-cache bypass, and both HTTP endpoints.
The final integration exactly reproduces all 12 earlier development outputs and
the 32K prototype output. There are 142 passing focused tests and 78 passing
lifecycle tests. The full offline suite has 3858 passes; its 14 failures and three
setup errors exactly match the untouched base. Ruff and Black pass; mypy has no
new diagnostics relative to the base. See
[`verification.json`](../../benchmarks/results/kvzap/verification.json).

Download the pinned base model, predictor, and LongBench `data.zip` revisions
listed in `benchmarks/bench_kvzap.py` before running the offline measurements.
From the repository root, with this checkout installed:

```bash
HF_HUB_OFFLINE=1 python benchmarks/bench_kvzap.py \
  --modes full quant4 quant8 kvzap --prompt-tokens 32768 --generation-tokens 128 \
  > performance.jsonl

HF_HUB_OFFLINE=1 python benchmarks/bench_kvzap.py \
  --manifest benchmarks/results/kvzap/dense_validation_manifest.json \
  --modes full quant4 kvzap > validation.jsonl

python benchmarks/summarize_kvzap.py validation.jsonl \
  --manifest benchmarks/results/kvzap/dense_validation_manifest.json \
  --output validation-summary.json
```

Run GPU measurements sequentially. Earlier runs that briefly overlapped were
excluded and repeated in isolation; their filenames and reasons remain in
[`excluded_runs.json`](../../benchmarks/results/kvzap/excluded_runs.json).
