# Experimental KeyDiff prompt KV pruning

`keydiff_kv_budget` retains a selected number of prompt KV rows per head after
prefill, using the scoring rule from
[NVIDIA kvpress's KeyDiffPress](https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/keydiff_press.py).
The implementation uses MLX operations and the existing mlx-lm generation loop;
kvpress and PyTorch are not required by this feature at runtime.

This is lossy compression, disabled by default. Validate answer quality on your
own prompts before using it. The supported initial scope is **text-only
SimpleEngine with full-attention Llama or dense Qwen3**, using mlx-lm 0.31.3.
Continuous batching, multimodal/hybrid/sliding-window models, speculative
decoding, SpecPrefill, and combining pruning with KV quantization are outside
this version's scope. Incompatible server modes/models return HTTP 400 before
streaming begins; invalid budgets return HTTP 422.

See the [M1 Pro development results](../benchmarks/keydiff-m1-pro.md) for measured
memory/latency benefits and the observed answer-quality losses.

## Use

Start a normal single-request server:

```bash
vllm-mlx serve mlx-community/Qwen3-4B-4bit --port 8000
```

Both OpenAI `/v1/chat/completions` and `/v1/completions` accept the optional
positive integer `keydiff_kv_budget`, in streaming and non-streaming requests:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="mlx-community/Qwen3-4B-4bit",
    messages=[{"role": "user", "content": long_document_and_question}],
    temperature=0,
    max_tokens=256,
    extra_body={"keydiff_kv_budget": 1024, "enable_thinking": False},
)
```

Omit the parameter to use the normal cache. It can also be passed to
`SimpleEngine.generate/chat/stream_generate/stream_chat` or
`MLXLanguageModel.generate/stream_generate`. KeyDiff requests bypass the system
prefix cache and prefix trie; other requests retain their existing behavior.
Caches are owned by each generation and released on completion or stream close.

## Semantics and limits

All prompt tokens are processed with the full cache. Immediately before the
first decode forward pass, **the last 128 prompt rows are protected**. Each
layer scores the earlier prefix's post-RoPE keys by negative cosine similarity
to that prefix's mean normalized key, accumulating in float32. It fills the
remaining budget with the highest-scoring rows independently for each KV head,
then gathers keys and values in chronological order. The protected window is
included in the budget; budgets of 128 or less retain only that many recent
rows. There is no learned calibration or iterative BlockPress step.

This is KeyDiff scoring with a recent-window policy, not a reproduction of every
variant in the paper or kvpress's integration. In particular,
[kvpress's QA pipeline](https://github.com/NVIDIA/kvpress/blob/main/kvpress/pipeline.py)
separates context compression from question generation. A generic chat server
receives a combined prompt; the fixed window is a heuristic, not an exact
context/question boundary. Long questions can still extend beyond it.

Logical token positions continue to advance through the original sequence for
RoPE; attention masks and physical writes use the number of retained rows.
Generated tokens are appended without further pruning. A budget of 1024 thus
means **1024 retained prompt rows plus generated rows**, with mlx-lm's ordinary
allocation padding. It is not a hard total-memory limit. Prompts that already
fit the budget are left intact, and the first predicted token uses the full
prompt attention result.

This reduces **resident decode KV memory**, not full-prefill peak memory. It
cannot make a prompt fit if full prefill already runs out of memory. Scoring
and gathering cost time and temporary memory. mlx-lm evaluates a decode step
ahead of its first yield, so that cost appears in measured TTFT. Benefits depend
on model, context length, budget, and generation length; latency improvements
must be measured rather than inferred from the retained-row ratio.

## Reproduce comparisons

```bash
# Decode performance: report medians/ranges across repetitions.
python benchmarks/bench_keydiff.py --model mlx-community/Qwen3-4B-4bit \
  --lengths 4096 8192 --budget 1024 --task decode --repeats 3 \
  --max-tokens 128 > decode.jsonl

# Synthetic retrieval at early, middle, and late context positions.
python benchmarks/bench_keydiff.py --model mlx-community/Qwen3-4B-4bit \
  --lengths 4096 8192 --budget 1024 --task retrieval \
  --depths 0.1 0.5 0.9 --max-tokens 32 > retrieval.jsonl

# Supply additional prompts and accepted answers from your own workload.
python benchmarks/bench_keydiff.py --model mlx-community/Qwen3-4B-4bit \
  --prompts workload.jsonl --budget 1024 > workload-results.jsonl
```

The harness exercises actual mlx-lm generation with the same KeyDiff cache used
by the server. It measures full KV, KeyDiff, one-shot sink+tail retention at the
same row/byte budget, and mlx-lm's existing 4/8-bit KV quantization (group size
64). The optional `rotating` mode uses mlx-lm's online rotating cache; unlike
the one-shot policies, it also evicts during prefill. Model weights remain
identical across policies.

The optional `plain-keydiff` mode removes recent-window protection for an
ablation against the whole-prompt scoring rule.

JSONL records contain prompt/output token counts, TTFT, decode tokens/s,
end-to-end time, allocated KV bytes, first-yield active Metal bytes, peak Metal
bytes, memory after cache release, generated text, answer substring match, and
normalized English token F1 against the best accepted answer.
Metal memory counters are not process RSS or system-wide unified-memory usage.
The built-in synthetic passkey is a smoke test only: it does not establish
general QA, summarization, coding, or multi-turn quality. Custom workload input
has one `{"id": "...", "prompt": "...", "answers": ["..."]}` object per line.

Run the download-free numerical and integration tests with:

```bash
pytest tests/test_keydiff.py
```

They compare scores with an independent PyTorch implementation of the kvpress
formula, check grouped-query attention and absolute RoPE offsets on tiny Llama
and Qwen3 models, and exercise request isolation, stream closure, and API
parameter forwarding. A passing test suite is not evidence of model quality.
