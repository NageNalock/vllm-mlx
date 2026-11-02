# Layerwise quantized-prefill evidence

The native implementation is frozen at `8f0b640`. The main report is
[Layerwise quantized KV prefill](../../../docs/benchmarks/quantized-prefill.md).

- `llama-32k-summary.json`: final standalone measurement compared with full KV,
  early 4-bit KV (start=0), and deferred 4-bit KV. One isolated run per mode.
- `native-llama-32k.jsonl`: final standalone candidate measurement.
- `layerwise-initial-32k.jsonl`: the earlier native-core measurement through the
  shared research driver; the standalone run reproduces its output and KV sizes.
- `llama-full-32k.jsonl`, `llama-early0-32k.jsonl`, `llama-deferred-32k.jsonl`:
  preserved reference measurements. Early start=0 is not the MLX-LM CLI default.
- `validation_manifest.json`: the same 48 examples used for implementation
  comparison; these are not relabeled as new held-out data after policy selection.
- `full-validation.jsonl`, `early0-validation.jsonl`: all 48 reference quality
  records per mode, extracted from the earlier comparison without excluding cases.
- `deferred-validation.jsonl`, `layerwise-validation.jsonl`: complete reference
  and native-candidate runs. Every prompt/output/first-logprob hash agrees.
- `quality-vs-deferred.json`, `quality-vs-full.json`: paired comparisons, including
  every individual score change. Quantization is not lossless relative to full KV.
- `validation_context_audit.json`: 48 questions, 47 distinct contexts.
- `provenance.json`: original filenames, hashes, and the extraction steps.
- `assets.json`: hardware, runtime versions, model and dataset revisions.
- `verification.json`, `http_smoke.json`, `cancellation_comparison.json`: numerical,
  lifecycle, API, static, and untouched-base checks.
- `large_model_protocol.json`: the pre-specified Qwen3 32B capacity comparison.
  Its runs are sequential and stop on sustained critical memory pressure or a
  20-minute duration budget. A guarded stop is censored evidence, not proof of OOM.
  Consult `large_model_status.json` when present for completion and stop reasons.

Performance outputs use fixed lengths and ignore EOS; quality outputs stop at
EOS. QA word F1 and code first-line similarity are not interchangeable with
accuracy or functional code correctness. Physical process peak includes loading
and warmup and differs from MLX active allocation and resident KV bytes.
