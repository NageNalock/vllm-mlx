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
- `early0-vs-full.json`: the same 48-case descriptive comparison for the early
  start=0 policy, including its individual answer changes.
- `validation_context_audit.json`: 48 questions, 47 distinct contexts.
- `provenance.json`: original filenames, hashes, and the extraction steps.
- `assets.json`: hardware, runtime versions, model and dataset revisions.
- `verification.json`, `http_smoke.json`, `cancellation_comparison.json`: numerical,
  lifecycle, API, static, and untouched-base checks.
- `cancellation_test_isolation.json`: the traced foreign-generator destructor
  error behind an initial full-suite lifetime assertion failure, and the test
  isolation change. The strict release assertion and runtime code are unchanged.
- `large_model_protocol.json`: the pre-specified Qwen3 32B capacity comparison.
  Its runs are sequential and stop on sustained critical memory pressure or a
  20-minute duration budget. A guarded stop is censored evidence, not proof of OOM.
- `large_model_status.json`: all three outcomes and every sampled pressure level.
  Layerwise and early start=5000 complete; deferred quantization reaches the
  duration limit. Pressure samples are formatted inline without changing values.
- `qwen32b-32k-summary.json`, `qwen32b-*.jsonl`: derived metrics and original runs.
  Both completed policies recover all four facts. The early policy already runs
  this workload; the layerwise run's 18.8-minute TTFT and 3.68 tok/s do not establish
  a newly enabled model tier or the intended fluent large-model experience.
- `qwen32b-*.time.txt`: exact process-resource footers for completed runs.
  `qwen32b-deferred-32k.stderr.txt` preserves the stopped run's stderr; no final
  physical-peak footer was available. Its tokenizer warning refers to the source
  pool before truncation, not an inference input longer than 32,768 tokens.
- [Code-audit protocol](model-tier/protocol.json) records the original Qwen3
  4B-to-8B comparison, source excerpts, answer key and pass criteria. The
  [precision follow-up](model-tier/precision-addendum.json) records the 8-bit
  and 14B checks added after the first 4-bit answer regression.
- [Code-audit results](model-tier/summary.json) links model hashes, practical
  thresholds and alternative configurations to [individual runs](model-tier/runs.jsonl).
  Each completed run includes its full answer, field scores, latency, throughput
  and the measured process's `/usr/bin/time -l` footer.

The 32K timing runs ignore EOS for fixed output lengths. The source-code audit
and the paired quality runs stop at EOS. QA word F1 and code first-line
similarity are not interchangeable with
accuracy or functional code correctness. Physical process peak includes loading
and warmup and differs from MLX active allocation and resident KV bytes.
