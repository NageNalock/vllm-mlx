# KVzap evaluation records

The implementation and evaluation source at `56b6c8e` are frozen for the fresh
validation. Model, predictor, and dataset revisions are pinned. The main report
is [KVzap on Apple Silicon](../../../docs/benchmarks/kvzap.md).

## Final candidate

- `layerwise_validation_protocol.json`: frozen algorithm, gates, and source hashes.
- `dense_validation_manifest.json`: the 48 newly selected examples. The filename
  predates the change in prefill execution order; none was evaluated before the
  final protocol was frozen. Selection excludes earlier IDs and document contexts.
- `layerwise-development.jsonl`: native integration reproduces all 12 development
  outputs and first-logit hashes from `dense-prefill-development.jsonl`.
- `layerwise-final-32k.jsonl`: isolated 32K performance of the native integration.
- `layerwise-retrieval-32k.jsonl`: supplementary four-fact retrieval in code context.
- `layerwise-validation.jsonl`: fresh full-KV / 4-bit-KV / KVzap paired outputs.
  Check `layerwise_validation_status.json` for completion and gate results before
  interpreting this file as a complete evaluation.
- `layerwise-quality.json`, `quant4-quality.json`, and
  `layerwise-vs-quant4-quality.json`: paired summaries, confidence intervals, and
  every individual score regression or improvement.
- KVzap's conditional 1024-token measurement was not run because the complete
  fresh quality screen failed. The additional deferred baseline below is separate.
- `deferred_quantization_protocol.json`: an additional baseline using MLX-LM's
  existing `quantized_kv_start=prompt length` option. It keeps dense full-precision
  prefill before quantizing the cache to 4 bits. The candidate and primary quality
  gates stay unchanged; this comparison separates prefill precision from the
  decode-time cache policy.
- `deferred-validation.jsonl`, `deferred-quality.json`, and
  `layerwise-vs-deferred-quality.json`: the completed 48-case additional baseline
  and its paired comparisons. `deferred_quantization_status.json` records
  completion. These results do not override the failed primary quality gate.
- `deferred-32k-quant4.jsonl`, `long-32k-quant4-deferred.jsonl`: isolated deferred
  baseline measurements with 128 and 1024 fixed output tokens, respectively.
- `verification.json`: numerical, lifecycle, API, full-suite, and static checks.

The separate native quantization implementation is in
[fork PR #3](https://github.com/NageNalock/vllm-mlx/pull/3). This branch preserves
the rejected pruning experiment and its unchanged quality gate.

`dense-32k-full.jsonl` and `dense-32k-quant4.jsonl` are isolated baselines. The
4-bit and 8-bit baselines include fused prefill. All performance prompts are
matched by their token hash. Fixed-length performance runs ignore EOS; quality
runs stop at EOS. Scores are QA word F1 or first-line code similarity, not a
single accuracy measure or the complete official LongBench benchmark.

## Earlier experiments

These records are retained to show the failed alternatives and development
history. They are not pooled into the fresh validation or selected as headline
measurements after seeing the new results.

- `validation.jsonl`, `validation_partial_summary.json`, `validation_stop.json`:
  the earlier 8-bit-prefill variant failed the complete 16-case Qasper subset and
  stopped after 21 paired cases. It is **not** a complete 48-case evaluation.
- `attribution-*.jsonl`: separate quantization and pruning effects on two observed
  failures. Both examples are development evidence thereafter.
- `online8-*`: the rejected variant that pruned between prefill chunks.
- `native-*`, `final-*`, `development_summary.json`,
  `dense-prefill-development*`, `dense-32k-kvzap.jsonl`, and `retrieval-32k.jsonl`:
  earlier prototypes, smoke checks, and measurements.
- `layerwise-32k-kvzap.jsonl` and `layerwise-32k-chunk-eval.jsonl`: experimental
  layerwise scheduling before integration. Source is retained in commits
  `0b2ba34` and `b521505`; obsolete experiment entry points were removed from
  the final implementation.
- `excluded_runs.json`: explicit exclusions for two overlapping or aborted timing
  runs. They were repeated in isolation.

`initial_protocol.json`, `dense_validation_protocol.json`, and
`performance_extension_protocol.json` describe earlier stages; the final
protocol explicitly supersedes them where applicable. `model_assets.json`
records checked model files, and `predictor_parity.json` records independent
MLX-versus-PyTorch linear-predictor comparisons.

Machine-specific manifest paths in three historical metadata records were
normalized to repository-relative paths for portability; see
`metadata_path_normalization.json`. Numerical measurements, model outputs,
prompts, and hashes were not changed. Process physical peaks from
`/usr/bin/time -l` are separate from MLX peak active allocation.
