# SPDX-License-Identifier: Apache-2.0
"""Compare MLX-LM's existing quantization-after-dense-prefill option.

Use bench_kvzap.py arguments with --modes quant4. The candidate, prompts,
measurement code, and metric definitions remain unchanged.
"""

import hashlib
from pathlib import Path

import bench_kvzap as benchmark

_generate_step = benchmark.generate_step
_emit = benchmark.emit


def deferred_generate_step(prompt, model, **kwargs):
    kwargs["quantized_kv_start"] = len(prompt)
    return _generate_step(prompt, model, **kwargs)


def emit(record):
    if record.get("mode") == "quant4":
        record["mode"] = "quant4_deferred"
    if record["event"] == "metadata":
        assert record["arguments"]["modes"] == ["quant4"]
        record["quantized_baseline"] = (
            "MLX-LM 4-bit cache, quantized_kv_start=prompt length; "
            "full-precision dense prefill before cache quantization"
        )
        record["source_sha256"]["benchmarks/" + Path(__file__).name] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
    _emit(record)


if __name__ == "__main__":
    benchmark.generate_step = deferred_generate_step
    benchmark.emit = emit
    benchmark.main()
