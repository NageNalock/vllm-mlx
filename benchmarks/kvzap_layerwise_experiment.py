# SPDX-License-Identifier: Apache-2.0
"""Diagnostic layer-major prefill: complete dense attention before freeing KV."""

import hashlib
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

import bench_kvzap as bench
from vllm_mlx.kvzap import KVzapCache, kvzap_prompt_cache


class LayerwiseDecoder(nn.Module):
    def __init__(self, original, tokens, step_size):
        super().__init__()
        self.original = original
        self.tokens = tokens
        self.step_size = step_size
        self.prefilled = False
        self.pending_output = None

    @property
    def layers(self):
        return self.original.layers

    @property
    def embed_tokens(self):
        return self.original.embed_tokens

    def __call__(self, inputs, cache=None, input_embeddings=None):
        if self.pending_output is not None:
            output = self.pending_output
            self.pending_output = None
            return output
        if self.prefilled:
            return self.original(inputs, cache, input_embeddings)
        length = len(self.tokens)
        if inputs.shape[1] not in (length - 1, length):
            raise ValueError(
                "Layerwise prefill requires outer prefill_step_size >= prompt length"
            )
        h = self.original.embed_tokens(mx.array(self.tokens)[None])
        chunks = [
            (start, min(start + self.step_size, length - 1))
            for start in range(0, length - 1, self.step_size)
        ] + [(length - 1, length)]
        for layer, layer_cache in zip(self.layers, cache):
            outputs = []
            for start, end in chunks:
                output = layer(
                    h[:, start:end],
                    "causal" if end - start > 1 else None,
                    cache=layer_cache,
                )
                mx.eval(output, layer_cache.state)
                outputs.append(output)
            h = mx.concatenate(outputs, axis=1)
            mx.eval(h)
            del outputs
            layer_cache.compact()
            mx.clear_cache()
        output = self.original.norm(h[:, -1:])
        mx.eval(output)
        self.prefilled = True
        self.tokens = None
        if inputs.shape[1] == length - 1:
            # MLX-LM separately requests the final prompt token after prefill.
            self.pending_output = output
        return output


@contextmanager
def layerwise_cache(model, tokens, predictor, threshold, step_size):
    with kvzap_prompt_cache(model, len(tokens), predictor, threshold) as caches:
        original = model.model
        try:
            model.model = LayerwiseDecoder(original, tokens, step_size)
            yield caches
        finally:
            model.model = original


_measure = bench.measure
_generate_step = bench.generate_step


def score_with_chunk_evaluation(self, hidden_states):
    if hidden_states.shape[0] != 1:
        raise ValueError("KVzap supports one request at a time")
    if self._heads is None and self.offset < self.prompt_length:
        self._scores.append(self.scorer(hidden_states).transpose(0, 2, 1))


def measure(model, tokenizer, predictor, tokens, mode, args, max_tokens, stop_at_eos):
    original = bench.kvzap_prompt_cache
    bench.kvzap_prompt_cache = lambda model, _, predictor, threshold: layerwise_cache(
        model, tokens, predictor, threshold, args.prefill_step_size
    )
    try:
        return _measure(
            model, tokenizer, predictor, tokens, mode, args, max_tokens, stop_at_eos
        )
    finally:
        bench.kvzap_prompt_cache = original


def generate_step(prompt, model, **kwargs):
    if isinstance(kwargs["prompt_cache"][0], KVzapCache):
        kwargs["prefill_step_size"] = len(prompt)
    return _generate_step(prompt, model, **kwargs)


if __name__ == "__main__":
    bench.emit(
        {
            "event": "variant",
            "name": "Layer-major full-precision prefill",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "purpose": "Development; same pruning threshold and window",
        }
    )
    bench.measure = measure
    bench.generate_step = generate_step
    KVzapCache.score = score_with_chunk_evaluation
    bench.main()
