# SPDX-License-Identifier: Apache-2.0

"""Dense prefill with layerwise conversion to MLX's native quantized KV cache."""

from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache


def validate_quantized_prefill_model(model):
    from mlx_lm.models.llama import Model as Llama
    from mlx_lm.models.qwen3 import Model as Qwen3

    if not isinstance(model, (Llama, Qwen3)) or (
        isinstance(model, Llama)
        and any(t != "full_attention" for t in model.args.layer_types)
    ):
        raise ValueError(
            "Layerwise KV quantization supports dense Llama and Qwen3 only"
        )
    args = model.args
    head_dim = args.head_dim or args.hidden_size // args.num_attention_heads
    if head_dim % 64:
        raise ValueError(
            "Layerwise KV quantization requires head dimensions divisible by 64"
        )


class _LayerwiseDecoder(nn.Module):
    def __init__(self, original, tokens, bits, step_size):
        super().__init__()
        self.original = original
        self.tokens = tokens
        self.bits = bits
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
        if input_embeddings is not None or inputs.shape[0] != 1:
            raise ValueError("Layerwise KV quantization requires a single text request")
        if self.pending_output is not None:
            output = self.pending_output
            self.pending_output = None
            return output
        if self.prefilled:
            return self.original(inputs, cache, input_embeddings)
        length = len(self.tokens)
        if inputs.shape[1] not in (length - 1, length):
            raise ValueError(
                "Layerwise KV quantization requires outer prefill_step_size >= prompt length"
            )
        h = self.original.embed_tokens(mx.array(self.tokens)[None])
        chunks = [
            (start, min(start + self.step_size, length - 1))
            for start in range(0, length - 1, self.step_size)
        ] + [(length - 1, length)]
        for index, layer in enumerate(self.layers):
            layer_cache = cache[index]
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
            cache[index] = layer_cache.to_quantized(group_size=64, bits=self.bits)
            mx.eval(cache[index].state)
            del layer_cache
            mx.clear_cache()
        output = self.original.norm(h[:, -1:])
        mx.eval(output)
        self.prefilled = True
        self.tokens = None
        if inputs.shape[1] == length - 1:
            # MLX-LM samples from a separate call for the final prompt token.
            self.pending_output = output
        return output


@contextmanager
def quantized_prompt_cache(model, prompt, bits, prefill_step_size=2048):
    """Prefill densely, quantizing each layer only after its last prompt query.

    Use a fresh prompt and outer prefill_step_size=len(prompt). Access to the
    model must be serialized while this context is active, as in SimpleEngine.
    Decoding uses the original model and MLX's native QuantizedKVCache.
    """
    validate_quantized_prefill_model(model)
    if bits not in (4, 8):
        raise ValueError("KV cache bits must be 4 or 8")
    if len(prompt) < 1 or prefill_step_size < 1:
        raise ValueError(
            "Layerwise KV quantization requires a non-empty prompt and positive prefill_step_size"
        )
    caches = make_prompt_cache(model)
    decoder = model.model
    try:
        model.model = _LayerwiseDecoder(decoder, prompt, bits, prefill_step_size)
        yield caches
    finally:
        model.model = decoder
