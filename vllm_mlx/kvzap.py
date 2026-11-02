# SPDX-License-Identifier: Apache-2.0

"""Request-local KVzap pruning for Llama attention."""

import json
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.cache import KVCache, _BaseCache

PREDICTOR = "nvidia/KVzap-linear-Llama-3.1-8B-Instruct"
PREDICTOR_REVISION = "92b20816c9d97412b1976ea15aa68b7b60510883"


def validate_kvzap_model(model):
    from mlx_lm.models.llama import Model

    if not isinstance(model, Model):
        raise ValueError("KVzap currently supports Llama 3.1 8B Instruct only")
    args = model.args
    if (
        (args.hidden_size, args.num_hidden_layers, args.num_key_value_heads)
        != (4096, 32, 8)
        or (args.rope_scaling or {}).get("rope_type") != "llama3"
        or any(t != "full_attention" for t in args.layer_types)
    ):
        raise ValueError("KVzap currently supports Llama 3.1 8B Instruct only")


def predictor_path():
    from huggingface_hub import hf_hub_download

    config = hf_hub_download(PREDICTOR, "config.json", revision=PREDICTOR_REVISION)
    hf_hub_download(PREDICTOR, "model.safetensors", revision=PREDICTOR_REVISION)
    return Path(config).parent


class KVzapCache(_BaseCache):
    """Remove low-scoring prompt entries, independently for each KV head.

    RoPE uses the total number of processed tokens. The per-head caches track
    physical storage separately. Prefill keeps the model's KV dtype and uses
    dense attention. Generated tokens are kept without pruning.
    This cache does not support batching, prefix reuse, or serialization.
    """

    def __init__(self, scorer, prompt_length, threshold=-7.0, window_size=128):
        self.scorer = scorer
        self.prompt_length = prompt_length
        self.threshold = threshold
        self.window_size = window_size
        self.offset = 0
        self._cache = KVCache()
        self._heads = None
        self._scores = []

    def score(self, hidden_states):
        if hidden_states.shape[0] != 1:
            raise ValueError("KVzap supports one request at a time")
        if self._heads is None and self.offset < self.prompt_length:
            self._scores.append(self.scorer(hidden_states).transpose(0, 2, 1))
            # Release each layer's input before evaluating later layers.
            mx.eval(self._scores[-1])

    def compact(self):
        keys = self._cache.keys[..., : self._cache.offset, :]
        values = self._cache.values[..., : self._cache.offset, :]
        scores = mx.concatenate(self._scores, axis=-1)
        keep = scores >= self.threshold
        keep[..., -self.window_size :] = True
        counts = keep.sum(axis=-1).tolist()[0]
        self._heads = []
        for head, count in enumerate(counts):
            indices = mx.argpartition(-keep[0, head].astype(mx.int32), count - 1)[
                :count
            ]
            indices = mx.sort(indices)
            cache = KVCache()
            cache.keys = keys[:, head : head + 1, indices, :]
            cache.values = values[:, head : head + 1, indices, :]
            cache.offset = count
            self._heads.append(cache)
        mx.eval([c.state for c in self._heads])
        self._cache = None
        self._scores = []

    def update_and_fetch(self, keys, values):
        if self._heads is None and self.offset >= self.prompt_length:
            # Complete dense prefill before pruning, as in the reference method.
            self.compact()
        self.offset += keys.shape[2]
        if self._heads is None:
            return self._cache.update_and_fetch(keys, values)
        updated = [
            c.update_and_fetch(keys[:, h : h + 1], values[:, h : h + 1])
            for h, c in enumerate(self._heads)
        ]
        return [k for k, _ in updated], [v for _, v in updated]

    def attention(self, queries, keys, values, scale, mask):
        if self._heads is None:
            return mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=scale, mask=mask
            )
        groups = mx.split(queries, len(keys), axis=1)
        return mx.concatenate(
            [
                mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
                for q, k, v in zip(groups, keys, values)
            ],
            axis=1,
        )

    def make_mask(self, N, **kwargs):
        return "causal" if N > 1 else None

    @property
    def state(self):
        caches = self._heads if self._heads is not None else [self._cache]
        return [c.state for c in caches], self._scores

    @state.setter
    def state(self, value):
        raise ValueError("KVzap does not support restoring a saved cache")

    def size(self):
        return self.offset

    def empty(self):
        return self.offset == 0

    def to_quantized(self, group_size=64, bits=4):
        raise ValueError(
            "KVzap requires full-precision prefill; do not also set kv_bits"
        )

    @property
    def nbytes(self):
        caches = self._heads if self._heads is not None else [self._cache]
        return sum(c.nbytes for c in caches) + sum(s.nbytes for s in self._scores)


def make_kvzap_cache(
    model, scorer_path, prompt_length, threshold=-7.0, window_size=128
):
    """Load a released linear KVzap predictor for a matching Llama model."""
    from mlx_lm.models.llama import Model

    scorer_path = Path(scorer_path)
    config = json.loads((scorer_path / "config.json").read_text())
    if (
        not isinstance(model, Model)
        or any(t != "full_attention" for t in model.args.layer_types)
        or config["hidden_dim"] is not None
        or config["n_modules"] != model.args.num_hidden_layers
        or config["input_dim"] != model.args.hidden_size
        or config["output_dim"] != model.args.num_key_value_heads
    ):
        raise ValueError("KVzap requires a matching linear predictor and Llama model")
    if prompt_length < 1 or window_size < 1:
        raise ValueError("KVzap requires a positive prompt length and window")
    weights = mx.load(str(scorer_path / "model.safetensors"))
    dtype = model.layers[0].input_layernorm.weight.dtype
    caches = []
    for i in range(config["n_modules"]):
        scorer = nn.Linear(config["input_dim"], config["output_dim"])
        scorer.weight = weights[f"layers.{i}.weight"].astype(dtype)
        scorer.bias = weights[f"layers.{i}.bias"].astype(dtype)
        caches.append(KVzapCache(scorer, prompt_length, threshold, window_size))
    mx.eval([c.scorer.parameters() for c in caches])
    return caches


class _KVzapAttention(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.original = attention

    def __call__(self, x, mask=None, cache=None):
        if not isinstance(cache, KVzapCache):
            return self.original(x, mask, cache)
        attention = self.original
        batch, length, _ = x.shape
        cache.score(x)
        queries = attention.q_proj(x).reshape(batch, length, attention.n_heads, -1)
        keys = attention.k_proj(x).reshape(batch, length, attention.n_kv_heads, -1)
        values = attention.v_proj(x).reshape(batch, length, attention.n_kv_heads, -1)
        queries = attention.rope(queries.transpose(0, 2, 1, 3), offset=cache.offset)
        keys = attention.rope(keys.transpose(0, 2, 1, 3), offset=cache.offset)
        keys, values = cache.update_and_fetch(keys, values.transpose(0, 2, 1, 3))
        output = cache.attention(queries, keys, values, attention.scale, mask)
        return attention.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))


@contextmanager
def kvzap_prompt_cache(model, prompt_length, scorer_path=None, threshold=-7.0):
    """Use KVzap for one request, restoring attention modules on exit.

    The caller must serialize access to the model while this context is active,
    as SimpleEngine does for generation. No global MLX-LM patches are needed.
    """
    if scorer_path is None:
        validate_kvzap_model(model)
        scorer_path = predictor_path()
    caches = make_kvzap_cache(model, scorer_path, prompt_length, threshold)
    originals = [layer.self_attn for layer in model.layers]
    try:
        for layer, attention in zip(model.layers, originals):
            layer.self_attn = _KVzapAttention(attention)
        yield caches
    finally:
        for layer, attention in zip(model.layers, originals):
            layer.self_attn = attention
