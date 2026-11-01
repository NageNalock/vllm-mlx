# SPDX-License-Identifier: Apache-2.0
"""MLX-native, one-shot KeyDiff pruning for single-request text generation."""

import mlx.core as mx
from mlx_lm.models.cache import KVCache, _BaseCache, make_prompt_cache


def keydiff_scores(keys: mx.array) -> mx.array:
    """Negative cosine similarity to the mean normalized key, per KV head.

    Implements NVIDIA kvpress's KeyDiffPress scoring rule on post-RoPE keys.
    Accumulate in float32, including when the cache uses float16/bfloat16.
    """
    keys = keys.astype(mx.float32)
    norms = mx.linalg.norm(keys, axis=-1, keepdims=True)
    mean_key = (keys / mx.maximum(norms, 1e-12)).mean(axis=-2, keepdims=True)
    mean_norm = mx.linalg.norm(mean_key, axis=-1, keepdims=True)
    similarities = (keys * mean_key).sum(axis=-1, keepdims=True)
    return -(similarities / (mx.maximum(norms, 1e-8) * mx.maximum(mean_norm, 1e-8)))[
        ..., 0
    ]


class KeyDiffKVCache(_BaseCache):
    """Prune the prompt once, before the first decode forward pass.

    ``offset`` counts all processed tokens for RoPE. The wrapped KVCache counts
    only retained rows for storage and attention masks. Generated tokens are
    appended without further eviction, so ``budget`` is a prompt retention
    budget, not a bound on the entire generation's memory.

    This request-local cache deliberately does not support trimming, prefix
    reuse, serialization, batching, or KV quantization.
    """

    # A server receives a complete prompt rather than kvpress's separate
    # context/question inputs. Preserve the tail so instructions and the
    # question are not subject to eviction. This is part of the total budget.
    recent_tokens = 128

    def __init__(self, budget: int, prompt_length: int):
        if budget < 1 or prompt_length < 1:
            raise ValueError("KeyDiff requires a positive budget and prompt length")
        self.budget = budget
        self.prompt_length = prompt_length
        self.offset = 0
        self._cache = KVCache()

    def _compact(self):
        if self.offset != self.prompt_length or self._cache.size() <= self.budget:
            return
        keys, values = self._cache.state
        recent = min(self.recent_tokens, self.budget)
        prefix_length = self.offset - recent
        indices = mx.broadcast_to(
            mx.arange(prefix_length, self.offset), (*keys.shape[:2], recent)
        )
        selected = self.budget - recent
        if selected:
            scores = keydiff_scores(keys[:, :, :prefix_length])
            selected_indices = mx.argpartition(-scores, kth=selected - 1, axis=-1)[
                ..., :selected
            ]
            indices = mx.concatenate([selected_indices, indices], axis=-1)
        # Keep chronological order independently for each KV head.
        indices = mx.sort(indices, axis=-1)[..., None]
        self._cache.state = (
            mx.take_along_axis(keys, indices, axis=2),
            mx.take_along_axis(values, indices, axis=2),
        )
        # Materialize the gathers so their lazy graphs release the full cache.
        mx.eval(self._cache.state)

    def update_and_fetch(self, keys, values):
        self._compact()
        self.offset += keys.shape[2]
        return self._cache.update_and_fetch(keys, values)

    def make_mask(self, *args, **kwargs):
        self._compact()
        return self._cache.make_mask(*args, **kwargs)

    @property
    def state(self):
        return self._cache.state

    @state.setter
    def state(self, value):
        raise ValueError("KeyDiff caches are request-local and cannot be restored")

    def size(self):
        return self._cache.size()

    def empty(self):
        return self._cache.empty()

    def to_quantized(self, group_size=64, bits=4):
        raise ValueError("KeyDiff cannot be combined with KV quantization")

    @property
    def nbytes(self):
        return self._cache.nbytes


def validate_keydiff_model(model) -> None:
    """Limit the initial implementation to tested full-attention architectures."""
    from mlx_lm.models.llama import Model as LlamaModel
    from mlx_lm.models.qwen3 import Model as Qwen3Model

    if not isinstance(model, (LlamaModel, Qwen3Model)) or any(
        type(c) is not KVCache for c in make_prompt_cache(model)
    ):
        raise ValueError("KeyDiff supports only full-attention Llama and dense Qwen3")


def make_keydiff_cache(model, budget: int, prompt_length: int) -> list[KeyDiffKVCache]:
    validate_keydiff_model(model)
    return [KeyDiffKVCache(budget, prompt_length) for _ in model.layers]
