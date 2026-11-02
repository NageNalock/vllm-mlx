# SPDX-License-Identifier: Apache-2.0
"""Diagnostic variant: preserve FP16 KV throughout dense prefill."""

import hashlib
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import KVCache

import vllm_mlx.kvzap as kvzap
from bench_kvzap import emit, main


class DensePrefillKVzapCache(kvzap.KVzapCache):
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
            self.compact()
        self.offset += keys.shape[2]
        if self._heads is None:
            return self._cache.update_and_fetch(keys, values)
        updated = [
            c.update_and_fetch(keys[:, h : h + 1], values[:, h : h + 1])
            for h, c in enumerate(self._heads)
        ]
        return [k for k, _ in updated], [v for _, v in updated]


if __name__ == "__main__":
    emit(
        {
            "event": "variant",
            "name": "Full-precision dense prefill; otherwise frozen KVzap policy",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "purpose": "Development and attribution, not held-out validation",
        }
    )
    kvzap.KVzapCache = DensePrefillKVzapCache
    main()
