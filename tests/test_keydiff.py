# SPDX-License-Identifier: Apache-2.0
"""Numerical and cache-position regressions; no model downloads required."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

from vllm_mlx.keydiff import (
    KeyDiffKVCache,
    keydiff_scores,
    make_keydiff_cache,
    validate_keydiff_model,
)


@pytest.fixture(autouse=True)
def local_generation_stream(monkeypatch):
    # Other engine tests bind mlx-lm's module-level stream on worker threads.
    from importlib import import_module

    module = import_module("mlx_lm.generate")
    monkeypatch.setattr(module, "generation_stream", mx.new_stream(mx.default_device()))


@pytest.fixture(params=["llama", "qwen3"])
def tiny_model(request):
    from importlib import import_module

    module = import_module(f"mlx_lm.models.{request.param}")
    mx.random.seed(7)
    args = module.ModelArgs.from_dict(
        dict(
            model_type=request.param,
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            rms_norm_eps=1e-5,
            vocab_size=128,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            tie_word_embeddings=True,
        )
    )
    model = module.Model(args)
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_scores_match_kvpress_torch_rule(dtype):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    keys = mx.array(np.random.default_rng(3).normal(size=(1, 4, 67, 32))).astype(dtype)
    keys[:, :, 0, :] = 0
    source = np.array(keys.astype(mx.float32))
    ref_keys = torch.from_numpy(source)
    # Independent reference: NVIDIA/kvpress KeyDiffPress.score.
    mean_key = F.normalize(ref_keys, p=2, dim=-1).mean(dim=2, keepdim=True)
    expected = -F.cosine_similarity(ref_keys, mean_key, dim=-1)
    np.testing.assert_allclose(np.array(keydiff_scores(keys)), expected, atol=2e-6)


@pytest.mark.parametrize("recent", [0, 128])
def test_compaction_gathers_keys_and_values_per_head_and_releases_storage(recent):
    rng = np.random.default_rng(42)
    keys = mx.array(rng.normal(size=(1, 3, 1024, 32)), mx.float32)
    values = mx.array(rng.normal(size=(1, 3, 1024, 48)), mx.float32)
    cache = KeyDiffKVCache(budget=256, prompt_length=1024)
    cache.recent_tokens = recent
    for start in range(0, 1024, 256):
        cache.update_and_fetch(
            keys[:, :, start : start + 256], values[:, :, start : start + 256]
        )
    mx.eval(cache.state)
    full_bytes = cache.nbytes
    assert cache.offset == cache.size() == 1024

    prefix_length = 1024 - recent
    selected = np.sort(
        np.argsort(np.array(keydiff_scores(keys[:, :, :prefix_length])), axis=-1)[
            ..., -(256 - recent) :
        ],
        axis=-1,
    )
    indices = np.concatenate(
        [selected, np.broadcast_to(np.arange(prefix_length, 1024), (1, 3, recent))],
        axis=-1,
    )
    next_keys, next_values = keys[:, :, :1], values[:, :, :1]
    actual_keys, actual_values = cache.update_and_fetch(next_keys, next_values)
    mx.eval(actual_keys, actual_values)
    np.testing.assert_array_equal(
        np.array(actual_keys[:, :, :-1]),
        np.take_along_axis(np.array(keys), indices[..., None], axis=2),
    )
    np.testing.assert_array_equal(
        np.array(actual_values[:, :, :-1]),
        np.take_along_axis(np.array(values), indices[..., None], axis=2),
    )
    assert cache.offset == 1025
    assert cache.size() == 257
    assert cache.nbytes <= full_bytes / 2
    assert not cache.is_trimmable()


@pytest.mark.parametrize("budget", [32, 128])
def test_inactive_budget_matches_full_generation(tiny_model, budget):
    prompt = mx.arange(32)
    full = list(generate_step(prompt, tiny_model, max_tokens=5, prefill_step_size=7))
    caches = make_keydiff_cache(tiny_model, budget, len(prompt))
    actual = list(
        generate_step(
            prompt, tiny_model, max_tokens=5, prefill_step_size=7, prompt_cache=caches
        )
    )
    assert [t for t, _ in actual] == [t for t, _ in full]
    for (_, a), (_, b) in zip(actual, full):
        np.testing.assert_array_equal(np.array(a), np.array(b))


def test_decode_preserves_rope_positions_and_physical_mask(tiny_model):
    prompt = (mx.arange(256) % 128)[None]
    caches = make_keydiff_cache(tiny_model, 160, 256)
    full_caches = make_prompt_cache(tiny_model)
    expected_prefill = tiny_model(prompt, cache=full_caches)
    actual_prefill = tiny_model(prompt, cache=caches)
    np.testing.assert_array_equal(np.array(actual_prefill), np.array(expected_prefill))

    class ReferenceCache:
        """An independent concatenate-only cache with absolute RoPE offset."""

        def __init__(self, full):
            self.offset = 256
            k, v = full.state
            selected = np.sort(
                np.argsort(np.array(keydiff_scores(k[:, :, :128])), axis=-1)[..., -32:],
                axis=-1,
            )
            indices = np.concatenate(
                [selected, np.broadcast_to(np.arange(128, 256), (*k.shape[:2], 128))],
                axis=-1,
            )
            self.keys = mx.array(
                np.take_along_axis(np.array(k), indices[..., None], axis=2)
            )
            self.values = mx.array(
                np.take_along_axis(np.array(v), indices[..., None], axis=2)
            )

        def update_and_fetch(self, keys, values):
            self.offset += keys.shape[2]
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
            return self.keys, self.values

        def make_mask(self, n, **kwargs):
            from mlx_lm.models.cache import create_attention_mask

            return create_attention_mask(n, offset=self.keys.shape[2], **kwargs)

    references = [ReferenceCache(c) for c in full_caches]
    # Include a multi-token append to exercise the mask's physical offset.
    for tokens in ([[41]], [[42, 43]], [[44]]):
        actual = tiny_model(mx.array(tokens), cache=caches)
        expected = tiny_model(mx.array(tokens), cache=references)
        np.testing.assert_allclose(np.array(actual), np.array(expected), atol=2e-5)
    assert all(c.offset == 260 and c.size() == 164 for c in caches)
    mask = caches[0].make_mask(2, return_array=True, window_size=None)
    assert mask.shape == (2, 166)
    assert not mask[0, -1].item()


@pytest.mark.parametrize("budget", [1, 12, 128])
def test_small_budget_preserves_only_the_recent_tail(budget):
    cache = KeyDiffKVCache(budget, 256)
    keys = mx.arange(256).reshape(1, 1, 256, 1).astype(mx.float32)
    cache.update_and_fetch(keys, keys)
    new = mx.array([256.0]).reshape(1, 1, 1, 1)
    actual, _ = cache.update_and_fetch(new, new)
    np.testing.assert_array_equal(
        np.array(actual).ravel(), np.arange(256 - budget, 257)
    )
    assert cache.offset == 257


def test_rejects_sliding_window_and_unknown_models():
    from mlx_lm.models.llama import Model, ModelArgs

    with pytest.raises(ValueError, match="full-attention"):
        validate_keydiff_model(object())
    model = Model(
        ModelArgs(
            model_type="llama",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=64,
            layer_types=["full_attention", "sliding_attention"],
            sliding_window=16,
        )
    )
    with pytest.raises(ValueError, match="full-attention"):
        validate_keydiff_model(model)


@pytest.mark.parametrize("budget,length", [(0, 8), (-1, 8), (8, 0)])
def test_rejects_invalid_cache_sizes(budget, length):
    with pytest.raises(ValueError, match="positive"):
        KeyDiffKVCache(budget, length)


def test_quantization_request_is_not_silently_ignored(tiny_model):
    caches = make_keydiff_cache(tiny_model, 8, 32)
    with pytest.raises(ValueError, match="KV quantization"):
        list(
            generate_step(
                mx.arange(32), tiny_model, max_tokens=2, prompt_cache=caches, kv_bits=4
            )
        )


@pytest.fixture
def tiny_wrapper(tiny_model):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from vllm_mlx.models.llm import MLXLanguageModel

    vocab = {"<unk>": 0, **{str(i): i for i in range(1, 128)}}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>")
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['content'] + ' ' }}{% endfor %}"
    )
    wrapper = MLXLanguageModel("tiny-model")
    wrapper.model, wrapper.tokenizer, wrapper._loaded = tiny_model, tokenizer, True
    return wrapper


def test_wrapper_preserves_processor_history_and_disabled_path(tiny_wrapper):
    prompt = " ".join(str(i) for i in range(40))
    histories = []

    def observe_history(tokens, logits):
        histories.append(tokens.tolist())
        return logits

    full = list(
        tiny_wrapper.stream_generate(
            prompt, max_tokens=4, temperature=0, logits_processors=[observe_history]
        )
    )
    full_histories = histories[:]
    histories.clear()
    inactive = list(
        tiny_wrapper.stream_generate(
            prompt, max_tokens=4, temperature=0, keydiff_kv_budget=64
        )
    )
    assert [c.token for c in full] == [c.token for c in inactive]
    chunks = list(
        tiny_wrapper.stream_generate(
            prompt,
            max_tokens=4,
            temperature=0,
            keydiff_kv_budget=12,
            logits_processors=[observe_history],
        )
    )
    assert chunks[0].token == full[0].token
    assert chunks[0].prompt_tokens == 40
    assert histories[:2] == full_histories[:2]
    output = tiny_wrapper.generate(
        prompt, max_tokens=4, temperature=0, keydiff_kv_budget=12
    )
    assert output.text == "".join(c.text for c in chunks)
    with pytest.raises(ValueError, match="fresh"):
        list(
            tiny_wrapper.stream_generate(prompt, prompt_cache=[], keydiff_kv_budget=12)
        )


@pytest.fixture
def tiny_engine(tiny_wrapper, monkeypatch):
    from vllm_mlx.engine.simple import SimpleEngine

    monkeypatch.setattr("vllm_mlx.engine.simple.is_mllm_model", lambda _: False)
    engine = SimpleEngine("tiny-model", prefix_trie_cache=True)
    engine._model, engine._loaded = tiny_wrapper, True
    engine._supports_system_kv_cache = True
    return engine


@pytest.mark.anyio
async def test_engine_chat_close_releases_cache_and_next_request_is_fresh(
    tiny_engine, monkeypatch
):
    import gc
    import weakref

    import vllm_mlx.keydiff as keydiff

    refs = []
    factory = keydiff.make_keydiff_cache

    def capture(*args):
        caches = factory(*args)
        refs.extend(weakref.ref(c) for c in caches)
        return caches

    monkeypatch.setattr(keydiff, "make_keydiff_cache", capture)
    messages = [
        {"role": "system", "content": "1 2 3"},
        {"role": "user", "content": " ".join(str(i) for i in range(40))},
    ]
    try:
        stream = tiny_engine.stream_chat(
            messages, temperature=0, max_tokens=20, keydiff_kv_budget=12
        )
        first = await anext(stream)
        assert first.prompt_tokens == 43
        assert refs and all(r().size() < r().offset for r in refs)
        await stream.aclose()
        gc.collect()
        assert all(r() is None for r in refs)
        assert not tiny_engine._system_kv_cache
        assert tiny_engine._prefix_trie_cache is None

        output = await tiny_engine.chat(
            messages, temperature=0, max_tokens=3, keydiff_kv_budget=12
        )
        assert output.completion_tokens == 3
        assert output.prompt_tokens == 43
        gc.collect()
        assert all(r() is None for r in refs)
        # A subsequent ordinary request still takes the unmodified full-KV path.
        before = len(refs)
        await tiny_engine.generate("1 2 3", temperature=0, max_tokens=3)
        assert len(refs) == before
    finally:
        await stream.aclose()
        await tiny_engine.stop()


@pytest.mark.parametrize(
    "setting", ["_is_mllm", "_mtp", "_max_kv_size", "_draft_model"]
)
def test_engine_rejects_incompatible_features(tiny_engine, setting):
    setattr(tiny_engine, setting, True)
    with pytest.raises(ValueError, match="requires text-only"):
        tiny_engine.validate_keydiff(12)


def test_http_options_and_unsupported_engine(tiny_engine):
    from fastapi import HTTPException
    from pydantic import ValidationError

    from vllm_mlx.server import (
        ChatCompletionRequest,
        CompletionRequest,
        _prepare_chat_completion_invocation,
        _validate_keydiff_request,
    )

    for cls, payload in [
        (CompletionRequest, {"prompt": "1 2 3"}),
        (ChatCompletionRequest, {"messages": [{"role": "user", "content": "1 2 3"}]}),
    ]:
        for invalid in (0, -1, 1.5, True):
            with pytest.raises(ValidationError):
                cls(model="tiny-model", keydiff_kv_budget=invalid, **payload)
        request = cls(model="tiny-model", keydiff_kv_budget=12, **payload)
        _validate_keydiff_request(tiny_engine, request)
        with pytest.raises(HTTPException) as exc:
            _validate_keydiff_request(object(), request)
        assert exc.value.status_code == 400
    prepared = _prepare_chat_completion_invocation(tiny_engine, request, 4)
    assert prepared.chat_kwargs["keydiff_kv_budget"] == 12


@pytest.mark.anyio
async def test_completion_forwards_budget_in_both_modes(tiny_engine, monkeypatch):
    from types import SimpleNamespace

    import vllm_mlx.server as server

    captured = []

    async def generate(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(
            text="ok",
            new_text="ok",
            finished=True,
            finish_reason="length",
            completion_tokens=1,
            prompt_tokens=3,
        )

    async def stream_generate(**kwargs):
        yield await generate(**kwargs)

    monkeypatch.setattr(tiny_engine, "generate", generate)
    monkeypatch.setattr(tiny_engine, "stream_generate", stream_generate)
    monkeypatch.setattr(server, "_model_name", "tiny-model")
    monkeypatch.setattr(server, "_model_manager", None)
    monkeypatch.setattr(server, "_residency_manager", None)
    monkeypatch.setattr(server, "_default_model_key", None)
    monkeypatch.setattr(server, "get_engine", lambda: tiny_engine)
    request = server.CompletionRequest(
        model="tiny-model", prompt="1 2 3", keydiff_kv_budget=12
    )
    await server.create_completion(request, raw_request=None)
    chunks = [
        c
        async for c in server.stream_completion(tiny_engine, request.prompt, request, 1)
    ]
    assert chunks
    assert len(captured) == 2
    assert all(c["keydiff_kv_budget"] == 12 for c in captured)


def test_unsupported_http_mode_fails_before_sse_headers(monkeypatch):
    from fastapi.testclient import TestClient

    import vllm_mlx.server as server

    monkeypatch.setattr(server, "_api_key", None)
    monkeypatch.setattr(server, "_model_name", "tiny-model")
    monkeypatch.setattr(server, "_model_manager", None)
    monkeypatch.setattr(server, "_residency_manager", None)
    monkeypatch.setattr(server, "_default_model_key", None)
    monkeypatch.setattr(server, "get_engine", lambda: object())
    client = TestClient(server.app)
    for endpoint, payload in [
        ("/v1/completions", {"prompt": "1 2 3"}),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "1 2 3"}]}),
    ]:
        for stream in (False, True):
            response = client.post(
                endpoint,
                json={
                    "model": "tiny-model",
                    "stream": stream,
                    "keydiff_kv_budget": 12,
                    **payload,
                },
            )
            assert response.status_code == 400
            assert response.headers["content-type"] == "application/json"
