# SPDX-License-Identifier: Apache-2.0

import gc
import json
import unittest
import weakref

import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from vllm_mlx.kvzap import KVzapCache, _KVzapAttention, _KVzapDecoder
from mlx_lm.models.llama import Model, ModelArgs


class TestKVzap(unittest.TestCase):
    def test_pruned_attention_matches_masked_dense_reference(self):
        mx.random.seed(12)
        length, heads, dim = 19, 3, 64
        scores = mx.array(
            [
                [-8.0 if (t + h) % (h + 2) else -6.0 for h in range(heads)]
                for t in range(length)
            ]
        )[None]
        scorer = nn.Linear(heads, heads, bias=False)
        scorer.weight = mx.eye(heads)
        cache = KVzapCache(scorer, window_size=2)
        k = mx.random.normal((1, heads, length, dim))
        v = mx.random.normal((1, heads, length, dim))
        cache.score(scores)
        cache.update_and_fetch(k, v)
        mx.eval(cache.state)
        cache.compact()

        # Two new queries test bottom-right causal alignment after head pruning.
        q = mx.random.normal((1, heads * 2, 2, dim))
        new_k = mx.random.normal((1, heads, 2, dim))
        new_v = mx.random.normal((1, heads, 2, dim))
        cache.score(mx.zeros((1, 2, heads)))
        keys, values = cache.update_and_fetch(new_k, new_v)
        actual = cache.attention(q, keys, values, dim**-0.5, "causal")
        keep = np.array(scores[0].T >= -7)
        keep[:, -2:] = True
        mask = np.concatenate([keep, np.ones((heads, 2), dtype=bool)], axis=1)
        mask = np.broadcast_to(mask[:, None, :], (heads, 2, length + 2)).copy()
        mask[:, 0, -1] = False
        mask = np.repeat(mask, 2, axis=0)[None]
        expected = mx.fast.scaled_dot_product_attention(
            q,
            mx.concatenate([k, new_k], axis=2),
            mx.concatenate([v, new_v], axis=2),
            scale=dim**-0.5,
            mask=mx.array(mask),
        )
        np.testing.assert_allclose(np.array(actual), np.array(expected), atol=2e-6)
        self.assertEqual(cache.offset, length + 2)
        self.assertEqual([c.offset for c in cache._heads], (keep.sum(1) + 2).tolist())
        for h, c in enumerate(cache._heads):
            retained = np.array(k[:, h : h + 1])[:, :, keep[h]]
            np.testing.assert_array_equal(
                np.array(c.keys[:, :, : len(retained[0, 0])]), retained
            )

    def test_dense_prefill_and_no_pruning_decode_match_full_reference(self):
        from mlx_lm.models.cache import KVCache

        mx.random.seed(7)
        args = ModelArgs(
            model_type="llama",
            hidden_size=128,
            num_hidden_layers=2,
            intermediate_size=256,
            num_attention_heads=2,
            num_key_value_heads=1,
            rms_norm_eps=1e-5,
            vocab_size=64,
        )
        model = Model(args)
        for layer in model.layers:
            layer.self_attn = _KVzapAttention(layer.self_attn)
        tokens = mx.random.randint(0, 64, (1, 19))
        full = [KVCache() for _ in model.layers]
        compressed = [
            KVzapCache(nn.Linear(128, 1), threshold=-float("inf")) for _ in model.layers
        ]
        for start, end in [(0, 8), (8, 16), (16, 17)]:
            expected = model(tokens[:, start:end], cache=full)
            actual = model(tokens[:, start:end], cache=compressed)
            np.testing.assert_allclose(np.array(actual), np.array(expected), atol=1e-5)
        self.assertTrue(all(c._heads is None for c in compressed))
        for cache in compressed:
            cache.compact()
        expected = model(tokens[:, 17:19], cache=full)
        actual = model(tokens[:, 17:19], cache=compressed)
        np.testing.assert_allclose(np.array(actual), np.array(expected), atol=1e-5)
        self.assertEqual(compressed[0].offset, 19)

    def test_physical_storage_drops_and_recent_window_survives(self):
        scorer = nn.Linear(64, 2)
        scorer.weight = mx.zeros((2, 64))
        scorer.bias = mx.full((2,), -8.0)
        cache = KVzapCache(scorer, window_size=128)
        cache.score(mx.zeros((1, 2048, 64)))
        cache.update_and_fetch(mx.ones((1, 2, 2048, 64)), mx.ones((1, 2, 2048, 64)))
        original = cache.nbytes
        mx.eval(cache.state)
        cache.compact()
        cache.update_and_fetch(mx.zeros((1, 2, 1, 64)), mx.zeros((1, 2, 1, 64)))
        self.assertLess(cache.nbytes, original / 4)
        self.assertEqual([c.offset for c in cache._heads], [129, 129])
        self.assertEqual(cache.offset, 2049)
        self.assertEqual(cache._scores, [])


@pytest.fixture(autouse=True)
def local_generation_stream(monkeypatch):
    from importlib import import_module

    module = import_module("mlx_lm.generate")
    monkeypatch.setattr(module, "generation_stream", mx.new_stream(mx.default_device()))


@pytest.mark.parametrize("length", [1, 17, 19])
@pytest.mark.parametrize("prune", [False, True])
def test_layerwise_prefill_preserves_logits_positions_and_processor_history(
    length, prune
):
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import KVCache

    mx.random.seed(9)
    model = Model(
        ModelArgs(
            model_type="llama",
            hidden_size=128,
            num_hidden_layers=3,
            intermediate_size=256,
            num_attention_heads=2,
            num_key_value_heads=1,
            rms_norm_eps=1e-5,
            vocab_size=64,
        )
    )
    mx.eval(model.parameters())
    tokens = mx.random.randint(0, 64, (length,))
    reference_histories = []

    def record_reference(history, logits):
        reference_histories.append(history.tolist())
        return logits

    reference = [
        (token, np.array(logprobs))
        for token, logprobs in generate_step(
            tokens,
            model,
            max_tokens=4,
            prefill_step_size=8,
            prompt_cache=[KVCache() for _ in model.layers],
            logits_processors=[record_reference],
        )
    ]
    decoder = model.model
    caches = []
    for layer in model.layers:
        layer.self_attn = _KVzapAttention(layer.self_attn)
        scorer = nn.Linear(128, 1)
        scorer.weight = mx.zeros((1, 128))
        scorer.bias = mx.array([-8.0])
        caches.append(
            KVzapCache(
                scorer,
                threshold=-7 if prune else -float("inf"),
                window_size=2,
            )
        )
    model.model = _KVzapDecoder(decoder, tokens, 8)
    histories = []

    def record_history(history, logits):
        histories.append(history.tolist())
        return logits

    actual = [
        (token, np.array(logprobs))
        for token, logprobs in generate_step(
            tokens,
            model,
            max_tokens=4,
            prefill_step_size=length,
            prompt_cache=caches,
            logits_processors=[record_history],
        )
    ]
    np.testing.assert_array_equal(actual[0][1], reference[0][1])
    assert histories[0] == reference_histories[0]
    assert all(c.offset == length + 4 for c in caches)
    if not prune:
        assert histories == reference_histories
        assert [t for t, _ in actual] == [t for t, _ in reference]
        for (_, a), (_, b) in zip(actual, reference):
            np.testing.assert_allclose(a, b, atol=1e-5)
    elif length > 2:
        assert all(c._heads[0].offset < c.offset for c in caches)


@pytest.fixture
def tiny_wrapper(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    import vllm_mlx.kvzap as kvzap
    from vllm_mlx.models.llm import MLXLanguageModel

    args = ModelArgs(
        model_type="llama",
        hidden_size=256,
        num_hidden_layers=2,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=128,
    )
    mx.random.seed(3)
    model = Model(args)
    mx.eval(model.parameters())
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "input_dim": 256,
                "output_dim": 2,
                "n_modules": 2,
                "hidden_dim": None,
            }
        )
    )
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            **{f"layers.{i}.weight": mx.zeros((2, 256)) for i in range(2)},
            **{f"layers.{i}.bias": mx.array([-8.0, -6.0]) for i in range(2)},
        },
    )
    monkeypatch.setattr(kvzap, "predictor_path", lambda: tmp_path)
    monkeypatch.setattr(kvzap, "validate_kvzap_model", lambda _: None)
    tokenizer = Tokenizer(
        WordLevel({"<unk>": 0, **{str(i): i for i in range(1, 128)}}, unk_token="<unk>")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>")
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['content'] + ' ' }}{% endfor %}"
    )
    wrapper = MLXLanguageModel("tiny-model")
    wrapper.model, wrapper.tokenizer, wrapper._loaded = model, tokenizer, True
    return wrapper


def test_wrapper_restores_attention_on_close_error_and_next_request(tiny_wrapper):
    model = tiny_wrapper.model
    decoder = model.model
    originals = [layer.self_attn for layer in model.layers]
    prompt = " ".join(str(i % 127 + 1) for i in range(160))
    baseline = list(tiny_wrapper.stream_generate(prompt, max_tokens=3, temperature=0))
    stream = tiny_wrapper.stream_generate(
        prompt, max_tokens=20, temperature=0, kvzap=True
    )
    next(stream)
    assert all(isinstance(layer.self_attn, _KVzapAttention) for layer in model.layers)
    stream.close()
    assert model.model is decoder
    assert all(
        layer.self_attn is original for layer, original in zip(model.layers, originals)
    )
    restored = list(tiny_wrapper.stream_generate(prompt, max_tokens=3, temperature=0))
    assert [c.token for c in restored] == [c.token for c in baseline]

    def fail(tokens, logits):
        raise RuntimeError("processor failed")

    with pytest.raises(RuntimeError, match="processor failed"):
        list(tiny_wrapper.stream_generate(prompt, kvzap=True, logits_processors=[fail]))
    assert model.model is decoder
    assert all(
        layer.self_attn is original for layer, original in zip(model.layers, originals)
    )
    streamed = list(
        tiny_wrapper.stream_generate(prompt, max_tokens=3, temperature=0, kvzap=True)
    )
    generated = tiny_wrapper.generate(prompt, max_tokens=3, temperature=0, kvzap=True)
    assert generated.text == "".join(c.text for c in streamed)
    with pytest.raises(ValueError, match="fresh"):
        list(tiny_wrapper.stream_generate(prompt, prompt_cache=[], kvzap=True))


def test_single_token_prompt(tiny_wrapper):
    output = list(
        tiny_wrapper.stream_generate([1], max_tokens=2, temperature=0, kvzap=True)
    )
    assert len(output) == 2
    assert output[0].prompt_tokens == 1


@pytest.fixture
def tiny_engine(tiny_wrapper, monkeypatch):
    from vllm_mlx.engine.simple import SimpleEngine

    monkeypatch.setattr("vllm_mlx.engine.simple.is_mllm_model", lambda _: False)
    engine = SimpleEngine("tiny-model", prefix_trie_cache=True)
    engine._model, engine._loaded = tiny_wrapper, True
    engine._supports_system_kv_cache = True
    return engine


@pytest.mark.anyio
async def test_chat_cancellation_releases_caches_and_restores_model(
    tiny_engine, monkeypatch
):
    import vllm_mlx.kvzap as kvzap

    refs = []
    factory = kvzap.make_kvzap_cache

    def capture(*args, **kwargs):
        caches = factory(*args, **kwargs)
        refs.extend(weakref.ref(c) for c in caches)
        return caches

    monkeypatch.setattr(kvzap, "make_kvzap_cache", capture)
    model = tiny_engine._model.model
    decoder = model.model
    originals = [layer.self_attn for layer in model.layers]
    messages = [
        {"role": "system", "content": "1 2 3"},
        {"role": "user", "content": " ".join(str(i % 127 + 1) for i in range(160))},
    ]
    stream = tiny_engine.stream_chat(messages, temperature=0, max_tokens=20, kvzap=True)
    try:
        first = await anext(stream)
        assert first.prompt_tokens == 163
        assert refs and all(r()._heads[0].offset < r().offset for r in refs)
        await stream.aclose()
        gc.collect()
        assert model.model is decoder
        assert all(r() is None for r in refs)
        assert all(
            layer.self_attn is original
            for layer, original in zip(model.layers, originals)
        )
        assert not tiny_engine._system_kv_cache
        assert tiny_engine._prefix_trie_cache is None
        output = await tiny_engine.chat(
            messages, temperature=0, max_tokens=3, kvzap=True
        )
        assert output.completion_tokens == 3
        before = len(refs)
        await tiny_engine.generate("1 2 3", temperature=0, max_tokens=3)
        assert len(refs) == before
    finally:
        await stream.aclose()
        await tiny_engine.stop()


@pytest.mark.parametrize(
    "setting", ["_is_mllm", "_mtp", "_max_kv_size", "_draft_model"]
)
def test_incompatible_engine_features_are_rejected(tiny_engine, setting):
    setattr(tiny_engine, setting, True)
    with pytest.raises(ValueError, match="requires text-only"):
        tiny_engine.validate_kvzap()


def test_unsupported_model_is_rejected_before_download(monkeypatch):
    import vllm_mlx.kvzap as kvzap

    def fail():
        pytest.fail("Unsupported model must not download a predictor")

    monkeypatch.setattr(kvzap, "predictor_path", fail)
    with pytest.raises(ValueError, match="Llama 3.1"):
        with kvzap.kvzap_prompt_cache(object(), 10):
            pass


def test_http_options_and_unsupported_engine(tiny_engine):
    from fastapi import HTTPException
    from pydantic import ValidationError

    from vllm_mlx.server import (
        ChatCompletionRequest,
        CompletionRequest,
        _prepare_chat_completion_invocation,
        _validate_kvzap_request,
    )

    for cls, payload in [
        (CompletionRequest, {"prompt": "1 2 3"}),
        (ChatCompletionRequest, {"messages": [{"role": "user", "content": "1 2 3"}]}),
    ]:
        for invalid in (0, 1, 1.5, "true"):
            with pytest.raises(ValidationError):
                cls(model="tiny-model", kvzap=invalid, **payload)
        request = cls(model="tiny-model", kvzap=True, **payload)
        _validate_kvzap_request(tiny_engine, request)
        with pytest.raises(HTTPException) as exc:
            _validate_kvzap_request(object(), request)
        assert exc.value.status_code == 400
    prepared = _prepare_chat_completion_invocation(tiny_engine, request, 4)
    assert prepared.chat_kwargs["kvzap"] is True


@pytest.mark.anyio
async def test_completion_forwards_option_in_both_modes(tiny_engine, monkeypatch):
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
    request = server.CompletionRequest(model="tiny-model", prompt="1 2 3", kvzap=True)
    await server.create_completion(request, raw_request=None)
    chunks = [
        c
        async for c in server.stream_completion(tiny_engine, request.prompt, request, 1)
    ]
    assert chunks
    assert len(captured) == 2
    assert all(c["kvzap"] is True for c in captured)


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
                    "kvzap": True,
                    **payload,
                },
            )
            assert response.status_code == 400
            assert response.headers["content-type"] == "application/json"
