from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any, cast

import pytest


def test_zero_gpu_runtime_exposes_generic_generation_and_exact_counting_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POC_SKIP_MODEL_LOAD", "1")
    sys.modules.pop("zero_gpu_runtime", None)
    runtime = importlib.import_module("zero_gpu_runtime")

    assert runtime.MODEL_ID == "spkc83/retail-bank-servicing-agent-9b-peft-v8-natural-generation"
    assert runtime.MODEL_REVISION == "badbc05ad1f861818ea244b462eda49bca6c6fca"
    assert not hasattr(runtime, "BANK")
    assert not hasattr(runtime.generate_text, "_zero_gpu_config")
    assert runtime.runtime_metadata() == {
        "runtime_device": "unavailable",
        "cuda_device_name": "unavailable",
        "model_loading_mode": "peft_adapter",
        "model_dtype": "bf16",
        "base_model_id": "spkc83/retail-bank-servicing-agent-9b",
        "base_model_revision": "1d56824995aa1adecfe20f62ca42fb1c0c443817",
        "adapter_id": "spkc83/retail-bank-servicing-agent-9b-peft-v8-natural-generation",
        "adapter_revision": "badbc05ad1f861818ea244b462eda49bca6c6fca",
    }
    with pytest.raises(RuntimeError, match="unavailable"):
        runtime.generate_text(
            [{"role": "user", "content": "Show my balance."}],
            [],
            512,
        )
    with pytest.raises(RuntimeError, match="unavailable"):
        runtime.count_tokens(
            [{"role": "user", "content": "Show my balance."}],
            [],
        )


def test_generation_kwargs_default_to_greedy_decoding_with_no_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POC_SKIP_MODEL_LOAD", "1")
    sys.modules.pop("zero_gpu_runtime", None)
    runtime = importlib.import_module("zero_gpu_runtime")

    kwargs = runtime.build_generation_kwargs(
        max_new_tokens=64,
        sample=False,
        pad_token_id=0,
        eos_token_id=2,
    )

    assert kwargs == {
        "max_new_tokens": 64,
        "do_sample": False,
        "pad_token_id": 0,
        "eos_token_id": 2,
        "use_cache": True,
    }


def test_generation_kwargs_sampling_adds_do_sample_and_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POC_SKIP_MODEL_LOAD", "1")
    sys.modules.pop("zero_gpu_runtime", None)
    runtime = importlib.import_module("zero_gpu_runtime")

    kwargs = runtime.build_generation_kwargs(
        max_new_tokens=64,
        sample=True,
        pad_token_id=0,
        eos_token_id=2,
    )

    assert kwargs == {
        "max_new_tokens": 64,
        "do_sample": True,
        "pad_token_id": 0,
        "eos_token_id": 2,
        "use_cache": True,
        "temperature": 0.7,
    }


def test_token_count_uses_input_ids_from_batch_encoding_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POC_SKIP_MODEL_LOAD", "1")
    sys.modules.pop("zero_gpu_runtime", None)
    runtime = importlib.import_module("zero_gpu_runtime")

    class FakeTokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return {
                "input_ids": [11, 12, 13, 14],
                "attention_mask": [1, 1, 1, 1],
            }

    cast(Any, runtime).tokenizer = FakeTokenizer()

    assert (
        runtime.count_tokens(
            [{"role": "system", "content": "system"}],
            [],
        )
        == 4
    )


def test_zero_gpu_runtime_loads_exact_base_and_unmerged_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_revision = "1" * 40
    adapter_revision = "2" * 40
    monkeypatch.delenv("POC_SKIP_MODEL_LOAD", raising=False)
    monkeypatch.setenv("RETAIL_BANK_BASE_MODEL_ID", "ibm-granite/base")
    monkeypatch.setenv("RETAIL_BANK_BASE_MODEL_REVISION", base_revision)
    monkeypatch.setenv("RETAIL_BANK_ADAPTER_ID", "spkc83/adapter")
    monkeypatch.setenv("RETAIL_BANK_ADAPTER_REVISION", adapter_revision)
    monkeypatch.setenv("RETAIL_BANK_MODEL_DTYPE", "bf16")
    calls: dict[str, tuple[object, ...] | dict[str, object]] = {}

    class FakeTokenizer:
        pad_token_id = 0
        eos_token = "<eos>"

    class FakeModel:
        def to(self, device: str) -> None:
            calls["device"] = (device,)

        def eval(self) -> None:
            calls["eval"] = ()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(repo: str, **kwargs: object) -> FakeTokenizer:
            calls["tokenizer"] = (repo, kwargs)
            return FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(repo: str, **kwargs: object) -> FakeModel:
            calls["base"] = (repo, kwargs)
            return FakeModel()

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model: FakeModel, repo: str, **kwargs: object) -> FakeModel:
            calls["adapter"] = (model, repo, kwargs)
            return model

    spaces = ModuleType("spaces")
    torch = ModuleType("torch")
    torch.bfloat16 = "bf16"  # type: ignore[attr-defined]
    torch.float16 = "fp16"  # type: ignore[attr-defined]
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = FakeAutoTokenizer  # type: ignore[attr-defined]
    transformers.AutoModelForCausalLM = FakeAutoModel  # type: ignore[attr-defined]
    peft = ModuleType("peft")
    peft.PeftModel = FakePeftModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spaces", spaces)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)
    sys.modules.pop("zero_gpu_runtime", None)

    runtime = importlib.import_module("zero_gpu_runtime")

    assert calls["tokenizer"] == ("spkc83/adapter", {"revision": adapter_revision})
    assert calls["base"] == (
        "ibm-granite/base",
        {
            "revision": base_revision,
            "dtype": "bf16",
            "low_cpu_mem_usage": True,
        },
    )
    adapter_call = calls["adapter"]
    assert isinstance(adapter_call, tuple)
    assert adapter_call[1:] == (
        "spkc83/adapter",
        {
            "revision": adapter_revision,
            "torch_device": "cpu",
            "autocast_adapter_dtype": False,
        },
    )
    assert runtime.model is adapter_call[0]
