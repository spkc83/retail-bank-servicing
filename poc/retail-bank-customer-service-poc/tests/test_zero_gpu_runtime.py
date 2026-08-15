from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
import torch


def test_zero_gpu_runtime_exposes_generic_generation_and_exact_counting_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POC_SKIP_MODEL_LOAD", "1")
    sys.modules.pop("zero_gpu_runtime", None)
    runtime = importlib.import_module("zero_gpu_runtime")

    assert runtime.MODEL_ID == "spkc83/retail-bank-servicing-agent-9b-peft"
    assert runtime.MODEL_REVISION == "cc95e446af2b5e1d8d9df2751a8192613ad386e3"
    assert not hasattr(runtime, "BANK")
    assert not hasattr(runtime.generate_text, "_zero_gpu_config")
    assert runtime.runtime_metadata() == {
        "runtime_device": "unavailable",
        "cuda_device_name": "unavailable",
        "model_loading_mode": "peft_adapter",
        "model_dtype": "bf16",
        "activation_observer_mode": "unavailable",
        "base_model_id": "spkc83/retail-bank-servicing-agent-9b",
        "base_model_revision": "1d56824995aa1adecfe20f62ca42fb1c0c443817",
        "adapter_id": "spkc83/retail-bank-servicing-agent-9b-peft",
        "adapter_revision": "cc95e446af2b5e1d8d9df2751a8192613ad386e3",
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


def test_zero_gpu_generation_returns_pass_scoped_activation_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POC_SKIP_MODEL_LOAD", "1")
    sys.modules.pop("zero_gpu_runtime", None)
    runtime = importlib.import_module("zero_gpu_runtime")

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
            return "prompt"

        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, _ids: object, **_kwargs: object) -> str:
            return "Observed ZeroGPU answer"

    class FakeModel:
        device = torch.device("cpu")

        def generate(self, **_kwargs: object) -> torch.Tensor:
            return torch.tensor([[1, 2, 3]])

    class FakeSession:
        def finalize(self) -> SimpleNamespace:
            return SimpleNamespace(
                as_dict=lambda: {
                    "schema": "retail-bank-activation-observation/v1",
                    "status": "observed",
                }
            )

    class FakeObserver:
        config = SimpleNamespace(mode="observe")

        @contextmanager
        def capture(self):
            yield FakeSession()

    cast(Any, runtime).tokenizer = FakeTokenizer()
    cast(Any, runtime).model = FakeModel()
    cast(Any, runtime).torch = torch
    cast(Any, runtime).activation_observer = FakeObserver()

    result = runtime.generate_text(
        [{"role": "system", "content": "system"}],
        None,
        8,
    )

    assert result.text == "Observed ZeroGPU answer"
    assert result.activation_observation == {
        "schema": "retail-bank-activation-observation/v1",
        "status": "observed",
    }


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
        config = SimpleNamespace(num_hidden_layers=40, hidden_size=4096)

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
    activation_guardrails = ModuleType("activation_guardrails")

    class FakeActivationProbeConfig:
        @staticmethod
        def from_env(_model: FakeModel) -> SimpleNamespace:
            return SimpleNamespace(mode="off")

    class FakeActivationObserver:
        def __init__(self, _model: FakeModel, config: object, **_kwargs: object) -> None:
            self.config = config

    activation_guardrails.ActivationProbeConfig = FakeActivationProbeConfig  # type: ignore[attr-defined]
    activation_guardrails.ActivationObserver = FakeActivationObserver  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spaces", spaces)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "activation_guardrails", activation_guardrails)
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
