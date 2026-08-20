from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

import local_gpu_runtime as runtime_module
from local_gpu_runtime import (
    MODEL_ID,
    MODEL_REVISION,
    LocalGraniteRuntime,
    build_generation_kwargs,
    build_model_load_kwargs,
    validate_cuda_runtime,
)


class FakeQuantizationConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeCuda:
    def __init__(
        self,
        *,
        available: bool = True,
        capability: tuple[int, int] = (7, 0),
        arch_list: list[str] | None = None,
    ) -> None:
        self.available = available
        self.capability = capability
        self.arch_list = arch_list or ["sm_70", "sm_80"]

    def is_available(self) -> bool:
        return self.available

    def get_device_capability(self, _index: int) -> tuple[int, int]:
        return self.capability

    def get_arch_list(self) -> list[str]:
        return self.arch_list

    def get_device_name(self, _index: int) -> str:
        return "NVIDIA TITAN V"


def fake_torch(cuda: FakeCuda | None = None):
    return SimpleNamespace(float16="fp16", cuda=cuda or FakeCuda())


def test_local_runtime_pins_released_model_identity() -> None:
    assert MODEL_ID == "spkc83/retail-bank-servicing-agent-9b-peft"
    assert MODEL_REVISION == "cc95e446af2b5e1d8d9df2751a8192613ad386e3"


def test_nf4_load_configuration_uses_fp16_double_quantization_on_cuda_zero() -> None:
    kwargs = build_model_load_kwargs(
        torch_module=fake_torch(),
        quantization_config_factory=FakeQuantizationConfig,
    )

    assert kwargs["dtype"] == "fp16"
    assert kwargs["device_map"] == {"": 0}
    assert kwargs["low_cpu_mem_usage"] is True
    assert kwargs["quantization_config"].kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "fp16",
    }


def test_generation_kwargs_default_to_greedy_decoding_with_no_temperature() -> None:
    kwargs = build_generation_kwargs(
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


def test_generation_kwargs_sampling_adds_do_sample_and_temperature() -> None:
    kwargs = build_generation_kwargs(
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


def test_cuda_validation_accepts_titan_v_sm70_build() -> None:
    metadata = validate_cuda_runtime(fake_torch())

    assert metadata == {
        "cuda_device_name": "NVIDIA TITAN V",
        "cuda_compute_capability": "7.0",
        "cuda_compiled_arch": "sm_70",
    }


@pytest.mark.parametrize(
    ("cuda", "message"),
    [
        (FakeCuda(available=False), "CUDA is unavailable"),
        (FakeCuda(arch_list=["sm_80"]), "sm_70"),
    ],
)
def test_cuda_validation_rejects_unusable_runtime(cuda: FakeCuda, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        validate_cuda_runtime(fake_torch(cuda))


def test_explicit_test_skip_does_not_construct_a_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_POC_SKIP_MODEL_LOAD", "1")

    runtime = LocalGraniteRuntime.load()

    assert runtime.runtime_metadata()["runtime_device"] == "unavailable"
    assert runtime.runtime_metadata()["weight_quantization"] == "not-loaded-test-mode"
    with pytest.raises(RuntimeError, match="unavailable"):
        runtime.generate(
            [{"role": "system", "content": "test"}],
            [],
            16,
        )


def test_local_runtime_attaches_adapter_to_quantized_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_revision = "3" * 40
    adapter_revision = "4" * 40
    calls: dict[str, tuple[object, ...]] = {}

    class FakeTokenizer:
        pad_token_id = 0
        eos_token = "<eos>"
        padding_side = "right"

    class FakeModel:
        device = "cuda:0"

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

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    torch = ModuleType("torch")
    torch.float16 = "fp16"  # type: ignore[attr-defined]
    torch.cuda = FakeCuda()  # type: ignore[attr-defined]
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = FakeAutoTokenizer  # type: ignore[attr-defined]
    transformers.AutoModelForCausalLM = FakeAutoModel  # type: ignore[attr-defined]
    transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig  # type: ignore[attr-defined]
    peft = ModuleType("peft")
    peft.PeftModel = FakePeftModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setattr(runtime_module, "BASE_MODEL_ID", "ibm-granite/base")
    monkeypatch.setattr(runtime_module, "BASE_MODEL_REVISION", base_revision)
    monkeypatch.setattr(runtime_module, "ADAPTER_ID", "spkc83/adapter")
    monkeypatch.setattr(runtime_module, "ADAPTER_REVISION", adapter_revision)
    monkeypatch.setattr(runtime_module, "USE_PEFT_ADAPTER", True)
    monkeypatch.delenv("LOCAL_POC_SKIP_MODEL_LOAD", raising=False)

    runtime = LocalGraniteRuntime.load()

    assert calls["tokenizer"][0] == "spkc83/adapter"
    assert calls["tokenizer"][1]["revision"] == adapter_revision  # type: ignore[index]
    assert calls["base"][0] == "ibm-granite/base"
    adapter_call = calls["adapter"]
    assert adapter_call[1:] == (
        "spkc83/adapter",
        {
            "revision": adapter_revision,
            "token": None,
            "autocast_adapter_dtype": False,
        },
    )
    assert runtime.model is adapter_call[0]
    assert runtime.weight_quantization == "bitsandbytes-nf4-double"
