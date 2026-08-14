from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_gpu_runtime import (
    MODEL_ID,
    MODEL_REVISION,
    LocalGraniteRuntime,
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
    assert MODEL_ID == "spkc83/retail-bank-servicing-agent-9b"
    assert MODEL_REVISION == "1799d068906c0da2a8739668857b096d20fed549"


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
