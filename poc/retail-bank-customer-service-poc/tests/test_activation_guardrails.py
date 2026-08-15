from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import activation_guardrails as guardrails
from activation_guardrails import ActivationObserver, ActivationProbeConfig


class GraniteDecoderLayer(nn.Module):
    def __init__(self, hidden_width: int, *, tuple_output: bool = False) -> None:
        super().__init__()
        self.hidden_size = hidden_width
        self.tuple_output = tuple_output

    def forward(self, hidden_states: torch.Tensor):
        output = hidden_states + 1
        return (output, "cache") if self.tuple_output else output


class FakeGraniteModel(nn.Module):
    def __init__(
        self,
        *,
        layer_count: int = 4,
        hidden_width: int = 4,
        tuple_output: bool = False,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=layer_count,
            hidden_size=hidden_width,
        )
        self.layers = nn.ModuleList(
            GraniteDecoderLayer(hidden_width, tuple_output=tuple_output) for _ in range(layer_count)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            output = layer(hidden_states)
            hidden_states = output[0] if isinstance(output, tuple) else output
        return hidden_states


def observe_config(model: FakeGraniteModel, *layers: int) -> ActivationProbeConfig:
    return ActivationProbeConfig(
        mode="observe",
        layer_indices=layers,
        layer_count=model.config.num_hidden_layers,
        hidden_width=model.config.hidden_size,
    )


def test_off_mode_installs_no_hooks_and_serializes_off_status() -> None:
    model = FakeGraniteModel()
    config = ActivationProbeConfig.from_env(model)

    with ActivationObserver(model, config).capture() as session:
        model(torch.zeros(1, 2, 4))

    assert session.finalize().as_dict() == {
        "schema": "retail-bank-activation-observation/v1",
        "status": "off",
        "mode": "off",
        "config": {
            "layer_indices": [1, 3],
            "layer_count": 4,
            "hidden_width": 4,
            "max_samples_per_layer": 256,
        },
        "runtime_composition": {},
        "layers": [],
    }
    assert all(not layer._forward_hooks for layer in model.layers)


@pytest.mark.parametrize("tuple_output", [False, True])
def test_tensor_and_tuple_outputs_capture_last_sequence_position(tuple_output: bool) -> None:
    model = FakeGraniteModel(layer_count=1, hidden_width=2, tuple_output=tuple_output)
    observer = ActivationObserver(model, observe_config(model, 0))

    with observer.capture() as session:
        model(torch.tensor([[[0.0, 0.0], [2.0, -4.0]]]))

    layer = session.finalize().as_dict()["layers"][0]
    assert layer["prefill"] == {
        "sample_count": 1,
        "rms": 3.0,
        "mean_abs": 3.0,
        "max_abs": 3.0,
        "all_finite": True,
        "seq_width_min": 2,
        "seq_width_max": 2,
    }


def test_prefill_and_decode_samples_are_aggregated_separately() -> None:
    model = FakeGraniteModel(layer_count=1, hidden_width=2)

    with ActivationObserver(model, observe_config(model, 0)).capture() as session:
        model(torch.zeros(1, 3, 2))
        model(torch.full((1, 1, 2), 3.0))

    layer = session.finalize().as_dict()["layers"][0]
    assert layer["prefill"]["sample_count"] == 1
    assert layer["prefill"]["seq_width_min"] == 3
    assert layer["decode"]["sample_count"] == 1
    assert layer["decode"]["seq_width_max"] == 1
    assert layer["decode"]["rms"] == 4.0


def test_observation_does_not_change_model_output() -> None:
    model = FakeGraniteModel(layer_count=2, hidden_width=3)
    inputs = torch.tensor([[[0.5, -1.0, 2.0], [3.0, 4.0, -2.0]]])
    baseline = model(inputs)

    with ActivationObserver(model, observe_config(model, 0, 1)).capture() as session:
        observed = model(inputs)

    assert torch.equal(observed, baseline)
    assert session.finalize().status == "observed"


def test_capture_is_bounded_to_256_samples_per_layer() -> None:
    model = FakeGraniteModel(layer_count=1, hidden_width=2)

    with ActivationObserver(model, observe_config(model, 0)).capture() as session:
        for _ in range(300):
            model(torch.ones(1, 1, 2))

    decode = session.finalize().as_dict()["layers"][0]["decode"]
    assert decode["sample_count"] == 256


def test_capture_removes_hooks_when_model_execution_raises() -> None:
    model = FakeGraniteModel(layer_count=2)
    observer = ActivationObserver(model, observe_config(model, 0, 1))

    with pytest.raises(RuntimeError, match="generation failed"), observer.capture():
        assert all(len(layer._forward_hooks) == 1 for layer in model.layers)
        raise RuntimeError("generation failed")

    assert all(not layer._forward_hooks for layer in model.layers)


def test_hook_failure_does_not_change_model_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeGraniteModel(layer_count=1, hidden_width=2)

    def fail_reduction(_output: object) -> torch.Tensor:
        raise RuntimeError("private details must not escape")

    monkeypatch.setattr(guardrails, "_reduce_output", fail_reduction)
    with ActivationObserver(model, observe_config(model, 0)).capture() as session:
        output = model(torch.ones(1, 1, 2))

    assert torch.equal(output, torch.full((1, 1, 2), 2.0))
    observation = session.finalize().as_dict()
    assert observation["status"] == "unavailable"
    assert observation["code"] == "hook_failure"
    assert "private details" not in json.dumps(observation)


def test_unsupported_output_and_no_samples_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeGraniteModel(layer_count=1)
    observer = ActivationObserver(model, observe_config(model, 0))

    monkeypatch.setattr(model.layers[0], "forward", lambda _hidden_states: {"hidden": "gone"})
    with observer.capture() as unsupported:
        model.layers[0](torch.zeros(1, 1, 4))
    with observer.capture() as empty:
        pass

    assert unsupported.finalize().as_dict()["code"] == "unsupported_output"
    assert empty.finalize().as_dict()["code"] == "no_samples"


def test_observation_contains_only_aggregate_configuration_and_runtime_data() -> None:
    model = FakeGraniteModel(layer_count=1, hidden_width=3)
    observer = ActivationObserver(
        model,
        observe_config(model, 0),
        runtime_composition={"device": "cuda:0", "quantization": "nf4"},
    )

    with observer.capture() as session:
        model(torch.tensor([[[91.125, -73.25, 44.875]]]))

    payload = session.finalize().as_dict()
    serialized = json.dumps(payload, sort_keys=True, allow_nan=False)
    assert set(payload) == {
        "schema",
        "status",
        "mode",
        "config",
        "runtime_composition",
        "layers",
    }
    assert "prompt" not in serialized
    assert "token" not in serialized
    assert "91.125" not in serialized
    assert payload["runtime_composition"] == {"device": "cuda:0", "quantization": "nf4"}


def test_module_shape_mismatch_is_reported_without_installing_hooks() -> None:
    model = FakeGraniteModel(layer_count=2)
    layer = model.layers[1]
    assert isinstance(layer, GraniteDecoderLayer)
    layer.hidden_size = 99

    with ActivationObserver(model, observe_config(model, 0, 1)).capture() as session:
        model(torch.zeros(1, 1, 4))

    assert session.finalize().as_dict()["code"] == "module_mismatch"
    assert all(not layer._forward_hooks for layer in model.layers)


def test_finalize_failure_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeGraniteModel(layer_count=1)

    with ActivationObserver(model, observe_config(model, 0)).capture() as session:
        model(torch.zeros(1, 1, 4))

    monkeypatch.setattr(guardrails, "_aggregate_rows", lambda _rows: 1 / 0)
    assert session.finalize().as_dict()["code"] == "finalize_failure"


def test_env_config_defaults_to_midpoint_and_final_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeGraniteModel(layer_count=6, hidden_width=8)
    monkeypatch.setenv("RETAIL_BANK_MI_MODE", "observe")
    monkeypatch.delenv("RETAIL_BANK_MI_LAYERS", raising=False)

    config = ActivationProbeConfig.from_env(model)

    assert config.mode == "observe"
    assert config.layer_indices == (2, 5)
    assert config.hidden_width == 8


@pytest.mark.parametrize(
    ("mode", "layers"),
    [
        ("enforce", "0"),
        ("observe", ""),
        ("observe", "one,2"),
        ("observe", "0,0"),
        ("observe", "0,4"),
    ],
)
def test_invalid_env_config_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    layers: str,
) -> None:
    model = FakeGraniteModel()
    monkeypatch.setenv("RETAIL_BANK_MI_MODE", mode)
    monkeypatch.setenv("RETAIL_BANK_MI_LAYERS", layers)

    with pytest.raises(ValueError, match="RETAIL_BANK_MI"):
        ActivationProbeConfig.from_env(model)


def test_config_rejects_buffers_over_16_mib() -> None:
    with pytest.raises(ValueError, match="16 MiB"):
        ActivationProbeConfig(
            mode="observe",
            layer_indices=tuple(range(1_639)),
            layer_count=1_639,
            hidden_width=4,
        )
