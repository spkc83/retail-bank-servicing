from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

SCHEMA = "retail-bank-activation-observation/v1"
MAX_BUFFER_BYTES = 16 * 1024 * 1024
REDUCED_VALUES_PER_SAMPLE = 5

Mode = Literal["off", "observe"]
Status = Literal["off", "observed", "unavailable"]
ErrorCode = Literal[
    "module_mismatch",
    "unsupported_output",
    "hook_failure",
    "no_samples",
    "finalize_failure",
]


@dataclass(frozen=True, slots=True)
class ActivationProbeConfig:
    mode: Mode
    layer_indices: tuple[int, ...]
    layer_count: int
    hidden_width: int
    max_samples_per_layer: int = 256

    def __post_init__(self) -> None:
        if self.mode not in ("off", "observe"):
            raise ValueError("RETAIL_BANK_MI_MODE must be 'off' or 'observe'")
        if self.layer_count < 1 or self.hidden_width < 1:
            raise ValueError("model activation dimensions must be positive")
        if not self.layer_indices:
            raise ValueError("RETAIL_BANK_MI_LAYERS must select at least one layer")
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("RETAIL_BANK_MI_LAYERS must not contain duplicates")
        if any(index < 0 or index >= self.layer_count for index in self.layer_indices):
            raise ValueError("RETAIL_BANK_MI_LAYERS contains an out-of-range layer")
        if not 1 <= self.max_samples_per_layer <= 256:
            raise ValueError("max_samples_per_layer must be between 1 and 256")
        buffer_bytes = (
            len(self.layer_indices) * self.max_samples_per_layer * REDUCED_VALUES_PER_SAMPLE * 8
        )
        if buffer_bytes > MAX_BUFFER_BYTES:
            raise ValueError("activation observation buffer must not exceed 16 MiB")

    @classmethod
    def from_env(cls, model: Any) -> ActivationProbeConfig:
        layer_count, hidden_width = _model_dimensions(model)
        raw_mode = os.environ.get("RETAIL_BANK_MI_MODE", "off").strip().lower()
        if raw_mode not in ("off", "observe"):
            raise ValueError("RETAIL_BANK_MI_MODE must be 'off' or 'observe'")

        raw_layers = os.environ.get("RETAIL_BANK_MI_LAYERS")
        if raw_layers is None:
            layer_indices = tuple(dict.fromkeys((max(0, layer_count // 2 - 1), layer_count - 1)))
        else:
            parts = [part.strip() for part in raw_layers.split(",")]
            if not parts or any(not part for part in parts):
                raise ValueError("RETAIL_BANK_MI_LAYERS must be comma-separated layer indices")
            try:
                layer_indices = tuple(int(part) for part in parts)
            except ValueError as error:
                raise ValueError(
                    "RETAIL_BANK_MI_LAYERS must be comma-separated layer indices"
                ) from error
        return cls(
            mode=cast(Mode, raw_mode),
            layer_indices=layer_indices,
            layer_count=layer_count,
            hidden_width=hidden_width,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_indices": list(self.layer_indices),
            "layer_count": self.layer_count,
            "hidden_width": self.hidden_width,
            "max_samples_per_layer": self.max_samples_per_layer,
        }


@dataclass(frozen=True, slots=True)
class PhaseAggregate:
    sample_count: int
    rms: float | None
    mean_abs: float | None
    max_abs: float | None
    all_finite: bool
    seq_width_min: int
    seq_width_max: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "rms": self.rms,
            "mean_abs": self.mean_abs,
            "max_abs": self.max_abs,
            "all_finite": self.all_finite,
            "seq_width_min": self.seq_width_min,
            "seq_width_max": self.seq_width_max,
        }


@dataclass(frozen=True, slots=True)
class LayerAggregate:
    layer_index: int
    prefill: PhaseAggregate | None
    decode: PhaseAggregate | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "prefill": None if self.prefill is None else self.prefill.as_dict(),
            "decode": None if self.decode is None else self.decode.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class ActivationObservation:
    status: Status
    mode: Mode
    config: ActivationProbeConfig
    runtime_composition: tuple[tuple[str, str], ...]
    layers: tuple[LayerAggregate, ...] = ()
    code: ErrorCode | None = None
    schema: str = SCHEMA

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "status": self.status,
            "mode": self.mode,
            "config": self.config.as_dict(),
            "runtime_composition": dict(self.runtime_composition),
            "layers": [layer.as_dict() for layer in self.layers],
        }
        if self.code is not None:
            payload["code"] = self.code
        return payload


class ActivationObserver:
    def __init__(
        self,
        model: nn.Module,
        config: ActivationProbeConfig,
        runtime_composition: Mapping[str, str] | None = None,
    ) -> None:
        self._model = model
        self.config = config
        composition = runtime_composition or {}
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in composition.items()
        ):
            raise ValueError("runtime_composition keys and values must be strings")
        self._runtime_composition = tuple(sorted(composition.items()))
        self._layers = () if config.mode == "off" else _discover_layers(model, config)

    @contextmanager
    def capture(self) -> Iterator[ActivationCapture]:
        session = ActivationCapture(self.config, self._runtime_composition)
        handles: list[Any] = []
        try:
            if self.config.mode == "observe":
                layers = self._layers
                if layers is None:
                    session.disable("module_mismatch")
                else:
                    try:
                        for layer_index, module in layers:
                            handles.append(module.register_forward_hook(session.hook(layer_index)))
                    except Exception:
                        session.disable("hook_failure")
            yield session
        finally:
            for handle in handles:
                try:
                    handle.remove()
                except Exception:
                    session.disable("hook_failure")


class ActivationCapture:
    def __init__(
        self,
        config: ActivationProbeConfig,
        runtime_composition: tuple[tuple[str, str], ...],
    ) -> None:
        self._config = config
        self._runtime_composition = runtime_composition
        self._rows: dict[int, list[Tensor]] = {index: [] for index in config.layer_indices}
        self._code: ErrorCode | None = None
        self._finalized: ActivationObservation | None = None

    def hook(
        self,
        layer_index: int,
    ) -> Callable[[nn.Module, tuple[Any, ...], object], None]:
        def capture_output(_module: nn.Module, _inputs: tuple[Any, ...], output: object) -> None:
            if self._code is not None:
                return None
            rows = self._rows[layer_index]
            if len(rows) >= self._config.max_samples_per_layer:
                return None
            try:
                rows.append(_reduce_output(output))
            except _UnsupportedOutputError:
                self.disable("unsupported_output")
            except Exception:
                self.disable("hook_failure")
            return None

        return capture_output

    def disable(self, code: ErrorCode) -> None:
        if self._code is None:
            self._code = code
            self._rows = {index: [] for index in self._config.layer_indices}

    def finalize(self) -> ActivationObservation:
        if self._finalized is not None:
            return self._finalized
        if self._config.mode == "off":
            observation = self._observation("off")
        elif self._code is not None:
            observation = self._observation("unavailable", code=self._code)
        elif not any(self._rows.values()):
            observation = self._observation("unavailable", code="no_samples")
        else:
            try:
                layers = tuple(
                    _aggregate_layer(layer_index, rows)
                    for layer_index, rows in self._rows.items()
                    if rows
                )
                observation = self._observation("observed", layers=layers)
            except Exception:
                observation = self._observation("unavailable", code="finalize_failure")
        self._finalized = observation
        self._rows = {index: [] for index in self._config.layer_indices}
        return observation

    def _observation(
        self,
        status: Status,
        *,
        layers: tuple[LayerAggregate, ...] = (),
        code: ErrorCode | None = None,
    ) -> ActivationObservation:
        return ActivationObservation(
            status=status,
            mode=self._config.mode,
            config=self._config,
            runtime_composition=self._runtime_composition,
            layers=layers,
            code=code,
        )


class _UnsupportedOutputError(Exception):
    pass


def _model_dimensions(model: Any) -> tuple[int, int]:
    config: Any = getattr(model, "config", None)
    try:
        layer_count = int(config.num_hidden_layers)
        hidden_width = int(config.hidden_size)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("model config must define num_hidden_layers and hidden_size") from error
    if layer_count < 1 or hidden_width < 1:
        raise ValueError("model activation dimensions must be positive")
    return layer_count, hidden_width


def _discover_layers(
    model: nn.Module,
    config: ActivationProbeConfig,
) -> tuple[tuple[int, nn.Module], ...] | None:
    modules = [
        module
        for _name, module in model.named_modules()
        if module.__class__.__name__ == "GraniteDecoderLayer"
    ]
    if len(modules) != config.layer_count:
        return None
    if any(_module_hidden_width(module) != config.hidden_width for module in modules):
        return None
    return tuple((index, modules[index]) for index in config.layer_indices)


def _module_hidden_width(module: nn.Module) -> int | None:
    for attribute in ("hidden_size", "hidden_width"):
        value = getattr(module, attribute, None)
        if isinstance(value, int):
            return value
    layer_norm = getattr(module, "input_layernorm", None)
    normalized_shape = getattr(layer_norm, "normalized_shape", None)
    if isinstance(normalized_shape, Sequence) and len(normalized_shape) == 1:
        return int(normalized_shape[0])
    return None


def _reduce_output(output: object) -> Tensor:
    tensor = _first_tensor(output)
    if tensor is None or tensor.ndim != 3 or tensor.shape[1] < 1:
        raise _UnsupportedOutputError
    hidden = tensor[:, -1, :].detach().float()
    absolute = hidden.abs()
    return torch.stack(
        (
            hidden.square().mean().sqrt(),
            absolute.mean(),
            absolute.max(),
            torch.isfinite(hidden).all().to(dtype=torch.float32),
            hidden.new_tensor(tensor.shape[1], dtype=torch.float32),
        )
    )


def _first_tensor(output: object) -> Tensor | None:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return next((item for item in output if isinstance(item, Tensor)), None)
    return None


def _aggregate_layer(layer_index: int, rows: list[Tensor]) -> LayerAggregate:
    # This is the session's sole device-to-CPU transfer for the layer.
    values = torch.stack(rows).detach().to(device="cpu", dtype=torch.float64)
    widths = values[:, 4]
    return LayerAggregate(
        layer_index=layer_index,
        prefill=_aggregate_rows(values[widths > 1]),
        decode=_aggregate_rows(values[widths == 1]),
    )


def _aggregate_rows(rows: Tensor) -> PhaseAggregate | None:
    if rows.shape[0] == 0:
        return None
    return PhaseAggregate(
        sample_count=int(rows.shape[0]),
        rms=_finite_float(rows[:, 0].mean()),
        mean_abs=_finite_float(rows[:, 1].mean()),
        max_abs=_finite_float(rows[:, 2].max()),
        all_finite=bool(rows[:, 3].bool().all()),
        seq_width_min=int(rows[:, 4].min()),
        seq_width_max=int(rows[:, 4].max()),
    )


def _finite_float(value: Tensor) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None
