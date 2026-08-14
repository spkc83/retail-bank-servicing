from __future__ import annotations

import importlib
import sys
from typing import Any, cast

import pytest


def test_zero_gpu_runtime_exposes_generic_generation_and_exact_counting_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POC_SKIP_MODEL_LOAD", "1")
    sys.modules.pop("zero_gpu_runtime", None)
    runtime = importlib.import_module("zero_gpu_runtime")

    assert runtime.MODEL_ID == "spkc83/retail-bank-servicing-agent-9b"
    assert runtime.MODEL_REVISION == "1799d068906c0da2a8739668857b096d20fed549"
    assert not hasattr(runtime, "BANK")
    assert not hasattr(runtime.generate_text, "_zero_gpu_config")
    assert runtime.runtime_metadata() == {
        "runtime_device": "unavailable",
        "cuda_device_name": "unavailable",
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
