from __future__ import annotations

import os
from typing import Any

MODEL_ID = os.environ.get(
    "RETAIL_BANK_MODEL_ID",
    "spkc83/retail-bank-servicing-agent-9b",
)
MODEL_REVISION = os.environ.get(
    "RETAIL_BANK_MODEL_REVISION",
    "1799d068906c0da2a8739668857b096d20fed549",
)
SKIP_MODEL_LOAD = os.environ.get("POC_SKIP_MODEL_LOAD") == "1"

if SKIP_MODEL_LOAD:

    class _Spaces:
        @staticmethod
        def GPU(**kwargs: Any) -> Any:
            def decorator(function: Any) -> Any:
                function._zero_gpu_config = dict(kwargs)
                return function

            return decorator

    spaces_runtime: Any = _Spaces()
    tokenizer = None
    model = None
else:
    import spaces
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spaces_runtime = spaces
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to("cuda")
    model.eval()


def count_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> int:
    if tokenizer is None:
        raise RuntimeError("ZeroGPU model tokenizer is unavailable")
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
    )
    input_ids = rendered.get("input_ids") if hasattr(rendered, "get") else rendered
    if not hasattr(input_ids, "__len__"):
        raise RuntimeError("tokenizer did not return countable input IDs")
    return len(input_ids)


def runtime_metadata() -> dict[str, str]:
    if model is None:
        return {
            "runtime_device": "unavailable",
            "cuda_device_name": "unavailable",
        }
    device = str(model.device)
    cuda_device_name = "unavailable"
    try:
        if model.device.type == "cuda" and torch.cuda.is_available():
            cuda_device_name = str(torch.cuda.get_device_name(model.device))
    except (AssertionError, RuntimeError):
        cuda_device_name = "unavailable"
    return {
        "runtime_device": device,
        "cuda_device_name": cuda_device_name,
    }


def generate_text(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    max_new_tokens: int,
) -> str:
    if tokenizer is None or model is None:
        raise RuntimeError("ZeroGPU model is unavailable")
    if not messages or messages[0].get("role") != "system":
        raise ValueError("model messages must begin with a system prompt")
    if not 1 <= max_new_tokens <= 512:
        raise ValueError("max_new_tokens must be between 1 and 512")

    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(rendered, return_tensors="pt")
    inputs = {name: tensor.to(model.device) for name, tensor in encoded.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    new_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    return str(tokenizer.decode(new_ids, skip_special_tokens=True)).strip()
