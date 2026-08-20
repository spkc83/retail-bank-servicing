from __future__ import annotations

import os
from typing import Any

MODEL_ID = os.environ.get(
    "RETAIL_BANK_MODEL_ID",
    "spkc83/retail-bank-servicing-agent-9b-peft-v8-natural-generation",
)
MODEL_REVISION = os.environ.get(
    "RETAIL_BANK_MODEL_REVISION",
    "badbc05ad1f861818ea244b462eda49bca6c6fca",
)
BASE_MODEL_ID = os.environ.get(
    "RETAIL_BANK_BASE_MODEL_ID",
    "spkc83/retail-bank-servicing-agent-9b",
).strip()
BASE_MODEL_REVISION = os.environ.get(
    "RETAIL_BANK_BASE_MODEL_REVISION",
    "1d56824995aa1adecfe20f62ca42fb1c0c443817",
).strip()
ADAPTER_ID = os.environ.get(
    "RETAIL_BANK_ADAPTER_ID",
    "spkc83/retail-bank-servicing-agent-9b-peft-v8-natural-generation",
).strip()
ADAPTER_REVISION = os.environ.get(
    "RETAIL_BANK_ADAPTER_REVISION",
    "badbc05ad1f861818ea244b462eda49bca6c6fca",
).strip()
MODEL_DTYPE = os.environ.get("RETAIL_BANK_MODEL_DTYPE", "bf16").strip().lower()
SKIP_MODEL_LOAD = os.environ.get("POC_SKIP_MODEL_LOAD") == "1"


def _validate_model_configuration() -> bool:
    adapter_values = (BASE_MODEL_ID, BASE_MODEL_REVISION, ADAPTER_ID, ADAPTER_REVISION)
    if any(adapter_values) and not all(adapter_values):
        raise ValueError(
            "PEFT loading requires RETAIL_BANK_BASE_MODEL_ID, "
            "RETAIL_BANK_BASE_MODEL_REVISION, RETAIL_BANK_ADAPTER_ID, and "
            "RETAIL_BANK_ADAPTER_REVISION"
        )
    revisions = (
        ("RETAIL_BANK_MODEL_REVISION", MODEL_REVISION),
        ("RETAIL_BANK_BASE_MODEL_REVISION", BASE_MODEL_REVISION),
        ("RETAIL_BANK_ADAPTER_REVISION", ADAPTER_REVISION),
    )
    for field, revision in revisions:
        if revision and (
            len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise ValueError(f"{field} must be an exact 40-character lowercase Git revision")
    if MODEL_DTYPE not in {"bf16", "fp16"}:
        raise ValueError("RETAIL_BANK_MODEL_DTYPE must be bf16 or fp16")
    return bool(ADAPTER_ID)


USE_PEFT_ADAPTER = _validate_model_configuration()

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
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spaces_runtime = spaces
    load_repo = BASE_MODEL_ID if USE_PEFT_ADAPTER else MODEL_ID
    load_revision = BASE_MODEL_REVISION if USE_PEFT_ADAPTER else MODEL_REVISION
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_ID if USE_PEFT_ADAPTER else load_repo,
        revision=ADAPTER_REVISION if USE_PEFT_ADAPTER else load_revision,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if MODEL_DTYPE == "bf16" else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        load_repo,
        revision=load_revision,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = (
        PeftModel.from_pretrained(
            base_model,
            ADAPTER_ID,
            revision=ADAPTER_REVISION,
            torch_device="cpu",
            autocast_adapter_dtype=False,
        )
        if USE_PEFT_ADAPTER
        else base_model
    )
    model.to("cuda")
    model.eval()


def build_generation_kwargs(
    *,
    max_new_tokens: int,
    sample: bool,
    pad_token_id: int | None,
    eos_token_id: int | None,
) -> dict[str, Any]:
    """Best-of-N candidate 1 stays greedy (sample=False); candidates 2+ sample."""

    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": sample,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
        "use_cache": True,
    }
    if sample:
        # Chosen for Best-of-N candidates 2+ only; candidate 1 never sets this.
        kwargs["temperature"] = 0.7
    return kwargs


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
    composition = {
        "model_loading_mode": "peft_adapter" if USE_PEFT_ADAPTER else "merged",
        "model_dtype": MODEL_DTYPE,
    }
    if USE_PEFT_ADAPTER:
        composition.update(
            {
                "base_model_id": BASE_MODEL_ID,
                "base_model_revision": BASE_MODEL_REVISION,
                "adapter_id": ADAPTER_ID,
                "adapter_revision": ADAPTER_REVISION,
            }
        )
    if model is None:
        return {
            "runtime_device": "unavailable",
            "cuda_device_name": "unavailable",
            **composition,
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
        **composition,
    }


def generate_text(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    max_new_tokens: int,
    *,
    sample: bool = False,
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
    generation_kwargs = build_generation_kwargs(
        max_new_tokens=max_new_tokens,
        sample=sample,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    new_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    return str(tokenizer.decode(new_ids, skip_special_tokens=True)).strip()
