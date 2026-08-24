from __future__ import annotations

import os
import threading
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
# Some published adapters nest adapter_config.json under a subdirectory instead of the
# repo root; PEFT only looks at the root unless told otherwise. Empty means root, which
# is what every adapter published before v10 uses.
ADAPTER_SUBFOLDER = os.environ.get("RETAIL_BANK_ADAPTER_SUBFOLDER", "").strip().strip("/")


def validate_model_configuration() -> bool:
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
    return bool(ADAPTER_ID)


USE_PEFT_ADAPTER = validate_model_configuration()


class LocalGraniteRuntime:
    """Deterministic Granite runtime quantized at load time for a 12GB CUDA GPU."""

    def __init__(
        self,
        *,
        tokenizer: Any | None,
        model: Any | None,
        torch_module: Any | None,
        cuda_metadata: dict[str, str],
        weight_quantization: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.torch = torch_module
        self.cuda_metadata = dict(cuda_metadata)
        self.weight_quantization = weight_quantization
        self._generation_lock = threading.Lock()

    @classmethod
    def load(cls) -> LocalGraniteRuntime:
        if os.environ.get("LOCAL_POC_SKIP_MODEL_LOAD") == "1":
            return cls(
                tokenizer=None,
                model=None,
                torch_module=None,
                cuda_metadata={
                    "cuda_device_name": "unavailable",
                    "cuda_compute_capability": "unavailable",
                    "cuda_compiled_arch": "unavailable",
                },
                weight_quantization="not-loaded-test-mode",
            )

        try:
            import torch
            from peft import PeftModel
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as error:
            raise RuntimeError(
                "Local Granite inference requires torch, transformers, and bitsandbytes"
            ) from error

        cuda_metadata = validate_cuda_runtime(torch)
        load_repo = BASE_MODEL_ID if USE_PEFT_ADAPTER else MODEL_ID
        load_revision = BASE_MODEL_REVISION if USE_PEFT_ADAPTER else MODEL_REVISION
        tokenizer = AutoTokenizer.from_pretrained(
            ADAPTER_ID if USE_PEFT_ADAPTER else load_repo,
            revision=ADAPTER_REVISION if USE_PEFT_ADAPTER else load_revision,
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=False,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            load_repo,
            revision=load_revision,
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=False,
            **build_model_load_kwargs(
                torch_module=torch,
                quantization_config_factory=BitsAndBytesConfig,
            ),
        )
        if USE_PEFT_ADAPTER:
            model = PeftModel.from_pretrained(
                model,
                ADAPTER_ID,
                revision=ADAPTER_REVISION,
                token=os.environ.get("HF_TOKEN"),
                autocast_adapter_dtype=False,
                **({"subfolder": ADAPTER_SUBFOLDER} if ADAPTER_SUBFOLDER else {}),
            )
        model.eval()
        return cls(
            tokenizer=tokenizer,
            model=model,
            torch_module=torch,
            cuda_metadata=cuda_metadata,
            weight_quantization="bitsandbytes-nf4-double",
        )

    def count_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> int:
        tokenizer, _model, _torch = self._required_components()
        rendered = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
        )
        input_ids = rendered.get("input_ids") if hasattr(rendered, "get") else rendered
        if hasattr(input_ids, "shape"):
            return int(input_ids.shape[-1])
        if not hasattr(input_ids, "__len__"):
            raise RuntimeError("tokenizer did not return countable input IDs")
        return len(input_ids)

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_new_tokens: int,
        *,
        sample: bool = False,
        prefill: str = "",
    ) -> str:
        tokenizer, model, torch = self._required_components()
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
        # A prefill opens the assistant turn for the model: it continues from the
        # given text instead of choosing how to start, so a routed retry cannot
        # name a tool other than the one the router exposed.
        rendered = f"{rendered}{prefill}" if prefill else rendered
        encoded = tokenizer(rendered, return_tensors="pt")
        device = _model_device(model)
        inputs = {
            name: tensor.to(device) for name, tensor in encoded.items() if hasattr(tensor, "to")
        }
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        generation_kwargs = build_generation_kwargs(
            max_new_tokens=max_new_tokens,
            sample=sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        with self._generation_lock, torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)
        prompt_width = int(inputs["input_ids"].shape[-1])
        new_ids = output_ids[0, prompt_width:]
        # Return only what the model produced; the caller joins it with the
        # prefill for parsing, so traces never attribute injected text to the model.
        return str(tokenizer.decode(new_ids, skip_special_tokens=True)).strip()

    def runtime_metadata(self) -> dict[str, str]:
        device = "unavailable" if self.model is None else str(_model_device(self.model))
        metadata = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_loading_mode": "peft_adapter" if USE_PEFT_ADAPTER else "merged",
            "runtime_device": device,
            "weight_quantization": self.weight_quantization,
            **self.cuda_metadata,
        }
        if USE_PEFT_ADAPTER:
            metadata.update(
                {
                    "base_model_id": BASE_MODEL_ID,
                    "base_model_revision": BASE_MODEL_REVISION,
                    "adapter_id": ADAPTER_ID,
                    "adapter_revision": ADAPTER_REVISION,
                }
            )
        if self.model is not None and self.torch is not None:
            allocated = self.torch.cuda.memory_allocated(0) / (1024**3)
            reserved = self.torch.cuda.memory_reserved(0) / (1024**3)
            metadata["cuda_memory_allocated_gib"] = f"{allocated:.2f}"
            metadata["cuda_memory_reserved_gib"] = f"{reserved:.2f}"
        return metadata

    def _required_components(self) -> tuple[Any, Any, Any]:
        if self.tokenizer is None or self.model is None or self.torch is None:
            raise RuntimeError("Local Granite model is unavailable")
        return self.tokenizer, self.model, self.torch


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


def build_model_load_kwargs(
    *,
    torch_module: Any,
    quantization_config_factory: Any,
) -> dict[str, Any]:
    dtype = torch_module.float16
    return {
        "dtype": dtype,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
        "quantization_config": quantization_config_factory(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
    }


def validate_cuda_runtime(torch_module: Any) -> dict[str, str]:
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; local Granite 9B requires a CUDA GPU")
    major, minor = torch_module.cuda.get_device_capability(0)
    compiled_arch = f"sm_{major}{minor}"
    arch_list = set(torch_module.cuda.get_arch_list())
    if compiled_arch not in arch_list:
        raise RuntimeError(
            f"The installed PyTorch build does not include {compiled_arch}; "
            f"compiled architectures: {sorted(arch_list)}"
        )
    return {
        "cuda_device_name": str(torch_module.cuda.get_device_name(0)),
        "cuda_compute_capability": f"{major}.{minor}",
        "cuda_compiled_arch": compiled_arch,
    }


def _model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device
