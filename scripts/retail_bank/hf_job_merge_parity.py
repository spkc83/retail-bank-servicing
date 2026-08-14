# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate==1.12.0",
#   "huggingface-hub==1.22.0",
#   "peft==0.18.1",
#   "safetensors==0.8.0",
#   "torch>=2.9,<3",
#   "transformers==5.13.0",
# ]
# ///
"""Measure BF16 adapter-versus-merged parity without rerunning training."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "ibm-granite/granite-4.1-8b"
BASE_REVISION = "1504002f650e656a0a3789d99574df12e3e94ed0"
DEFAULT_OUTPUT_ROOT = Path("/data/retail-bank-agent-9b-3a6a7efe")
PROMPTS = (
    "Hello, how can you help me today?",
    "Show my account balances.",
    "List my five most recent transactions.",
    "My card ending in 4821 was stolen. Freeze it.",
    "Cancel the pending transfer to River Consulting.",
    "What is the weather tomorrow?",
    "I need a replacement card, but I have not said which card yet.",
    "The transfer failed. What should I do next?",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--merged-subdir", default="merged")
    parser.add_argument("--adapter-subdir", default="adapter")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument(
        "--inference-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--max-input-tokens", type=int, default=192)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def require_artifacts(
    output_root: Path,
    merged_subdir: str,
    adapter_subdir: str = "adapter",
) -> tuple[Path, Path]:
    adapter_dir = output_root / adapter_subdir
    merged_dir = output_root / merged_subdir
    required = (
        adapter_dir / "adapter_model.safetensors",
        adapter_dir / "adapter_config.json",
        merged_dir / "model.safetensors",
        merged_dir / "config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"merge parity artifacts are missing: {missing}")
    return adapter_dir, merged_dir


def render_inputs(
    tokenizer: Any,
    prompts: tuple[str, ...],
    *,
    max_input_tokens: int,
) -> list[dict[str, torch.Tensor]]:
    rendered: list[dict[str, torch.Tensor]] = []
    for prompt in prompts:
        text = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "You are a retail-bank customer-service assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        rendered.append({name: tensor for name, tensor in encoded.items()})
    return rendered


def run_model(
    model: Any,
    batches: list[dict[str, torch.Tensor]],
    *,
    max_new_tokens: int,
    pad_token_id: int,
    eos_token_id: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    device = next(model.parameters()).device
    logits: list[torch.Tensor] = []
    generations: list[torch.Tensor] = []
    model.eval()
    for batch in batches:
        on_device = {name: tensor.to(device) for name, tensor in batch.items()}
        with torch.inference_mode():
            output = model(**on_device)
            generated = model.generate(
                **on_device,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                use_cache=True,
            )
        logits.append(output.logits.detach().float().cpu())
        generations.append(generated[:, on_device["input_ids"].shape[-1] :].detach().cpu())
    return logits, generations


def quantile(values: torch.Tensor, q: float, *, sample_limit: int = 1_000_000) -> float:
    stride = max(1, (values.numel() + sample_limit - 1) // sample_limit)
    sample = values[::stride]
    return float(torch.quantile(sample, q).item())


def compare_outputs(
    adapter_logits: list[torch.Tensor],
    merged_logits: list[torch.Tensor],
    adapter_generations: list[torch.Tensor],
    merged_generations: list[torch.Tensor],
) -> dict[str, Any]:
    differences = torch.cat(
        [
            (adapter - merged).abs().reshape(-1)
            for adapter, merged in zip(
                adapter_logits,
                merged_logits,
                strict=True,
            )
        ]
    )
    adapter_argmax = torch.cat([logits.argmax(dim=-1).reshape(-1) for logits in adapter_logits])
    merged_argmax = torch.cat([logits.argmax(dim=-1).reshape(-1) for logits in merged_logits])
    generation_matches = [
        bool(torch.equal(adapter, merged))
        for adapter, merged in zip(
            adapter_generations,
            merged_generations,
            strict=True,
        )
    ]
    return {
        "prompt_count": len(adapter_logits),
        "compared_logit_count": int(differences.numel()),
        "finite_logit_difference_count": int(torch.isfinite(differences).sum().item()),
        "all_logit_differences_finite": bool(torch.isfinite(differences).all().item()),
        "max_abs_logit_diff": float(differences.max().item()),
        "mean_abs_logit_diff": float(differences.mean().item()),
        "p99_abs_logit_diff": quantile(differences, 0.99),
        "p999_abs_logit_diff": quantile(differences, 0.999),
        "allclose_atol_0_005": bool(
            all(
                torch.allclose(adapter, merged, atol=5e-3, rtol=0.0)
                for adapter, merged in zip(
                    adapter_logits,
                    merged_logits,
                    strict=True,
                )
            )
        ),
        "allclose_atol_0_01": bool(
            all(
                torch.allclose(adapter, merged, atol=1e-2, rtol=0.0)
                for adapter, merged in zip(
                    adapter_logits,
                    merged_logits,
                    strict=True,
                )
            )
        ),
        "allclose_atol_0_02": bool(
            all(
                torch.allclose(adapter, merged, atol=2e-2, rtol=0.0)
                for adapter, merged in zip(
                    adapter_logits,
                    merged_logits,
                    strict=True,
                )
            )
        ),
        "argmax_token_agreement": float((adapter_argmax == merged_argmax).float().mean().item()),
        "greedy_generation_equal_by_prompt": generation_matches,
        "all_greedy_generations_equal": all(generation_matches),
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("HF_TOKEN must be passed as a Hugging Face Job secret")
    adapter_dir, merged_dir = require_artifacts(
        args.output_root,
        args.merged_subdir,
        args.adapter_subdir,
    )
    inference_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.inference_dtype]
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise RuntimeError("tokenizer requires pad and EOS token IDs")
    batches = render_inputs(
        tokenizer,
        PROMPTS,
        max_input_tokens=args.max_input_tokens,
    )

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.base_revision,
        dtype=inference_dtype,
        device_map={"": torch.cuda.current_device()},
    )
    adapter_model = PeftModel.from_pretrained(
        base,
        adapter_dir,
        autocast_adapter_dtype=False,
    )
    adapter_logits, adapter_generations = run_model(
        adapter_model,
        batches,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=int(tokenizer.pad_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
    )
    del adapter_model
    del base
    gc.collect()
    torch.cuda.empty_cache()

    merged_model = AutoModelForCausalLM.from_pretrained(
        merged_dir,
        dtype=inference_dtype,
        device_map={"": torch.cuda.current_device()},
    )
    merged_logits, merged_generations = run_model(
        merged_model,
        batches,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=int(tokenizer.pad_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
    )
    report = {
        "contract": "banking-v3-bf16-merge-parity/v1",
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir),
        "dtype": args.inference_dtype,
        "adapter_autocast_dtype": False,
        "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "prompts": list(PROMPTS),
        "metrics": compare_outputs(
            adapter_logits,
            merged_logits,
            adapter_generations,
            merged_generations,
        ),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    report_path = args.output_root / (
        f"merge_parity_diagnostics_{args.merged_subdir}_{args.inference_dtype}.json"
    )
    write_report(report_path, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
