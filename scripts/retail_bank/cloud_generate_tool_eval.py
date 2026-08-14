#!/usr/bin/env python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate==1.12.0",
#   "bitsandbytes==0.50.0",
#   "datasets==4.5.0",
#   "huggingface-hub==1.22.0",
#   "peft==0.18.1",
#   "safetensors==0.8.0",
#   "torch==2.12.1",
#   "transformers==5.13.0",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu126" }
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
# ///
"""Generate read-only banking V5 frozen tool-eval predictions from a merged Hub model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hello_slm.banking_counterfactual_eval_data import (
    COUNTERFACTUAL_GATE_CONTRACT,
    COUNTERFACTUAL_MANIFEST_CONTRACT,
    counterfactual_gate_failures,
    validate_counterfactual_manifest,
)
from hello_slm.banking_tool_eval import (
    StaticPredictionModel,
    TaggedJsonToolAdapter,
    evaluate_records,
    load_predictions_jsonl,
    release_gate_failures,
)
from hello_slm.banking_tool_sft_data import public_tool_manifest
from hello_slm.banking_tool_wire import ToolWireAdapter
from hello_slm.config import canonical_json_bytes

DEFAULT_MODEL_REPO = "spkc83/retail-bank-servicing-agent-9b-peft"
DEFAULT_MODEL_REVISION = "cc95e446af2b5e1d8d9df2751a8192613ad386e3"
DEFAULT_BASE_MODEL_REPO = "spkc83/retail-bank-servicing-agent-9b"
DEFAULT_BASE_MODEL_REVISION = "1d56824995aa1adecfe20f62ca42fb1c0c443817"
DEFAULT_DATASET_REPO = "spkc83/retail-bank-servicing-alignment-sft"
DEFAULT_OUTPUT_DIR = "artifacts/banking-v5-tool-eval"
DEFAULT_FAMILY = "granite"
REVISION_HEX_LENGTH = 40

PUBLIC_BANKING_TOOL_MANIFEST: tuple[dict[str, Any], ...] = tuple(public_tool_manifest())


class ToolEvalGenerationError(ValueError):
    """Raised when frozen tool-eval generation inputs are invalid."""


class GenerationBackend(Protocol):
    tokenizer: Any

    def generate_text(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_new_tokens: int,
    ) -> str:
        """Generate one assistant continuation for already-rendered banking messages."""


@dataclass(frozen=True)
class EvalConfig:
    model_repo: str
    model_revision: str
    dataset_repo: str
    dataset_revision: str
    manifest: Path | None
    output_dir: Path
    predictions_jsonl: Path | None
    metadata_json: Path | None
    split: str
    family: str
    device: str
    dtype: str
    max_new_tokens_first: int
    max_new_tokens_final: int
    max_tool_passes: int
    max_tool_calls: int
    limit: int | None
    trust_remote_code: bool
    push_to_hub: bool
    enforce_release_gates: bool
    token: str | None
    load_in_4bit: bool = False
    base_model_repo: str | None = None
    base_model_revision: str | None = None
    adapter_repo: str | None = None
    adapter_revision: str | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--base-model-repo", default=DEFAULT_BASE_MODEL_REPO)
    parser.add_argument("--base-model-revision", default=DEFAULT_BASE_MODEL_REVISION)
    parser.add_argument("--adapter-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--adapter-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--merged-model-only", action="store_true")
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--predictions-jsonl", type=Path)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--family", choices=("granite",), default=DEFAULT_FAMILY)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-new-tokens-first", type=int, default=192)
    parser.add_argument("--max-new-tokens-final", type=int, default=220)
    parser.add_argument("--max-tool-passes", type=int, default=4)
    parser.add_argument("--max-tool-calls", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load CUDA linear weights with bitsandbytes NF4 double quantization.",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--enforce-release-gates", action="store_true")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        model_repo=str(args.model_repo),
        model_revision=str(args.model_revision),
        dataset_repo=str(args.dataset_repo),
        dataset_revision=str(args.dataset_revision),
        manifest=args.manifest,
        output_dir=args.output_dir,
        predictions_jsonl=args.predictions_jsonl,
        metadata_json=args.metadata_json,
        split=str(args.split),
        family=str(args.family),
        device=str(args.device),
        dtype=str(args.dtype),
        max_new_tokens_first=int(args.max_new_tokens_first),
        max_new_tokens_final=int(args.max_new_tokens_final),
        max_tool_passes=int(args.max_tool_passes),
        max_tool_calls=int(args.max_tool_calls),
        limit=int(args.limit) if args.limit is not None else None,
        trust_remote_code=bool(args.trust_remote_code),
        push_to_hub=bool(args.push_to_hub),
        enforce_release_gates=bool(args.enforce_release_gates),
        token=args.token,
        load_in_4bit=bool(args.load_in_4bit),
        base_model_repo=None if args.merged_model_only else args.base_model_repo,
        base_model_revision=None if args.merged_model_only else args.base_model_revision,
        adapter_repo=None if args.merged_model_only else args.adapter_repo,
        adapter_revision=None if args.merged_model_only else args.adapter_revision,
    )


def validate_exact_revision(value: str, *, field: str) -> None:
    if len(value) != REVISION_HEX_LENGTH or any(char not in "0123456789abcdef" for char in value):
        raise ToolEvalGenerationError(
            f"{field} must be an exact 40-character lowercase Git revision"
        )


def run_eval(config: EvalConfig, backend: GenerationBackend | None = None) -> dict[str, Any]:
    validate_config(config)
    manifest_path = resolve_manifest(config)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    is_counterfactual = manifest_payload.get("contract") == COUNTERFACTUAL_MANIFEST_CONTRACT
    if is_counterfactual:
        validate_counterfactual_manifest(manifest_path)
    records = load_manifest_records(manifest_path, config.split)
    if config.limit is not None:
        records = records[: config.limit]
    if not records:
        raise ToolEvalGenerationError("no records selected for evaluation")

    owns_backend = backend is None
    if backend is None:
        backend = TransformersGenerationBackend(config)
    adapter = ToolWireAdapter(
        backend.tokenizer,
        family=config.family,
        public_tool_manifest=PUBLIC_BANKING_TOOL_MANIFEST,
    )

    output_paths = output_paths_for(config)
    output_paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed_predictions(output_paths["predictions"])
    written = 0
    first_phase = 0
    final_phase = 0
    started = time.time()

    with output_paths["predictions"].open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            record_id = required_str(record, "record_id")
            if record_id not in completed:
                row = generate_record_prediction_row(
                    backend,
                    adapter,
                    record,
                    max_new_tokens_first=config.max_new_tokens_first,
                    max_new_tokens_final=config.max_new_tokens_final,
                    max_tool_passes=config.max_tool_passes,
                    max_tool_calls=config.max_tool_calls,
                )
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                completed.add(record_id)
                written += 1
            first_phase += 1
            if expected_requires_tool(record):
                final_phase += 1

    if owns_backend and hasattr(backend, "close"):
        backend.close()  # type: ignore[attr-defined]

    report = evaluate_records(
        records,
        model=StaticPredictionModel(load_predictions_jsonl(output_paths["predictions"])),
        adapter=TaggedJsonToolAdapter(template_hash=adapter.template_hash),
        checkpoint_revision=active_model_revision(config),
    )
    if is_counterfactual:
        gate_contract = COUNTERFACTUAL_GATE_CONTRACT
        gate_failures = counterfactual_gate_failures(report, records)
    else:
        gate_contract = "banking-tool-release-gate/v1"
        gate_failures = release_gate_failures(report)
    write_json(output_paths["report"], report)
    metadata = build_metadata(
        config,
        manifest_path=manifest_path,
        records=records,
        adapter=adapter,
        predictions_path=output_paths["predictions"],
        first_phase=first_phase,
        final_phase=final_phase,
        written=written,
        elapsed_seconds=time.time() - started,
        report_path=output_paths["report"],
    )
    metadata["release_gate"] = {
        "contract": gate_contract,
        "enforced": config.enforce_release_gates,
        "eligible": not gate_failures,
        "failures": gate_failures,
    }
    write_json(output_paths["metadata"], metadata)
    if config.push_to_hub:
        publish_eval_artifacts(config, output_paths)
    if config.enforce_release_gates and gate_failures:
        raise ToolEvalGenerationError(f"{gate_contract} failed: " + "; ".join(gate_failures))
    return metadata


def validate_config(config: EvalConfig) -> None:
    validate_exact_revision(config.model_revision, field="--model-revision")
    composition = (
        config.base_model_repo,
        config.base_model_revision,
        config.adapter_repo,
        config.adapter_revision,
    )
    if any(composition) and not all(composition):
        raise ToolEvalGenerationError(
            "PEFT evaluation requires --base-model-repo, --base-model-revision, "
            "--adapter-repo, and --adapter-revision"
        )
    if config.adapter_repo is not None:
        assert config.base_model_revision is not None
        assert config.adapter_revision is not None
        validate_exact_revision(config.base_model_revision, field="--base-model-revision")
        validate_exact_revision(config.adapter_revision, field="--adapter-revision")
    if config.dataset_revision.startswith("sha256:"):
        if config.manifest is None:
            raise ToolEvalGenerationError("sha256 dataset identity requires a local --manifest")
        if not config.manifest.is_file():
            raise ToolEvalGenerationError(f"manifest is unavailable: {config.manifest}")
        expected = f"sha256:{sha256_file(config.manifest)}"
        if config.dataset_revision != expected:
            raise ToolEvalGenerationError(
                "--dataset-revision sha256 does not match the local manifest"
            )
    else:
        validate_exact_revision(config.dataset_revision, field="--dataset-revision")
    if config.max_new_tokens_first < 1:
        raise ToolEvalGenerationError("--max-new-tokens-first must be at least 1")
    if config.max_new_tokens_final < 1:
        raise ToolEvalGenerationError("--max-new-tokens-final must be at least 1")
    if config.max_tool_passes < 1:
        raise ToolEvalGenerationError("--max-tool-passes must be at least 1")
    if config.max_tool_calls < 1:
        raise ToolEvalGenerationError("--max-tool-calls must be at least 1")
    if config.limit is not None and config.limit < 1:
        raise ToolEvalGenerationError("--limit must be at least 1")
    if config.load_in_4bit and config.device != "cuda":
        raise ToolEvalGenerationError("--load-in-4bit requires --device cuda")


def resolve_manifest(config: EvalConfig) -> Path:
    if config.manifest is not None:
        if not config.manifest.is_file():
            raise ToolEvalGenerationError(f"manifest is unavailable: {config.manifest}")
        return config.manifest
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ToolEvalGenerationError(
            "huggingface-hub is required to download the dataset"
        ) from exc
    dataset_root = Path(
        snapshot_download(
            repo_id=config.dataset_repo,
            repo_type="dataset",
            revision=config.dataset_revision,
            token=config.token,
        )
    )
    manifest = dataset_root / "manifest.json"
    if not manifest.is_file():
        raise ToolEvalGenerationError(f"dataset manifest is unavailable: {manifest}")
    return manifest


def output_paths_for(config: EvalConfig) -> dict[str, Path]:
    dataset_identity = config.dataset_revision.removeprefix("sha256:")
    slug = f"{active_model_revision(config)[:12]}-{dataset_identity[:12]}-{config.split}"
    return {
        "predictions": config.predictions_jsonl or config.output_dir / f"predictions-{slug}.jsonl",
        "metadata": config.metadata_json or config.output_dir / f"metadata-{slug}.json",
        "report": config.output_dir / f"report-{slug}.json",
    }


def load_manifest_records(manifest_path: Path, split: str) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent
    paths: list[Path] = []
    if "splits" in manifest and split in manifest["splits"]:
        entry = manifest["splits"][split]
        value = entry["path"] if isinstance(entry, Mapping) else entry
        paths.append(resolve_data_path(base_dir, value))
    elif "tool_sft" in manifest:
        for entry in manifest["tool_sft"]:
            if entry.get("name") == split and entry.get("included", True):
                paths.append(resolve_data_path(base_dir, entry["path"]))
    else:
        raise ToolEvalGenerationError(f"manifest {manifest_path} does not declare split {split!r}")

    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    if not records:
        raise ToolEvalGenerationError(f"manifest split {split!r} is empty")
    return records


def resolve_data_path(base_dir: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base_dir / path


def first_phase_messages(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    messages = messages_list(record)
    if expected_requires_tool(record):
        for index, message in enumerate(messages):
            if (
                message.get("role") == "assistant"
                and message.get("tool_calls")
                and message.get("loss", True) is not False
            ):
                return [dict(item) for item in messages[:index]]
    else:
        return [dict(item) for item in messages[: final_assistant_index(record)]]
    raise ToolEvalGenerationError(f"{record.get('record_id')} has no first assistant target")


def grounded_final_phase_messages(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not expected_requires_tool(record):
        raise ToolEvalGenerationError("grounded final phase requires a tool record")
    messages = messages_list(record)
    final_index = final_assistant_index(record)
    prefix = messages[:final_index]
    if not any(message.get("role") == "tool" for message in prefix):
        raise ToolEvalGenerationError(f"{record.get('record_id')} has no canonical tool results")
    return [dict(item) for item in prefix]


def final_assistant_index(record: Mapping[str, Any]) -> int:
    messages = messages_list(record)
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            return index
    raise ToolEvalGenerationError(f"{record.get('record_id')} has no final assistant message")


def messages_list(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ToolEvalGenerationError(f"{record.get('record_id')} is missing messages")
    return messages


def expected_requires_tool(record: Mapping[str, Any]) -> bool:
    expected = record.get("expected")
    if not isinstance(expected, Mapping):
        raise ToolEvalGenerationError(f"{record.get('record_id')} is missing expected metadata")
    return bool(expected.get("requires_tool"))


def generate_record_prediction_row(
    backend: GenerationBackend,
    adapter: ToolWireAdapter,
    record: Mapping[str, Any],
    *,
    max_new_tokens_first: int,
    max_new_tokens_final: int,
    max_tool_passes: int,
    max_tool_calls: int,
) -> dict[str, Any]:
    expected = record.get("expected", {})
    trajectory = generate_iterative_trajectory(
        backend,
        adapter,
        record,
        max_new_tokens_first=max_new_tokens_first,
        max_new_tokens_final=max_new_tokens_final,
        max_tool_passes=max_tool_passes,
        max_tool_calls=max_tool_calls,
    )
    requires_tool = bool(expected.get("requires_tool")) if isinstance(expected, Mapping) else None
    expected_path = expected.get("path") if isinstance(expected, Mapping) else None
    expected_tool_calls = expected.get("tool_calls", []) if isinstance(expected, Mapping) else []
    expected_grounding_facts = (
        expected.get("grounding_facts", []) if isinstance(expected, Mapping) else []
    )
    return {
        "contract": "banking-tool-eval-prediction/v1",
        "record_id": required_str(record, "record_id"),
        "requires_tool": requires_tool,
        "expected_path": expected_path,
        "expected_tool_calls": expected_tool_calls,
        "expected_grounding_facts": expected_grounding_facts,
        **trajectory,
        "created_at": datetime.now(UTC).isoformat(),
    }


def generate_iterative_trajectory(
    backend: GenerationBackend,
    adapter: ToolWireAdapter,
    record: Mapping[str, Any],
    *,
    max_new_tokens_first: int,
    max_new_tokens_final: int,
    max_tool_passes: int,
    max_tool_calls: int,
) -> dict[str, Any]:
    transcript = first_phase_messages(record)
    expected_calls = expected_tool_calls(record)
    canonical_results = canonical_tool_results(record)
    raw_passes: list[str] = []
    pass_reports: list[dict[str, Any]] = []
    ordered_calls: list[dict[str, Any]] = []
    appended_results: list[dict[str, Any]] = []
    next_expected_index = 0
    stop_reason = "max_tool_passes"
    final_raw: str | None = None
    final_parsed: dict[str, Any] | None = None
    final_parse_error: str | None = None
    requires_tool = expected_requires_tool(record)

    for pass_index in range(max_tool_passes):
        max_new_tokens = max_new_tokens_first if pass_index == 0 else max_new_tokens_final
        prompt_count = len(transcript)
        raw = backend.generate_text(transcript, max_new_tokens=max_new_tokens)
        parsed, parse_error = parse_or_error(adapter, raw)
        raw_passes.append(raw)
        pass_report: dict[str, Any] = {
            "pass_index": pass_index,
            "prompt_message_count": prompt_count,
            "raw_output": raw,
            "parsed_assistant": parsed,
            "parse_error": parse_error,
            "appended_result_indexes": [],
        }
        pass_reports.append(pass_report)
        if parse_error is not None or parsed is None:
            stop_reason = "parse_error"
            break

        tool_calls = list(parsed.get("tool_calls", []) or [])
        if not tool_calls:
            final_raw = raw
            final_parsed = parsed
            stop_reason = "final_answer"
            break

        transcript.append(parsed)
        all_pass_calls_matched = True
        for call in tool_calls:
            simplified = simplify_tool_call(call)
            ordered_calls.append(simplified)
            if len(ordered_calls) > max_tool_calls:
                stop_reason = "max_tool_calls"
                all_pass_calls_matched = False
                break
            expected_call = (
                expected_calls[next_expected_index]
                if next_expected_index < len(expected_calls)
                else None
            )
            if expected_call is None or simplified != expected_call:
                stop_reason = "unmatched_tool_call"
                all_pass_calls_matched = False
                break
            if next_expected_index >= len(canonical_results):
                stop_reason = "missing_canonical_tool_result"
                all_pass_calls_matched = False
                break
            result = rebind_tool_result(canonical_results[next_expected_index], call)
            transcript.append(result)
            appended_results.append(
                {
                    "expected_index": next_expected_index,
                    "tool_call_id": result["tool_call_id"],
                    "name": result["name"],
                    "content": result["content"],
                }
            )
            pass_report["appended_result_indexes"].append(next_expected_index)
            next_expected_index += 1
        if not all_pass_calls_matched:
            break

    if final_raw is None and final_parse_error is None:
        final_parse_error = (
            pass_reports[-1]["parse_error"] if pass_reports else "no_generation_pass"
        )
    return {
        "first_assistant_prompt_message_count": (
            pass_reports[0]["prompt_message_count"] if pass_reports else len(transcript)
        ),
        "grounded_final_prompt_message_count": (
            pass_reports[-1]["prompt_message_count"]
            if requires_tool and final_raw is not None
            else 0
        ),
        "first_assistant_raw_output": raw_passes[0] if raw_passes else "",
        "grounded_final_raw_output": final_raw if requires_tool else None,
        "raw_output": "\n".join(raw_passes),
        "raw_passes": raw_passes,
        "pass_reports": pass_reports,
        "ordered_emitted_tool_calls": ordered_calls,
        "appended_tool_results": appended_results,
        "matched_expected_tool_call_count": next_expected_index,
        "stop_reason": stop_reason,
        "first_assistant_parsed": (pass_reports[0]["parsed_assistant"] if pass_reports else None),
        "first_assistant_parse_error": (
            pass_reports[0]["parse_error"] if pass_reports else "no_generation_pass"
        ),
        "grounded_final_parsed": final_parsed if requires_tool else None,
        "grounded_final_parse_error": final_parse_error if requires_tool else None,
    }


def expected_tool_calls(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected = record.get("expected", {})
    if not isinstance(expected, Mapping):
        return []
    calls = expected.get("tool_calls", [])
    return [
        {"name": str(call["name"]), "arguments": dict(call.get("arguments", {}))}
        for call in calls
        if isinstance(call, Mapping)
    ]


def canonical_tool_results(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    target_ids = {
        str(call["id"])
        for message in messages_list(record)
        if message.get("role") == "assistant"
        and message.get("tool_calls")
        and message.get("loss", True) is not False
        for call in message.get("tool_calls", [])
        if isinstance(call, Mapping) and isinstance(call.get("id"), str)
    }
    return [
        dict(message)
        for message in messages_list(record)
        if message.get("role") == "tool" and str(message.get("tool_call_id", "")) in target_ids
    ]


def simplify_tool_call(call: Mapping[str, Any]) -> dict[str, Any]:
    function = call.get("function", {})
    if not isinstance(function, Mapping):
        return {"name": "", "arguments": {}}
    arguments = function.get("arguments", {})
    return {
        "name": str(function.get("name", "")),
        "arguments": dict(arguments) if isinstance(arguments, Mapping) else {},
    }


def rebind_tool_result(
    canonical_result: Mapping[str, Any],
    emitted_call: Mapping[str, Any],
) -> dict[str, Any]:
    function = emitted_call.get("function", {})
    name = function.get("name") if isinstance(function, Mapping) else canonical_result["name"]
    return {
        "role": "tool",
        "tool_call_id": str(emitted_call["id"]),
        "name": str(name),
        "content": canonical_result.get("content"),
        "loss": False,
    }


def parse_or_error(
    adapter: ToolWireAdapter,
    raw_output: str,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return adapter.parse_assistant(raw_output), None
    except ValueError as exc:
        return None, str(exc)


class TransformersGenerationBackend:
    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.adapter: ToolWireAdapter | None = None
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise ToolEvalGenerationError("transformers and torch are required") from exc
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise ToolEvalGenerationError("CUDA device requested but unavailable")
        load_repo = self.config.base_model_repo or self.config.model_repo
        load_revision = self.config.base_model_revision or self.config.model_revision
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.adapter_repo or load_repo,
            revision=self.config.adapter_revision or load_revision,
            token=self.config.token,
            trust_remote_code=self.config.trust_remote_code,
        )
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        load_kwargs = model_load_kwargs(
            self.config,
            torch_module=torch,
            quantization_config_factory=BitsAndBytesConfig,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            load_repo,
            revision=load_revision,
            token=self.config.token,
            trust_remote_code=self.config.trust_remote_code,
            **load_kwargs,
        )
        model = (
            PeftModel.from_pretrained(
                base_model,
                self.config.adapter_repo,
                revision=self.config.adapter_revision,
                token=self.config.token,
                autocast_adapter_dtype=False,
            )
            if self.config.adapter_repo is not None
            else base_model
        )
        if self.config.device != "cuda":
            model.to(self.config.device)
        model.eval()
        self.tokenizer = tokenizer
        self.model = model
        self.adapter = ToolWireAdapter(
            tokenizer,
            family=self.config.family,
            public_tool_manifest=PUBLIC_BANKING_TOOL_MANIFEST,
        )

    def generate_text(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_new_tokens: int,
    ) -> str:
        if self.model is None or self.tokenizer is None or self.adapter is None:
            raise ToolEvalGenerationError("generation backend is not initialized")
        import torch

        rendered = self.adapter.render_generation(messages)
        inputs = {
            key: value.to(self.model.device)
            for key, value in rendered.items()
            if hasattr(value, "to") and key != "tools"
        }
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=getattr(self.tokenizer, "pad_token_id", None)
                or getattr(self.tokenizer, "eos_token_id", None),
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                use_cache=True,
            )
        prompt_width = int(inputs["input_ids"].shape[-1])
        new_tokens = output_ids[0, prompt_width:]
        return str(self.tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()

    def close(self) -> None:
        self.model = None


def model_load_kwargs(
    config: EvalConfig,
    *,
    torch_module: Any,
    quantization_config_factory: Any,
) -> dict[str, Any]:
    dtype_by_name = {
        "bf16": torch_module.bfloat16,
        "fp16": torch_module.float16,
        "fp32": torch_module.float32,
    }
    dtype = dtype_by_name[config.dtype]
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": {"": 0} if config.device == "cuda" else None,
    }
    if config.load_in_4bit:
        kwargs["quantization_config"] = quantization_config_factory(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    return kwargs


def read_completed_predictions(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ToolEvalGenerationError(
                    f"existing predictions JSONL is corrupt at line {line_number}: {path}"
                ) from exc
            if "phase" in row:
                raise ToolEvalGenerationError(
                    f"existing predictions JSONL uses the old phase-row contract: {path}"
                )
            completed.add(required_str(row, "record_id"))
    return completed


def build_metadata(
    config: EvalConfig,
    *,
    manifest_path: Path,
    records: Sequence[Mapping[str, Any]],
    adapter: ToolWireAdapter,
    predictions_path: Path,
    first_phase: int,
    final_phase: int,
    written: int,
    elapsed_seconds: float,
    report_path: Path,
) -> dict[str, Any]:
    return {
        "contract": "banking-tool-eval-run-metadata/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "model": {
            "repo": config.model_repo,
            "revision": config.model_revision,
            "family": config.family,
            "loading_mode": "peft_adapter" if config.adapter_repo else "merged",
            "base": (
                {
                    "repo": config.base_model_repo,
                    "revision": config.base_model_revision,
                }
                if config.adapter_repo
                else None
            ),
            "adapter": (
                {
                    "repo": config.adapter_repo,
                    "revision": config.adapter_revision,
                }
                if config.adapter_repo
                else None
            ),
        },
        "dataset": {
            "repo": config.dataset_repo,
            "revision": config.dataset_revision,
            "split": config.split,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "fingerprint": sha256_json(records),
        },
        "tool_manifest": {
            "count": len(PUBLIC_BANKING_TOOL_MANIFEST),
            "sha256": sha256_json(PUBLIC_BANKING_TOOL_MANIFEST),
            "rendered_with_tokenizer_template_hash": adapter.template_hash,
        },
        "decode": {
            "do_sample": False,
            "max_new_tokens_first": config.max_new_tokens_first,
            "max_new_tokens_final": config.max_new_tokens_final,
            "max_tool_passes": config.max_tool_passes,
            "max_tool_calls": config.max_tool_calls,
            "dtype": config.dtype,
            "device": config.device,
            "weight_quantization": ("bitsandbytes-nf4-double" if config.load_in_4bit else "none"),
        },
        "phases": {
            "first_assistant_records": first_phase,
            "grounded_final_records": final_phase,
        },
        "outputs": {
            "predictions_jsonl": str(predictions_path),
            "predictions_sha256": sha256_file(predictions_path),
            "report_json": str(report_path),
            "report_sha256": sha256_file(report_path),
            "new_rows_written": written,
            "hub_path_prefix": (
                "evaluation/"
                f"{active_model_revision(config)[:12]}-"
                f"{config.dataset_revision.removeprefix('sha256:')[:12]}"
            ),
        },
        "elapsed_seconds": round(elapsed_seconds, 3),
        "read_only_contract": {
            "tool_execution": False,
            "deterministic_output_repair": False,
            "canonical_results_only_for_exact_emitted_calls": True,
            "teacher_forced_unseen_assistant_tool_calls": False,
        },
    }


def publish_eval_artifacts(config: EvalConfig, output_paths: Mapping[str, Path]) -> None:
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as exc:
        raise ToolEvalGenerationError(
            "huggingface-hub is required to publish evaluation artifacts"
        ) from exc
    dataset_identity = config.dataset_revision.removeprefix("sha256:")
    active_revision = active_model_revision(config)
    path_prefix = f"evaluation/{active_revision[:12]}-{dataset_identity[:12]}"
    operations = [
        CommitOperationAdd(
            path_in_repo=f"{path_prefix}/{path.name}",
            path_or_fileobj=path,
        )
        for path in (
            output_paths["predictions"],
            output_paths["metadata"],
            output_paths["report"],
        )
    ]
    HfApi(token=config.token).create_commit(
        repo_id=config.model_repo,
        repo_type="model",
        operations=operations,
        commit_message="Add frozen banking-v3 tool-use evaluation",
        commit_description=(
            f"Model revision {active_revision}; dataset revision {config.dataset_revision}."
        ),
    )


def active_model_revision(config: EvalConfig) -> str:
    return config.adapter_revision or config.model_revision


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def required_str(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ToolEvalGenerationError(f"missing required string field: {field}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    try:
        metadata = run_eval(config_from_args(parse_args(argv)))
    except (OSError, ToolEvalGenerationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
