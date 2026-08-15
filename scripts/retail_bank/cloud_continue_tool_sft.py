#!/usr/bin/env python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate==1.12.0",
#   "datasets==4.5.0",
#   "huggingface-hub==1.22.0",
#   "peft==0.18.1",
#   "safetensors==0.8.0",
#   "torch>=2.9,<3",
#   "trackio>=0.33,<0.34",
#   "transformers==5.13.0",
#   "trl==0.26.2",
# ]
# ///
"""Continue the released V5 Granite PEFT adapter on remediation data.

The worker composes an exact root-level LoRA adapter over an exact BF16 base,
retains every record in the training split, evaluates after training, and can
atomically publish only adapter artifacts to a new public repository. It never
merges or republishes base-model weights.
"""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cloud_train_tool_sft import (  # type: ignore[import-not-found]
    PUBLIC_BANKING_TOOL_MANIFEST,
    TRAINING_SEED,
    collate_pretokenized,
    load_manifest_records,
    seed_training,
    sha256_file,
    tf32_supported,
    tokenize_records,
)
from hello_slm.banking_tool_wire import ToolWireAdapter

REMOTE_CONFIRMATION_ENV = "RETAIL_BANK_ALLOW_REMOTE_CONTINUATION_SFT"
REMOTE_CONFIRMATION_VALUE = "banking-v5-peft-remediation"
DATASET_REPO = "spkc83/retail-bank-servicing-alignment-sft"
ADAPTER_REPO = "spkc83/retail-bank-servicing-agent-9b-peft-v5-remediation"
BASE_MODEL = "spkc83/retail-bank-servicing-agent-9b"
BASE_REVISION = "1d56824995aa1adecfe20f62ca42fb1c0c443817"
DEFAULT_SOURCE_ADAPTER_REVISION = "d965816bd6a9252bfb4327c1b0d64f9d34f4a1a2"
DEFAULT_HUB_DEST = "spkc83/retail-bank-servicing-agent-9b-peft-v5-candidate3"
DEFAULT_MANIFEST = "data/banking-servicing-alignment-v5/manifest.json"
DEFAULT_OUTPUT_DIR = "/data/retail-bank-agent-9b-peft-v5-candidate3"
ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_args.bin",
)
POLICY_FAQ_FAMILIES = frozenset(
    {
        "faq_mortgage",
        "faq_mortgage_age",
        "faq_deposit_opening",
        "faq_savings_interest",
        "no_tool_banking_faq",
        "policy_detour",
    }
)


@dataclass(frozen=True)
class ContinuationConfig:
    manifest: Path
    output_dir: Path
    source_adapter_repo: str
    source_adapter_revision: str
    base_model: str
    base_revision: str
    family: str
    hub_dest: str
    max_steps: int
    max_train_seconds: int
    batch_size: int
    gradient_accumulation_steps: int
    max_seq_len: int
    learning_rate: float
    checkpoint_every: int
    positive_multiplier: int
    ambiguity_multiplier: int
    policy_faq_multiplier: int
    tool_outcome_multiplier: int
    dry_run: bool
    allow_remote_execution: bool
    push_to_hub: bool
    trackio_project: str | None
    trackio_run_name: str | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-adapter-repo", default=ADAPTER_REPO)
    parser.add_argument("--source-adapter-revision", default=DEFAULT_SOURCE_ADAPTER_REVISION)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--family", choices=("granite",), default="granite")
    parser.add_argument("--hub-dest", default=DEFAULT_HUB_DEST)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--max-train-seconds", type=int, default=3_600)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--positive-multiplier", type=int, default=10)
    parser.add_argument("--ambiguity-multiplier", type=int, default=5)
    parser.add_argument("--policy-faq-multiplier", type=int, default=4)
    parser.add_argument("--tool-outcome-multiplier", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute-remote", action="store_false", dest="dry_run")
    parser.add_argument("--allow-remote-execution", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--trackio-project")
    parser.add_argument("--trackio-run-name")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ContinuationConfig:
    return ContinuationConfig(
        manifest=Path(args.manifest),
        output_dir=Path(args.output_dir),
        source_adapter_repo=str(args.source_adapter_repo),
        source_adapter_revision=str(args.source_adapter_revision),
        base_model=str(args.base_model),
        base_revision=str(args.base_revision),
        family=str(args.family),
        hub_dest=str(args.hub_dest),
        max_steps=int(args.max_steps),
        max_train_seconds=int(args.max_train_seconds),
        batch_size=int(args.batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        max_seq_len=int(args.max_seq_len),
        learning_rate=float(args.learning_rate),
        checkpoint_every=int(args.checkpoint_every),
        positive_multiplier=int(args.positive_multiplier),
        ambiguity_multiplier=int(args.ambiguity_multiplier),
        policy_faq_multiplier=int(args.policy_faq_multiplier),
        tool_outcome_multiplier=int(args.tool_outcome_multiplier),
        dry_run=bool(args.dry_run),
        allow_remote_execution=bool(args.allow_remote_execution),
        push_to_hub=bool(args.push_to_hub),
        trackio_project=args.trackio_project,
        trackio_run_name=args.trackio_run_name,
    )


def require_exact_revision(value: str, *, field: str) -> None:
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be an exact 40-character lowercase Git revision")


def require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"{field} must be an exact lowercase SHA256")
    if any(char not in "0123456789abcdef" for char in value):
        raise RuntimeError(f"{field} must be an exact lowercase SHA256")
    return value


def validate_pinned_model_inputs(config: ContinuationConfig) -> None:
    owner, separator, name = config.source_adapter_repo.partition("/")
    if not separator or not owner or not name or "/" in name:
        raise RuntimeError("source adapter repository must use owner/name form")
    if config.base_model != BASE_MODEL or config.base_revision != BASE_REVISION:
        raise RuntimeError(f"base must be exactly {BASE_MODEL}@{BASE_REVISION}")
    require_exact_revision(config.source_adapter_revision, field="--source-adapter-revision")
    require_exact_revision(config.base_revision, field="--base-revision")


def remote_execution_allowed(config: ContinuationConfig) -> bool:
    return bool(
        not config.dry_run
        and config.allow_remote_execution
        and os.environ.get(REMOTE_CONFIRMATION_ENV) == REMOTE_CONFIRMATION_VALUE
    )


def assert_remote_execution_allowed(config: ContinuationConfig) -> None:
    if not remote_execution_allowed(config):
        raise PermissionError(
            "Continuation SFT requires --execute-remote, --allow-remote-execution, "
            f"and {REMOTE_CONFIRMATION_ENV}={REMOTE_CONFIRMATION_VALUE}."
        )


def expected_path(record: Mapping[str, Any]) -> str:
    expected = record.get("expected")
    return str(expected.get("path", "")) if isinstance(expected, Mapping) else ""


def scenario_family(record: Mapping[str, Any]) -> str:
    metadata = record.get("metadata")
    return str(metadata.get("scenario_family", "")) if isinstance(metadata, Mapping) else ""


def coreference_target(record: Mapping[str, Any]) -> str:
    metadata = record.get("metadata")
    return str(metadata.get("coreference_target", "")) if isinstance(metadata, Mapping) else ""


def is_coreference_positive_record(record: Mapping[str, Any]) -> bool:
    return (
        scenario_family(record) == "deictic_replace_action"
        and coreference_target(record) == "replace_card"
    )


def is_coreference_ambiguity_record(record: Mapping[str, Any]) -> bool:
    return (
        scenario_family(record) == "deictic_replace_ambiguity"
        and coreference_target(record) == "clarification"
    )


def is_policy_faq_record(record: Mapping[str, Any]) -> bool:
    return scenario_family(record) in POLICY_FAQ_FAMILIES or expected_path(record) in {
        "no_tool_banking_faq",
        "retrieval_grounded_policy",
    }


def is_tool_outcome_record(record: Mapping[str, Any]) -> bool:
    return scenario_family(record) == "tool_outcome_consistency"


def mask_coreference_positive_final_loss(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    masked = deepcopy(list(records))
    for record in masked:
        if not is_coreference_positive_record(record):
            continue
        tool_targets = [
            message
            for message in record["messages"]
            if message.get("role") == "assistant" and message.get("tool_calls")
        ]
        if len(tool_targets) != 1 or tool_targets[0].get("loss") is not True:
            raise RuntimeError("coreference positive must supervise exactly one tool decision")
        for message in reversed(record["messages"]):
            if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                message["loss"] = False
                break
    return masked


def is_regression_record(record: Mapping[str, Any]) -> bool:
    return expected_path(record) in {
        "tool_success",
        "tool_error",
        "no_tool_banking_faq",
        "ood",
        "hard_negative",
    }


def build_continuation_mix(
    records: Sequence[dict[str, Any]],
    *,
    positive_multiplier: int,
    ambiguity_multiplier: int,
    policy_faq_multiplier: int,
    tool_outcome_multiplier: int,
    seed: int = TRAINING_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if (
        min(
            positive_multiplier,
            ambiguity_multiplier,
            policy_faq_multiplier,
            tool_outcome_multiplier,
        )
        < 1
    ):
        raise ValueError("continuation multipliers must be >= 1")
    mixed: list[dict[str, Any]] = []
    stats = {
        "input_records": len(records),
        "coreference_positive_records": 0,
        "coreference_ambiguity_records": 0,
        "policy_faq_records": 0,
        "tool_outcome_records": 0,
        "regression_records": 0,
        "total_weighted_records": 0,
        "positive_multiplier": positive_multiplier,
        "ambiguity_multiplier": ambiguity_multiplier,
        "policy_faq_multiplier": policy_faq_multiplier,
        "tool_outcome_multiplier": tool_outcome_multiplier,
        "all_input_records_retained": True,
    }
    for record in records:
        positive = is_coreference_positive_record(record)
        ambiguity = is_coreference_ambiguity_record(record)
        policy_faq = is_policy_faq_record(record)
        tool_outcome = is_tool_outcome_record(record)
        regression = is_regression_record(record)
        stats["coreference_positive_records"] += int(positive)
        stats["coreference_ambiguity_records"] += int(ambiguity)
        stats["policy_faq_records"] += int(policy_faq)
        stats["tool_outcome_records"] += int(tool_outcome)
        stats["regression_records"] += int(regression)
        weight = max(
            positive_multiplier if positive else 1,
            ambiguity_multiplier if ambiguity else 1,
            policy_faq_multiplier if policy_faq else 1,
            tool_outcome_multiplier if tool_outcome else 1,
        )
        mixed.extend([record] * weight)
    random.Random(seed).shuffle(mixed)
    stats["total_weighted_records"] = len(mixed)
    return mixed, stats


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def dataset_identity(manifest_path: Path) -> dict[str, str]:
    repository = os.environ.get("RETAIL_BANK_TOOL_SFT_DATASET_REPO", "")
    revision = os.environ.get("RETAIL_BANK_TOOL_SFT_DATASET_REVISION", "")
    if repository != DATASET_REPO:
        raise RuntimeError(f"dataset repository must be exactly {DATASET_REPO}")
    require_exact_revision(revision, field="dataset revision")
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 is None:
        raise RuntimeError(f"dataset manifest is unavailable: {manifest_path}")
    return {
        "repository": repository,
        "revision": revision,
        "manifest_sha256": manifest_sha256,
    }


def build_dry_run_plan(config: ContinuationConfig) -> dict[str, Any]:
    return {
        "worker": "cloud_continue_tool_sft",
        "mode": "dry_run" if config.dry_run else "execution_requested",
        "source_adapter_repo": config.source_adapter_repo,
        "source_adapter_revision": config.source_adapter_revision,
        "base_model": config.base_model,
        "base_revision": config.base_revision,
        "manifest": str(config.manifest),
        "output_dir": str(config.output_dir),
        "hub_dest": config.hub_dest,
        "training": {
            "max_steps": config.max_steps,
            "max_train_seconds": config.max_train_seconds,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "max_seq_len": config.max_seq_len,
            "learning_rate": config.learning_rate,
            "checkpoint_every": config.checkpoint_every,
            "positive_multiplier": config.positive_multiplier,
            "ambiguity_multiplier": config.ambiguity_multiplier,
            "policy_faq_multiplier": config.policy_faq_multiplier,
            "tool_outcome_multiplier": config.tool_outcome_multiplier,
            "retained_regression_mix": [
                "single-tool",
                "tool-error",
                "FAQ",
                "OOD",
                "hard-negative",
            ],
        },
        "release": {
            "format": "root-level PEFT adapter",
            "merge": False,
            "evaluation_before_publish": True,
            "new_repository_required": True,
            "frozen_release_gates": "run separately without threshold changes",
            "push_to_hub": config.push_to_hub,
        },
        "remote_guard": {
            "requires_flag": "--allow-remote-execution",
            "requires_execution_switch": "--execute-remote",
            "requires_env": f"{REMOTE_CONFIRMATION_ENV}={REMOTE_CONFIRMATION_VALUE}",
            "currently_allowed": remote_execution_allowed(config),
        },
    }


def build_training_args(config: ContinuationConfig) -> Any:
    from trl import SFTConfig  # type: ignore[import-not-found]

    return SFTConfig(
        output_dir=str(config.output_dir / "trainer"),
        max_steps=config.max_steps,
        max_length=config.max_seq_len,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=max(1, round(config.max_steps * 0.03)),
        bf16=True,
        tf32=tf32_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        assistant_only_loss=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        packing=False,
        logging_steps=10,
        save_steps=config.checkpoint_every,
        eval_strategy="steps",
        eval_steps=max(1, min(config.checkpoint_every, config.max_steps)),
        save_total_limit=2,
        remove_unused_columns=False,
        optim="adamw_torch_fused",
        report_to="trackio" if config.trackio_project else [],
        project=config.trackio_project or "huggingface",
        run_name=config.trackio_run_name or "granite-peft-v5-candidate3",
        push_to_hub=False,
    )


def ensure_unique_release_output(config: ContinuationConfig) -> None:
    if config.output_dir.exists():
        raise RuntimeError(
            f"refusing to overwrite existing continuation output: {config.output_dir}"
        )
    if config.hub_dest == config.source_adapter_repo:
        raise RuntimeError("destination repository must differ from source adapter repository")


def snapshot_source_adapter(config: ContinuationConfig) -> Path:
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    require_exact_revision(config.source_adapter_revision, field="--source-adapter-revision")
    snapshot = Path(
        snapshot_download(
            repo_id=config.source_adapter_repo,
            repo_type="model",
            revision=config.source_adapter_revision,
            allow_patterns=[*ADAPTER_FILES, "training_result.json"],
            token=os.environ.get("HF_TOKEN"),
        )
    )
    missing = [name for name in ADAPTER_FILES if not (snapshot / name).is_file()]
    if missing:
        raise RuntimeError(f"source adapter is missing root-level files: {missing}")
    release = read_json(snapshot / "training_result.json")
    base = release.get("base_model")
    if isinstance(base, Mapping):
        source_base_repo = base.get("repository")
        source_base_revision = base.get("revision")
    else:
        source_base_repo = base
        source_base_revision = release.get("base_revision")
    if source_base_repo != config.base_model or source_base_revision != config.base_revision:
        raise RuntimeError("source adapter provenance does not match the pinned base")
    recorded_adapter_sha = release.get("adapter_model_sha256")
    if recorded_adapter_sha is None:
        recorded_adapter_sha = release.get("adapter_sha256")
    expected_sha = require_sha256(recorded_adapter_sha, field="source adapter SHA256")
    if sha256(snapshot / "adapter_model.safetensors") != expected_sha:
        raise RuntimeError("source adapter digest does not match its release provenance")
    return snapshot


def continuation_fingerprint(
    config: ContinuationConfig,
    adapter: ToolWireAdapter,
    source_snapshot: Path,
    mix_report: Mapping[str, Any],
    dataset: Mapping[str, str],
) -> dict[str, Any]:
    source_commit = os.environ.get("RETAIL_BANK_SOURCE_COMMIT", "")
    require_exact_revision(source_commit, field="source commit")
    return {
        "contract": "banking-v5-peft-remediation-fingerprint/v1",
        "source_commit": source_commit,
        "base_model": config.base_model,
        "base_revision": config.base_revision,
        "base_weight_dtype": "bfloat16",
        "family": config.family,
        "source_adapter": {
            "repository": config.source_adapter_repo,
            "revision": config.source_adapter_revision,
            "adapter_model_sha256": sha256(source_snapshot / "adapter_model.safetensors"),
        },
        "dataset_identity": dict(dataset),
        "template_hash": adapter.template_hash,
        "training_seed": TRAINING_SEED,
        "continuation": {
            "max_steps": config.max_steps,
            "max_train_seconds": config.max_train_seconds,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "max_seq_len": config.max_seq_len,
            "sampling": dict(mix_report),
        },
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_eval_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    eval_loss = metrics.get("eval_loss")
    if isinstance(eval_loss, bool) or not isinstance(eval_loss, int | float):
        raise RuntimeError("post-train evaluation must provide numeric eval_loss")
    if not math.isfinite(float(eval_loss)):
        raise RuntimeError("post-train eval_loss must be finite")
    non_finite = [
        name
        for name, value in metrics.items()
        if isinstance(value, int | float)
        and not isinstance(value, bool)
        and not math.isfinite(float(value))
    ]
    if non_finite:
        raise RuntimeError(f"post-train evaluation contains non-finite metrics: {non_finite}")
    return dict(metrics)


def score_coreference_behavior(
    records: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, Any]:
    positive_results: list[bool] = []
    ambiguity_results: list[bool] = []
    pair_results: dict[str, list[bool]] = {}
    parse_failures = 0
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping) or "coreference_pair_id" not in metadata:
            continue
        record_id = str(record["record_id"])
        target = str(metadata.get("coreference_target", ""))
        prediction = predictions.get(record_id)
        if not isinstance(prediction, Mapping):
            parse_failures += 1
            passed = False
        else:
            calls = list(prediction.get("tool_calls", ()) or ())
            if target == "replace_card":
                expected = record.get("expected")
                expected_calls = (
                    list(expected.get("tool_calls", ()) or ())
                    if isinstance(expected, Mapping)
                    else []
                )
                actual_calls = [
                    {
                        "name": call.get("function", {}).get("name"),
                        "arguments": call.get("function", {}).get("arguments"),
                    }
                    for call in calls
                    if isinstance(call, Mapping) and isinstance(call.get("function"), Mapping)
                ]
                passed = actual_calls == expected_calls
            elif target == "clarification":
                content = str(prediction.get("content") or "").lower()
                passed = not calls and "which" in content and "card" in content
            else:
                raise RuntimeError(f"unexpected coreference target in validation: {target}")
        if target == "replace_card":
            positive_results.append(passed)
        elif target == "clarification":
            ambiguity_results.append(passed)
        else:
            raise RuntimeError(f"unexpected coreference target in validation: {target}")
        pair_results.setdefault(str(metadata["coreference_pair_id"]), []).append(passed)
    if not positive_results or not ambiguity_results or not pair_results:
        raise RuntimeError("coreference behavioral gate requires paired validation records")
    incomplete_pairs = [pair_id for pair_id, results in pair_results.items() if len(results) != 2]
    if incomplete_pairs:
        raise RuntimeError(f"coreference validation pairs are incomplete: {incomplete_pairs}")
    pair_passes = [all(results) for results in pair_results.values()]
    return {
        "positive_tool_argument_accuracy": sum(positive_results) / len(positive_results),
        "ambiguity_accuracy": sum(ambiguity_results) / len(ambiguity_results),
        "pair_flip_accuracy": sum(pair_passes) / len(pair_passes),
        "positive_records": len(positive_results),
        "ambiguity_records": len(ambiguity_results),
        "pairs": len(pair_passes),
        "parse_failures": parse_failures,
    }


def validate_coreference_behavioral_gate(
    metrics: Mapping[str, Any],
    *,
    minimum: float = 0.95,
) -> dict[str, Any]:
    required = (
        "positive_tool_argument_accuracy",
        "ambiguity_accuracy",
        "pair_flip_accuracy",
    )
    failures = []
    for name in required:
        value = metrics.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) < minimum
        ):
            failures.append(f"{name}={value!r}")
    if failures:
        raise RuntimeError(
            f"coreference behavioral gate requires each accuracy >= {minimum:.2f}: "
            + ", ".join(failures)
        )
    return dict(metrics)


def run_coreference_behavioral_gate(
    model: Any,
    tokenizer: Any,
    adapter: ToolWireAdapter,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    predictions: dict[str, Mapping[str, Any] | None] = {}
    was_training = bool(model.training)
    model.eval()
    device = next(model.parameters()).device
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping) or "coreference_pair_id" not in metadata:
            continue
        messages = list(record.get("messages", ()))
        last_user = max(
            index for index, message in enumerate(messages) if message.get("role") == "user"
        )
        rendered = adapter.render_generation(messages[: last_user + 1])
        inputs = {
            key: value.to(device)
            for key, value in rendered.items()
            if key != "tools" and hasattr(value, "to")
        }
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        prompt_width = int(inputs["input_ids"].shape[-1])
        try:
            predictions[str(record["record_id"])] = adapter.parse_assistant(
                output_ids[0, prompt_width:]
            )
        except (TypeError, ValueError):
            predictions[str(record["record_id"])] = None
    if was_training:
        model.train()
    return validate_coreference_behavioral_gate(score_coreference_behavior(records, predictions))


def render_model_card(result: Mapping[str, Any]) -> str:
    return f"""---
license: apache-2.0
base_model: {result["base_model"]}
datasets:
- {result["dataset_identity"]["repository"]}
pipeline_tag: text-generation
tags:
- granite
- retail-banking
- tool-calling
- peft
- lora
---

# Retail Bank Servicing Agent 9B — V5 Remediation PEFT Adapter

This repository contains a continued LoRA adapter only. Load the exact BF16 base
revision and attach this root-level adapter revision with PEFT. No merged base
weights are included.

- Base: `{result["base_model"]}@{result["base_revision"]}`
- Parent adapter: `{result["source_adapter_repo"]}@{result["source_adapter_revision"]}`
- Dataset: `{result["dataset_identity"]["repository"]}@{result["dataset_identity"]["revision"]}`
- Optimizer steps: `{result["steps"]}`
- Adapter SHA256: `{result["adapter_sha256"]}`

Training evaluation completed before publication. Release eligibility still
requires the unchanged frozen behavioral evaluation gates.
"""


def json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def upload_release(
    config: ContinuationConfig,
    *,
    result: Mapping[str, Any],
    result_path: Path,
    metadata_path: Path,
) -> str:
    from huggingface_hub import (  # type: ignore[import-not-found]
        CommitOperationAdd,
        HfApi,
    )

    if config.hub_dest == config.source_adapter_repo:
        raise RuntimeError("destination repository must differ from source adapter repository")
    if result.get("pushed_to_hub") != config.hub_dest:
        raise RuntimeError("training result must record the atomic publication destination")
    eval_metrics = result.get("eval_metrics")
    if not isinstance(eval_metrics, Mapping):
        raise RuntimeError("training result is missing post-train evaluation metrics")
    validate_eval_metrics(eval_metrics)
    behavioral_gate = result.get("coreference_behavioral_gate")
    if not isinstance(behavioral_gate, Mapping):
        raise RuntimeError("training result is missing the coreference behavioral gate")
    validate_coreference_behavioral_gate(behavioral_gate)
    if read_json(result_path) != dict(result):
        raise RuntimeError("training result file does not match the atomic release payload")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(
        config.hub_dest,
        repo_type="model",
        private=False,
        exist_ok=False,
    )
    adapter_dir = config.output_dir / "adapter"
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=adapter_dir / name)
        for name in ADAPTER_FILES
    ]
    operations.extend(
        [
            CommitOperationAdd(path_in_repo="training_result.json", path_or_fileobj=result_path),
            CommitOperationAdd(
                path_in_repo="continuation_training_metadata.json",
                path_or_fileobj=metadata_path,
            ),
            CommitOperationAdd(
                path_in_repo="README.md", path_or_fileobj=render_model_card(result).encode()
            ),
        ]
    )
    commit = api.create_commit(
        repo_id=config.hub_dest,
        repo_type="model",
        operations=operations,
        commit_message="Publish V5 remediation PEFT adapter atomically",
    )
    revision = str(commit.oid)
    require_exact_revision(revision, field="published adapter revision")
    return revision


def run_remote_continuation(config: ContinuationConfig) -> dict[str, Any]:
    assert_remote_execution_allowed(config)
    validate_pinned_model_inputs(config)
    require_exact_revision(os.environ.get("RETAIL_BANK_SOURCE_COMMIT", ""), field="source commit")
    ensure_unique_release_output(config)
    dataset = dataset_identity(config.manifest)
    config.output_dir.mkdir(parents=True, exist_ok=False)
    seed_training(TRAINING_SEED)

    from datasets import Dataset as HfDataset  # type: ignore[import-not-found]
    from peft import PeftModel  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTTrainer  # type: ignore[import-not-found]

    source_snapshot = snapshot_source_adapter(config)
    tokenizer = AutoTokenizer.from_pretrained(source_snapshot, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    wire_adapter = ToolWireAdapter(
        tokenizer,
        family=config.family,
        public_tool_manifest=PUBLIC_BANKING_TOOL_MANIFEST,
        pad_to_max_length=False,
    )
    train_records = load_manifest_records(config.manifest, "train")
    validation_records = load_manifest_records(config.manifest, "validation")
    mixed_train_records, mix_report = build_continuation_mix(
        train_records,
        positive_multiplier=config.positive_multiplier,
        ambiguity_multiplier=config.ambiguity_multiplier,
        policy_faq_multiplier=config.policy_faq_multiplier,
        tool_outcome_multiplier=config.tool_outcome_multiplier,
    )
    train_examples = tokenize_records(
        mask_coreference_positive_final_loss(mixed_train_records),
        wire_adapter,
        max_seq_len=config.max_seq_len,
    )
    validation_examples = tokenize_records(
        mask_coreference_positive_final_loss(validation_records),
        wire_adapter,
        max_seq_len=config.max_seq_len,
    )
    train_dataset = HfDataset.from_dict(
        {
            name: [example[name].tolist() for example in train_examples]
            for name in ("input_ids", "attention_mask", "labels")
        }
    )
    validation_dataset = HfDataset.from_dict(
        {
            name: [example[name].tolist() for example in validation_examples]
            for name in ("input_ids", "attention_mask", "labels")
        }
    )
    base = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        revision=config.base_revision,
        dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
    )
    model = PeftModel.from_pretrained(
        base,
        source_snapshot,
        is_trainable=True,
        autocast_adapter_dtype=False,
    )
    model.enable_input_require_grads()
    model.config.use_cache = False

    class WallClockStopCallback(TrainerCallback):
        def __init__(self, limit_seconds: int) -> None:
            self.limit_seconds = limit_seconds
            self.started_at = 0.0

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, state, control, kwargs
            self.started_at = time.monotonic()

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, state, kwargs
            if time.monotonic() - self.started_at >= self.limit_seconds:
                control.should_training_stop = True
                control.should_save = True
            return control

    fingerprint = continuation_fingerprint(
        config, wire_adapter, source_snapshot, mix_report, dataset
    )
    trainer = SFTTrainer(
        model=model,
        args=build_training_args(config),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=partial(
            collate_pretokenized,
            pad_token_id=int(tokenizer.pad_token_id),
        ),
        processing_class=tokenizer,
        callbacks=[WallClockStopCallback(config.max_train_seconds)],
    )
    train_output = trainer.train()
    if config.trackio_project:
        from transformers.integrations import TrackioCallback  # type: ignore[import-not-found]

        trainer.remove_callback(TrackioCallback)
    eval_metrics = validate_eval_metrics(trainer.evaluate())
    behavioral_gate = run_coreference_behavioral_gate(
        model,
        tokenizer,
        wire_adapter,
        validation_records,
    )
    adapter_dir = config.output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    missing = [name for name in ADAPTER_FILES if not (adapter_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"continued adapter is missing release files: {missing}")
    actual_step = int(trainer.state.global_step)
    metadata = {
        "contract": "banking-v5-peft-remediation-metadata/v1",
        "step": actual_step,
        "created_at_unix": int(time.time()),
        "worker": "cloud_continue_tool_sft",
        "fingerprint": fingerprint,
        "train_metrics": dict(train_output.metrics),
        "eval_metrics": eval_metrics,
        "coreference_behavioral_gate": behavioral_gate,
    }
    metadata_path = config.output_dir / "continuation_training_metadata.json"
    write_json(metadata_path, metadata)
    result = {
        "contract": "banking-v5-peft-remediation-result/v1",
        "worker": "cloud_continue_tool_sft",
        "steps": actual_step,
        "source_adapter_repo": config.source_adapter_repo,
        "source_adapter_revision": config.source_adapter_revision,
        "base_model": config.base_model,
        "base_revision": config.base_revision,
        "base_weight_dtype": "bfloat16",
        "dataset_identity": dataset,
        "template_hash": wire_adapter.template_hash,
        "sampling": mix_report,
        "train_metrics": dict(train_output.metrics),
        "eval_metrics": eval_metrics,
        "coreference_behavioral_gate": behavioral_gate,
        "adapter_sha256": sha256(adapter_dir / "adapter_model.safetensors"),
        "merged_model": None,
        "frozen_release_gates": "pending unchanged frozen evaluation",
        "pushed_to_hub": config.hub_dest if config.push_to_hub else False,
        "publication": {
            "requested": config.push_to_hub,
            "destination_repo": config.hub_dest if config.push_to_hub else None,
            "atomic_bundle": True,
        },
    }
    result_path = config.output_dir / "continuation_training_result.json"
    write_json(result_path, result)
    del trainer
    del model
    del base
    gc.collect()
    torch.cuda.empty_cache()

    if config.push_to_hub:
        revision = upload_release(
            config,
            result=result,
            result_path=result_path,
            metadata_path=metadata_path,
        )
        result["published_adapter_revision"] = revision
        write_json(result_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(parse_args(argv))
    validate_pinned_model_inputs(config)
    if config.max_steps < 1 or config.max_train_seconds < 60:
        raise ValueError("continuation caps must allow at least one step and 60 seconds")
    if config.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if config.dry_run:
        print(json.dumps(build_dry_run_plan(config), indent=2, sort_keys=True))
    else:
        print(json.dumps(run_remote_continuation(config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
