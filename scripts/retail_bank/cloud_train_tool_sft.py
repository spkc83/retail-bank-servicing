#!/usr/bin/env python
"""Guarded V5 grounded-policy and dialogue-resume SFT worker.

Default behavior is a dry-run plan. Full BF16 LoRA or QLoRA execution requires
an explicit CLI switch plus an environment confirmation; local tiny smoke uses
small offline stand-ins and never downloads a base model, launches a job, or
pushes to the Hub.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from hello_slm.banking_bare_probe_gate import run_bare_probe_gate
from hello_slm.banking_generation_guidance import messages_with_record_turn_guidance
from hello_slm.banking_tool_sft_data import public_tool_manifest
from hello_slm.banking_tool_wire import IGNORED_LABEL, ToolWireAdapter

REMOTE_CONFIRMATION_ENV = "RETAIL_BANK_ALLOW_REMOTE_TOOL_SFT"
REMOTE_CONFIRMATION_VALUE = "banking-v5-grounded-dialogue-sft"
# Overridable so a gated rerun can escape a bad optimization path instead of
# deterministically reproducing it; the seed lands in the training fingerprint.
TRAINING_SEED = int(os.environ.get("RETAIL_BANK_TRAINING_SEED", "7303"))
DEFAULT_MANIFEST = "data/banking-servicing-alignment-v5/manifest.json"
DEFAULT_OUTPUT_DIR = "artifacts/banking-servicing-agent-v5"
DEFAULT_HUB_DEST = "spkc83/retail-bank-servicing-agent-9b"
DEFAULT_BASE_MODEL = "spkc83/retail-bank-servicing-agent-9b"
DEFAULT_BASE_REVISION = "1d56824995aa1adecfe20f62ca42fb1c0c443817"
DEFAULT_FAMILY = "granite"
DATASET_REPO = "spkc83/retail-bank-servicing-alignment-sft"
COREFERENCE_GATE_MINIMUM = 0.95

LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

PUBLIC_BANKING_TOOL_MANIFEST: tuple[dict[str, Any], ...] = tuple(public_tool_manifest())


@dataclass(frozen=True)
class WorkerConfig:
    manifest: Path
    output_dir: Path
    base_model: str
    base_revision: str
    family: str
    max_steps: int
    max_train_seconds: int
    batch_size: int
    max_seq_len: int
    learning_rate: float
    checkpoint_every: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    precision: str
    gradient_accumulation_steps: int
    warmup_ratio: float
    resume_from: Path | None
    dry_run: bool
    run_tiny_smoke: bool
    allow_remote_execution: bool
    push_to_hub: bool
    hub_dest: str
    merge_adapter: bool
    trackio_project: str | None
    trackio_run_name: str | None
    positive_multiplier: int = 1
    ambiguity_multiplier: int = 1
    policy_faq_multiplier: int = 1
    tool_outcome_multiplier: int = 1


class ToolSftDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, examples: Sequence[dict[str, Tensor]]) -> None:
        self._examples = tuple(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return self._examples[index]


class SimpleToolTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    chat_template = "simple-tool-chat-v1"

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        return_tensors: str | None = None,
    ) -> str | Tensor:
        del tools
        chunks: list[str] = []
        for message in messages:
            role = str(message["role"])
            if role == "assistant" and message.get("tool_calls"):
                payload = {"tool_calls": message["tool_calls"]}
                chunks.append(f"assistant:{json.dumps(payload, sort_keys=True)}")
            elif role == "tool":
                chunks.append(
                    f"tool {message['name']}[{message['tool_call_id']}]:{message['content']}"
                )
            else:
                chunks.append(f"{role}:{message.get('content', '')}")
        if add_generation_prompt:
            chunks.append("assistant:")
        rendered = "\n".join(chunks)
        if not tokenize:
            return rendered
        ids = self(rendered, add_special_tokens=False)["input_ids"]
        tensor = torch.tensor([ids], dtype=torch.long)
        return tensor if return_tensors == "pt" else tensor

    def __call__(
        self, text: str, *, add_special_tokens: bool = False, **_: Any
    ) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [max(3, min(255, ord(char))) for char in text]}

    def decode(self, tokens: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(int(token)) for token in tokens if int(token) > 2)

    def save_pretrained(self, path: str | Path) -> None:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        (output / "simple_tool_tokenizer.json").write_text(
            json.dumps({"chat_template": self.chat_template}, sort_keys=True),
            encoding="utf-8",
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--base-revision", default=DEFAULT_BASE_REVISION)
    parser.add_argument("--family", choices=("granite",), default=DEFAULT_FAMILY)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument(
        "--max-train-seconds",
        type=int,
        default=14_400,
        help="Stop optimizer work at four hours so merge, validation, and Hub upload fit a 5h job.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--precision",
        choices=("bf16-lora", "qlora"),
        default="bf16-lora",
        help="BF16 LoRA is the primary RTX PRO 6000 96GB lane; QLoRA is optional.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--resume-from")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute-remote", action="store_false", dest="dry_run")
    parser.add_argument("--run-tiny-smoke", action="store_true")
    parser.add_argument("--allow-remote-execution", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-dest", default=DEFAULT_HUB_DEST)
    parser.add_argument("--skip-merge-adapter", action="store_false", dest="merge_adapter")
    parser.add_argument("--trackio-project")
    parser.add_argument("--trackio-run-name")
    parser.add_argument(
        "--positive-multiplier",
        type=int,
        default=1,
        help="Repeat coreference positives this many times in the training mix.",
    )
    parser.add_argument("--ambiguity-multiplier", type=int, default=1)
    parser.add_argument("--policy-faq-multiplier", type=int, default=1)
    parser.add_argument("--tool-outcome-multiplier", type=int, default=1)
    return parser.parse_args(argv)


def worker_config_from_args(args: argparse.Namespace) -> WorkerConfig:
    return WorkerConfig(
        manifest=Path(args.manifest),
        output_dir=Path(args.output_dir),
        base_model=str(args.base_model),
        base_revision=str(args.base_revision),
        family=str(args.family),
        max_steps=int(args.max_steps),
        max_train_seconds=int(args.max_train_seconds),
        batch_size=int(args.batch_size),
        max_seq_len=int(args.max_seq_len),
        learning_rate=float(args.learning_rate),
        checkpoint_every=int(args.checkpoint_every),
        lora_rank=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        precision=str(args.precision),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        warmup_ratio=float(args.warmup_ratio),
        resume_from=Path(args.resume_from) if args.resume_from else None,
        dry_run=bool(args.dry_run),
        run_tiny_smoke=bool(args.run_tiny_smoke),
        allow_remote_execution=bool(args.allow_remote_execution),
        push_to_hub=bool(args.push_to_hub),
        hub_dest=str(args.hub_dest),
        merge_adapter=bool(args.merge_adapter),
        trackio_project=args.trackio_project,
        trackio_run_name=args.trackio_run_name,
        positive_multiplier=int(args.positive_multiplier),
        ambiguity_multiplier=int(args.ambiguity_multiplier),
        policy_faq_multiplier=int(args.policy_faq_multiplier),
        tool_outcome_multiplier=int(args.tool_outcome_multiplier),
    )


def remote_execution_allowed(config: WorkerConfig) -> bool:
    return bool(
        not config.dry_run
        and config.allow_remote_execution
        and os.environ.get(REMOTE_CONFIRMATION_ENV) == REMOTE_CONFIRMATION_VALUE
    )


def assert_remote_execution_allowed(config: WorkerConfig) -> None:
    if not remote_execution_allowed(config):
        raise PermissionError(
            "Tool SFT execution requires --execute-remote, --allow-remote-execution, "
            f"and {REMOTE_CONFIRMATION_ENV}={REMOTE_CONFIRMATION_VALUE}."
        )


def package_status() -> dict[str, bool]:
    return {
        name: importlib.util.find_spec(name) is not None
        for name in ("transformers", "trl", "peft", "bitsandbytes", "trackio")
    }


def build_dry_run_plan(config: WorkerConfig) -> dict[str, Any]:
    return {
        "worker": "cloud_train_tool_sft",
        "mode": "dry_run" if config.dry_run else "execution_requested",
        "base_model": config.base_model,
        "base_revision": config.base_revision,
        "family": config.family,
        "manifest": str(config.manifest),
        "output_dir": str(config.output_dir),
        "hub_dest": config.hub_dest,
        "push_to_hub": config.push_to_hub,
        "tool_count": len(PUBLIC_BANKING_TOOL_MANIFEST),
        "training": {
            "stack": [
                "Transformers",
                "TRL SFTTrainer",
                "PEFT LoRA",
                *(["bitsandbytes QLoRA"] if config.precision == "qlora" else ["BF16 base weights"]),
            ],
            "precision": config.precision,
            "max_steps": config.max_steps,
            "max_train_seconds": config.max_train_seconds,
            "batch_size": config.batch_size,
            "max_seq_len": config.max_seq_len,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "target_modules": list(LORA_TARGET_MODULES),
            "checkpoint_every": config.checkpoint_every,
            "trackio_project": config.trackio_project,
            "trackio_run_name": config.trackio_run_name,
            "mix_multipliers": {
                "positive": config.positive_multiplier,
                "ambiguity": config.ambiguity_multiplier,
                "policy_faq": config.policy_faq_multiplier,
                "tool_outcome": config.tool_outcome_multiplier,
            },
        },
        "publication_guard": {
            "hub_dest_differs_from_base": config.hub_dest != config.base_model,
            "requires_empty_destination_repository": True,
            "coreference_behavioral_gate_minimum": COREFERENCE_GATE_MINIMUM,
        },
        "remote_guard": {
            "requires_flag": "--allow-remote-execution",
            "requires_execution_switch": "--execute-remote",
            "requires_env": f"{REMOTE_CONFIRMATION_ENV}={REMOTE_CONFIRMATION_VALUE}",
            "currently_allowed": remote_execution_allowed(config),
        },
        "package_status": package_status(),
        "remote_actions_when_allowed": [
            "load pinned base model and tokenizer",
            "render family chat-template tool definitions",
            "tokenize canonical tool SFT records with assistant-only labels",
            f"train a PEFT adapter through TRL SFTTrainer with {config.precision}",
            "write checkpoint metadata with resume fingerprints",
            "optionally merge adapter and verify reload parity",
            "optionally push only when guarded execution and --push-to-hub are set",
        ],
        "will_not_do_without_guard": [
            "download 9B base weights",
            "start a paid/cloud job",
            "write to Hugging Face Hub",
            "merge or publish a checkpoint",
        ],
    }


def load_manifest_records(manifest_path: Path, split: str) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("training_allowed") is False or manifest.get("contract") == (
        "banking-counterfactual-eval-manifest/v1"
    ):
        raise ValueError(f"evaluation-only manifest cannot be used for training: {manifest_path}")
    base_dir = manifest_path.parent
    paths: list[Path] = []
    if "splits" in manifest and split in manifest["splits"]:
        entry = manifest["splits"][split]
        value = entry["path"] if isinstance(entry, Mapping) else entry
        declared = Path(value)
        paths.append(
            declared.resolve() if declared.is_absolute() else (base_dir / declared).resolve()
        )
    elif "tool_sft" in manifest:
        for entry in manifest["tool_sft"]:
            if entry.get("name") == split and entry.get("included", True):
                declared = Path(entry["path"])
                paths.append(
                    declared.resolve()
                    if declared.is_absolute()
                    else (base_dir / declared).resolve()
                )
    else:
        raise ValueError(f"manifest {manifest_path} does not declare split {split!r}")

    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    if not records:
        raise ValueError(f"manifest split {split!r} is empty")
    non_trainable = [
        str(record.get("record_id", "<missing>"))
        for record in records
        if record.get("metadata", {}).get("trainable") is False
    ]
    if non_trainable:
        raise ValueError(
            "evaluation-only records cannot be used for training: " + ", ".join(non_trainable[:3])
        )
    return records


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_identity(manifest_path: Path) -> dict[str, str | None]:
    return {
        "repository": os.environ.get("RETAIL_BANK_TOOL_SFT_DATASET_REPO"),
        "revision": os.environ.get("RETAIL_BANK_TOOL_SFT_DATASET_REVISION"),
        "manifest_sha256": sha256_file(manifest_path),
    }


def validate_hub_destination(config: WorkerConfig) -> None:
    """The from-scratch lane must publish a new repository, never its own training base."""

    if config.hub_dest == config.base_model:
        raise RuntimeError(
            f"hub destination {config.hub_dest!r} must differ from the training base model; "
            "publishing into the base repository would overwrite the weights this run "
            "trains from"
        )


def validate_dataset_identity(manifest_path: Path) -> dict[str, str]:
    """Pin the dataset the same way the continuation worker does before spending GPU time."""

    identity = dataset_identity(manifest_path)
    repository = identity.get("repository")
    revision = identity.get("revision")
    if repository != DATASET_REPO:
        raise RuntimeError(f"dataset repository must be exactly {DATASET_REPO}, got {repository!r}")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(char not in "0123456789abcdef" for char in revision)
    ):
        raise RuntimeError(
            f"dataset revision must be an exact 40-character lowercase revision, got {revision!r}"
        )
    if identity.get("manifest_sha256") is None:
        raise RuntimeError(f"dataset manifest is unavailable: {manifest_path}")
    return {
        "repository": repository,
        "revision": revision,
        "manifest_sha256": str(identity["manifest_sha256"]),
    }


def continuation_module() -> Any:
    """Import the continuation worker lazily.

    ``cloud_continue_tool_sft`` imports this module at module scope, so a top-level
    import here would be circular. Importing inside a call keeps the shared mix and
    behavioural-gate helpers available without duplicating them.
    """

    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import cloud_continue_tool_sft  # noqa: PLC0415

    return cloud_continue_tool_sft


def build_training_mix(
    config: WorkerConfig,
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Return the training records plus mix stats, or the untouched records when unweighted."""

    multipliers = (
        config.positive_multiplier,
        config.ambiguity_multiplier,
        config.policy_faq_multiplier,
        config.tool_outcome_multiplier,
    )
    if min(multipliers) < 1:
        raise ValueError("training mix multipliers must be >= 1")
    if max(multipliers) == 1:
        return [dict(record) for record in records], None
    continuation = continuation_module()
    masked = continuation.mask_coreference_positive_final_loss([dict(record) for record in records])
    mixed, stats = continuation.build_continuation_mix(
        masked,
        positive_multiplier=config.positive_multiplier,
        ambiguity_multiplier=config.ambiguity_multiplier,
        policy_faq_multiplier=config.policy_faq_multiplier,
        tool_outcome_multiplier=config.tool_outcome_multiplier,
        seed=TRAINING_SEED,
    )
    return mixed, dict(stats)


def enable_generation_cache(model: Any) -> None:
    """Gradient checkpointing forces ``use_cache=False``; greedy gate decoding needs it back."""

    disable = getattr(model, "gradient_checkpointing_disable", None)
    if callable(disable):
        disable()
    for holder in ("config", "generation_config"):
        holder_value = getattr(model, holder, None)
        if holder_value is not None and hasattr(holder_value, "use_cache"):
            holder_value.use_cache = True
    base = getattr(model, "base_model", None)
    base_config = getattr(base, "config", None)
    if base_config is not None and hasattr(base_config, "use_cache"):
        base_config.use_cache = True
    model.eval()


def coreference_gate_records(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        record
        for record in records
        if isinstance(record.get("metadata"), Mapping)
        and "coreference_pair_id" in record["metadata"]
    ]


def run_coreference_behavioral_gates(
    *,
    model: Any,
    tokenizer: Any,
    adapter: Any,
    validation_records: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    output_dir: Path,
    step: int,
    shadow_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run the dev and shadow coreference gates, persisting both reports.

    ``shadow_records`` are normally loaded (and validated) before training starts so a
    malformed manifest fails before the GPU spend; they are re-read here only when the
    caller did not pre-load them.

    Raises ``RuntimeError`` when either gate falls below the minimum, which keeps the
    trained adapter on the job bucket for diagnosis and stops the run before publication.
    """

    continuation = continuation_module()
    dev_records = coreference_gate_records(validation_records)
    if not dev_records:
        raise RuntimeError("validation split has no coreference gate records")
    evaluations = output_dir / "behavioral-evaluations"
    dev_report = continuation.generate_coreference_behavior_report(
        model,
        tokenizer,
        adapter,
        dev_records,
        cumulative_step=step,
    )
    dev_gate = continuation.persist_and_validate_coreference_report(
        dev_report,
        evaluations / f"dev-step-{step}.json",
    )
    if shadow_records is None:
        shadow_records = continuation.load_shadow_gate_records(manifest_path)
    shadow_report = continuation.generate_coreference_behavior_report(
        model,
        tokenizer,
        adapter,
        shadow_records,
        cumulative_step=step,
    )
    shadow_gate = continuation.persist_and_validate_coreference_report(
        shadow_report,
        evaluations / f"shadow-step-{step}.json",
    )
    return {"dev": dict(dev_gate), "shadow": dict(shadow_gate)}


def destination_repo_state(api: Any, repo_id: str) -> str:
    from huggingface_hub.errors import RepositoryNotFoundError  # type: ignore[import-not-found]

    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
    except RepositoryNotFoundError:
        return "absent"
    files = list(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    return "empty" if not files else "nonempty"


def require_publishable_destination(api: Any, repo_id: str) -> str:
    state = destination_repo_state(api, repo_id)
    if state == "nonempty":
        raise RuntimeError(f"destination model repository is not empty: {repo_id}")
    return state


def preflight_destination_repo(config: WorkerConfig) -> str:
    from huggingface_hub import HfApi  # type: ignore[import-not-found]

    return require_publishable_destination(
        HfApi(token=os.environ.get("HF_TOKEN")),
        config.hub_dest,
    )


def training_fingerprint(config: WorkerConfig, adapter: ToolWireAdapter) -> dict[str, Any]:
    """Identify a run by everything that changes the weights it produces.

    This previously recorded the base, dataset, LoRA shape, precision and seed
    and nothing else, so a 1e-4/batch-2/2000-step run and a 2e-5/batch-8/8000-step
    run fingerprinted identically. Two consequences: an adapter could not be
    traced back to the run that made it, and ``validate_resume_fingerprint``
    would resume a checkpoint into a materially different optimisation and call
    it a match. The ``optimization`` block mirrors the continuation lane's,
    which has carried these fields all along.

    Widening this deliberately invalidates resume against checkpoints written
    by the older, narrower fingerprint -- they genuinely do not record enough
    to prove compatibility.
    """
    return {
        "base_model": config.base_model,
        "base_revision": config.base_revision,
        "family": config.family,
        "template_hash": adapter.template_hash,
        "dataset_identity": dataset_identity(config.manifest),
        "training_seed": TRAINING_SEED,
        "precision": config.precision,
        "lora": {
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": list(LORA_TARGET_MODULES),
        },
        "optimization": {
            "max_steps": config.max_steps,
            "max_train_seconds": config.max_train_seconds,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "max_seq_len": config.max_seq_len,
            "warmup_ratio": config.warmup_ratio,
        },
        "training_mix": {
            "positive_multiplier": config.positive_multiplier,
            "ambiguity_multiplier": config.ambiguity_multiplier,
            "policy_faq_multiplier": config.policy_faq_multiplier,
            "tool_outcome_multiplier": config.tool_outcome_multiplier,
        },
    }


def validate_resume_fingerprint(resume_from: Path, expected: Mapping[str, Any]) -> None:
    metadata_path = resume_from / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("fingerprint") != expected:
        raise ValueError("resume fingerprint does not match current training inputs")


def save_trainer_checkpoint_metadata(
    output_dir: Path,
    *,
    step: int,
    fingerprint: Mapping[str, Any],
) -> Path:
    checkpoint = output_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    path = checkpoint / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "contract": "banking-tool-sft-resume/v1",
                "step": step,
                "worker": "cloud_train_tool_sft",
                "fingerprint": dict(fingerprint),
                "optimizer_scheduler_rng_state": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def tokenize_records(
    records: Sequence[Mapping[str, Any]],
    adapter: ToolWireAdapter,
    *,
    max_seq_len: int,
    limit: int | None = None,
) -> list[dict[str, Tensor]]:
    selected = records[:limit] if limit is not None else records
    examples: list[dict[str, Tensor]] = []
    for record in selected:
        messages = messages_with_record_turn_guidance(record)
        rendered = adapter.render_training(
            messages,
            max_seq_len=max_seq_len,
            tools=training_tools_for_record(record, adapter),
        )
        examples.append(
            {
                "input_ids": rendered.input_ids,
                "attention_mask": rendered.attention_mask,
                "labels": rendered.labels,
            }
        )
    return examples


def training_tools_for_record(
    record: Mapping[str, Any],
    adapter: ToolWireAdapter,
) -> list[Mapping[str, Any]] | None:
    """Resolve V7 exact schemas; None preserves legacy all-tool rendering."""

    expected = record.get("expected")
    if not isinstance(expected, Mapping):
        return None
    contract = expected.get("generation_contract")
    if contract is None:
        return None
    if not isinstance(contract, Mapping):
        raise ValueError("generation_contract must be an object")
    names = contract.get("tool_names")
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError("generation_contract.tool_names must be a list of strings")
    if len(names) != len(set(names)):
        raise ValueError("generation_contract.tool_names must be unique")
    available = {str(tool["name"]): tool for tool in adapter.public_tool_manifest}
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(f"generation_contract references unknown tools: {unknown}")
    constraints = contract.get("argument_constraints", {})
    if not isinstance(constraints, Mapping):
        raise ValueError("generation_contract.argument_constraints must be an object")
    if not names:
        if constraints:
            raise ValueError("no-tool generation_contract cannot constrain arguments")
        return []
    if len(names) != 1:
        raise ValueError("generation_contract must expose exactly one or no tools")
    selected = available[names[0]]
    parameters = dict(selected.get("parameters", {}))
    properties = {
        str(name): dict(value)
        for name, value in parameters.get("properties", {}).items()
        if isinstance(value, Mapping)
    }
    unknown_arguments = [name for name in constraints if name not in properties]
    if unknown_arguments:
        raise ValueError(f"generation_contract constrains unknown arguments: {unknown_arguments}")
    for name, constraint in constraints.items():
        if not isinstance(constraint, Mapping) or set(constraint) != {"const"}:
            raise ValueError("argument constraints must contain exactly one const")
        properties[str(name)] = {**properties[str(name)], "const": constraint["const"]}
    exact_parameters = {
        **parameters,
        "properties": properties,
        "additionalProperties": False,
    }
    if constraints:
        exact_parameters["required"] = list(constraints)
    else:
        exact_parameters.pop("required", None)
    return [{**selected, "parameters": exact_parameters}]


def collate_pretokenized(
    batch: Sequence[Mapping[str, Any]],
    *,
    pad_token_id: int = 0,
) -> dict[str, Tensor]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    tensors = [
        {
            name: (
                value.detach().clone()
                if isinstance(value, Tensor)
                else torch.tensor(value, dtype=torch.long)
            )
            for name, value in item.items()
            if name in {"input_ids", "attention_mask", "labels"}
        }
        for item in batch
    ]
    max_length = max(int(item["input_ids"].numel()) for item in tensors)
    output = {
        "input_ids": torch.full(
            (len(tensors), max_length),
            int(pad_token_id),
            dtype=torch.long,
        ),
        "attention_mask": torch.zeros((len(tensors), max_length), dtype=torch.long),
        "labels": torch.full(
            (len(tensors), max_length),
            IGNORED_LABEL,
            dtype=torch.long,
        ),
    }
    for row, item in enumerate(tensors):
        length = int(item["input_ids"].numel())
        output["input_ids"][row, :length] = item["input_ids"]
        output["attention_mask"][row, :length] = item["attention_mask"]
        output["labels"][row, :length] = item["labels"]
    return output


def save_checkpoint_metadata(
    output_dir: Path,
    *,
    step: int,
    config: WorkerConfig,
    adapter: ToolWireAdapter,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    checkpoint = output_dir / "checkpoints" / f"step-{step:06d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    metadata = {
        "step": step,
        "created_at_unix": int(time.time()),
        "worker": "cloud_train_tool_sft",
        "fingerprint": training_fingerprint(config, adapter),
        "resume_validation": {
            "base_revision": config.base_revision,
            "manifest_sha256": sha256_file(config.manifest),
            "template_hash": adapter.template_hash,
            "optimizer_scheduler_rng_state": False,
        },
        "extra": dict(extra or {}),
    }
    path = checkpoint / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def seed_training(seed: int = TRAINING_SEED) -> torch.Generator:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.Generator(device="cpu").manual_seed(seed)


def log_trackio(config: WorkerConfig, event: str, payload: Mapping[str, Any]) -> None:
    if not config.trackio_project:
        return
    try:
        import trackio  # type: ignore[import-not-found]
    except Exception:
        print(
            json.dumps({"event": "trackio_unavailable", "payload": payload}, sort_keys=True),
            file=sys.stderr,
        )
        return
    run = trackio.init(project=config.trackio_project)
    run.log({f"{event}/{key}": value for key, value in payload.items()})


def configure_trackio_environment(config: WorkerConfig) -> None:
    if config.trackio_project:
        os.environ["TRACKIO_PROJECT"] = config.trackio_project


def tf32_supported() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8)


def build_training_configs(config: WorkerConfig) -> dict[str, Any]:
    from peft import LoraConfig  # type: ignore[import-not-found]
    from trl import SFTConfig  # type: ignore[import-not-found]

    quantization = None
    if config.precision == "qlora":
        from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(LORA_TARGET_MODULES),
    )
    training_args = SFTConfig(
        output_dir=str(config.output_dir),
        max_steps=config.max_steps,
        max_length=config.max_seq_len,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=max(1, round(config.max_steps * config.warmup_ratio)),
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
        eval_steps=max(config.checkpoint_every, 1_000),
        save_total_limit=2,
        remove_unused_columns=False,
        optim=("paged_adamw_8bit" if config.precision == "qlora" else "adamw_torch_fused"),
        report_to="trackio" if config.trackio_project else [],
        project=config.trackio_project or "huggingface",
        seed=TRAINING_SEED,
        data_seed=TRAINING_SEED,
        run_name=config.trackio_run_name or f"{config.family}-tool-sft",
        push_to_hub=False,
    )
    return {"quantization": quantization, "lora": lora, "training_args": training_args}


def merge_adapter_with_reload_parity(
    *,
    config: WorkerConfig,
    tokenizer: Any,
    input_ids: Tensor,
    attention_mask: Tensor,
    auto_model_cls: Any,
    peft_model_cls: Any,
    minimum_argmax_agreement: float = 0.999,
    maximum_logit_difference: float = 0.3,
) -> dict[str, Any]:
    adapter_dir = config.output_dir / "adapter"
    merged_dir = config.output_dir / "merged"
    base = auto_model_cls.from_pretrained(
        config.base_model,
        revision=config.base_revision,
        dtype=torch.float32,
        device_map={"": torch.cuda.current_device()},
    )
    adapter_model = peft_model_cls.from_pretrained(
        base,
        adapter_dir,
        autocast_adapter_dtype=True,
    )
    merged = adapter_model.merge_and_unload(safe_merge=True)
    merged.to(dtype=torch.float16)
    merged.config.torch_dtype = torch.float16
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    del adapter_model
    del merged
    del base
    gc.collect()
    torch.cuda.empty_cache()

    reference_base = auto_model_cls.from_pretrained(
        config.base_model,
        revision=config.base_revision,
        dtype=torch.float16,
        device_map={"": torch.cuda.current_device()},
    )
    reference_adapter = peft_model_cls.from_pretrained(
        reference_base,
        adapter_dir,
        autocast_adapter_dtype=False,
    )
    reference_adapter.eval()
    reference_device = next(reference_adapter.parameters()).device
    reference_batch = {
        "input_ids": input_ids.to(reference_device),
        "attention_mask": attention_mask.to(reference_device),
    }
    with torch.inference_mode():
        adapter_logits = reference_adapter(**reference_batch).logits.detach().float().cpu()
        adapter_generation = (
            reference_adapter.generate(
                **reference_batch,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=getattr(tokenizer, "pad_token_id", None),
            )
            .detach()
            .cpu()
        )
    del reference_adapter
    del reference_base
    gc.collect()
    torch.cuda.empty_cache()

    reloaded = auto_model_cls.from_pretrained(
        merged_dir,
        dtype=torch.float16,
        device_map={"": torch.cuda.current_device()},
    )
    reloaded.eval()
    reload_device = next(reloaded.parameters()).device
    reload_batch = {
        "input_ids": input_ids.to(reload_device),
        "attention_mask": attention_mask.to(reload_device),
    }
    with torch.inference_mode():
        reloaded_logits = reloaded(**reload_batch).logits.detach().float().cpu()
        reloaded_generation = (
            reloaded.generate(
                **reload_batch,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=getattr(tokenizer, "pad_token_id", None),
            )
            .detach()
            .cpu()
        )
    differences = (adapter_logits - reloaded_logits).abs()
    finite = bool(torch.isfinite(differences).all().item())
    max_abs_logit_diff = float(differences.max().item())
    argmax_agreement = float(
        (adapter_logits.argmax(dim=-1) == reloaded_logits.argmax(dim=-1)).float().mean().item()
    )
    generation_equal = torch.equal(adapter_generation, reloaded_generation)
    if (
        not finite
        or not generation_equal
        or argmax_agreement < minimum_argmax_agreement
        or max_abs_logit_diff > maximum_logit_difference
    ):
        raise RuntimeError(
            "merged checkpoint reload parity failed: "
            f"finite={finite}, generation_equal={bool(generation_equal)}, "
            f"argmax_agreement={argmax_agreement}, "
            f"max_abs_logit_diff={max_abs_logit_diff}"
        )
    report = {
        "all_logit_differences_finite": finite,
        "generation_equal": bool(generation_equal),
        "argmax_token_agreement": argmax_agreement,
        "max_abs_logit_diff": max_abs_logit_diff,
        "minimum_argmax_agreement": minimum_argmax_agreement,
        "maximum_logit_difference": maximum_logit_difference,
        "merge_accumulation_dtype": "float32",
        "release_weight_dtype": "float16",
    }
    del reloaded
    gc.collect()
    torch.cuda.empty_cache()
    return report


def write_training_result(path: Path, result: Mapping[str, Any]) -> Path:
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def model_card_release_text(config: WorkerConfig) -> tuple[str, str, str]:
    """Return (frontmatter extra, description sentence, release paragraph) for the card."""

    if config.merge_adapter:
        return (
            "",
            f"It is a merged {config.precision} LoRA adaptation of "
            f"`{config.base_model}` at revision `{config.base_revision}`. The model has "
            "approximately 8.8 billion parameters and uses the base model's native tagged "
            "JSON tool-call format.",
            "The released root checkpoint is merged FP16 weights. The trained BF16 adapter is\n"
            "also stored under `adapter/`.",
        )
    return (
        "library_name: peft\n",
        f"It is a {config.precision} LoRA adapter (rank {config.lora_rank}, alpha "
        f"{config.lora_alpha}) trained on top of `{config.base_model}` at revision "
        f"`{config.base_revision}`; the base has approximately 8.8 billion parameters "
        "and uses its native tagged JSON tool-call format.",
        "The repository root is the trained BF16 LoRA adapter only: there are no merged\n"
        "weights and no `config.json`. Load it with `PeftModel.from_pretrained(base, repo,\n"
        "revision=...)` on top of the base model at the pinned revision. The same adapter is\n"
        "duplicated under `adapter/`.",
    )


def write_model_card(
    config: WorkerConfig,
    *,
    train_records: int,
    validation_records: int,
    result: Mapping[str, Any],
) -> Path:
    frontmatter_extra, description, release_text = model_card_release_text(config)
    dataset_repo = os.environ.get(
        "RETAIL_BANK_TOOL_SFT_DATASET_REPO",
        "spkc83/retail-bank-servicing-alignment-sft",
    )
    dataset_revision = os.environ.get("RETAIL_BANK_TOOL_SFT_DATASET_REVISION", "unrecorded")
    source_commit = os.environ.get("RETAIL_BANK_SOURCE_COMMIT", "unrecorded")
    card = f"""---
license: apache-2.0
base_model: {config.base_model}
datasets:
- {dataset_repo}
pipeline_tag: text-generation
{frontmatter_extra}tags:
- retail-banking
- tool-calling
- conversational
- peft
---

# Retail Bank Agent 9B

This is a research checkpoint for a synthetic retail-bank customer-service
demonstration. {description}

## Training

- Dataset: `{dataset_repo}` at `{dataset_revision}`
- Training records: {train_records}
- Validation records: {validation_records}
- Tool manifest: nine synthetic retail-banking tools
- Assistant-only target masking: tool-call and final-assistant spans
- Maximum sequence length: {config.max_seq_len}
- Optimizer steps: {result.get("steps", config.max_steps)}
- LoRA rank/alpha: {config.lora_rank}/{config.lora_alpha}
- Source commit: `{source_commit}`
- Chat-template SHA-256: `{result.get("template_hash", "unrecorded")}`

{release_text}

## Intended use and limitations

The model is intended only for the linked synthetic banking POC. It must be
given the published tool schemas and tool results. It has no access to real
banking systems, is not financial advice, and may make incorrect tool choices
or unsupported claims. Evaluate tool-call syntax, arguments, backend
execution, grounded final responses, OOD behavior, and multi-turn behavior
before relying on a revision.
"""
    path = config.output_dir / "README.md"
    path.write_text(card, encoding="utf-8")
    return path


def run_tiny_smoke(config: WorkerConfig) -> dict[str, Any]:
    tokenizer = SimpleToolTokenizer()
    adapter = ToolWireAdapter(
        tokenizer,
        family=config.family,
        public_tool_manifest=PUBLIC_BANKING_TOOL_MANIFEST,
    )
    records = tiny_smoke_records()
    examples = tokenize_records(records, adapter, max_seq_len=config.max_seq_len)
    save_checkpoint_metadata(
        config.output_dir,
        step=1,
        config=config,
        adapter=adapter,
        extra={"smoke_examples": len(examples)},
    )
    final_dir = config.output_dir / "final"
    adapter_dir = config.output_dir / "adapter"
    tokenizer.save_pretrained(final_dir)
    tokenizer.save_pretrained(adapter_dir)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"family": config.family, "template_hash": adapter.template_hash}),
        encoding="utf-8",
    )
    parity = adapter.parse_assistant(
        json.dumps(
            {
                "tool_calls": [
                    {
                        "id": "call_smoke_0",
                        "index": 0,
                        "type": "function",
                        "function": {"name": "freeze_card", "arguments": {"last4": "4821"}},
                    }
                ]
            }
        )
    )
    log_trackio(config, "tiny_smoke", {"examples": len(examples), "step": 1})
    return {
        "mode": "tiny_smoke",
        "steps": 1,
        "examples": len(examples),
        "first_input_tokens": int(examples[0]["attention_mask"].sum().item()),
        "assistant_label_tokens": int((examples[0]["labels"] != -100).sum().item()),
        "checkpoint": str(config.output_dir / "checkpoints" / "step-000001" / "metadata.json"),
        "adapter_dir": str(adapter_dir),
        "final_dir": str(final_dir),
        "template_hash": adapter.template_hash,
        "merge_reload_parity": parity["tool_calls"][0]["function"]["name"] == "freeze_card",
        "pushed_to_hub": False,
    }


def run_remote_training(config: WorkerConfig) -> dict[str, Any]:
    assert_remote_execution_allowed(config)
    validate_hub_destination(config)
    dataset_pin = validate_dataset_identity(config.manifest)
    # Fail before the GPU spend when the shadow gate contract is broken, too.
    shadow_gate_records = continuation_module().load_shadow_gate_records(config.manifest)
    if config.push_to_hub:
        # Fail before the GPU spend rather than after it when the destination is taken.
        preflight_destination_repo(config)
    configure_trackio_environment(config)
    # Seed before anything stochastic: LoRA initialisation, dropout, and the
    # trainer's own sampler all draw from these generators. This lane recorded
    # `training_seed` in its fingerprint for months while never calling this,
    # so two runs of the same configuration could and did diverge -- the most
    # likely explanation for behaviour churn that was read as data problems.
    seed_training(TRAINING_SEED)
    configs = build_training_configs(config)
    from datasets import Dataset as HfDataset  # type: ignore[import-not-found]
    from peft import PeftModel  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainerCallback,
    )
    from trl import SFTTrainer  # type: ignore[import-not-found]

    if config.resume_from is not None:
        # Validated after tokenizer loads because template hash is part of the fingerprint.
        pass

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, revision=config.base_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    adapter = ToolWireAdapter(
        tokenizer,
        family=config.family,
        public_tool_manifest=PUBLIC_BANKING_TOOL_MANIFEST,
        pad_to_max_length=False,
    )
    fingerprint = training_fingerprint(config, adapter)
    if config.resume_from is not None:
        validate_resume_fingerprint(config.resume_from, fingerprint)
    train_records = load_manifest_records(config.manifest, "train")
    validation_records = load_manifest_records(config.manifest, "validation")
    mixed_train_records, mix_stats = build_training_mix(config, train_records)
    train_examples = tokenize_records(
        mixed_train_records,
        adapter,
        max_seq_len=config.max_seq_len,
    )
    validation_examples = tokenize_records(
        validation_records,
        adapter,
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
    model_kwargs: dict[str, Any] = {
        "revision": config.base_revision,
        "dtype": torch.bfloat16,
        "device_map": {"": torch.cuda.current_device()},
    }
    if configs["quantization"] is not None:
        model_kwargs["quantization_config"] = configs["quantization"]
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        **model_kwargs,
    )

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

    class ResumeMetadataCallback(TrainerCallback):
        def on_save(
            self,
            args: Any,
            state: Any,
            control: Any,
            **kwargs: Any,
        ) -> Any:
            del args, kwargs
            save_trainer_checkpoint_metadata(
                config.output_dir,
                step=int(state.global_step),
                fingerprint=fingerprint,
            )
            return control

    trainer = SFTTrainer(
        model=model,
        args=configs["training_args"],
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=partial(
            collate_pretokenized,
            pad_token_id=int(tokenizer.pad_token_id),
        ),
        peft_config=configs["lora"],
        processing_class=tokenizer,
        callbacks=[
            WallClockStopCallback(config.max_train_seconds),
            ResumeMetadataCallback(),
        ],
    )
    train_output = trainer.train(
        resume_from_checkpoint=str(config.resume_from) if config.resume_from else None
    )
    if config.trackio_project:
        # Trackio's Trainer callback closes its run in on_train_end. A separate
        # post-training evaluate() still emits an on_log event, which otherwise
        # tries to write to that closed run.
        from transformers.integrations import (  # type: ignore[import-not-found]
            TrackioCallback,
        )

        trainer.remove_callback(TrackioCallback)
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(config.output_dir / "adapter"))
    tokenizer.save_pretrained(config.output_dir / "adapter")
    actual_step = int(trainer.state.global_step)
    enable_generation_cache(trainer.model)
    result: dict[str, Any] = {
        "steps": actual_step,
        "adapter_dir": str(config.output_dir / "adapter"),
        "template_hash": adapter.template_hash,
        "train_metrics": dict(train_output.metrics),
        "eval_metrics": dict(eval_metrics),
        "dataset_identity": dataset_pin,
        "training_mix": mix_stats,
        "merged_adapter": config.merge_adapter,
        "pushed_to_hub": False,
    }
    result_path = config.output_dir / "training_result.json"
    try:
        behavioral_gates = run_coreference_behavioral_gates(
            model=trainer.model,
            tokenizer=tokenizer,
            adapter=adapter,
            validation_records=validation_records,
            manifest_path=config.manifest,
            output_dir=config.output_dir,
            step=actual_step,
            shadow_records=shadow_gate_records,
        )
    except RuntimeError as gate_error:
        # The adapter and the failing behavioural report already sit on the job bucket.
        # Keep the run's metrics next to them so the bundle can be diagnosed, and
        # published by hand only on a deliberate decision, without a second GPU run.
        result["behavioral_gate_failure"] = str(gate_error)
        write_training_result(result_path, result)
        raise
    # Third gate: the guidance-free behaviours. The v12 run proved these churn
    # between otherwise-identical runs while every coreference gate stays at
    # 1.0, so they block the upload exactly the way the coreference gates do.
    bare_probe_report = run_bare_probe_gate(trainer.model, tokenizer)
    evaluations_dir = config.output_dir / "behavioral-evaluations"
    evaluations_dir.mkdir(parents=True, exist_ok=True)
    (evaluations_dir / f"bare-probes-step-{actual_step}.json").write_text(
        json.dumps(bare_probe_report, indent=2, sort_keys=True), encoding="utf-8"
    )
    result["bare_probe_behavioral_gate"] = {
        key: bare_probe_report[key]
        for key in ("contract", "pass", "gated_total", "gated_passed", "failures")
    }
    if not bare_probe_report["pass"]:
        failures = "; ".join(
            f"{row['case']}: {row['failure']}" for row in bare_probe_report["failures"]
        )
        result["behavioral_gate_failure"] = f"bare-probe gate failed: {failures}"
        write_training_result(result_path, result)
        raise RuntimeError(f"bare-probe behavioural gate failed: {failures}")
    save_checkpoint_metadata(
        config.output_dir,
        step=actual_step,
        config=config,
        adapter=adapter,
        extra={
            "optimizer_scheduler_rng_state": True,
            "train_metrics": dict(train_output.metrics),
            "eval_metrics": dict(eval_metrics),
            "coreference_behavioral_gate": behavioral_gates["dev"],
            "shadow_coreference_behavioral_gate": behavioral_gates["shadow"],
        },
    )
    result["coreference_behavioral_gate"] = behavioral_gates["dev"]
    result["shadow_coreference_behavioral_gate"] = behavioral_gates["shadow"]
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if config.merge_adapter:
        parity = merge_adapter_with_reload_parity(
            config=config,
            tokenizer=tokenizer,
            input_ids=torch.tensor(
                [train_dataset[0]["input_ids"][:128]],
                dtype=torch.long,
            ),
            attention_mask=torch.tensor(
                [train_dataset[0]["attention_mask"][:128]],
                dtype=torch.long,
            ),
            auto_model_cls=AutoModelForCausalLM,
            peft_model_cls=PeftModel,
        )
        result["merged_dir"] = str(config.output_dir / "merged")
        result["merge_reload_parity"] = parity
    model_card = write_model_card(
        config,
        train_records=len(mixed_train_records),
        validation_records=len(validation_records),
        result=result,
    )
    write_training_result(result_path, result)
    if config.push_to_hub:
        from huggingface_hub import HfApi  # type: ignore[import-not-found]

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        require_publishable_destination(api, config.hub_dest)
        api.create_repo(config.hub_dest, repo_type="model", private=False, exist_ok=True)
        # Repo root holds merged FP16 weights, or the adapter itself under
        # --skip-merge-adapter. The adapter/ copy is uploaded either way so consumers
        # can always load the PEFT weights from the same path.
        api.upload_folder(
            repo_id=config.hub_dest,
            repo_type="model",
            folder_path=config.output_dir / ("merged" if config.merge_adapter else "adapter"),
            ignore_patterns=[".*", "**/.*"],
        )
        api.upload_folder(
            repo_id=config.hub_dest,
            repo_type="model",
            folder_path=config.output_dir / "adapter",
            path_in_repo="adapter",
        )
        api.upload_file(
            repo_id=config.hub_dest,
            repo_type="model",
            path_or_fileobj=model_card,
            path_in_repo="README.md",
        )
        api.upload_file(
            repo_id=config.hub_dest,
            repo_type="model",
            path_or_fileobj=(
                config.output_dir / "checkpoints" / f"step-{actual_step:06d}" / "metadata.json"
            ),
            path_in_repo="training_metadata.json",
        )
        result["pushed_to_hub"] = config.hub_dest
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        api.upload_file(
            repo_id=config.hub_dest,
            repo_type="model",
            path_or_fileobj=result_path,
            path_in_repo="training_result.json",
        )
    return result


def tiny_smoke_records() -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "banking-tool-sft/v1",
            "record_id": "smoke_freeze_001",
            "messages": [
                {"role": "user", "content": "Freeze my debit card ending in 4821.", "loss": False},
                {
                    "role": "assistant",
                    "content": None,
                    "loss": True,
                    "tool_calls": [
                        {
                            "id": "call_smoke_freeze_0",
                            "index": 0,
                            "type": "function",
                            "function": {
                                "name": "freeze_card",
                                "arguments": {"last4": "4821"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_smoke_freeze_0",
                    "name": "freeze_card",
                    "content": {"ok": True, "result": {"card": {"last4": "4821"}}},
                    "loss": False,
                },
                {
                    "role": "assistant",
                    "content": "Your debit card ending in 4821 is now frozen.",
                    "loss": True,
                },
            ],
        }
    ]


def main(argv: Sequence[str] | None = None) -> int:
    config = worker_config_from_args(parse_args(argv))
    if config.run_tiny_smoke:
        print(json.dumps(run_tiny_smoke(config), indent=2, sort_keys=True))
        return 0
    if config.dry_run and not remote_execution_allowed(config):
        print(json.dumps(build_dry_run_plan(config), indent=2, sort_keys=True))
        return 0
    print(json.dumps(run_remote_training(config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
