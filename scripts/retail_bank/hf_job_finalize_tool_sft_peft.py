# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "huggingface-hub==1.22.0",
# ]
# ///
"""Publish a validated Granite BF16 LoRA adapter without merging base weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi

DEFAULT_DESTINATION_REPO = "spkc83/retail-bank-servicing-agent-9b-peft"
ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_args.bin",
)
RELEASE_CONTRACT = "banking-v5-peft-adapter-release/v1"
FINALIZER_REPORT_CONTRACT = "banking-v5-peft-finalizer-report/v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--adapter-subdir", default="adapter")
    parser.add_argument("--selected-step", type=int, default=750)
    parser.add_argument("--destination-repo", default=DEFAULT_DESTINATION_REPO)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--training-job", required=True)
    return parser.parse_args(argv)


def require_exact_revision(value: str, *, field: str) -> None:
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be an exact 40-character lowercase Git revision")


def require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{field} must be a SHA256 string")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise RuntimeError(f"{field} must be an exact lowercase SHA256")
    return value


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{field} must be a JSON object")
    return value


def require_nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{field} must be a non-empty string")
    return value


def validate_training_artifacts(
    *,
    adapter_dir: Path,
    metadata: Mapping[str, Any],
    training_result: Mapping[str, Any] | None,
    selected_step: int,
) -> dict[str, Any]:
    if selected_step < 1:
        raise ValueError("--selected-step must be positive")
    missing = [name for name in ADAPTER_FILES if not (adapter_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"required adapter files are missing from {adapter_dir}: {missing}")
    if metadata.get("step") != selected_step:
        raise RuntimeError(f"training metadata does not represent selected step {selected_step}")
    if training_result is not None and training_result.get("steps") != selected_step:
        raise RuntimeError(f"training result does not represent selected step {selected_step}")

    fingerprint = require_mapping(metadata.get("fingerprint"), field="metadata fingerprint")
    base_model = require_nonempty_string(
        fingerprint.get("base_model"), field="fingerprint.base_model"
    )
    base_revision = require_nonempty_string(
        fingerprint.get("base_revision"), field="fingerprint.base_revision"
    )
    require_exact_revision(base_revision, field="fingerprint.base_revision")
    if fingerprint.get("family") != "granite":
        raise RuntimeError("fingerprint.family must be granite")
    if fingerprint.get("precision") != "bf16-lora":
        raise RuntimeError("fingerprint.precision must be bf16-lora for this release path")
    template_hash = require_sha256(
        fingerprint.get("template_hash"), field="fingerprint.template_hash"
    )

    dataset = require_mapping(
        fingerprint.get("dataset_identity"), field="fingerprint.dataset_identity"
    )
    dataset_repo = require_nonempty_string(
        dataset.get("repository"), field="dataset_identity.repository"
    )
    dataset_revision = require_nonempty_string(
        dataset.get("revision"), field="dataset_identity.revision"
    )
    require_exact_revision(dataset_revision, field="dataset_identity.revision")
    dataset_manifest_sha256 = require_sha256(
        dataset.get("manifest_sha256"), field="dataset_identity.manifest_sha256"
    )

    lora = require_mapping(fingerprint.get("lora"), field="fingerprint.lora")
    rank = int(lora.get("rank", 0))
    alpha = int(lora.get("alpha", 0))
    dropout = float(lora.get("dropout", -1.0))
    targets = lora.get("target_modules")
    if rank < 1 or alpha < 1 or not 0.0 <= dropout < 1.0:
        raise RuntimeError("fingerprint contains invalid LoRA hyperparameters")
    if (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(item, str) for item in targets)
    ):
        raise RuntimeError("fingerprint.lora.target_modules must be a non-empty string list")

    adapter_config = read_json(adapter_dir / "adapter_config.json")
    if adapter_config.get("base_model_name_or_path") != base_model:
        raise RuntimeError("adapter_config base model does not match training fingerprint")
    if str(adapter_config.get("peft_type", "")).upper() != "LORA":
        raise RuntimeError("adapter_config.peft_type must be LORA")
    if int(adapter_config.get("r", 0)) != rank:
        raise RuntimeError("adapter_config LoRA rank does not match training fingerprint")
    if int(adapter_config.get("lora_alpha", 0)) != alpha:
        raise RuntimeError("adapter_config LoRA alpha does not match training fingerprint")
    if float(adapter_config.get("lora_dropout", -1.0)) != dropout:
        raise RuntimeError("adapter_config LoRA dropout does not match training fingerprint")
    configured_targets = adapter_config.get("target_modules")
    if not isinstance(configured_targets, list) or set(configured_targets) != set(targets):
        raise RuntimeError("adapter_config target modules do not match training fingerprint")

    extra = require_mapping(metadata.get("extra"), field="metadata.extra")
    train_metrics = require_mapping(extra.get("train_metrics"), field="metadata train metrics")
    eval_metrics = require_mapping(extra.get("eval_metrics"), field="metadata eval metrics")
    if training_result is not None:
        if training_result.get("train_metrics") != train_metrics:
            raise RuntimeError("training result train metrics do not match selected metadata")
        if training_result.get("eval_metrics") != eval_metrics:
            raise RuntimeError("training result eval metrics do not match selected metadata")

    file_sha256 = {name: sha256(adapter_dir / name) for name in ADAPTER_FILES}
    return {
        "base_model": base_model,
        "base_revision": base_revision,
        "dataset_repo": dataset_repo,
        "dataset_revision": dataset_revision,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "template_hash": template_hash,
        "lora": {
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "target_modules": list(targets),
        },
        "train_metrics": dict(train_metrics),
        "eval_metrics": dict(eval_metrics),
        "adapter_file_sha256": file_sha256,
    }


def build_release(
    *,
    validated: Mapping[str, Any],
    destination_repo: str,
    source_commit: str,
    training_job: str,
    selected_step: int,
    immutable_revision: str | None,
) -> dict[str, Any]:
    return {
        "contract": RELEASE_CONTRACT,
        "destination_repo": destination_repo,
        "source_commit": source_commit,
        "training_job": training_job,
        "steps": selected_step,
        "base_model": {
            "repository": validated["base_model"],
            "revision": validated["base_revision"],
            "weight_dtype": "bfloat16",
        },
        "dataset": {
            "repository": validated["dataset_repo"],
            "revision": validated["dataset_revision"],
            "manifest_sha256": validated["dataset_manifest_sha256"],
        },
        "peft_composition": {
            "method": "LoRA",
            "base_weight_dtype": "bfloat16",
            "adapter_file": "adapter_model.safetensors",
            **validated["lora"],
        },
        "template_hash": validated["template_hash"],
        "train_metrics": validated["train_metrics"],
        "eval_metrics": validated["eval_metrics"],
        "adapter_model_sha256": validated["adapter_file_sha256"]["adapter_model.safetensors"],
        "adapter_file_sha256": validated["adapter_file_sha256"],
        "final_immutable_hub_revision": immutable_revision,
    }


def render_model_card(release: Mapping[str, Any]) -> str:
    base = release["base_model"]
    dataset = release["dataset"]
    composition = release["peft_composition"]
    return f"""---
license: apache-2.0
base_model: {base["repository"]}
datasets:
- {dataset["repository"]}
pipeline_tag: text-generation
tags:
- granite
- retail-banking
- tool-calling
- conversational
- peft
- lora
---

# Retail Bank Servicing Agent 9B — PEFT Adapter

This repository contains the Granite 9B LoRA adapter only. Inference composes it
with the exact base checkpoint below using BF16 base weights; it does not contain
merged 9B weights.

- Final immutable adapter-bundle revision: `{release["final_immutable_hub_revision"]}`
- Base model: `{base["repository"]}`
- Base revision: `{base["revision"]}`
- SFT dataset: `{dataset["repository"]}`
- Dataset revision: `{dataset["revision"]}`
- Dataset manifest SHA256: `{dataset["manifest_sha256"]}`
- Source commit: `{release["source_commit"]}`
- Training job: `{release["training_job"]}`
- Optimizer steps: `{release["steps"]}`
- Training loss: `{release["train_metrics"]["train_loss"]}`
- Validation loss: `{release["eval_metrics"]["eval_loss"]}`
- Validation token accuracy: `{release["eval_metrics"]["eval_mean_token_accuracy"]}`
- Adapter SHA256: `{release["adapter_model_sha256"]}`
- PEFT composition: `{composition["method"]}` rank `{composition["rank"]}`, alpha
  `{composition["alpha"]}`, dropout `{composition["dropout"]}`, BF16 base weights

Load the pinned base revision first, then attach this adapter revision with PEFT.
The finalizer's JSON report distinguishes the immutable adapter-bundle commit from
the metadata-only repository-head commit.

This model is experimental, trained on synthetic retail-banking data, and has no
connection to a real bank.
"""


def json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    require_exact_revision(args.source_commit, field="--source-commit")
    if not args.training_job.strip():
        raise ValueError("--training-job must be non-empty")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be passed as a Hugging Face Job secret")

    adapter_dir = args.output_root / args.adapter_subdir
    metadata_path = (
        args.output_root / "checkpoints" / f"step-{args.selected_step:06d}" / "metadata.json"
    )
    training_result_path = args.output_root / "training_result.json"
    metadata = read_json(metadata_path)
    training_result = read_json(training_result_path) if training_result_path.is_file() else None
    validated = validate_training_artifacts(
        adapter_dir=adapter_dir,
        metadata=metadata,
        training_result=training_result,
        selected_step=args.selected_step,
    )

    api = HfApi(token=token)
    api.create_repo(
        args.destination_repo,
        repo_type="model",
        private=False,
        exist_ok=False,
    )
    pending_release = build_release(
        validated=validated,
        destination_repo=args.destination_repo,
        source_commit=args.source_commit,
        training_job=args.training_job,
        selected_step=args.selected_step,
        immutable_revision=None,
    )
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=adapter_dir / name)
        for name in ADAPTER_FILES
    ]
    operations.extend(
        [
            CommitOperationAdd(
                path_in_repo="training_metadata.json", path_or_fileobj=metadata_path
            ),
            CommitOperationAdd(
                path_in_repo="training_result.json", path_or_fileobj=json_bytes(pending_release)
            ),
            CommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=render_model_card(pending_release).encode(),
            ),
        ]
    )
    bundle_commit = api.create_commit(
        repo_id=args.destination_repo,
        repo_type="model",
        operations=operations,
        commit_message="Publish validated Granite 9B PEFT adapter release",
    )
    immutable_revision = str(bundle_commit.oid)
    require_exact_revision(immutable_revision, field="adapter bundle revision")

    final_release = build_release(
        validated=validated,
        destination_repo=args.destination_repo,
        source_commit=args.source_commit,
        training_job=args.training_job,
        selected_step=args.selected_step,
        immutable_revision=immutable_revision,
    )
    provenance_commit = api.create_commit(
        repo_id=args.destination_repo,
        repo_type="model",
        parent_commit=immutable_revision,
        operations=[
            CommitOperationAdd(
                path_in_repo="training_result.json", path_or_fileobj=json_bytes(final_release)
            ),
            CommitOperationAdd(
                path_in_repo="README.md", path_or_fileobj=render_model_card(final_release).encode()
            ),
        ],
        commit_message="Record immutable PEFT adapter bundle revision",
    )
    final_revision = str(provenance_commit.oid)
    require_exact_revision(final_revision, field="final repository revision")
    print(
        json.dumps(
            {
                "contract": FINALIZER_REPORT_CONTRACT,
                "destination_repo": args.destination_repo,
                "weights_commit": immutable_revision,
                "final_revision": final_revision,
                "release": final_release,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
