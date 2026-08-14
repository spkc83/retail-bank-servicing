# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "huggingface-hub==1.22.0",
# ]
# ///
"""Validate persisted SFT artifacts and publish them without retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

MODEL_REPO = "spkc83/retail-bank-agent-9b"
DEFAULT_OUTPUT_ROOT = Path("/mnt/artifacts/retail-bank-agent-9b-3a6a7efe")
MERGED_ALLOWLIST = (
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)
ADAPTER_ALLOWLIST = (
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_args.bin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--merged-subdir", default="merged-fp16")
    parser.add_argument("--adapter-subdir", default="adapter")
    parser.add_argument("--selected-step", type=int, default=3_000)
    parser.add_argument(
        "--parity-report",
        default="merge_parity_diagnostics_merged-fp16_float16.json",
    )
    parser.add_argument("--model-repo", default=MODEL_REPO)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--training-job", required=True)
    parser.add_argument("--remerge-job", required=True)
    parser.add_argument("--parity-job", required=True)
    parser.add_argument("--minimum-argmax-agreement", type=float, default=0.999)
    parser.add_argument("--maximum-logit-difference", type=float, default=0.3)
    parser.add_argument("--maximum-p999-difference", type=float, default=0.07)
    return parser.parse_args()


def require_exact_revision(value: str, *, field: str) -> None:
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be an exact 40-character lowercase Git revision")


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


def require_files(root: Path, names: tuple[str, ...]) -> None:
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"required release files are missing from {root}: {missing}")


def validate_parity(
    report: dict[str, Any],
    *,
    minimum_argmax_agreement: float,
    maximum_logit_difference: float,
    maximum_p999_difference: float,
) -> dict[str, Any]:
    if report.get("contract") != "banking-v3-bf16-merge-parity/v1":
        raise RuntimeError("unexpected merge parity report contract")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError("merge parity report has no metrics object")
    failures: list[str] = []
    if metrics.get("all_logit_differences_finite") is not True:
        failures.append("non-finite logit differences")
    if metrics.get("all_greedy_generations_equal") is not True:
        failures.append("greedy generations differ")
    prompt_count = int(metrics.get("prompt_count", 0))
    if prompt_count < 8:
        failures.append(f"only {prompt_count} prompts were compared")
    argmax_agreement = float(metrics.get("argmax_token_agreement", 0.0))
    if argmax_agreement < minimum_argmax_agreement:
        failures.append(
            f"argmax agreement {argmax_agreement} < {minimum_argmax_agreement}"
        )
    max_difference = float(metrics.get("max_abs_logit_diff", float("inf")))
    if max_difference > maximum_logit_difference:
        failures.append(
            f"maximum logit difference {max_difference} > {maximum_logit_difference}"
        )
    p999_difference = float(metrics.get("p999_abs_logit_diff", float("inf")))
    if p999_difference > maximum_p999_difference:
        failures.append(
            f"p999 logit difference {p999_difference} > {maximum_p999_difference}"
        )
    if failures:
        raise RuntimeError("merge parity release gate failed: " + "; ".join(failures))
    return metrics


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_model_card(
    path: Path,
    *,
    metadata: dict[str, Any],
    source_commit: str,
    training_job: str,
    remerge_job: str,
    parity_job: str,
    weights_revision: str,
) -> None:
    fingerprint = metadata["fingerprint"]
    dataset = fingerprint["dataset_identity"]
    metrics = metadata["extra"]
    path.write_text(
        f"""---
license: apache-2.0
base_model: {fingerprint["base_model"]}
datasets:
- {dataset["repository"]}
pipeline_tag: text-generation
tags:
- retail-banking
- tool-calling
- conversational
- peft
---

# Retail Bank Servicing Agent 9B

Merged FP16 LoRA adaptation for the synthetic retail-bank customer-service POC.

- Weights revision: `{weights_revision}`
- Base revision: `{fingerprint["base_revision"]}`
- Dataset revision: `{dataset["revision"]}`
- Source revision: `{source_commit}`
- Training job: `{training_job}`
- FP32-to-FP16 remerge job: `{remerge_job}`
- Merge parity job: `{parity_job}`
- Optimizer steps: `{metadata["step"]}`
- Training loss: `{metrics["train_metrics"]["train_loss"]}`
- Validation loss: `{metrics["eval_metrics"]["eval_loss"]}`
- Validation token accuracy: `{metrics["eval_metrics"]["eval_mean_token_accuracy"]}`

The root checkpoint contains merged weights. The unmerged LoRA adapter is retained
under `adapter/`. This model is experimental, uses synthetic data, and has no
connection to a real bank.
""",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    require_exact_revision(args.source_commit, field="--source-commit")
    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("HF_TOKEN must be passed as a Hugging Face Job secret")

    if args.selected_step < 1:
        raise ValueError("--selected-step must be positive")
    merged_dir = args.output_root / args.merged_subdir
    adapter_dir = args.output_root / args.adapter_subdir
    metadata_path = (
        args.output_root
        / "checkpoints"
        / f"step-{args.selected_step:06d}"
        / "metadata.json"
    )
    parity_path = args.output_root / args.parity_report
    require_files(merged_dir, MERGED_ALLOWLIST)
    require_files(adapter_dir, ADAPTER_ALLOWLIST)
    metadata = read_json(metadata_path)
    parity = read_json(parity_path)
    parity_metrics = validate_parity(
        parity,
        minimum_argmax_agreement=args.minimum_argmax_agreement,
        maximum_logit_difference=args.maximum_logit_difference,
        maximum_p999_difference=args.maximum_p999_difference,
    )
    if metadata.get("step") != args.selected_step:
        raise RuntimeError(
            "training metadata does not represent selected step "
            f"{args.selected_step}"
        )

    model_sha256 = sha256(merged_dir / "model.safetensors")
    adapter_sha256 = sha256(adapter_dir / "adapter_model.safetensors")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(
        args.model_repo,
        repo_type="model",
        private=False,
        exist_ok=True,
    )
    weights_commit = api.upload_folder(
        repo_id=args.model_repo,
        repo_type="model",
        folder_path=merged_dir,
        allow_patterns=list(MERGED_ALLOWLIST),
        commit_message="Publish merged retail-bank SFT weights",
    )
    weights_revision = str(weights_commit.oid)
    require_exact_revision(weights_revision, field="weights revision")
    api.upload_folder(
        repo_id=args.model_repo,
        repo_type="model",
        folder_path=adapter_dir,
        path_in_repo="adapter",
        allow_patterns=list(ADAPTER_ALLOWLIST),
        commit_message="Retain retail-bank LoRA adapter",
    )

    release = {
        "contract": "banking-v3-tool-sft-release/v1",
        "steps": args.selected_step,
        "source_commit": args.source_commit,
        "training_job": args.training_job,
        "remerge_job": args.remerge_job,
        "parity_job": args.parity_job,
        "weights_revision": weights_revision,
        "model_repo": args.model_repo,
        "merged_model_sha256": model_sha256,
        "adapter_model_sha256": adapter_sha256,
        "template_hash": metadata["fingerprint"]["template_hash"],
        "train_metrics": metadata["extra"]["train_metrics"],
        "eval_metrics": metadata["extra"]["eval_metrics"],
        "merge_reload_parity": parity_metrics,
        "merge_parity_acceptance": {
            "minimum_argmax_agreement": args.minimum_argmax_agreement,
            "maximum_logit_difference": args.maximum_logit_difference,
            "maximum_p999_difference": args.maximum_p999_difference,
            "all_greedy_generations_equal": True,
            "all_logit_differences_finite": True,
        },
        "release_weight_dtype": "float16",
        "pushed_to_hub": args.model_repo,
    }
    release_path = args.output_root / "training_result.json"
    card_path = args.output_root / "README.release.md"
    write_json(release_path, release)
    write_model_card(
        card_path,
        metadata=metadata,
        source_commit=args.source_commit,
        training_job=args.training_job,
        remerge_job=args.remerge_job,
        parity_job=args.parity_job,
        weights_revision=weights_revision,
    )
    api.upload_file(
        repo_id=args.model_repo,
        repo_type="model",
        path_or_fileobj=metadata_path,
        path_in_repo="training_metadata.json",
        commit_message="Add retail-bank training provenance",
    )
    api.upload_file(
        repo_id=args.model_repo,
        repo_type="model",
        path_or_fileobj=parity_path,
        path_in_repo="merge_parity_diagnostics.json",
        commit_message="Add BF16 merge parity diagnostics",
    )
    api.upload_file(
        repo_id=args.model_repo,
        repo_type="model",
        path_or_fileobj=release_path,
        path_in_repo="training_result.json",
        commit_message="Add retail-bank training result",
    )
    api.upload_file(
        repo_id=args.model_repo,
        repo_type="model",
        path_or_fileobj=card_path,
        path_in_repo="README.md",
        commit_message="Document banking-v3 SFT release",
    )
    final_revision = str(api.model_info(args.model_repo).sha)
    require_exact_revision(final_revision, field="final model revision")
    print(
        json.dumps(
            {
                **release,
                "final_model_revision": final_revision,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
