#!/usr/bin/env python
"""Publish the allowlisted POC source and persist exact Space runtime pins."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_SOURCE_DIR = Path("poc/retail-bank-customer-service-poc")
DEFAULT_BASE_MODEL_ID = "spkc83/retail-bank-servicing-agent-9b"
DEFAULT_BASE_MODEL_REVISION = "1d56824995aa1adecfe20f62ca42fb1c0c443817"
DEFAULT_ADAPTER_ID = "spkc83/retail-bank-servicing-agent-9b-peft"
DEFAULT_ADAPTER_REVISION = "cc95e446af2b5e1d8d9df2751a8192613ad386e3"
ALLOW_PATTERNS = ["*.py", "*.md", "*.txt", "*.toml", "*.lock", "*.json", "LICENSE"]
IGNORE_PATTERNS = [
    "tests/**",
    "**/tests/**",
    ".*",
    ".*/**",
    "**/.*",
    "**/__pycache__/**",
    "**/*.pyc",
]


class DeployError(ValueError):
    """Raised when a Space deployment is unsafe or incomplete."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space-id", required=True)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--base-model-id", default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--base-model-revision", default=DEFAULT_BASE_MODEL_REVISION)
    parser.add_argument("--adapter-id", default=DEFAULT_ADAPTER_ID)
    parser.add_argument("--adapter-revision", default=DEFAULT_ADAPTER_REVISION)
    parser.add_argument("--merged-model-only", action="store_true")
    parser.add_argument("--model-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--router-id", required=True)
    parser.add_argument("--router-revision", required=True)
    parser.add_argument("--wait-timeout", type=float, default=600.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-publish", action="store_true")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    return parser.parse_args(argv)


def exact_revision(value: str, *, field: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise DeployError(f"{field} must be an exact 40-character lowercase revision")
    return value


def runtime_pins(args: argparse.Namespace, *, space_commit: str | None = None) -> dict[str, str]:
    base_model_id = None if args.merged_model_only else args.base_model_id
    base_model_revision = None if args.merged_model_only else args.base_model_revision
    adapter_id = None if args.merged_model_only else args.adapter_id
    adapter_revision = None if args.merged_model_only else args.adapter_revision
    composition = (
        base_model_id,
        base_model_revision,
        adapter_id,
        adapter_revision,
    )
    if any(composition) and not all(composition):
        raise DeployError(
            "PEFT deployment requires --base-model-id, --base-model-revision, "
            "--adapter-id, and --adapter-revision"
        )
    pins = {
        "RETAIL_BANK_MODEL_ID": str(args.model_id),
        "RETAIL_BANK_MODEL_REVISION": exact_revision(
            str(args.model_revision),
            field="model revision",
        ),
        "RETAIL_BANK_MODEL_DTYPE": str(args.model_dtype),
        "RETAIL_BANK_BASE_MODEL_ID": str(base_model_id or ""),
        "RETAIL_BANK_BASE_MODEL_REVISION": (
            exact_revision(str(base_model_revision), field="base model revision")
            if base_model_revision
            else ""
        ),
        "RETAIL_BANK_ADAPTER_ID": str(adapter_id or ""),
        "RETAIL_BANK_ADAPTER_REVISION": (
            exact_revision(str(adapter_revision), field="adapter revision")
            if adapter_revision
            else ""
        ),
        "RETAIL_BANK_ROUTER_ID": str(args.router_id),
        "RETAIL_BANK_ROUTER_REVISION": exact_revision(
            str(args.router_revision),
            field="router revision",
        ),
    }
    if space_commit is not None:
        pins["SPACE_COMMIT_SHA"] = exact_revision(space_commit, field="Space commit")
    return pins


def plan(args: argparse.Namespace) -> dict[str, Any]:
    if not args.source_dir.is_dir():
        raise DeployError(f"Space source directory is unavailable: {args.source_dir}")
    if args.wait_timeout <= 0:
        raise DeployError("--wait-timeout must be positive")
    return {
        "contract": "retail-bank-zero-gpu-deploy/v1",
        "space_id": str(args.space_id),
        "source_dir": str(args.source_dir),
        "allow_patterns": ALLOW_PATTERNS,
        "ignore_patterns": IGNORE_PATTERNS,
        "runtime_pins": runtime_pins(args),
    }


def deploy(args: argparse.Namespace, api: Any) -> dict[str, Any]:
    deployment_plan = plan(args)
    if not args.execute or not args.allow_publish:
        raise DeployError("deployment requires --execute and --allow-publish")
    commit = api.upload_folder(
        repo_id=args.space_id,
        repo_type="space",
        folder_path=args.source_dir,
        path_in_repo=".",
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        commit_message="Deploy pinned retail-bank servicing POC",
        token=args.token,
    )
    commit_oid = exact_revision(str(getattr(commit, "oid", "")), field="Space commit")
    pins = runtime_pins(args, space_commit=commit_oid)
    for key, value in pins.items():
        api.add_space_variable(
            repo_id=args.space_id,
            key=key,
            value=value,
            token=args.token,
        )
    runtime = api.wait_for_space(
        args.space_id,
        timeout=args.wait_timeout,
        token=args.token,
    )
    return {
        **deployment_plan,
        "mode": "executed",
        "space_commit": commit_oid,
        "runtime_pins": pins,
        "runtime_stage": str(getattr(runtime, "stage", "unknown")),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    deployment_plan = plan(args)
    if args.execute:
        try:
            from huggingface_hub import HfApi
        except ModuleNotFoundError as error:
            raise DeployError(
                "huggingface_hub is required; install the documented dev or scale extra"
            ) from error

        result: Mapping[str, Any] = deploy(args, HfApi())
    else:
        result = {**deployment_plan, "mode": "plan"}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DeployError as error:
        print(json.dumps({"error": str(error)}, sort_keys=True))
        raise SystemExit(2) from error
