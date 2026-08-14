#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hello_slm.banking_servicing_alignment_data import (  # noqa: E402
    DEFAULT_BASE_SFT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SYNTHETIC_BANK_PATH,
    write_servicing_alignment_dataset,
)

DEFAULT_RELEASE_LOCK: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare local Granite v5 servicing-alignment SFT data."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--base-sft-dir",
        type=Path,
        default=DEFAULT_BASE_SFT_DIR,
        help="Governed released SFT copy merged with the alignment augmentation.",
    )
    parser.add_argument(
        "--synthetic-bank",
        type=Path,
        default=DEFAULT_SYNTHETIC_BANK_PATH,
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Publish the generated dataset after local validation.",
    )
    parser.add_argument(
        "--expected-release-lock",
        type=Path,
        default=DEFAULT_RELEASE_LOCK,
        help="Optional tracked lock whose composite split hashes must match.",
    )
    parser.add_argument(
        "--repo-id",
        default="spkc83/retail-bank-servicing-alignment-sft",
        help="Dataset repo used only with --push-to-hub.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = write_servicing_alignment_dataset(
        args.output_dir,
        base_sft_dir=args.base_sft_dir,
        synthetic_bank_path=args.synthetic_bank,
    )
    if args.expected_release_lock is not None:
        verify_release_lock(manifest, args.expected_release_lock)
    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=False,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=args.output_dir,
            commit_message="Publish servicing alignment SFT data",
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def verify_release_lock(manifest: dict[str, object], lock_path: Path) -> None:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    entries = manifest["tool_sft"]
    if not isinstance(entries, list):
        raise ValueError("manifest tool_sft must be a list")
    actual = {
        str(entry["name"]): str(entry["sha256"]) for entry in entries if isinstance(entry, dict)
    }
    if actual != lock.get("prepared_split_sha256"):
        raise ValueError("servicing alignment split digests drifted from release lock")
    report = manifest["report"]
    if not isinstance(report, dict):
        raise ValueError("manifest report must be an object")
    if report.get("base_manifest_sha256") != lock.get("base_manifest_sha256"):
        raise ValueError("servicing alignment base manifest digest drifted")


if __name__ == "__main__":
    raise SystemExit(main())
