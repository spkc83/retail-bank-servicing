# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate==1.12.0",
#   "datasets==4.5.0",
#   "huggingface-hub==1.22.0",
#   "safetensors==0.8.0",
#   "torch>=2.9,<3",
#   "transformers==5.13.0",
# ]
# ///
"""Bootstrap pinned banking V5 frozen tool evaluation inside a Hugging Face GPU Job."""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

SOURCE_REPO = "spkc83/retail-bank-servicing"
MODEL_REPO = "spkc83/retail-bank-servicing-agent-9b"
DATASET_REPO = "spkc83/retail-bank-servicing-alignment-sft"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--model-repo", default=MODEL_REPO)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-repo", default=DATASET_REPO)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--output-dir", default="/data/retail-bank-agent-9b-tool-eval")
    parser.add_argument("--max-new-tokens-first", type=int, default=192)
    parser.add_argument("--max-new-tokens-final", type=int, default=220)
    return parser.parse_args()


def validate_git_revision(value: str, *, field: str) -> None:
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be the exact 40-character lowercase Git revision")


def download_source(source_commit: str, destination: Path) -> Path:
    validate_git_revision(source_commit, field="--source-commit")
    url = f"https://github.com/{SOURCE_REPO}/archive/{source_commit}.tar.gz"
    destination.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "retail-bank-tool-eval-job"})
    with urllib.request.urlopen(request, timeout=120) as response:
        archive = response.read()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        bundle.extractall(destination, filter="data")
    roots = [path for path in destination.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise RuntimeError("source archive did not contain exactly one repository root")
    return roots[0]


def main() -> int:
    args = parse_args()
    validate_git_revision(args.model_revision, field="--model-revision")
    validate_git_revision(args.dataset_revision, field="--dataset-revision")
    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("HF_TOKEN must be passed as a Hugging Face Job secret")

    with tempfile.TemporaryDirectory(prefix="retail-bank-agent-eval-source-") as temp_dir:
        source_root = download_source(args.source_commit, Path(temp_dir) / "source")
        env = {
            **os.environ,
            "PYTHONPATH": str(source_root / "src"),
            "RETAIL_BANK_SOURCE_COMMIT": args.source_commit,
            "RETAIL_BANK_TOOL_EVAL_MODEL_REPO": args.model_repo,
            "RETAIL_BANK_TOOL_EVAL_MODEL_REVISION": args.model_revision,
            "RETAIL_BANK_TOOL_EVAL_DATASET_REPO": args.dataset_repo,
            "RETAIL_BANK_TOOL_EVAL_DATASET_REVISION": args.dataset_revision,
        }
        command = [
            sys.executable,
            str(source_root / "scripts/retail_bank/cloud_generate_tool_eval.py"),
            "--model-repo",
            args.model_repo,
            "--model-revision",
            args.model_revision,
            "--dataset-repo",
            args.dataset_repo,
            "--dataset-revision",
            args.dataset_revision,
            "--output-dir",
            args.output_dir,
            "--max-new-tokens-first",
            str(args.max_new_tokens_first),
            "--max-new-tokens-final",
            str(args.max_new_tokens_final),
            "--dtype",
            "fp16",
            "--push-to-hub",
            "--enforce-release-gates",
        ]
        subprocess.run(command, cwd=source_root, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
