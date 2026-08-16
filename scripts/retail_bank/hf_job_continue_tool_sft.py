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
"""Bootstrap exact V6 generation-contract PEFT inputs in a Hugging Face GPU Job."""

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

from huggingface_hub import snapshot_download

SOURCE_REPO = "spkc83/retail-bank-servicing"
V6_CONTINUATION_PROTOCOL = "retail-bank-peft-v6-generation-contract/v1"
DATASET_REPO = "spkc83/retail-bank-servicing-alignment-sft"
ADAPTER_REPO = "spkc83/retail-bank-servicing-agent-9b-peft-v5-remediation"
DEFAULT_SOURCE_ADAPTER_REVISION = "d965816bd6a9252bfb4327c1b0d64f9d34f4a1a2"
DEFAULT_DESTINATION_REPO = "spkc83/retail-bank-servicing-agent-9b-peft-v6-generation-contract"
DEFAULT_PROBE_CHECKPOINT_DIR = (
    "/data/retail-bank-agent-9b-continuation-34484bb0-d965816b-715064e5/trainer/checkpoint-600"
)
DEFAULT_PROBE_CHECKPOINT_STEP = 600
CANDIDATE3_PROBE_DATASET_REVISION = "715064e50e7ed2f815dfd3ce19b61f345a466b9d"
BASE_MODEL = "spkc83/retail-bank-servicing-agent-9b"
BASE_REVISION = "1d56824995aa1adecfe20f62ca42fb1c0c443817"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--source-adapter-repo", default=ADAPTER_REPO)
    parser.add_argument("--source-adapter-revision", default=DEFAULT_SOURCE_ADAPTER_REVISION)
    parser.add_argument("--destination-repo", default=DEFAULT_DESTINATION_REPO)
    parser.add_argument(
        "--output-dir",
        default="/data/retail-bank-agent-9b-peft-v6-generation-contract",
    )
    parser.add_argument("--max-steps", type=int, default=964)
    parser.add_argument("--max-train-seconds", type=int, default=3_600)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--positive-multiplier", type=int, default=2)
    parser.add_argument("--ambiguity-multiplier", type=int, default=2)
    parser.add_argument("--policy-faq-multiplier", type=int, default=4)
    parser.add_argument("--tool-outcome-multiplier", type=int, default=6)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--publish-only", action="store_true")
    parser.add_argument("--probe-checkpoint-dir", default=DEFAULT_PROBE_CHECKPOINT_DIR)
    parser.add_argument("--probe-checkpoint-step", type=int, default=DEFAULT_PROBE_CHECKPOINT_STEP)
    return parser.parse_args()


def require_exact_revision(value: str, *, field: str) -> None:
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be an exact 40-character lowercase Git revision")


def validate_source_adapter(repository: str, revision: str) -> None:
    require_exact_revision(revision, field="--source-adapter-revision")
    if repository != ADAPTER_REPO or revision != DEFAULT_SOURCE_ADAPTER_REVISION:
        raise ValueError(
            f"source adapter must be exactly {ADAPTER_REPO}@{DEFAULT_SOURCE_ADAPTER_REVISION}"
        )


def execution_mode_args(args: argparse.Namespace) -> list[str]:
    if args.publish_only:
        return ["--publish-only", "--push-to-hub"]
    if args.probe_only:
        return [
            "--probe-only",
            "--probe-checkpoint-dir",
            str(args.probe_checkpoint_dir),
            "--probe-checkpoint-step",
            str(args.probe_checkpoint_step),
        ]
    return ["--push-to-hub"]


def download_source(source_commit: str, destination: Path) -> Path:
    require_exact_revision(source_commit, field="--source-commit")
    url = f"https://github.com/{SOURCE_REPO}/archive/{source_commit}.tar.gz"
    destination.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "retail-bank-tool-sft-continuation-job"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        archive = response.read()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        bundle.extractall(destination, filter="data")
    roots = [path for path in destination.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise RuntimeError("source archive did not contain exactly one repository root")
    return roots[0]


def validate_v6_source(source_root: Path) -> None:
    worker = source_root / "scripts/retail_bank/cloud_continue_tool_sft.py"
    marker = f'V6_CONTINUATION_PROTOCOL = "{V6_CONTINUATION_PROTOCOL}"'
    if not worker.is_file() or marker not in worker.read_text(encoding="utf-8"):
        raise RuntimeError(
            "source commit does not implement the required V6 continuation worker protocol"
        )


def main() -> int:
    args = parse_args()
    require_exact_revision(args.dataset_revision, field="--dataset-revision")
    validate_source_adapter(args.source_adapter_repo, args.source_adapter_revision)
    if args.probe_only and args.publish_only:
        raise ValueError("--probe-only and --publish-only are mutually exclusive")
    if args.destination_repo == args.source_adapter_repo:
        raise ValueError("--destination-repo must differ from the source adapter repository")
    if args.probe_only and args.dataset_revision != CANDIDATE3_PROBE_DATASET_REVISION:
        raise ValueError(
            "checkpoint probes require the exact candidate3 dataset revision "
            f"{CANDIDATE3_PROBE_DATASET_REVISION}"
        )
    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("HF_TOKEN must be passed as a Hugging Face Job secret")
    with tempfile.TemporaryDirectory(prefix="retail-bank-agent-continuation-") as temp_dir:
        temp_root = Path(temp_dir)
        source_root = download_source(args.source_commit, temp_root / "source")
        validate_v6_source(source_root)
        dataset_root = Path(
            snapshot_download(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                revision=args.dataset_revision,
                local_dir=temp_root / "dataset",
                token=os.environ["HF_TOKEN"],
            )
        )
        manifest = dataset_root / "manifest.json"
        if not manifest.is_file():
            raise RuntimeError(f"dataset manifest is unavailable: {manifest}")
        env = {
            **os.environ,
            "PYTHONPATH": str(source_root / "src"),
            "RETAIL_BANK_ALLOW_REMOTE_CONTINUATION_SFT": "banking-v6-generation-contract-peft",
            "RETAIL_BANK_SOURCE_COMMIT": args.source_commit,
            "RETAIL_BANK_TOOL_SFT_DATASET_REPO": DATASET_REPO,
            "RETAIL_BANK_TOOL_SFT_DATASET_REVISION": args.dataset_revision,
        }
        command = [
            sys.executable,
            str(source_root / "scripts/retail_bank/cloud_continue_tool_sft.py"),
            "--execute-remote",
            "--allow-remote-execution",
            "--manifest",
            str(manifest),
            "--output-dir",
            args.output_dir,
            "--source-adapter-repo",
            args.source_adapter_repo,
            "--source-adapter-revision",
            args.source_adapter_revision,
            "--hub-dest",
            args.destination_repo,
            "--base-model",
            BASE_MODEL,
            "--base-revision",
            BASE_REVISION,
            "--family",
            "granite",
            "--max-steps",
            str(args.max_steps),
            "--max-train-seconds",
            str(args.max_train_seconds),
            "--batch-size",
            "2",
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--max-seq-len",
            "2048",
            "--learning-rate",
            "2e-6",
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--positive-multiplier",
            str(args.positive_multiplier),
            "--ambiguity-multiplier",
            str(args.ambiguity_multiplier),
            "--policy-faq-multiplier",
            str(args.policy_faq_multiplier),
            "--tool-outcome-multiplier",
            str(args.tool_outcome_multiplier),
            "--trackio-project",
            "retail-bank-agent-v6-generation-contract",
            "--trackio-run-name",
            f"granite-peft-v6-generation-contract-{args.source_commit[:8]}",
        ]
        command.extend(execution_mode_args(args))
        subprocess.run(command, cwd=source_root, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
