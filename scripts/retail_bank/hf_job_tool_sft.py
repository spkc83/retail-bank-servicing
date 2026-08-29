# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate==1.12.0",
#   "bitsandbytes==0.50.0",
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
"""Bootstrap the pinned SFT source inside a Hugging Face GPU Job."""

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
DATASET_REPO = "spkc83/retail-bank-servicing-alignment-sft"
MODEL_REPO = "spkc83/retail-bank-servicing-agent-9b"
BASE_MODEL = "spkc83/retail-bank-servicing-agent-9b"
BASE_REVISION = "1d56824995aa1adecfe20f62ca42fb1c0c443817"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--dataset-repo", default=DATASET_REPO)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--base-family", default="granite")
    parser.add_argument("--manifest")
    parser.add_argument("--output-dir", default="/data/retail-bank-agent-9b")
    parser.add_argument("--hub-dest", default=MODEL_REPO)
    parser.add_argument("--resume-from")
    parser.add_argument("--max-steps", type=int, default=3_000)
    parser.add_argument("--max-train-seconds", type=int, default=14_400)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--trackio-project", default="retail-bank-agent-v5")
    parser.add_argument("--trackio-run-name")
    parser.add_argument(
        "--confirmation-token",
        default="banking-v5-grounded-dialogue-sft",
        help="Value required by the worker confirmation env guard.",
    )
    parser.add_argument(
        "--skip-merge-adapter",
        action="store_true",
        help="Publish the trained adapter without merging it into FP16 base weights.",
    )
    parser.add_argument("--training-seed", type=int, default=7303)
    parser.add_argument("--positive-multiplier", type=int, default=1)
    parser.add_argument("--ambiguity-multiplier", type=int, default=1)
    parser.add_argument("--policy-faq-multiplier", type=int, default=1)
    parser.add_argument("--tool-outcome-multiplier", type=int, default=1)
    return parser


def parse_args_from(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def parse_args() -> argparse.Namespace:
    return parse_args_from()


MULTIPLIER_ARGUMENTS = (
    ("--positive-multiplier", "positive_multiplier"),
    ("--ambiguity-multiplier", "ambiguity_multiplier"),
    ("--policy-faq-multiplier", "policy_faq_multiplier"),
    ("--tool-outcome-multiplier", "tool_outcome_multiplier"),
)


def validate_arguments(args: argparse.Namespace) -> None:
    """Reject a self-overwriting destination and an out-of-range mix before any network call."""

    if args.hub_dest == args.base_model:
        raise ValueError(
            f"--hub-dest {args.hub_dest!r} must differ from the training base model; "
            "publishing into the base repository would overwrite the weights this run "
            "trains from"
        )
    for flag, attribute in MULTIPLIER_ARGUMENTS:
        value = int(getattr(args, attribute))
        if not 1 <= value <= 99:
            raise ValueError(f"{flag} multiplier must be a whole number from 1 to 99")


def build_worker_command(
    args: argparse.Namespace,
    *,
    source_root: Path,
    manifest: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(source_root / "scripts/retail_bank/cloud_train_tool_sft.py"),
        "--execute-remote",
        "--allow-remote-execution",
        "--push-to-hub",
        "--manifest",
        str(manifest),
        "--output-dir",
        args.output_dir,
        "--hub-dest",
        args.hub_dest,
        "--base-model",
        args.base_model,
        "--base-revision",
        args.base_revision,
        "--family",
        args.base_family,
        "--precision",
        "bf16-lora",
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
        str(args.learning_rate),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--trackio-project",
        args.trackio_project,
        "--trackio-run-name",
        args.trackio_run_name or f"{args.base_family}-tool-sft-{args.source_commit[:8]}",
    ]
    for flag, attribute in MULTIPLIER_ARGUMENTS:
        command.extend([flag, str(int(getattr(args, attribute)))])
    if args.skip_merge_adapter:
        command.append("--skip-merge-adapter")
    if args.resume_from:
        command.extend(["--resume-from", args.resume_from])
    return command


def download_source(source_commit: str, destination: Path) -> Path:
    if len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit):
        raise ValueError("--source-commit must be an exact 40-character lowercase Git commit")
    url = f"https://github.com/{SOURCE_REPO}/archive/{source_commit}.tar.gz"
    destination.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "retail-bank-tool-sft-job"})
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
    validate_arguments(args)
    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("HF_TOKEN must be passed as a Hugging Face Job secret")
    with tempfile.TemporaryDirectory(prefix="retail-bank-agent-source-") as temp_dir:
        temp_root = Path(temp_dir)
        source_root = download_source(args.source_commit, temp_root / "source")
        dataset_root = Path(
            snapshot_download(
                repo_id=args.dataset_repo,
                repo_type="dataset",
                revision=args.dataset_revision,
                local_dir=temp_root / "dataset",
                token=os.environ["HF_TOKEN"],
            )
        )
        manifest = dataset_root / args.manifest if args.manifest else dataset_root / "manifest.json"
        if not manifest.is_file():
            raise RuntimeError(f"dataset manifest is unavailable: {manifest}")
        env = {
            **os.environ,
            "PYTHONPATH": str(source_root / "src"),
            "RETAIL_BANK_ALLOW_REMOTE_TOOL_SFT": args.confirmation_token,
            "RETAIL_BANK_SOURCE_COMMIT": args.source_commit,
            "RETAIL_BANK_TOOL_SFT_DATASET_REPO": args.dataset_repo,
            "RETAIL_BANK_TOOL_SFT_DATASET_REVISION": args.dataset_revision,
            "RETAIL_BANK_TRAINING_SEED": str(args.training_seed),
        }
        command = build_worker_command(args, source_root=source_root, manifest=manifest)
        subprocess.run(command, cwd=source_root, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
