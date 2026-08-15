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
"""Bootstrap exact V5 PEFT-remediation inputs inside a Hugging Face GPU Job."""

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

from huggingface_hub import HfApi, snapshot_download

SOURCE_REPO = "spkc83/retail-bank-servicing"
CANDIDATE5_PROTOCOL = "retail-bank-peft-candidate5/v1"
DATASET_REPO = "spkc83/retail-bank-servicing-alignment-sft"
ADAPTER_REPO = "spkc83/retail-bank-servicing-agent-9b-peft-v5-remediation"
DEFAULT_SOURCE_ADAPTER_REVISION = "d965816bd6a9252bfb4327c1b0d64f9d34f4a1a2"
DEFAULT_DESTINATION_REPO = "spkc83/retail-bank-servicing-agent-9b-peft-v5-candidate5"
DEFAULT_PROBE_CHECKPOINT_DIR = (
    "/data/retail-bank-agent-9b-continuation-34484bb0-d965816b-715064e5/trainer/checkpoint-600"
)
DEFAULT_PROBE_CHECKPOINT_STEP = 600
CANDIDATE3_PROBE_DATASET_REVISION = "715064e50e7ed2f815dfd3ce19b61f345a466b9d"
CANCELED_CANDIDATE5_DATASET_REVISION = "70c9cd9a9075ddbc1bf9aece0253dd62bd769c9d"
DEFAULT_RESUME_CHECKPOINT_DIR = (
    "/data/retail-bank-agent-9b-candidate5-4e86f632-d965816b-70c9cd9a/trainer/checkpoint-350"
)
DEFAULT_RESUME_CHECKPOINT_STEP = 350
RESUME_BUCKET_ID = "spkc83/jobs-artifacts"
RESUME_BUCKET_PREFIX = (
    "retail-bank-agent-9b-candidate5-4e86f632-d965816b-70c9cd9a/trainer/checkpoint-350"
)
RESUME_CHECKPOINT_XET_HASHES = {
    "adapter_config.json": "a9682f1296289fce43ec798430798468b64c795c13d528c17e4b2024030c3529",
    "adapter_model.safetensors": "f1af57fc28b05efbac63192b5652b1cf49d3b2504778fa3629b207fb3536940d",
    "optimizer.pt": "1100817feb246ec0d2ccc847d750dd01ab294af5f38d051a4bee4a9a2b09d532",
    "rng_state.pth": "0cd6c1c69085489da8c3ce055699dc6b0a46446c64096bf6087bddc7bb6007b5",
    "scheduler.pt": "25adb42bf10cd7f5dd953b978d0bbd3572b83d502ec96121dfdcc2ebf28cf9d1",
    "trainer_state.json": "0dbffbed61861d125aa1c65067c03df2620ebc9eb80bb6e2fdddafb52453fa44",
    "training_args.bin": "57cd397e7cb202aa8682719e1eb2f8132c62f8328c47bbbaa1546a9f9bc0f6ca",
}
BASE_MODEL = "spkc83/retail-bank-servicing-agent-9b"
BASE_REVISION = "1d56824995aa1adecfe20f62ca42fb1c0c443817"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--source-adapter-repo", default=ADAPTER_REPO)
    parser.add_argument("--source-adapter-revision", default=DEFAULT_SOURCE_ADAPTER_REVISION)
    parser.add_argument("--destination-repo", default=DEFAULT_DESTINATION_REPO)
    parser.add_argument("--output-dir", default="/data/retail-bank-agent-9b-candidate5")
    parser.add_argument("--max-steps", type=int, default=964)
    parser.add_argument("--max-train-seconds", type=int, default=3_600)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--positive-multiplier", type=int, default=2)
    parser.add_argument("--ambiguity-multiplier", type=int, default=1)
    parser.add_argument("--policy-faq-multiplier", type=int, default=4)
    parser.add_argument("--tool-outcome-multiplier", type=int, default=6)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--publish-only", action="store_true")
    parser.add_argument("--resume-canceled-candidate5", action="store_true")
    parser.add_argument("--probe-checkpoint-dir", default=DEFAULT_PROBE_CHECKPOINT_DIR)
    parser.add_argument("--probe-checkpoint-step", type=int, default=DEFAULT_PROBE_CHECKPOINT_STEP)
    parser.add_argument("--resume-checkpoint-dir", default=DEFAULT_RESUME_CHECKPOINT_DIR)
    parser.add_argument(
        "--resume-checkpoint-step", type=int, default=DEFAULT_RESUME_CHECKPOINT_STEP
    )
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
    if args.resume_canceled_candidate5:
        return [
            "--resume-canceled-candidate5",
            "--resume-checkpoint-dir",
            str(args.resume_checkpoint_dir),
            "--resume-checkpoint-step",
            str(args.resume_checkpoint_step),
            "--push-to-hub",
        ]
    return ["--push-to-hub"]


def validate_resume_bucket_checkpoint(api: HfApi) -> None:
    paths = [f"{RESUME_BUCKET_PREFIX}/{name}" for name in RESUME_CHECKPOINT_XET_HASHES]
    entries = list(api.get_bucket_paths_info(RESUME_BUCKET_ID, paths))
    by_name = {Path(entry.path).name: entry for entry in entries}
    missing = sorted(set(RESUME_CHECKPOINT_XET_HASHES) - set(by_name))
    if missing:
        raise RuntimeError(f"resume bucket checkpoint is incomplete: {missing}")
    mismatches = [
        name
        for name, expected_hash in RESUME_CHECKPOINT_XET_HASHES.items()
        if by_name[name].xet_hash != expected_hash
    ]
    if mismatches:
        raise RuntimeError(f"resume bucket checkpoint xet hash mismatch: {mismatches}")


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


def validate_candidate5_source(source_root: Path) -> None:
    worker = source_root / "scripts/retail_bank/cloud_continue_tool_sft.py"
    marker = f'CANDIDATE5_PROTOCOL = "{CANDIDATE5_PROTOCOL}"'
    if not worker.is_file() or marker not in worker.read_text(encoding="utf-8"):
        raise RuntimeError(
            "source commit does not implement the required candidate5 worker protocol"
        )


def main() -> int:
    args = parse_args()
    require_exact_revision(args.dataset_revision, field="--dataset-revision")
    validate_source_adapter(args.source_adapter_repo, args.source_adapter_revision)
    modes = (args.probe_only, args.publish_only, args.resume_canceled_candidate5)
    if sum(bool(mode) for mode in modes) > 1:
        raise ValueError("probe, publish-only, and resume modes are mutually exclusive")
    if args.destination_repo == args.source_adapter_repo:
        raise ValueError("--destination-repo must differ from the source adapter repository")
    if args.probe_only and args.dataset_revision != CANDIDATE3_PROBE_DATASET_REVISION:
        raise ValueError(
            "checkpoint probes require the exact candidate3 dataset revision "
            f"{CANDIDATE3_PROBE_DATASET_REVISION}"
        )
    if args.resume_canceled_candidate5:
        if args.dataset_revision != CANCELED_CANDIDATE5_DATASET_REVISION:
            raise ValueError(
                "candidate5 resume requires dataset revision "
                f"{CANCELED_CANDIDATE5_DATASET_REVISION}"
            )
        if (
            args.resume_checkpoint_dir != DEFAULT_RESUME_CHECKPOINT_DIR
            or args.resume_checkpoint_step != DEFAULT_RESUME_CHECKPOINT_STEP
            or args.max_steps != 964
        ):
            raise ValueError("candidate5 resume checkpoint and 964-step horizon are immutable")
    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("HF_TOKEN must be passed as a Hugging Face Job secret")
    if args.resume_canceled_candidate5:
        validate_resume_bucket_checkpoint(HfApi(token=os.environ["HF_TOKEN"]))
    with tempfile.TemporaryDirectory(prefix="retail-bank-agent-continuation-") as temp_dir:
        temp_root = Path(temp_dir)
        source_root = download_source(args.source_commit, temp_root / "source")
        validate_candidate5_source(source_root)
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
            "RETAIL_BANK_ALLOW_REMOTE_CONTINUATION_SFT": "banking-v5-peft-remediation",
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
            "retail-bank-agent-v5-remediation",
            "--trackio-run-name",
            f"granite-peft-candidate5-{args.source_commit[:8]}",
        ]
        command.extend(execution_mode_args(args))
        subprocess.run(command, cwd=source_root, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
