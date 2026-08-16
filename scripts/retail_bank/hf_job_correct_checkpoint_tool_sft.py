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
"""Bootstrap a provenance-gated, weights-only V7 checkpoint correction job."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

SOURCE_REPO = "spkc83/retail-bank-servicing"
DATASET_REPO = "spkc83/retail-bank-servicing-alignment-sft"
ADAPTER_REPO = "spkc83/retail-bank-servicing-agent-9b-peft-v5-remediation"
BASE_MODEL = "spkc83/retail-bank-servicing-agent-9b"
BASE_REVISION = "1d56824995aa1adecfe20f62ca42fb1c0c443817"
TRAINING_SOURCE_COMMIT = "0bb83262981e16fa85ad3a411ecd1be6ddbf0b88"
DATASET_REVISION = "7b58aa748d69bf56d9b32eb65989b8c07e04f835"
PARENT_ADAPTER_REVISION = "d965816bd6a9252bfb4327c1b0d64f9d34f4a1a2"
FAILED_JOB_ID = "6a813914c97db76cbdf32acc"
FAILED_OUTPUT_ROOT = Path(
    "/data/retail-bank-agent-9b-peft-v6-generation-contract-0bb83262-d965816b-7b58aa74"
)
DEFAULT_CHECKPOINT_STEP = 964
DEFAULT_CHECKPOINT = FAILED_OUTPUT_ROOT / f"trainer/checkpoint-{DEFAULT_CHECKPOINT_STEP}"
DEFAULT_DESTINATION = "spkc83/retail-bank-servicing-agent-9b-peft-v7-grounded-generation"
CORRECTION_PROTOCOL = "retail-bank-peft-v7-checkpoint-correction/v1"
CORRECTION_SOURCE_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "trainer_state.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-source-commit", required=True)
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--checkpoint-step", type=int, default=DEFAULT_CHECKPOINT_STEP)
    parser.add_argument("--destination-repo", default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--output-dir",
        default=(
            "/data/retail-bank-agent-9b-peft-v7-checkpoint-correction-"
            f"{TRAINING_SOURCE_COMMIT[:8]}-{DATASET_REVISION[:8]}-step{DEFAULT_CHECKPOINT_STEP}"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--max-train-seconds", type=int, default=3_600)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    return parser.parse_args()


def require_exact_revision(value: str, *, field: str) -> None:
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be an exact 40-character lowercase Git revision")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(source_commit: str, destination: Path) -> Path:
    require_exact_revision(source_commit, field="--recovery-source-commit")
    url = f"https://github.com/{SOURCE_REPO}/archive/{source_commit}.tar.gz"
    destination.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "retail-bank-v7-checkpoint-correction"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        archive = response.read()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        bundle.extractall(destination, filter="data")
    roots = [path for path in destination.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise RuntimeError("source archive did not contain exactly one repository root")
    return roots[0]


def validate_recovery_source(source_root: Path) -> None:
    worker = source_root / "scripts/retail_bank/cloud_continue_tool_sft.py"
    marker = f'V7_CHECKPOINT_CORRECTION_PROTOCOL = "{CORRECTION_PROTOCOL}"'
    if not worker.is_file() or marker not in worker.read_text(encoding="utf-8"):
        raise RuntimeError("source commit does not implement the V7 checkpoint correction protocol")


def expected_failed_job_command() -> list[str]:
    return [
        "uv",
        "run",
        (
            "https://raw.githubusercontent.com/spkc83/retail-bank-servicing/"
            f"{TRAINING_SOURCE_COMMIT}/scripts/retail_bank/hf_job_continue_tool_sft.py"
        ),
        "--source-commit",
        TRAINING_SOURCE_COMMIT,
        "--dataset-revision",
        DATASET_REVISION,
        "--source-adapter-repo",
        ADAPTER_REPO,
        "--source-adapter-revision",
        PARENT_ADAPTER_REVISION,
        "--destination-repo",
        DEFAULT_DESTINATION,
        "--output-dir",
        str(FAILED_OUTPUT_ROOT),
        "--max-steps",
        "964",
    ]


def _iso(value: datetime | None, *, field: str) -> str:
    if value is None or value.tzinfo is None:
        raise RuntimeError(f"failed job inspection is missing timezone-aware {field}")
    return value.isoformat()


def build_provenance(checkpoint: Path, expected_step: int, job: Any) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    if checkpoint.parent != FAILED_OUTPUT_ROOT.resolve() / "trainer":
        raise RuntimeError("checkpoint is outside the exact failed V7 trainer root")
    if checkpoint.name != f"checkpoint-{expected_step}":
        raise RuntimeError("checkpoint directory does not match its expected step")
    missing = [name for name in CORRECTION_SOURCE_FILES if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(f"checkpoint is missing correction source files: {missing}")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if state.get("global_step") != expected_step:
        raise RuntimeError("checkpoint trainer_state global_step mismatch")
    volumes = [
        {
            "type": volume.type,
            "source": volume.source,
            "mount_path": volume.mount_path,
            "read_only": bool(volume.read_only),
        }
        for volume in job.volumes
    ]
    expected_volume = {
        "type": "bucket",
        "source": "spkc83/jobs-artifacts",
        "mount_path": "/data",
        "read_only": False,
    }
    if (
        job.id != FAILED_JOB_ID
        or job.status.stage != "ERROR"
        or job.flavor != "rtx-pro-6000"
        or list(job.command) != expected_failed_job_command()
        or volumes != [expected_volume]
    ):
        raise RuntimeError("failed job inspection does not match exact V7 training provenance")
    return {
        "contract": CORRECTION_PROTOCOL,
        "created_at": datetime.now(UTC).isoformat(),
        "training_identity": {
            "training_source_commit": TRAINING_SOURCE_COMMIT,
            "dataset_repository": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "parent_adapter_repository": ADAPTER_REPO,
            "parent_adapter_revision": PARENT_ADAPTER_REVISION,
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
        },
        "inspected_job": {
            "id": job.id,
            "created_at": _iso(job.created_at, field="created_at"),
            "started_at": _iso(job.started_at, field="started_at"),
            "finished_at": _iso(job.finished_at, field="finished_at"),
            "stage": job.status.stage,
            "flavor": job.flavor,
            "command": list(job.command),
            "durable_volume": expected_volume,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "global_step": expected_step,
            "files": {
                name: {
                    "sha256": sha256(checkpoint / name),
                    "modified_at": datetime.fromtimestamp(
                        (checkpoint / name).stat().st_mtime,
                        tz=UTC,
                    ).isoformat(),
                }
                for name in CORRECTION_SOURCE_FILES
            },
        },
        "resume_semantics": {
            "adapter_weights": "loaded",
            "tokenizer": "loaded",
            "optimizer": "fresh",
            "scheduler": "fresh",
            "rng_state": "fresh seeded run",
            "exact_resume_claimed": False,
        },
    }


def main() -> int:
    args = parse_args()
    require_exact_revision(args.recovery_source_commit, field="--recovery-source-commit")
    if args.max_steps < 1 or args.max_train_seconds < 60 or args.checkpoint_every < 1:
        raise ValueError("correction caps and checkpoint interval must be positive")
    if args.destination_repo == ADAPTER_REPO:
        raise ValueError("destination repository must differ from the parent adapter repository")
    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("HF_TOKEN must be passed as a Hugging Face Job secret")
    checkpoint = Path(args.checkpoint_dir)
    with tempfile.TemporaryDirectory(prefix="retail-bank-v7-correction-") as temp_dir:
        temp_root = Path(temp_dir)
        source_root = download_source(args.recovery_source_commit, temp_root / "source")
        validate_recovery_source(source_root)
        job = HfApi(token=os.environ["HF_TOKEN"]).inspect_job(job_id=FAILED_JOB_ID)
        provenance = build_provenance(checkpoint, args.checkpoint_step, job)
        provenance_path = temp_root / "checkpoint_correction_provenance.json"
        provenance_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        dataset_root = Path(
            snapshot_download(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                revision=DATASET_REVISION,
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
            "RETAIL_BANK_SOURCE_COMMIT": args.recovery_source_commit,
            "RETAIL_BANK_TRAINING_SOURCE_COMMIT": TRAINING_SOURCE_COMMIT,
            "RETAIL_BANK_TOOL_SFT_DATASET_REPO": DATASET_REPO,
            "RETAIL_BANK_TOOL_SFT_DATASET_REVISION": DATASET_REVISION,
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
            ADAPTER_REPO,
            "--source-adapter-revision",
            PARENT_ADAPTER_REVISION,
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
            "2",
            "--max-seq-len",
            "2048",
            "--learning-rate",
            "1e-6",
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--positive-multiplier",
            "2",
            "--ambiguity-multiplier",
            "1",
            "--policy-faq-multiplier",
            "4",
            "--tool-outcome-multiplier",
            "6",
            "--correction-checkpoint-dir",
            str(checkpoint),
            "--correction-checkpoint-step",
            str(args.checkpoint_step),
            "--correction-provenance",
            str(provenance_path),
            "--trackio-project",
            "retail-bank-agent-v7-checkpoint-correction",
            "--trackio-run-name",
            f"granite-v7-correction-{args.recovery_source_commit[:8]}-step{args.checkpoint_step}",
            "--push-to-hub",
        ]
        subprocess.run(command, cwd=source_root, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
