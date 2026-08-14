from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

WORKER_PATH = Path("scripts/retail_bank/cloud_recover_continuation_export.py")
REMERGE_PATH = Path("scripts/retail_bank/hf_job_remerge_tool_sft.py")
PARITY_PATH = Path("scripts/retail_bank/hf_job_merge_parity.py")
JOB_PATH = Path("scripts/retail_bank/hf_job_recover_continuation_export.py")
LAUNCHER_PATH = Path("scripts/retail_bank/run_remote_continuation_export_recovery.sh")


def _load_worker() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "cloud_recover_continuation_export",
        WORKER_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


WORKER = _load_worker()


def test_recovery_uses_unchanged_parity_gates() -> None:
    assert WORKER.MINIMUM_ARGMAX_AGREEMENT == 0.999
    assert WORKER.MAXIMUM_LOGIT_DIFFERENCE == 0.3
    assert WORKER.MAXIMUM_P999_DIFFERENCE == 0.0703125


def test_recovery_candidate_order_starts_with_fp16_native() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")
    fp16_position = source.index('"merged_subdir": f"merged-{step_label}-fp16-native"')
    fp32_position = source.index('"merged_subdir": f"merged-{step_label}-fp32-fp16"')

    assert fp16_position < fp32_position
    assert '"merge_dtype": "float16"' in source[fp16_position:fp32_position]
    assert '"inference_dtype": "float16"' in source[fp16_position:fp32_position]
    assert "args.recovery_source_commit[:8]" in source


def test_recovery_publishes_only_after_validate_parity() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")
    run_body = source.split("def run_candidate", 1)[1].split("def publish", 1)[0]
    main_body = source.split("def main", 1)[1]

    assert "validate_parity(" in run_body
    assert main_body.index('if candidate["passed"]') < main_body.index(
        "publish(args, candidate=candidate"
    )
    assert "bfloat16" not in main_body


def test_recovery_weights_and_evidence_use_one_atomic_hub_commit() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")
    publish_body = source.split("def publish", 1)[1].split("def main", 1)[0]

    assert "CommitOperationAdd(" in publish_body
    assert "api.create_commit(" in publish_body
    assert "api.upload_folder(" not in publish_body
    assert "api.upload_file(" not in publish_body
    assert '"training_metadata.json"' in publish_body
    assert '"merge_parity_diagnostics.json"' in publish_body
    assert '"training_result.json"' in publish_body
    assert "Returned by the atomic Hub commit" not in publish_body
    assert publish_body.index("weights_revision = str(release_commit.oid)") < (
        publish_body.index('commit_message="Record exact continuation release revision"')
    )


def test_remerge_dtype_defaults_preserve_existing_release_path() -> None:
    source = REMERGE_PATH.read_text(encoding="utf-8")

    assert 'default="float32"' in source
    assert 'default="float16"' in source
    assert 'default="fp16_remerge.json"' in source


def test_parity_preserves_trained_adapter_dtype() -> None:
    source = PARITY_PATH.read_text(encoding="utf-8")

    assert "autocast_adapter_dtype=False" in source
    assert '"adapter_autocast_dtype": False' in source


def test_recovery_launcher_is_export_only_and_capped() -> None:
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
    job = JOB_PATH.read_text(encoding="utf-8")

    assert "--timeout 1h" in launcher
    assert "--volume hf://buckets/spkc83/jobs-artifacts:/data" in launcher
    assert "cloud_recover_continuation_export.py" in job
    assert "cloud_continue_tool_sft.py" not in job
    assert "trainer.train" not in job
    assert "rm " not in launcher
    assert 'selected_adapter_subdir="${8:-checkpoint-${selected_step}}"' in launcher
    assert '--selected-adapter-subdir "$selected_adapter_subdir"' in launcher
    assert '--selected-step "$selected_step"' in launcher
    assert 'output_root="$6"' in launcher
    assert "/scripts/retail_bank/hf_job_recover_continuation_export.py" in launcher
    assert "/scripts/banking_v2/hf_job_recover_continuation_export.py" not in launcher
    assert 'script_url="$legacy_script_url"' not in launcher


def test_recovery_rejects_symbolic_revisions() -> None:
    with pytest.raises(ValueError, match="exact 40-character"):
        WORKER.require_exact_revision("main", field="--recovery-source-commit")


def test_recovery_cross_checks_persisted_and_job_provenance() -> None:
    args = SimpleNamespace(
        parent_model_revision="a" * 40,
        dataset_revision="b" * 40,
        base_model="base",
        base_revision="c" * 40,
        training_source_commit="d" * 40,
        output_root=Path("/data/run"),
        training_job="job-123",
    )
    metadata = {
        "created_at_unix": 150,
        "step": 800,
        "fingerprint": {
            "source_model": {"revision": "a" * 40},
            "dataset_identity": {
                "revision": "b" * 40,
                "manifest_sha256": "e" * 64,
            },
            "base_model": "base",
            "base_revision": "c" * 40,
        },
    }
    job = SimpleNamespace(
        id="job-123",
        started_at=datetime.fromtimestamp(100, tz=UTC),
        finished_at=datetime.fromtimestamp(200, tz=UTC),
        command=[
            "uv",
            "run",
            "worker.py",
            "--source-commit",
            "d" * 40,
            "--dataset-revision",
            "b" * 40,
            "--source-model-revision",
            "a" * 40,
            "--output-dir",
            "/data/run",
            "--max-steps",
            "800",
        ],
    )

    verified = WORKER.validate_training_provenance(
        args,
        metadata,
        job,
        {
            "metadata": 150,
            "adapter": 151,
            "trainer_state": 152,
        },
    )

    assert verified["training_job_command_verified"] is True
    assert verified["training_job_artifact_window_verified"] is True
    assert verified["dataset_manifest_sha256"] == "e" * 64
    assert verified["max_steps"] == 800


def test_recovery_rejects_job_metadata_provenance_mismatch() -> None:
    args = SimpleNamespace(
        parent_model_revision="a" * 40,
        dataset_revision="b" * 40,
        base_model="base",
        base_revision="c" * 40,
        training_source_commit="d" * 40,
        output_root=Path("/data/run"),
        training_job="job-123",
    )
    metadata = {
        "created_at_unix": 150,
        "step": 800,
        "fingerprint": {
            "source_model": {"revision": "a" * 40},
            "dataset_identity": {
                "revision": "f" * 40,
                "manifest_sha256": "e" * 64,
            },
            "base_model": "base",
            "base_revision": "c" * 40,
        },
    }
    job = SimpleNamespace(
        id="job-123",
        started_at=datetime.fromtimestamp(100, tz=UTC),
        finished_at=datetime.fromtimestamp(200, tz=UTC),
        command=[
            "--source-commit",
            "d" * 40,
            "--dataset-revision",
            "b" * 40,
            "--source-model-revision",
            "a" * 40,
            "--output-dir",
            "/data/run",
            "--max-steps",
            "800",
        ],
    )

    with pytest.raises(RuntimeError, match="provenance mismatch"):
        WORKER.validate_training_provenance(
            args,
            metadata,
            job,
            {"metadata": 150, "adapter": 151, "trainer_state": 152},
        )


def test_recovery_rejects_artifacts_outside_training_job_window() -> None:
    args = SimpleNamespace(
        parent_model_revision="a" * 40,
        dataset_revision="b" * 40,
        base_model="base",
        base_revision="c" * 40,
        training_source_commit="d" * 40,
        output_root=Path("/data/run"),
        training_job="job-123",
    )
    metadata = {
        "created_at_unix": 150,
        "step": 800,
        "fingerprint": {
            "source_model": {"revision": "a" * 40},
            "dataset_identity": {
                "revision": "b" * 40,
                "manifest_sha256": "e" * 64,
            },
            "base_model": "base",
            "base_revision": "c" * 40,
        },
    }
    job = SimpleNamespace(
        id="job-123",
        started_at=datetime.fromtimestamp(100, tz=UTC),
        finished_at=datetime.fromtimestamp(200, tz=UTC),
        command=[
            "--source-commit",
            "d" * 40,
            "--dataset-revision",
            "b" * 40,
            "--source-model-revision",
            "a" * 40,
            "--output-dir",
            "/data/run",
            "--max-steps",
            "800",
        ],
    )

    with pytest.raises(RuntimeError, match="outside training job window"):
        WORKER.validate_training_provenance(
            args,
            metadata,
            job,
            {"metadata": 150, "adapter": 250, "trainer_state": 152},
        )
