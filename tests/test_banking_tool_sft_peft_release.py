from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


def load_finalizer() -> ModuleType:
    path = Path("scripts/retail_bank/hf_job_finalize_tool_sft_peft.py")
    spec = importlib.util.spec_from_file_location("hf_job_finalize_tool_sft_peft", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FINALIZER = load_finalizer()


def write_training_artifacts(root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    adapter = root / "adapter"
    adapter.mkdir(parents=True)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    adapter_config = {
        "base_model_name_or_path": "spkc83/retail-bank-servicing-agent-9b",
        "peft_type": "LORA",
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "target_modules": target_modules,
    }
    (adapter / "adapter_config.json").write_text(json.dumps(adapter_config), encoding="utf-8")
    for name in FINALIZER.ADAPTER_FILES:
        path = adapter / name
        if not path.exists():
            path.write_bytes(f"content:{name}".encode())

    train_metrics = {"train_loss": 0.13, "train_runtime": 905.58}
    eval_metrics = {"eval_loss": 0.32, "eval_mean_token_accuracy": 0.9624}
    metadata = {
        "step": 750,
        "fingerprint": {
            "base_model": "spkc83/retail-bank-servicing-agent-9b",
            "base_revision": "a" * 40,
            "family": "granite",
            "precision": "bf16-lora",
            "template_hash": "b" * 64,
            "dataset_identity": {
                "repository": "spkc83/retail-bank-servicing-alignment-sft",
                "revision": "c" * 40,
                "manifest_sha256": "d" * 64,
            },
            "lora": {
                "rank": 32,
                "alpha": 64,
                "dropout": 0.05,
                "target_modules": target_modules,
            },
        },
        "extra": {"train_metrics": train_metrics, "eval_metrics": eval_metrics},
    }
    metadata_path = root / "checkpoints" / "step-000750" / "metadata.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    training_result = {
        "steps": 750,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }
    (root / "training_result.json").write_text(json.dumps(training_result), encoding="utf-8")
    return adapter, metadata, training_result


def test_parse_args_defaults_to_dedicated_peft_destination(tmp_path: Path) -> None:
    args = FINALIZER.parse_args(
        [
            "--output-root",
            str(tmp_path),
            "--source-commit",
            "e" * 40,
            "--training-job",
            "job-123",
        ]
    )

    assert args.adapter_subdir == "adapter"
    assert args.selected_step == 750
    assert args.destination_repo == "spkc83/retail-bank-servicing-agent-9b-peft"


def test_validation_rejects_adapter_from_a_different_base(tmp_path: Path) -> None:
    adapter, metadata, training_result = write_training_artifacts(tmp_path)
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = "unexpected/base"
    (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(RuntimeError, match="base model does not match"):
        FINALIZER.validate_training_artifacts(
            adapter_dir=adapter,
            metadata=metadata,
            training_result=training_result,
            selected_step=750,
        )


def test_validation_rejects_unpinned_dataset_revision(tmp_path: Path) -> None:
    adapter, metadata, training_result = write_training_artifacts(tmp_path)
    metadata["fingerprint"]["dataset_identity"]["revision"] = "main"

    with pytest.raises(ValueError, match="exact 40-character"):
        FINALIZER.validate_training_artifacts(
            adapter_dir=adapter,
            metadata=metadata,
            training_result=training_result,
            selected_step=750,
        )


def test_validation_recovers_release_when_legacy_training_result_is_absent(
    tmp_path: Path,
) -> None:
    adapter, metadata, _ = write_training_artifacts(tmp_path)

    validated = FINALIZER.validate_training_artifacts(
        adapter_dir=adapter,
        metadata=metadata,
        training_result=None,
        selected_step=750,
    )

    assert validated["train_metrics"]["train_loss"] == 0.13
    assert validated["eval_metrics"]["eval_mean_token_accuracy"] == 0.9624


def test_main_publishes_atomic_root_adapter_then_exact_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter, _, _ = write_training_artifacts(tmp_path)

    class FakeApi:
        def __init__(self) -> None:
            self.create_repo_calls: list[dict[str, Any]] = []
            self.create_commit_calls: list[dict[str, Any]] = []

        def create_repo(self, repo_id: str, **kwargs: Any) -> None:
            self.create_repo_calls.append({"repo_id": repo_id, **kwargs})

        def create_commit(self, **kwargs: Any) -> SimpleNamespace:
            self.create_commit_calls.append(kwargs)
            revision = "f" * 40 if len(self.create_commit_calls) == 1 else "1" * 40
            return SimpleNamespace(oid=revision)

    api = FakeApi()
    monkeypatch.setattr(FINALIZER, "HfApi", lambda token: api)
    monkeypatch.setenv("HF_TOKEN", "test-token")

    result = FINALIZER.main(
        [
            "--output-root",
            str(tmp_path),
            "--destination-repo",
            "example/granite-peft",
            "--source-commit",
            "e" * 40,
            "--training-job",
            "job-123",
        ]
    )

    assert result == 0
    assert api.create_repo_calls == [
        {
            "repo_id": "example/granite-peft",
            "repo_type": "model",
            "private": False,
            "exist_ok": False,
        }
    ]
    assert len(api.create_commit_calls) == 2
    bundle_call = api.create_commit_calls[0]
    bundle_operations = {
        operation.path_in_repo: operation.path_or_fileobj for operation in bundle_call["operations"]
    }
    assert set(bundle_operations) == {
        *FINALIZER.ADAPTER_FILES,
        "training_metadata.json",
        "training_result.json",
        "README.md",
    }
    assert all(Path(bundle_operations[name]) == adapter / name for name in FINALIZER.ADAPTER_FILES)
    pending_release = json.loads(bundle_operations["training_result.json"])
    assert pending_release["final_immutable_hub_revision"] is None

    provenance_call = api.create_commit_calls[1]
    assert provenance_call["parent_commit"] == "f" * 40
    provenance_operations = {
        operation.path_in_repo: operation.path_or_fileobj
        for operation in provenance_call["operations"]
    }
    release = json.loads(provenance_operations["training_result.json"])
    assert release["contract"] == "banking-v5-peft-adapter-release/v1"
    assert release["final_immutable_hub_revision"] == "f" * 40
    assert release["base_model"] == {
        "repository": "spkc83/retail-bank-servicing-agent-9b",
        "revision": "a" * 40,
        "weight_dtype": "bfloat16",
    }
    assert release["dataset"]["revision"] == "c" * 40
    assert release["dataset"]["manifest_sha256"] == "d" * 64
    assert release["source_commit"] == "e" * 40
    assert release["training_job"] == "job-123"
    assert release["steps"] == 750
    assert release["peft_composition"]["base_weight_dtype"] == "bfloat16"
    assert release["adapter_model_sha256"] == FINALIZER.sha256(
        adapter / "adapter_model.safetensors"
    )
    card = provenance_operations["README.md"].decode()
    assert "Final immutable adapter-bundle revision: `" + "f" * 40 + "`" in card

    report = json.loads(capsys.readouterr().out)
    assert report["weights_commit"] == "f" * 40
    assert report["final_revision"] == "1" * 40
