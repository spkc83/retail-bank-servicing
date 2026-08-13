from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from model_service import MODEL_TOOLS  # type: ignore[import-not-found]

from hello_slm.banking_tool_sft_data import public_tool_manifest

WORKER_PATH = Path("scripts/retail_bank/cloud_train_tool_sft.py")


def _load_worker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cloud_train_tool_sft", WORKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load tool SFT worker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


worker = _load_worker()
WorkerConfig = worker.WorkerConfig


def _config(tmp_path: Path, *, execute_remote: bool = False, allow_remote: bool = False) -> Any:
    return WorkerConfig(
        manifest=tmp_path / "manifest.json",
        output_dir=tmp_path / "out",
        base_model="ibm-granite/granite-4.1-8b",
        base_revision="unit-test-revision",
        family="granite",
        max_steps=1,
        max_train_seconds=60,
        batch_size=1,
        max_seq_len=512,
        learning_rate=1e-4,
        checkpoint_every=1,
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.05,
        precision="bf16-lora",
        gradient_accumulation_steps=1,
        warmup_ratio=0.03,
        resume_from=None,
        dry_run=not execute_remote,
        run_tiny_smoke=False,
        allow_remote_execution=allow_remote,
        push_to_hub=False,
        hub_dest="spkc83/retail-bank-tool-sft-9b",
        merge_adapter=True,
        trackio_project=None,
        trackio_run_name=None,
    )


def test_remote_execution_requires_flag_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, execute_remote=True, allow_remote=True)
    monkeypatch.delenv("RETAIL_BANK_ALLOW_REMOTE_TOOL_SFT", raising=False)

    assert not worker.remote_execution_allowed(config)
    with pytest.raises(PermissionError, match="RETAIL_BANK_ALLOW_REMOTE_TOOL_SFT"):
        worker.assert_remote_execution_allowed(config)

    monkeypatch.setenv("RETAIL_BANK_ALLOW_REMOTE_TOOL_SFT", "banking-v3-tool-sft")
    assert worker.remote_execution_allowed(config)


def test_dry_run_plan_declares_bf16_lora_lane_and_no_unguarded_remote_actions(
    tmp_path: Path,
) -> None:
    plan = worker.build_dry_run_plan(_config(tmp_path))

    assert plan["worker"] == "cloud_train_tool_sft"
    assert plan["manifest"].endswith("manifest.json")
    assert plan["tool_count"] == 9
    assert "BF16 base weights" in plan["training"]["stack"]
    assert plan["training"]["precision"] == "bf16-lora"
    assert plan["remote_guard"]["currently_allowed"] is False
    assert "write to Hugging Face Hub" in plan["will_not_do_without_guard"]


def test_cli_defaults_align_generator_path_and_final_repo() -> None:
    args = worker.parse_args([])
    config = worker.worker_config_from_args(args)

    assert str(config.manifest) == "data/banking-v3-tool-sft/manifest.json"
    assert config.hub_dest == "spkc83/retail-bank-agent-9b"


def test_qlora_is_an_explicit_optional_precision_lane(tmp_path: Path) -> None:
    config = WorkerConfig(**{**_config(tmp_path).__dict__, "precision": "qlora"})
    plan = worker.build_dry_run_plan(config)

    assert "bitsandbytes QLoRA" in plan["training"]["stack"]
    assert plan["training"]["precision"] == "qlora"


def test_manifest_loader_and_adapter_tokenization(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    record = worker.tiny_smoke_records()[0]
    train_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"splits": {"train": {"path": "train.jsonl"}}}),
        encoding="utf-8",
    )
    tokenizer = worker.SimpleToolTokenizer()
    adapter = worker.ToolWireAdapter(
        tokenizer,
        family="granite",
        public_tool_manifest=worker.PUBLIC_BANKING_TOOL_MANIFEST,
    )

    records = worker.load_manifest_records(manifest_path, "train")
    examples = worker.tokenize_records(records, adapter, max_seq_len=512)

    assert records == [record]
    assert examples[0]["input_ids"].shape == examples[0]["labels"].shape
    assert (examples[0]["labels"] == -100).any()
    assert (examples[0]["labels"] != -100).any()


def test_manifest_loader_resolves_relative_paths_from_manifest_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "nested"
    data_dir.mkdir()
    train_path = data_dir / "train.jsonl"
    record = worker.tiny_smoke_records()[0]
    train_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"tool_sft": [{"name": "train", "path": "nested/train.jsonl"}]}),
        encoding="utf-8",
    )

    assert worker.load_manifest_records(manifest_path, "train") == [record]


def test_manifest_loader_rejects_evaluation_only_data(tmp_path: Path) -> None:
    data_path = tmp_path / "test.jsonl"
    data_path.write_text(json.dumps(worker.tiny_smoke_records()[0]) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "contract": "banking-counterfactual-eval-manifest/v1",
                "training_allowed": False,
                "allowed_use": ["counterfactual-evaluation"],
                "splits": {"test": {"path": "test.jsonl"}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evaluation-only manifest"):
        worker.load_manifest_records(manifest_path, "test")


def test_pretokenized_collator_pads_only_to_the_longest_batch_item() -> None:
    batch = [
        {
            "input_ids": [11, 12, 13],
            "attention_mask": [1, 1, 1],
            "labels": [-100, 12, 13],
        },
        {
            "input_ids": [21],
            "attention_mask": [1],
            "labels": [21],
        },
    ]

    collated = worker.collate_pretokenized(batch, pad_token_id=7)

    assert collated["input_ids"].tolist() == [[11, 12, 13], [21, 7, 7]]
    assert collated["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert collated["labels"].tolist() == [[-100, 12, 13], [21, -100, -100]]


def test_tiny_smoke_cli_writes_checkpoint_and_parity_metadata(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/retail_bank/cloud_train_tool_sft.py",
            "--run-tiny-smoke",
            "--output-dir",
            str(tmp_path / "worker"),
            "--max-seq-len",
            "512",
            "--checkpoint-every",
            "1",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    metadata_path = tmp_path / "worker" / "checkpoints" / "step-000001" / "metadata.json"

    assert payload["mode"] == "tiny_smoke"
    assert payload["steps"] == 1
    assert payload["merge_reload_parity"] is True
    assert payload["pushed_to_hub"] is False
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["worker"] == "cloud_train_tool_sft"
    assert metadata["fingerprint"]["family"] == "granite"
    assert metadata["resume_validation"]["template_hash"] == payload["template_hash"]


def test_worker_cli_default_is_dry_run() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/retail_bank/cloud_train_tool_sft.py"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["mode"] == "dry_run"
    assert payload["remote_guard"]["currently_allowed"] is False
    assert payload["manifest"] == "data/banking-v3-tool-sft/manifest.json"
    assert payload["hub_dest"] == "spkc83/retail-bank-agent-9b"
    assert "download 9B base weights" in payload["will_not_do_without_guard"]


def test_trackio_reporter_is_configured_on_sft_args(tmp_path: Path) -> None:
    if importlib.util.find_spec("trl") is None:
        pytest.skip("TRL is not installed in this environment")
    config = WorkerConfig(
        **{
            **_config(tmp_path).__dict__,
            "trackio_project": "banking-v3",
            "trackio_run_name": "granite-smoke",
        }
    )

    args = worker.build_training_configs(config)["training_args"]

    assert args.report_to == ["trackio"] or args.report_to == "trackio"
    assert args.run_name == "granite-smoke"


def test_remote_trainer_construction_does_not_pass_quantization_config() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")
    trainer_call = source.split("trainer = SFTTrainer(", 1)[1].split(
        "train_output = trainer.train", 1
    )[0]

    assert "data_collator=partial(" in trainer_call
    assert 'peft_config=configs["lora"]' in trainer_call
    assert "quantization_config" not in trainer_call


def test_remote_model_load_has_no_blanket_quantized_fallback() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")
    remote_body = source.split("def run_remote_training", 1)[1].split("def tiny_smoke_records", 1)[
        0
    ]

    assert "except Exception" not in remote_body
    assert 'if configs["quantization"] is not None' in remote_body


def test_hub_upload_ignores_hidden_checkpoint_temp_files() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")

    assert 'ignore_patterns=[".*", "**/.*"]' in source


def test_poc_serves_the_exact_sft_tool_manifest() -> None:
    serving_manifest = tuple(MODEL_TOOLS)

    assert tuple(public_tool_manifest()) == worker.PUBLIC_BANKING_TOOL_MANIFEST
    assert serving_manifest == worker.PUBLIC_BANKING_TOOL_MANIFEST


def test_trainer_checkpoint_metadata_is_resume_compatible(tmp_path: Path) -> None:
    fingerprint = {"base_revision": "pinned", "dataset": {"sha256": "abc"}}
    checkpoint_metadata = worker.save_trainer_checkpoint_metadata(
        tmp_path,
        step=500,
        fingerprint=fingerprint,
    )

    assert checkpoint_metadata == tmp_path / "checkpoint-500" / "metadata.json"
    worker.validate_resume_fingerprint(checkpoint_metadata.parent, fingerprint)
    payload = json.loads(checkpoint_metadata.read_text(encoding="utf-8"))
    assert payload["contract"] == "banking-tool-sft-resume/v1"
    assert payload["optimizer_scheduler_rng_state"] is True


def test_dataset_identity_is_stable_across_job_local_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "job-a" / "manifest.json"
    second = tmp_path / "job-b" / "manifest.json"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text('{"contract":"test"}', encoding="utf-8")
    second.write_text('{"contract":"test"}', encoding="utf-8")
    monkeypatch.setenv(
        "RETAIL_BANK_TOOL_SFT_DATASET_REPO",
        "spkc83/retail-bank-agent-sft",
    )
    monkeypatch.setenv(
        "RETAIL_BANK_TOOL_SFT_DATASET_REVISION",
        "fcf065db",
    )

    assert worker.dataset_identity(first) == worker.dataset_identity(second)
    assert "manifest_path" not in worker.dataset_identity(first)


def test_configs_pin_tool_sft_contract_and_disable_push_by_default() -> None:
    for config_path in Path("configs").glob("banking-tool-sft-*.toml"):
        with config_path.open("rb") as handle:
            config = tomllib.load(handle)

        assert config["run"]["format"] == "banking_v3_tool_sft_config"
        assert config["run"]["push_to_hub_default"] is False
        assert config["dataset"]["schema_version"] == "banking-tool-sft/v1"
        assert config["dataset"]["name"] == "data/banking-v3-tool-sft/manifest.json"
        assert config["dataset"]["tool_manifest"] == "public_banking_poc_nine_tools"
        assert config["training"]["stack"] == "trl_sfttrainer_peft_bf16_lora"
        assert config["training"]["precision"] == "bf16"
        assert config["training"]["push_to_hub"] is False
        assert config["training"]["max_train_seconds"] == 14_400
