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

    monkeypatch.setenv(
        "RETAIL_BANK_ALLOW_REMOTE_TOOL_SFT",
        "banking-v5-grounded-dialogue-sft",
    )
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

    assert str(config.manifest) == "data/banking-servicing-alignment-v5/manifest.json"
    assert config.hub_dest == "spkc83/retail-bank-servicing-agent-9b"


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


def test_v7_generation_contract_selects_exact_one_or_no_tool_schema_with_legacy_fallback() -> None:
    adapter = worker.ToolWireAdapter(
        worker.SimpleToolTokenizer(),
        family="granite",
        public_tool_manifest=worker.PUBLIC_BANKING_TOOL_MANIFEST,
    )
    base = worker.tiny_smoke_records()[0]
    execute = {
        **base,
        "expected": {
            "generation_contract": {
                "version": "banking-v7-route-to-generation/v1",
                "mode": "execute_tool",
                "entity_state": "resolved",
                "tool_names": ["freeze_card"],
                "argument_constraints": {"last4": {"const": "4821"}},
            }
        },
    }
    converse = {
        **base,
        "expected": {
            "generation_contract": {
                "version": "banking-v7-route-to-generation/v1",
                "mode": "converse",
                "entity_state": "not_required",
                "tool_names": [],
                "argument_constraints": {},
            }
        },
    }

    tools = worker.training_tools_for_record(execute, adapter)
    assert tools is not None
    assert [tool["name"] for tool in tools] == ["freeze_card"]
    assert tools[0]["parameters"] == {
        "type": "object",
        "properties": {"last4": {"type": ["string", "null"], "const": "4821"}},
        "required": ["last4"],
        "additionalProperties": False,
    }
    assert worker.training_tools_for_record(converse, adapter) == []
    assert worker.training_tools_for_record(base, adapter) is None


def test_v7_tokenization_renders_generation_contract_guidance() -> None:
    tokenizer = worker.SimpleToolTokenizer()
    adapter = worker.ToolWireAdapter(
        tokenizer,
        family="granite",
        public_tool_manifest=worker.PUBLIC_BANKING_TOOL_MANIFEST,
    )
    record = worker.tiny_smoke_records()[0]
    record["messages"].insert(0, {"role": "system", "content": "Banking system", "loss": False})
    record["expected"] = {
        "generation_contract": {
            "mode": "execute_tool",
            "entity_state": "resolved",
            "tool_names": ["freeze_card"],
            "argument_constraints": {"last4": {"const": "4821"}},
        }
    }

    rendered = worker.tokenize_records([record], adapter, max_seq_len=1024)

    decoded = tokenizer.decode(rendered[0]["input_ids"])
    assert "TURN GUIDANCE: Use only freeze_card for this turn" in decoded


def test_v6_generation_contract_rejects_unknown_training_tool() -> None:
    adapter = worker.ToolWireAdapter(
        worker.SimpleToolTokenizer(),
        family="granite",
        public_tool_manifest=worker.PUBLIC_BANKING_TOOL_MANIFEST,
    )
    record = {
        "expected": {
            "generation_contract": {
                "tool_names": ["close_account"],
            }
        }
    }

    with pytest.raises(ValueError, match="unknown tools"):
        worker.training_tools_for_record(record, adapter)


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
    assert payload["manifest"] == "data/banking-servicing-alignment-v5/manifest.json"
    assert payload["hub_dest"] == "spkc83/retail-bank-servicing-agent-9b"
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

        assert config["run"]["format"] == "banking_v5_tool_sft_config"
        assert config["run"]["push_to_hub_default"] is False
        assert config["dataset"]["schema_version"] == "banking-tool-sft/v1"
        assert config["dataset"]["name"] == ("data/banking-servicing-alignment-v5/manifest.json")
        assert config["dataset"]["tool_manifest"] == "public_banking_poc_nine_tools"
        assert config["training"]["stack"] == "trl_sfttrainer_peft_bf16_lora"
        assert config["training"]["precision"] == "bf16"
        assert config["training"]["push_to_hub"] is False
        assert config["training"]["max_train_seconds"] == 14_400


def _continuation() -> ModuleType:
    return worker.continuation_module()


def _mix_records() -> list[dict[str, Any]]:
    return [
        {
            "record_id": "positive_1",
            "expected": {"path": "tool_success"},
            "metadata": {
                "scenario_family": "deictic_replace_action",
                "coreference_target": "replace_card",
                "coreference_pair_id": "pair_1",
            },
            "messages": [
                {"role": "user", "content": "do that one too", "loss": False},
                {
                    "role": "assistant",
                    "content": None,
                    "loss": True,
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "index": 0,
                            "type": "function",
                            "function": {"name": "replace_card", "arguments": {"last4": "4821"}},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_0",
                    "name": "replace_card",
                    "content": {"ok": True},
                    "loss": False,
                },
                {"role": "assistant", "content": "Replacement ordered.", "loss": True},
            ],
        },
        {
            "record_id": "ambiguity_1",
            "expected": {"path": "clarification"},
            "metadata": {
                "scenario_family": "deictic_replace_ambiguity",
                "coreference_target": "clarification",
                "coreference_pair_id": "pair_1",
            },
            "messages": [
                {"role": "user", "content": "replace it", "loss": False},
                {"role": "assistant", "content": "Which card?", "loss": True},
            ],
        },
        {
            "record_id": "regression_1",
            "expected": {"path": "tool_success"},
            "metadata": {"scenario_family": "tool_success"},
            "messages": [
                {"role": "user", "content": "freeze my card", "loss": False},
                {"role": "assistant", "content": "Done.", "loss": True},
            ],
        },
    ]


def test_hub_destination_must_differ_from_the_training_base(tmp_path: Path) -> None:
    config = _config(tmp_path)
    same = worker.WorkerConfig(**{**vars(config), "hub_dest": config.base_model})

    worker.validate_hub_destination(config)
    with pytest.raises(RuntimeError, match="must differ from the training base model"):
        worker.validate_hub_destination(same)


def test_dataset_identity_validation_pins_repository_and_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"contract":"test"}', encoding="utf-8")
    monkeypatch.setenv("RETAIL_BANK_TOOL_SFT_DATASET_REVISION", "0" * 40)
    monkeypatch.setenv(
        "RETAIL_BANK_TOOL_SFT_DATASET_REPO",
        "spkc83/retail-bank-servicing-alignment-sft",
    )

    identity = worker.validate_dataset_identity(manifest)
    assert identity["repository"] == "spkc83/retail-bank-servicing-alignment-sft"

    monkeypatch.setenv("RETAIL_BANK_TOOL_SFT_DATASET_REPO", "spkc83/some-other-dataset")
    with pytest.raises(RuntimeError, match="dataset repository must be exactly"):
        worker.validate_dataset_identity(manifest)

    monkeypatch.setenv(
        "RETAIL_BANK_TOOL_SFT_DATASET_REPO",
        "spkc83/retail-bank-servicing-alignment-sft",
    )
    monkeypatch.setenv("RETAIL_BANK_TOOL_SFT_DATASET_REVISION", "0f99604")
    with pytest.raises(RuntimeError, match="40-character"):
        worker.validate_dataset_identity(manifest)


def test_unweighted_multipliers_keep_the_manifest_order_and_record_no_mix(tmp_path: Path) -> None:
    config = _config(tmp_path)
    records = _mix_records()

    mixed, stats = worker.build_training_mix(config, records)

    assert stats is None
    assert [record["record_id"] for record in mixed] == [
        "positive_1",
        "ambiguity_1",
        "regression_1",
    ]
    assert mixed[0]["messages"][-1]["loss"] is True


def test_weighted_multipliers_mask_positives_and_record_mix_stats(tmp_path: Path) -> None:
    config = worker.WorkerConfig(
        **{
            **vars(_config(tmp_path)),
            "positive_multiplier": 3,
            "ambiguity_multiplier": 6,
            "policy_faq_multiplier": 1,
            "tool_outcome_multiplier": 1,
        }
    )
    records = _mix_records()

    mixed, stats = worker.build_training_mix(config, records)

    assert stats is not None
    assert stats["input_records"] == 3
    assert stats["coreference_positive_records"] == 1
    assert stats["coreference_ambiguity_records"] == 1
    assert stats["total_weighted_records"] == 10
    assert stats["positive_multiplier"] == 3
    assert stats["ambiguity_multiplier"] == 6
    assert len(mixed) == 10
    counts = {record["record_id"]: 0 for record in records}
    for record in mixed:
        counts[record["record_id"]] += 1
    assert counts == {"positive_1": 3, "ambiguity_1": 6, "regression_1": 1}
    positives = [record for record in mixed if record["record_id"] == "positive_1"]
    assert all(record["messages"][-1]["loss"] is False for record in positives)
    assert records[0]["messages"][-1]["loss"] is True


def test_destination_repo_states_absent_empty_and_nonempty() -> None:
    import httpx
    from huggingface_hub.errors import RepositoryNotFoundError

    class FakeApi:
        def __init__(self, state: str) -> None:
            self.state = state

        def repo_info(self, **_kwargs: Any) -> None:
            if self.state == "absent":
                request = httpx.Request("GET", "https://huggingface.co/api/models/example/repo")
                raise RepositoryNotFoundError(
                    "absent", response=httpx.Response(404, request=request)
                )

        def list_repo_files(self, **_kwargs: Any) -> list[str]:
            return [] if self.state == "empty" else ["model.safetensors"]

    assert worker.require_publishable_destination(FakeApi("absent"), "example/repo") == "absent"
    assert worker.require_publishable_destination(FakeApi("empty"), "example/repo") == "empty"
    with pytest.raises(RuntimeError, match="not empty"):
        worker.require_publishable_destination(FakeApi("nonempty"), "example/repo")


def _canned_report(accuracy: float, step: int) -> dict[str, Any]:
    return {
        "contract": "banking-v5-coreference-behavior-report/v1",
        "cumulative_step": step,
        "metrics": {
            "positive_tool_argument_accuracy": accuracy,
            "ambiguity_accuracy": accuracy,
            "pair_flip_accuracy": accuracy,
            "positive_records": 1,
            "ambiguity_records": 1,
            "pairs": 1,
            "parse_failures": 0,
        },
        "records": [],
    }


def _install_fake_gate(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dev_accuracy: float,
    shadow_accuracy: float,
    shadow_records: list[dict[str, Any]],
) -> list[str]:
    continuation = _continuation()
    calls: list[str] = []

    def fake_report(
        model: Any,
        tokenizer: Any,
        adapter: Any,
        records: Any,
        *,
        cumulative_step: int,
    ) -> dict[str, Any]:
        del model, tokenizer, adapter
        first = str(list(records)[0]["record_id"])
        calls.append(first)
        accuracy = shadow_accuracy if first.startswith("shadow") else dev_accuracy
        return _canned_report(accuracy, cumulative_step)

    monkeypatch.setattr(continuation, "generate_coreference_behavior_report", fake_report)
    monkeypatch.setattr(
        continuation,
        "load_shadow_gate_records",
        lambda _manifest: shadow_records,
    )
    return calls


def test_behavioral_gates_persist_reports_and_return_passing_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shadow_records = [{"record_id": "shadow_1", "metadata": {"coreference_pair_id": "pair_s"}}]
    calls = _install_fake_gate(
        monkeypatch,
        dev_accuracy=1.0,
        shadow_accuracy=1.0,
        shadow_records=shadow_records,
    )

    gates = worker.run_coreference_behavioral_gates(
        model=object(),
        tokenizer=object(),
        adapter=object(),
        validation_records=_mix_records(),
        manifest_path=tmp_path / "manifest.json",
        output_dir=tmp_path / "out",
        step=1200,
    )

    assert calls == ["positive_1", "shadow_1"]
    assert gates["dev"]["pair_flip_accuracy"] == 1.0
    assert gates["shadow"]["pair_flip_accuracy"] == 1.0
    dev_path = tmp_path / "out" / "behavioral-evaluations" / "dev-step-1200.json"
    shadow_path = tmp_path / "out" / "behavioral-evaluations" / "shadow-step-1200.json"
    assert json.loads(dev_path.read_text(encoding="utf-8"))["cumulative_step"] == 1200
    assert json.loads(shadow_path.read_text(encoding="utf-8"))["cumulative_step"] == 1200


@pytest.mark.parametrize(
    ("dev_accuracy", "shadow_accuracy", "expected_reports"),
    [
        (0.5, 1.0, ["dev-step-1200.json"]),
        (1.0, 0.5, ["dev-step-1200.json", "shadow-step-1200.json"]),
    ],
)
def test_behavioral_gate_failure_raises_and_keeps_the_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dev_accuracy: float,
    shadow_accuracy: float,
    expected_reports: list[str],
) -> None:
    shadow_records = [{"record_id": "shadow_1", "metadata": {"coreference_pair_id": "pair_s"}}]
    _install_fake_gate(
        monkeypatch,
        dev_accuracy=dev_accuracy,
        shadow_accuracy=shadow_accuracy,
        shadow_records=shadow_records,
    )

    with pytest.raises(RuntimeError, match="each accuracy >= 0.95"):
        worker.run_coreference_behavioral_gates(
            model=object(),
            tokenizer=object(),
            adapter=object(),
            validation_records=_mix_records(),
            manifest_path=tmp_path / "manifest.json",
            output_dir=tmp_path / "out",
            step=1200,
        )

    evaluations = tmp_path / "out" / "behavioral-evaluations"
    assert sorted(path.name for path in evaluations.iterdir()) == sorted(expected_reports)


def test_behavioral_gates_use_preloaded_shadow_records_without_rereading_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_gate(
        monkeypatch,
        dev_accuracy=1.0,
        shadow_accuracy=1.0,
        shadow_records=[{"record_id": "shadow_manifest", "metadata": {}}],
    )
    preloaded = [{"record_id": "shadow_preloaded", "metadata": {"coreference_pair_id": "p"}}]

    worker.run_coreference_behavioral_gates(
        model=object(),
        tokenizer=object(),
        adapter=object(),
        validation_records=_mix_records(),
        manifest_path=tmp_path / "manifest.json",
        output_dir=tmp_path / "out",
        step=7,
        shadow_records=preloaded,
    )

    assert calls == ["positive_1", "shadow_preloaded"]


def test_model_card_describes_adapter_only_release(tmp_path: Path) -> None:
    base = _config(tmp_path)
    adapter_only = WorkerConfig(**{**base.__dict__, "merge_adapter": False})
    adapter_only.output_dir.mkdir(parents=True, exist_ok=True)
    card = worker.write_model_card(
        adapter_only, train_records=1, validation_records=1, result={}
    ).read_text(encoding="utf-8")
    assert "library_name: peft" in card
    assert "PeftModel.from_pretrained" in card
    assert "merged FP16" not in card
    assert "8.8 billion" in card

    merged = WorkerConfig(**{**base.__dict__, "merge_adapter": True})
    card = worker.write_model_card(
        merged, train_records=1, validation_records=1, result={}
    ).read_text(encoding="utf-8")
    assert "library_name: peft" not in card
    assert "The released root checkpoint is merged FP16 weights" in card


def test_behavioral_gate_requires_coreference_validation_records(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no coreference gate records"):
        worker.run_coreference_behavioral_gates(
            model=object(),
            tokenizer=object(),
            adapter=object(),
            validation_records=[{"record_id": "plain_1", "metadata": {}}],
            manifest_path=tmp_path / "manifest.json",
            output_dir=tmp_path / "out",
            step=1,
        )


def test_remote_training_gates_behavior_before_any_publication() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")
    gate = source.index("behavioral_gates = run_coreference_behavioral_gates(")
    post_train_merge = "if config.merge_adapter:\n        parity = merge_adapter_with_reload"
    assert gate < source.index(post_train_merge)
    assert gate < source.index("api.upload_folder(")
    assert gate < source.index("api.create_repo(")
    assert source.index("eval_metrics = trainer.evaluate()") < gate
    assert source.index('trainer.save_model(str(config.output_dir / "adapter"))') < gate
    assert gate < source.index("    del model\n")
    assert source.index("require_publishable_destination(api, config.hub_dest)") < source.index(
        "api.create_repo("
    )
    assert source.index("validate_hub_destination(config)") < source.index(
        "AutoTokenizer.from_pretrained"
    )
    assert source.index("validate_dataset_identity(config.manifest)") < source.index(
        "AutoTokenizer.from_pretrained"
    )
    assert source.index("preflight_destination_repo(config)") < source.index(
        "AutoTokenizer.from_pretrained"
    )
    # The shadow gate contract is checked before the GPU spend, and a failing gate
    # still leaves training_result.json on the bucket for diagnosis.
    assert source.index("load_shadow_gate_records(config.manifest)") < source.index(
        "AutoTokenizer.from_pretrained"
    )
    assert (
        gate < source.index('result["behavioral_gate_failure"]') < source.index("    del model\n")
    )
