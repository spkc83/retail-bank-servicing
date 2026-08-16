from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import httpx
import pytest

WORKER_PATH = Path("scripts/retail_bank/cloud_continue_tool_sft.py")
JOB_PATH = Path("scripts/retail_bank/hf_job_continue_tool_sft.py")
LAUNCHER_PATH = Path("scripts/retail_bank/run_remote_continuation_job.sh")


def _repository_not_found() -> Exception:
    from huggingface_hub.errors import RepositoryNotFoundError

    request = httpx.Request("GET", "https://huggingface.co/api/models/example/repo")
    return RepositoryNotFoundError("absent", response=httpx.Response(404, request=request))


def _load_worker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cloud_continue_tool_sft", WORKER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


WORKER = _load_worker()


def _load_job() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hf_job_continue_tool_sft", JOB_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


JOB = _load_job()


def _record(
    record_id: str,
    *,
    path: str,
    assistant_tool_calls: int = 0,
    final: str = "Done.",
    requires_tool: bool = False,
) -> dict[str, object]:
    tool_calls = [
        {
            "id": f"call_{record_id}_{index}",
            "index": index,
            "type": "function",
            "function": {"name": "list_cards", "arguments": {}},
        }
        for index in range(assistant_tool_calls)
    ]
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "demo", "loss": False},
        {"role": "user", "content": "help", "loss": False},
    ]
    if tool_calls:
        messages.append(
            {"role": "assistant", "content": None, "loss": True, "tool_calls": tool_calls}
        )
        for call in tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "name": "list_cards",
                    "tool_call_id": call["id"],
                    "content": {"ok": True, "result": {}},
                    "loss": False,
                }
            )
    messages.append({"role": "assistant", "content": final, "loss": True})
    return {
        "record_id": record_id,
        "messages": messages,
        "expected": {"path": path, "requires_tool": requires_tool},
        "metadata": {"scenario_family": record_id},
    }


def test_continuation_requires_exact_model_revision() -> None:
    with pytest.raises(ValueError, match="exact 40-character"):
        WORKER.require_exact_revision("main", field="--source-adapter-revision")

    WORKER.require_exact_revision("00c4ba1be926fc26dbc1f5311a4fd037462be1c1", field="ok")

    config = WORKER.config_from_args(WORKER.parse_args([]))
    with pytest.raises(RuntimeError, match="base must be exactly"):
        WORKER.validate_pinned_model_inputs(replace(config, base_revision="0" * 40))
    with pytest.raises(RuntimeError, match="owner/name"):
        WORKER.validate_pinned_model_inputs(replace(config, source_adapter_repo="invalid"))
    with pytest.raises(RuntimeError, match="source adapter must be exactly"):
        WORKER.validate_pinned_model_inputs(
            replace(config, source_adapter_repo="example/other-adapter")
        )
    with pytest.raises(RuntimeError, match="source adapter must be exactly"):
        WORKER.validate_pinned_model_inputs(replace(config, source_adapter_revision="0" * 40))


def test_job_bootstrap_rejects_source_adapter_repo_and_revision_overrides() -> None:
    with pytest.raises(ValueError, match="source adapter must be exactly"):
        JOB.validate_source_adapter("example/other-adapter", JOB.DEFAULT_SOURCE_ADAPTER_REVISION)
    with pytest.raises(ValueError, match="source adapter must be exactly"):
        JOB.validate_source_adapter(JOB.ADAPTER_REPO, "0" * 40)


def test_job_bootstrap_rejects_pre_v6_worker_protocol(tmp_path: Path) -> None:
    worker = tmp_path / "scripts/retail_bank/cloud_continue_tool_sft.py"
    worker.parent.mkdir(parents=True)
    worker.write_text('CANDIDATE4_PROTOCOL = "legacy"\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="V6 continuation worker protocol"):
        JOB.validate_v6_source(tmp_path)

    worker.write_text(
        f'V6_CONTINUATION_PROTOCOL = "{JOB.V6_CONTINUATION_PROTOCOL}"\n',
        encoding="utf-8",
    )
    JOB.validate_v6_source(tmp_path)


def test_continuation_mix_oversamples_paired_coreference_targets() -> None:
    positive = _record(
        "positive",
        path="multi_turn",
        assistant_tool_calls=1,
        final="I found the active card and froze it.",
        requires_tool=True,
    )
    positive["metadata"] = {
        "scenario_family": "deictic_replace_action",
        "coreference_target": "replace_card",
    }
    clarification = _record(
        "clarification",
        path="clarification",
        final="Which card should I replace? Please provide the last four digits shown in the app.",
    )
    clarification["metadata"] = {
        "scenario_family": "deictic_replace_ambiguity",
        "coreference_target": "clarification",
    }
    single_tool = _record(
        "single",
        path="tool_success",
        assistant_tool_calls=1,
        final="Your balance is ready.",
        requires_tool=True,
    )
    faq = _record(
        "faq",
        path="no_tool_banking_faq",
        final="Overdraft fees depend on account disclosures.",
    )

    mixed, stats = WORKER.build_continuation_mix(
        [positive, clarification, single_tool, faq],
        positive_multiplier=10,
        ambiguity_multiplier=5,
        policy_faq_multiplier=4,
        tool_outcome_multiplier=6,
        seed=123,
    )
    counts = {
        "positive": sum(1 for record in mixed if record["record_id"] == "positive"),
        "clarification": sum(1 for record in mixed if record["record_id"] == "clarification"),
        "single": sum(1 for record in mixed if record["record_id"] == "single"),
        "faq": sum(1 for record in mixed if record["record_id"] == "faq"),
    }

    assert counts == {"positive": 10, "clarification": 5, "single": 1, "faq": 4}
    assert stats["coreference_positive_records"] == 1
    assert stats["coreference_ambiguity_records"] == 1
    assert stats["policy_faq_records"] == 1
    assert stats["regression_records"] == 2


def test_continuation_mix_oversamples_policy_and_tool_outcome_families() -> None:
    balances = _record(
        "read_accounts",
        path="tool_success",
        assistant_tool_calls=1,
        final="Everyday Checking has USD 3,245.67 available.",
        requires_tool=True,
    )
    mortgage_age = _record(
        "faq_mortgage_age",
        path="no_tool_banking_faq",
        final="Applicants are typically at least 18.",
    )
    unrelated = _record("write_card", path="tool_success", assistant_tool_calls=1)
    outcome = _record("outcome", path="tool_error", assistant_tool_calls=1)
    outcome["metadata"] = {"scenario_family": "tool_outcome_consistency"}

    mixed, stats = WORKER.build_continuation_mix(
        [balances, mortgage_age, unrelated, outcome],
        positive_multiplier=10,
        ambiguity_multiplier=5,
        policy_faq_multiplier=4,
        tool_outcome_multiplier=6,
        seed=123,
    )
    counts = {
        record_id: sum(1 for record in mixed if record["record_id"] == record_id)
        for record_id in ("read_accounts", "faq_mortgage_age", "write_card", "outcome")
    }

    assert counts == {
        "read_accounts": 1,
        "faq_mortgage_age": 4,
        "write_card": 1,
        "outcome": 6,
    }
    assert stats["policy_faq_records"] == 1
    assert stats["tool_outcome_records"] == 1


def test_continuation_mix_focuses_remediation_and_retains_every_regression_record() -> None:
    referential = _record("referential", path="multi_turn", assistant_tool_calls=1)
    referential["metadata"] = {"scenario_family": "history_entity_action"}
    outcome = _record("outcome", path="tool_error", assistant_tool_calls=1)
    outcome["metadata"] = {"scenario_family": "tool_outcome_consistency"}
    regression = _record("regression", path="ood")

    mixed, stats = WORKER.build_continuation_mix(
        [referential, outcome, regression],
        positive_multiplier=10,
        ambiguity_multiplier=5,
        policy_faq_multiplier=4,
        tool_outcome_multiplier=6,
        seed=123,
    )

    assert sum(row["record_id"] == "referential" for row in mixed) == 1
    assert sum(row["record_id"] == "outcome" for row in mixed) == 6
    assert sum(row["record_id"] == "regression" for row in mixed) == 1
    assert stats["all_input_records_retained"] is True


def test_unpaired_clarification_is_not_focus_oversampled() -> None:
    unsafe = _record(
        "unsafe",
        path="clarification",
        final="Please provide your password and customer ID.",
    )

    mixed, stats = WORKER.build_continuation_mix(
        [unsafe],
        positive_multiplier=10,
        ambiguity_multiplier=5,
        policy_faq_multiplier=4,
        tool_outcome_multiplier=6,
    )

    assert len(mixed) == 1
    assert stats["coreference_ambiguity_records"] == 0


def test_coreference_positive_masks_only_post_tool_final_loss() -> None:
    positive = _record(
        "positive",
        path="multi_turn",
        assistant_tool_calls=1,
        requires_tool=True,
    )
    positive["metadata"] = {
        "scenario_family": "deictic_replace_action",
        "coreference_target": "replace_card",
    }
    ambiguity = _record("ambiguity", path="clarification")
    ambiguity["metadata"] = {
        "scenario_family": "deictic_replace_ambiguity",
        "coreference_target": "clarification",
    }

    masked = WORKER.mask_coreference_positive_final_loss([positive, ambiguity])

    positive_assistants = [
        message for message in masked[0]["messages"] if message["role"] == "assistant"
    ]
    assert positive_assistants[0]["loss"] is True
    assert positive_assistants[0]["tool_calls"]
    assert positive_assistants[-1]["loss"] is False
    assert masked[1]["messages"][-1]["loss"] is True
    assert positive["messages"][-1]["loss"] is True


def test_worker_dry_run_exposes_capped_continuation_plan() -> None:
    config = WORKER.config_from_args(WORKER.parse_args([]))
    plan = WORKER.build_dry_run_plan(config)

    assert plan["worker"] == "cloud_continue_tool_sft"
    assert plan["source_adapter_repo"] == (
        "spkc83/retail-bank-servicing-agent-9b-peft-v5-remediation"
    )
    assert plan["source_adapter_revision"] == "d965816bd6a9252bfb4327c1b0d64f9d34f4a1a2"
    assert plan["checkpoint_probe"]["available"] is True
    assert plan["checkpoint_probe"]["default_path"].endswith("/trainer/checkpoint-600")
    assert plan["checkpoint_probe"]["required_dataset_revision"] == (
        "715064e50e7ed2f815dfd3ce19b61f345a466b9d"
    )
    assert plan["training_parent"] == "pinned d965 adapter"
    assert plan["base_model"] == "spkc83/retail-bank-servicing-agent-9b"
    assert plan["base_revision"] == "1d56824995aa1adecfe20f62ca42fb1c0c443817"
    assert plan["hub_dest"] == ("spkc83/retail-bank-servicing-agent-9b-peft-v6-generation-contract")
    assert plan["training"]["max_steps"] == 964
    assert plan["training"]["max_train_seconds"] == 3_600
    assert plan["training"]["learning_rate"] == 2e-6
    assert plan["training"]["positive_multiplier"] == 2
    assert plan["training"]["ambiguity_multiplier"] == 1
    assert plan["training"]["policy_faq_multiplier"] == 4
    assert plan["training"]["tool_outcome_multiplier"] == 6
    assert plan["training"]["seed"] == 20_260_815
    assert plan["training"]["generation_contract"].startswith("per-record tool exposure")
    assert plan["training"]["selection_gate"] == "two consecutive dev passes"
    assert plan["release"]["format"] == "root-level PEFT adapter"
    assert plan["release"]["merge"] is False
    assert plan["release"]["evaluation_before_publish"] is True
    assert plan["remote_guard"]["currently_allowed"] is False


def test_consecutive_gate_accepts_final_non_interval_step_and_ignores_duplicate() -> None:
    tracker = WORKER.ConsecutiveGateTracker()

    assert tracker.observe(step=900, passed=False) is False
    assert tracker.observe(step=950, passed=True) is False
    assert tracker.observe(step=950, passed=True) is False
    assert tracker.consecutive_passes == 1
    assert tracker.observe(step=964, passed=True) is True
    assert tracker.selected_step == 964
    assert tracker.consecutive_passes == 2
    assert tracker.observe(step=964, passed=True) is False
    assert tracker.consecutive_passes == 2


def test_consecutive_gate_resets_after_a_failed_checkpoint() -> None:
    tracker = WORKER.ConsecutiveGateTracker()

    assert tracker.observe(step=850, passed=True) is False
    assert tracker.observe(step=900, passed=False) is False
    assert tracker.observe(step=950, passed=True) is False
    assert tracker.observe(step=964, passed=True) is True
    assert tracker.first_passing_step == 850
    assert tracker.selected_step == 964


def test_continuation_job_bootstrap_is_pinned_to_worker_and_dependencies() -> None:
    source = JOB_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }

    assert "# /// script" in source
    assert '"trl==0.26.2"' in source
    assert assignments["ADAPTER_REPO"] == (
        "spkc83/retail-bank-servicing-agent-9b-peft-v5-remediation"
    )
    assert assignments["DEFAULT_SOURCE_ADAPTER_REVISION"] == (
        "d965816bd6a9252bfb4327c1b0d64f9d34f4a1a2"
    )
    assert assignments["DEFAULT_PROBE_CHECKPOINT_STEP"] == 600
    assert assignments["DEFAULT_PROBE_CHECKPOINT_DIR"].endswith("/trainer/checkpoint-600")
    assert assignments["CANDIDATE3_PROBE_DATASET_REVISION"] == (
        "715064e50e7ed2f815dfd3ce19b61f345a466b9d"
    )
    assert assignments["BASE_MODEL"] == "spkc83/retail-bank-servicing-agent-9b"
    assert assignments["BASE_REVISION"] == "1d56824995aa1adecfe20f62ca42fb1c0c443817"
    assert assignments["DATASET_REPO"] == "spkc83/retail-bank-servicing-alignment-sft"
    assert assignments["DEFAULT_DESTINATION_REPO"] == (
        "spkc83/retail-bank-servicing-agent-9b-peft-v6-generation-contract"
    )
    assert "cloud_continue_tool_sft.py" in source
    assert "cloud_train_tool_sft.py" not in source
    assert "--source-adapter-revision" in source
    assert "--source-adapter-repo" in source
    assert "--destination-repo" in source
    assert "RETAIL_BANK_ALLOW_REMOTE_CONTINUATION_SFT" in source
    assert '"2e-6"' in source


def test_job_probe_command_never_requests_publication() -> None:
    probe = SimpleNamespace(
        probe_only=True,
        publish_only=False,
        probe_checkpoint_dir="/data/run/trainer/checkpoint-550",
        probe_checkpoint_step=550,
    )
    training = SimpleNamespace(
        probe_only=False,
        publish_only=False,
        probe_checkpoint_dir="unused",
        probe_checkpoint_step=0,
    )

    probe_args = JOB.execution_mode_args(probe)

    assert probe_args == [
        "--probe-only",
        "--probe-checkpoint-dir",
        "/data/run/trainer/checkpoint-550",
        "--probe-checkpoint-step",
        "550",
    ]
    assert "--push-to-hub" not in probe_args
    assert JOB.execution_mode_args(training) == ["--push-to-hub"]
    recovery = SimpleNamespace(publish_only=True, probe_only=False)
    assert JOB.execution_mode_args(recovery) == ["--publish-only", "--push-to-hub"]


def test_worker_probe_requires_exact_source_and_candidate3_dataset_revisions() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")
    probe_body = source.split("def run_checkpoint_probe", 1)[1].split(
        "def run_remote_continuation", 1
    )[0]

    assert 'require_exact_revision(os.environ.get("RETAIL_BANK_SOURCE_COMMIT", "")' in probe_body
    assert "CANDIDATE3_PROBE_DATASET_REVISION" in probe_body
    assert '"published": False' in probe_body


def test_remote_continuation_launcher_mounts_durable_bucket_and_uses_five_hour_cap() -> None:
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert 'job_timeout="5h"' in launcher
    assert 'job_timeout="30m"' in launcher
    assert 'job_flavor="cpu-basic"' in launcher
    assert "--volume hf://buckets/spkc83/jobs-artifacts:/data" in launcher
    assert "SOURCE_ADAPTER_REVISION must be the exact 40-character lowercase Git commit" in launcher
    assert "SOURCE_ADAPTER_REPO must be a Hugging Face repository id" in launcher
    assert "DESTINATION_REPO must differ from the source adapter repository" in launcher
    assert (
        "retail-bank-agent-9b-peft-v6-generation-contract-${source_commit:0:8}-"
        "${source_adapter_revision:0:8}-${dataset_revision:0:8}"
    ) in launcher
    assert "hf jobs uv run" in launcher
    assert "/scripts/retail_bank/hf_job_continue_tool_sft.py" in launcher
    assert "/scripts/banking_v2/hf_job_continue_tool_sft.py" not in launcher
    assert 'script_url="$legacy_script_url"' not in launcher
    assert "rm " not in launcher
    assert 'max_steps="${5:-964}"' in launcher
    assert "d965816bd6a9252bfb4327c1b0d64f9d34f4a1a2" in launcher
    assert "checkpoint-600" in launcher
    assert "PROBE_ONLY" in launcher
    assert "715064e50e7ed2f815dfd3ce19b61f345a466b9d" in launcher
    assert "Source adapter must be exactly" in launcher


def test_worker_evaluates_before_atomic_adapter_upload_without_merging() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")
    remote_body = source.split("def run_remote_continuation", 1)[1].split("def main", 1)[0]
    upload_branch = remote_body.index("revision = upload_release(")

    assert (
        remote_body.index("eval_metrics = validate_eval_metrics(trainer.evaluate())")
        < upload_branch
    )
    assert remote_body.index("behavioral_gate = persist_and_validate_coreference_report(") < (
        upload_branch
    )
    assert "class BehavioralGateCallback" in remote_body
    assert remote_body.index("preflight_destination_repo(config)") < remote_body.index(
        "snapshot_source_adapter(config)"
    )
    assert 'self.output_dir / "behavioral-evaluations"' in remote_body
    assert "ConsecutiveGateTracker()" in remote_body
    assert "trainer.remove_callback(BehavioralGateCallback)" in remote_body
    assert remote_body.index("eval_metrics = validate_eval_metrics(trainer.evaluate())") < (
        remote_body.index("trainer.remove_callback(BehavioralGateCallback)")
    )
    assert "shadow-selected-step-" in remote_body
    assert remote_body.index("shadow_report = generate_coreference_behavior_report(") < (
        upload_branch
    )
    assert "merge_and_unload" not in source
    assert "merged-fp16" not in source
    assert "CommitOperationAdd" in source
    assert "api.create_commit(" in source
    assert "api.upload_folder(" not in source


def test_continuation_behavioral_generation_uses_per_record_tool_contract() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")
    behavior_body = source.split("def generate_coreference_behavior_report", 1)[1].split(
        "def render_model_card", 1
    )[0]

    assert "tools=training_tools_for_record(record, adapter)" in behavior_body


def test_continuation_contract_resolution_matches_training_with_legacy_fallback() -> None:
    adapter = SimpleNamespace(
        public_tool_manifest=[
            {"name": "freeze_card"},
            {"name": "list_transactions"},
        ]
    )
    execute = {
        "expected": {
            "generation_contract": {
                "mode": "execute_tool",
                "tool_names": ["freeze_card"],
            }
        }
    }
    converse = {
        "expected": {
            "generation_contract": {
                "mode": "converse",
                "tool_names": [],
            }
        }
    }

    assert WORKER.training_tools_for_record(execute, adapter) == [{"name": "freeze_card"}]
    assert WORKER.training_tools_for_record(converse, adapter) == []
    assert WORKER.training_tools_for_record({"expected": {}}, adapter) is None


def test_v7_shadow_loader_enforces_predicted_e2e_schema(tmp_path: Path) -> None:
    shadow = tmp_path / "granite-v7-shadow.jsonl"
    row = {
        "record_id": "shadow-v7",
        "metadata": {"trainable": False},
        "expected": {
            "generation_contract": {
                "version": "banking-v7-route-to-generation/v1",
                "mode": "execute_tool",
                "entity_state": "resolved",
                "tool_names": ["replace_card"],
                "argument_constraints": {"last4": {"const": "8462"}},
            }
        },
    }
    shadow.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "behavioral_gates": [
                    {
                        "name": "granite-v7-shadow",
                        "path": shadow.name,
                        "record_count": 1,
                        "sha256": hashlib.sha256(shadow.read_bytes()).hexdigest(),
                        "allowed_use": [
                            "checkpoint-selection",
                            "generalization-evaluation",
                        ],
                        "trainable": False,
                        "gate_contract": "banking-v7-granite-predicted-e2e-gate/v1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert WORKER.load_granite_v7_shadow_records(manifest) == [row]


def test_worker_enables_input_grads_for_trainable_peft_checkpointing() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")
    remote_body = source.split("def run_remote_continuation", 1)[1].split(
        "train_output = trainer.train", 1
    )[0]

    assert remote_body.index("PeftModel.from_pretrained(") < remote_body.index(
        "model.enable_input_require_grads()"
    )
    assert remote_body.index("model.enable_input_require_grads()") < remote_body.index(
        "trainer = SFTTrainer("
    )


def test_continuation_upload_is_new_root_level_peft_bundle() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")
    upload_body = source.split("def upload_release", 1)[1].split("def run_remote_continuation", 1)[
        0
    ]

    assert "exist_ok=False" in upload_body
    assert "source adapter repository" in upload_body
    assert "path_in_repo=name" in upload_body
    assert 'path_in_repo="training_result.json"' in upload_body
    assert 'path_in_repo="continuation_training_metadata.json"' in upload_body
    assert 'path_in_repo="README.md"' in upload_body
    assert 'path_in_repo="adapter"' not in upload_body


def test_source_adapter_validation_checks_root_digest_and_pinned_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in WORKER.ADAPTER_FILES:
        (tmp_path / name).write_bytes(f"fixture:{name}".encode())
    adapter_sha = hashlib.sha256((tmp_path / "adapter_model.safetensors").read_bytes()).hexdigest()
    (tmp_path / "training_result.json").write_text(
        json.dumps(
            {
                "base_model": {
                    "repository": WORKER.BASE_MODEL,
                    "revision": WORKER.BASE_REVISION,
                },
                "adapter_model_sha256": adapter_sha,
            }
        ),
        encoding="utf-8",
    )
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **_kwargs: str(tmp_path))
    config = WORKER.config_from_args(WORKER.parse_args([]))
    assert WORKER.snapshot_source_adapter(config) == tmp_path

    payload = json.loads((tmp_path / "training_result.json").read_text())
    payload["base_model"]["revision"] = "0" * 40
    (tmp_path / "training_result.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="pinned base"):
        WORKER.snapshot_source_adapter(config)


def test_source_adapter_validation_accepts_prior_remediation_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in WORKER.ADAPTER_FILES:
        (tmp_path / name).write_bytes(f"continued:{name}".encode())
    adapter_sha = hashlib.sha256((tmp_path / "adapter_model.safetensors").read_bytes()).hexdigest()
    (tmp_path / "training_result.json").write_text(
        json.dumps(
            {
                "contract": "banking-v5-peft-remediation-result/v1",
                "base_model": WORKER.BASE_MODEL,
                "base_revision": WORKER.BASE_REVISION,
                "adapter_sha256": adapter_sha,
            }
        ),
        encoding="utf-8",
    )
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **_kwargs: str(tmp_path))
    config = replace(
        WORKER.config_from_args(WORKER.parse_args([])),
        source_adapter_repo="spkc83/retail-bank-servicing-agent-9b-peft-v5-remediation",
        source_adapter_revision="d965816bd6a9252bfb4327c1b0d64f9d34f4a1a2",
    )

    assert WORKER.snapshot_source_adapter(config) == tmp_path


def test_atomic_upload_creates_new_repo_and_places_adapter_at_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    for name in WORKER.ADAPTER_FILES:
        (adapter_dir / name).write_bytes(f"release:{name}".encode())
    result_path = tmp_path / "continuation_training_result.json"
    metadata_path = tmp_path / "continuation_training_metadata.json"
    adapter_sha = hashlib.sha256(
        (adapter_dir / "adapter_model.safetensors").read_bytes()
    ).hexdigest()
    result: dict[str, Any] = {
        "base_model": WORKER.BASE_MODEL,
        "base_revision": WORKER.BASE_REVISION,
        "source_adapter_repo": WORKER.ADAPTER_REPO,
        "source_adapter_revision": WORKER.DEFAULT_SOURCE_ADAPTER_REVISION,
        "dataset_identity": {"repository": WORKER.DATASET_REPO, "revision": "d" * 40},
        "steps": 250,
        "adapter_sha256": adapter_sha,
        "eval_metrics": {"eval_loss": 0.21, "eval_runtime": 12.0},
        "coreference_behavioral_gate": {
            "positive_tool_argument_accuracy": 1.0,
            "ambiguity_accuracy": 1.0,
            "pair_flip_accuracy": 1.0,
        },
        "shadow_coreference_behavioral_gate": {
            "positive_tool_argument_accuracy": 1.0,
            "ambiguity_accuracy": 1.0,
            "pair_flip_accuracy": 1.0,
        },
        "consecutive_dev_passes": 2,
        "pushed_to_hub": "example/new-remediation-adapter",
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    metadata_path.write_text("{}", encoding="utf-8")

    class FakeApi:
        def __init__(self) -> None:
            self.create_repo_call: dict[str, Any] | None = None
            self.create_commit_call: dict[str, Any] | None = None
            self.exists = False

        def repo_info(self, **_kwargs: Any) -> None:
            if not self.exists:
                raise _repository_not_found()

        def list_repo_files(self, **_kwargs: Any) -> list[str]:
            return []

        def create_repo(self, repo_id: str, **kwargs: Any) -> None:
            self.create_repo_call = {"repo_id": repo_id, **kwargs}
            self.exists = True

        def create_commit(self, **kwargs: Any) -> SimpleNamespace:
            self.create_commit_call = kwargs
            return SimpleNamespace(oid="e" * 40)

    api = FakeApi()
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token: api)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    config = replace(
        WORKER.config_from_args(WORKER.parse_args([])),
        output_dir=tmp_path,
        hub_dest="example/new-remediation-adapter",
    )

    revision = WORKER.upload_release(
        config,
        result=result,
        result_path=result_path,
        metadata_path=metadata_path,
    )

    assert revision == "e" * 40
    assert api.create_repo_call == {
        "repo_id": "example/new-remediation-adapter",
        "repo_type": "model",
        "private": False,
        "exist_ok": False,
    }
    assert api.create_commit_call is not None
    paths = {operation.path_in_repo for operation in api.create_commit_call["operations"]}
    assert paths == {
        *WORKER.ADAPTER_FILES,
        "training_result.json",
        "continuation_training_metadata.json",
        "README.md",
    }


def _release_bundle(tmp_path: Path) -> tuple[Any, dict[str, Any], Path, Path]:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    for name in WORKER.ADAPTER_FILES:
        (adapter_dir / name).write_bytes(f"release:{name}".encode())
    result: dict[str, Any] = {
        "contract": "banking-v6-generation-contract-peft-result/v1",
        "worker": "cloud_continue_tool_sft",
        "base_model": WORKER.BASE_MODEL,
        "base_revision": WORKER.BASE_REVISION,
        "source_adapter_repo": WORKER.ADAPTER_REPO,
        "source_adapter_revision": WORKER.DEFAULT_SOURCE_ADAPTER_REVISION,
        "dataset_identity": {"repository": WORKER.DATASET_REPO, "revision": "d" * 40},
        "steps": 964,
        "adapter_sha256": hashlib.sha256(
            (adapter_dir / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "eval_metrics": {"eval_loss": 0.21},
        "coreference_behavioral_gate": {
            "positive_tool_argument_accuracy": 1.0,
            "ambiguity_accuracy": 1.0,
            "pair_flip_accuracy": 1.0,
        },
        "shadow_coreference_behavioral_gate": {
            "positive_tool_argument_accuracy": 1.0,
            "ambiguity_accuracy": 1.0,
            "pair_flip_accuracy": 1.0,
        },
        "consecutive_dev_passes": 2,
        "merged_model": None,
        "pushed_to_hub": "example/new-remediation-adapter",
        "publication": {
            "requested": True,
            "destination_repo": "example/new-remediation-adapter",
            "atomic_bundle": True,
        },
    }
    result_path = tmp_path / "continuation_training_result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    metadata_path = tmp_path / "continuation_training_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "contract": "banking-v6-generation-contract-peft-metadata/v1",
                "worker": "cloud_continue_tool_sft",
                "step": result["steps"],
                "eval_metrics": result["eval_metrics"],
                "coreference_behavioral_gate": result["coreference_behavioral_gate"],
                "shadow_coreference_behavioral_gate": result["shadow_coreference_behavioral_gate"],
                "fingerprint": {
                    "contract": "banking-v6-generation-contract-peft-fingerprint/v1",
                    "source_commit": "a" * 40,
                    "base_model": WORKER.BASE_MODEL,
                    "base_revision": WORKER.BASE_REVISION,
                    "family": "granite",
                    "training_seed": WORKER.V6_TRAINING_SEED,
                    "source_adapter": {
                        "repository": WORKER.ADAPTER_REPO,
                        "revision": WORKER.DEFAULT_SOURCE_ADAPTER_REVISION,
                    },
                    "dataset_identity": result["dataset_identity"],
                },
            }
        ),
        encoding="utf-8",
    )
    config = replace(
        WORKER.config_from_args(WORKER.parse_args([])),
        output_dir=tmp_path,
        hub_dest="example/new-remediation-adapter",
    )
    return config, result, result_path, metadata_path


def test_atomic_upload_rejects_adapter_mutation_before_hub_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, result, result_path, metadata_path = _release_bundle(tmp_path)
    (tmp_path / "adapter" / "adapter_model.safetensors").write_bytes(b"mutated")
    monkeypatch.setenv("HF_TOKEN", "test-token")

    with pytest.raises(RuntimeError, match="adapter digest changed"):
        WORKER.upload_release(
            config,
            result=result,
            result_path=result_path,
            metadata_path=metadata_path,
        )


def test_atomic_upload_rechecks_adapter_after_hub_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, result, result_path, metadata_path = _release_bundle(tmp_path)

    class FakeApi:
        commit_called = False

        def repo_info(self, **_kwargs: Any) -> None:
            return None

        def list_repo_files(self, **_kwargs: Any) -> list[str]:
            (tmp_path / "adapter" / "adapter_model.safetensors").write_bytes(b"late mutation")
            return []

        def create_commit(self, **_kwargs: Any) -> SimpleNamespace:
            self.commit_called = True
            return SimpleNamespace(oid="e" * 40)

    api = FakeApi()
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token: api)
    monkeypatch.setenv("HF_TOKEN", "test-token")

    with pytest.raises(RuntimeError, match="adapter digest changed"):
        WORKER.upload_release(
            config,
            result=result,
            result_path=result_path,
            metadata_path=metadata_path,
        )
    assert api.commit_called is False


def test_atomic_upload_validates_optional_release_file_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, result, result_path, metadata_path = _release_bundle(tmp_path)
    result["release_file_sha256"] = {"adapter_config.json": "0" * 64}
    result_path.write_text(json.dumps(result), encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "test-token")

    with pytest.raises(RuntimeError, match="release file digest mismatch"):
        WORKER.upload_release(
            config,
            result=result,
            result_path=result_path,
            metadata_path=metadata_path,
        )


def test_destination_repo_states_absent_empty_and_nonempty() -> None:
    class FakeApi:
        def __init__(self, state: str) -> None:
            self.state = state

        def repo_info(self, **_kwargs: Any) -> None:
            if self.state == "absent":
                raise _repository_not_found()

        def list_repo_files(self, **_kwargs: Any) -> list[str]:
            return [] if self.state == "empty" else ["README.md"]

    assert WORKER.require_publishable_destination(FakeApi("absent"), "example/repo") == "absent"
    assert WORKER.require_publishable_destination(FakeApi("empty"), "example/repo") == "empty"
    with pytest.raises(RuntimeError, match="not empty"):
        WORKER.require_publishable_destination(FakeApi("nonempty"), "example/repo")


def test_atomic_upload_retries_an_existing_empty_repo_after_commit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, result, result_path, metadata_path = _release_bundle(tmp_path)

    class FakeApi:
        def __init__(self) -> None:
            self.exists = False
            self.create_count = 0
            self.commit_count = 0

        def repo_info(self, **_kwargs: Any) -> None:
            if not self.exists:
                raise _repository_not_found()

        def list_repo_files(self, **_kwargs: Any) -> list[str]:
            return []

        def create_repo(self, *_args: Any, **_kwargs: Any) -> None:
            self.exists = True
            self.create_count += 1

        def create_commit(self, **_kwargs: Any) -> SimpleNamespace:
            self.commit_count += 1
            if self.commit_count == 1:
                raise RuntimeError("simulated commit failure")
            return SimpleNamespace(oid="e" * 40)

    api = FakeApi()
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token: api)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    with pytest.raises(RuntimeError, match="simulated commit failure"):
        WORKER.upload_release(
            config,
            result=result,
            result_path=result_path,
            metadata_path=metadata_path,
        )
    assert (
        WORKER.upload_release(
            config,
            result=result,
            result_path=result_path,
            metadata_path=metadata_path,
        )
        == "e" * 40
    )
    assert api.create_count == 1
    assert api.commit_count == 2


def test_publish_only_recovery_reuses_bundle_after_atomic_commit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, result, result_path, metadata_path = _release_bundle(tmp_path)
    config = replace(
        config,
        dry_run=False,
        allow_remote_execution=True,
        push_to_hub=True,
        publish_only=True,
    )
    monkeypatch.setenv(WORKER.REMOTE_CONFIRMATION_ENV, WORKER.REMOTE_CONFIRMATION_VALUE)
    monkeypatch.setenv("RETAIL_BANK_SOURCE_COMMIT", "a" * 40)
    monkeypatch.setenv("RETAIL_BANK_TOOL_SFT_DATASET_REPO", WORKER.DATASET_REPO)
    monkeypatch.setenv("RETAIL_BANK_TOOL_SFT_DATASET_REVISION", "d" * 40)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    WORKER.write_publication_bundle_manifest(
        config,
        result_path=result_path,
        metadata_path=metadata_path,
    )

    class FakeApi:
        def __init__(self) -> None:
            self.exists = False
            self.commits = 0

        def repo_info(self, **_kwargs: Any) -> None:
            if not self.exists:
                raise _repository_not_found()

        def list_repo_files(self, **_kwargs: Any) -> list[str]:
            return []

        def create_repo(self, *_args: Any, **_kwargs: Any) -> None:
            self.exists = True

        def create_commit(self, **_kwargs: Any) -> SimpleNamespace:
            self.commits += 1
            if self.commits == 1:
                raise RuntimeError("simulated commit failure")
            return SimpleNamespace(oid="e" * 40)

    api = FakeApi()
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token: api)
    monkeypatch.setattr(
        WORKER,
        "dataset_identity",
        lambda _path: {
            "repository": WORKER.DATASET_REPO,
            "revision": "d" * 40,
        },
    )
    with pytest.raises(RuntimeError, match="simulated commit failure"):
        WORKER.upload_release(
            config,
            result=result,
            result_path=result_path,
            metadata_path=metadata_path,
        )

    recovered = WORKER.run_publish_recovery(config)

    assert recovered["publish_recovery"] is True
    assert recovered["published_adapter_revision"] == "e" * 40
    assert api.commits == 2


def test_publish_only_recovery_rejects_incomplete_and_mismatched_bundles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incomplete_config = replace(
        WORKER.config_from_args(WORKER.parse_args([])),
        output_dir=tmp_path / "incomplete",
    )
    with pytest.raises(RuntimeError, match="bundle is incomplete"):
        WORKER.validate_publication_bundle(
            incomplete_config,
            source_commit="a" * 40,
            dataset_revision="d" * 40,
        )

    complete = tmp_path / "complete"
    complete.mkdir()
    config, _result, result_path, metadata_path = _release_bundle(complete)
    monkeypatch.setenv("RETAIL_BANK_SOURCE_COMMIT", "a" * 40)
    monkeypatch.setenv("RETAIL_BANK_TOOL_SFT_DATASET_REVISION", "d" * 40)
    bundle_path = WORKER.write_publication_bundle_manifest(
        config,
        result_path=result_path,
        metadata_path=metadata_path,
    )
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["destination_repo"] = "example/wrong-destination"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    with pytest.raises(RuntimeError, match="bundle identity mismatch"):
        WORKER.validate_publication_bundle(
            config,
            source_commit="a" * 40,
            dataset_revision="d" * 40,
        )


@pytest.mark.parametrize(
    "metrics",
    [{}, {"eval_loss": float("nan")}, {"eval_loss": 0.2, "eval_runtime": float("inf")}],
)
def test_continuation_rejects_missing_or_non_finite_post_train_eval(
    metrics: dict[str, float],
) -> None:
    with pytest.raises(RuntimeError, match="post-train"):
        WORKER.validate_eval_metrics(metrics)


def test_coreference_behavioral_gate_scores_tool_arguments_ambiguity_and_pair_flip() -> None:
    positive = _record("positive", path="multi_turn", assistant_tool_calls=1)
    positive["metadata"] = {
        "coreference_pair_id": "pair-1",
        "coreference_target": "replace_card",
    }
    positive["expected"] = {
        "path": "multi_turn",
        "tool_calls": [{"name": "replace_card", "arguments": {"last4": "6107"}}],
    }
    ambiguous = _record(
        "ambiguous",
        path="clarification",
        final="Which card should I replace? Please share the last four digits.",
    )
    ambiguous["metadata"] = {
        "coreference_pair_id": "pair-1",
        "coreference_target": "clarification",
    }
    predictions = {
        "positive": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": "replace_card", "arguments": {"last4": "6107"}}}],
        },
        "ambiguous": {
            "role": "assistant",
            "content": "Which card should I replace? Please share the last four digits.",
            "tool_calls": [],
        },
    }

    metrics = WORKER.score_coreference_behavior([positive, ambiguous], predictions)

    assert metrics["positive_tool_argument_accuracy"] == 1.0
    assert metrics["ambiguity_accuracy"] == 1.0
    assert metrics["pair_flip_accuracy"] == 1.0
    assert WORKER.validate_coreference_behavioral_gate(metrics) == metrics

    with pytest.raises(RuntimeError, match="requires each accuracy"):
        WORKER.validate_coreference_behavioral_gate({**metrics, "pair_flip_accuracy": 0.94})


def test_behavioral_report_persists_record_predictions_before_gate_failure(
    tmp_path: Path,
) -> None:
    positive = _record("positive", path="multi_turn", assistant_tool_calls=1)
    positive["metadata"] = {
        "coreference_pair_id": "pair-1",
        "coreference_target": "replace_card",
    }
    positive["expected"] = {
        "path": "multi_turn",
        "tool_calls": [{"name": "replace_card", "arguments": {"last4": "6107"}}],
    }
    ambiguous = _record("ambiguous", path="clarification")
    ambiguous["metadata"] = {
        "coreference_pair_id": "pair-1",
        "coreference_target": "clarification",
    }
    predictions = {
        "positive": {"role": "assistant", "content": "Which card?", "tool_calls": []},
        "ambiguous": {
            "role": "assistant",
            "content": "Which card should I replace?",
            "tool_calls": [],
        },
    }
    report = WORKER.build_coreference_behavior_report(
        [positive, ambiguous],
        predictions,
        raw_outputs={"positive": "Which card?", "ambiguous": "Which card should I replace?"},
        parse_errors={},
        cumulative_step=650,
    )
    path = tmp_path / "behavioral-evaluations" / "step-0650.json"

    with pytest.raises(RuntimeError, match="requires each accuracy"):
        WORKER.persist_and_validate_coreference_report(report, path)

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["cumulative_step"] == 650
    assert persisted["metrics"]["positive_tool_argument_accuracy"] == 0.0
    assert [row["record_id"] for row in persisted["records"]] == [
        "positive",
        "ambiguous",
    ]
    assert persisted["records"][0]["raw_output"] == "Which card?"
    assert persisted["records"][0]["passed"] is False


def test_source_checkpoint_requires_exact_step_and_records_adapter_digest(
    tmp_path: Path,
) -> None:
    (tmp_path / "adapter_model.safetensors").write_bytes(b"candidate3-checkpoint")
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "trainer_state.json").write_text(json.dumps({"global_step": 600}), encoding="utf-8")

    identity = WORKER.validate_source_checkpoint(tmp_path, expected_step=600)

    assert identity["path"] == str(tmp_path.resolve())
    assert identity["step"] == 600
    assert identity["optimizer_resumed"] is False
    assert identity["adapter_sha256"] == hashlib.sha256(b"candidate3-checkpoint").hexdigest()
    with pytest.raises(RuntimeError, match="step mismatch"):
        WORKER.validate_source_checkpoint(tmp_path, expected_step=550)


def test_shadow_gate_loader_requires_non_trainable_exact_digest(tmp_path: Path) -> None:
    records = [_record(f"shadow-{index}", path="clarification") for index in range(32)]
    for record in records:
        record["metadata"]["trainable"] = False
    shadow = tmp_path / "coreference-shadow.jsonl"
    shadow.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    digest = hashlib.sha256(shadow.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "behavioral_gates": [
                    {
                        "name": "coreference-shadow",
                        "path": shadow.name,
                        "record_count": 32,
                        "sha256": digest,
                        "allowed_use": ["post-selection-evaluation-once"],
                        "trainable": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert len(WORKER.load_shadow_gate_records(manifest)) == 32
    payload = json.loads(manifest.read_text())
    payload["behavioral_gates"][0]["trainable"] = True
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-trainable"):
        WORKER.load_shadow_gate_records(manifest)

    payload["behavioral_gates"][0]["trainable"] = False
    records[0]["metadata"]["trainable"] = True
    shadow.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    payload["behavioral_gates"][0]["sha256"] = hashlib.sha256(shadow.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="rows must set metadata.trainable=false"):
        WORKER.load_shadow_gate_records(manifest)


def test_worker_dataset_identity_rejects_a_hash_invalid_manifest_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    shutil.copytree(Path("data/banking-servicing-alignment-v5"), dataset)
    train = dataset / "train.jsonl"
    payload = bytearray(train.read_bytes())
    payload[0] = ord("[") if payload[0] != ord("[") else ord("{")
    train.write_bytes(payload)
    monkeypatch.setenv("RETAIL_BANK_TOOL_SFT_DATASET_REPO", WORKER.DATASET_REPO)
    monkeypatch.setenv("RETAIL_BANK_TOOL_SFT_DATASET_REVISION", "d" * 40)

    with pytest.raises(ValueError, match="train sha256 mismatch"):
        WORKER.dataset_identity(dataset / "manifest.json")
