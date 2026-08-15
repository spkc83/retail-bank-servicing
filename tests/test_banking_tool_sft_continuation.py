from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

WORKER_PATH = Path("scripts/retail_bank/cloud_continue_tool_sft.py")
JOB_PATH = Path("scripts/retail_bank/hf_job_continue_tool_sft.py")
LAUNCHER_PATH = Path("scripts/retail_bank/run_remote_continuation_job.sh")


def _load_worker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cloud_continue_tool_sft", WORKER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


WORKER = _load_worker()


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
    assert plan["base_model"] == "spkc83/retail-bank-servicing-agent-9b"
    assert plan["base_revision"] == "1d56824995aa1adecfe20f62ca42fb1c0c443817"
    assert plan["hub_dest"] == "spkc83/retail-bank-servicing-agent-9b-peft-v5-candidate3"
    assert plan["training"]["max_steps"] == 600
    assert plan["training"]["max_train_seconds"] == 3_600
    assert plan["training"]["learning_rate"] == 5e-6
    assert plan["training"]["positive_multiplier"] == 10
    assert plan["training"]["ambiguity_multiplier"] == 5
    assert plan["training"]["policy_faq_multiplier"] == 4
    assert plan["training"]["tool_outcome_multiplier"] == 6
    assert plan["release"]["format"] == "root-level PEFT adapter"
    assert plan["release"]["merge"] is False
    assert plan["release"]["evaluation_before_publish"] is True
    assert plan["remote_guard"]["currently_allowed"] is False


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
    assert assignments["BASE_MODEL"] == "spkc83/retail-bank-servicing-agent-9b"
    assert assignments["BASE_REVISION"] == "1d56824995aa1adecfe20f62ca42fb1c0c443817"
    assert assignments["DATASET_REPO"] == "spkc83/retail-bank-servicing-alignment-sft"
    assert "cloud_continue_tool_sft.py" in source
    assert "cloud_train_tool_sft.py" not in source
    assert "--source-adapter-revision" in source
    assert "--source-adapter-repo" in source
    assert "--destination-repo" in source
    assert "RETAIL_BANK_ALLOW_REMOTE_CONTINUATION_SFT" in source
    assert '"5e-6"' in source


def test_remote_continuation_launcher_mounts_durable_bucket_and_uses_five_hour_cap() -> None:
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "--timeout 5h" in launcher
    assert "--volume hf://buckets/spkc83/jobs-artifacts:/data" in launcher
    assert "SOURCE_ADAPTER_REVISION must be the exact 40-character lowercase Git commit" in launcher
    assert "SOURCE_ADAPTER_REPO must be a Hugging Face repository id" in launcher
    assert "DESTINATION_REPO must differ from the source adapter repository" in launcher
    assert (
        "retail-bank-agent-9b-continuation-${source_commit:0:8}-"
        "${source_adapter_revision:0:8}-${dataset_revision:0:8}"
    ) in launcher
    assert "hf jobs uv run" in launcher
    assert "/scripts/retail_bank/hf_job_continue_tool_sft.py" in launcher
    assert "/scripts/banking_v2/hf_job_continue_tool_sft.py" not in launcher
    assert 'script_url="$legacy_script_url"' not in launcher
    assert "rm " not in launcher
    assert 'max_steps="${5:-600}"' in launcher
    assert "d965816bd6a9252bfb4327c1b0d64f9d34f4a1a2" in launcher


def test_worker_evaluates_before_atomic_adapter_upload_without_merging() -> None:
    source = WORKER_PATH.read_text(encoding="utf-8")
    remote_body = source.split("def run_remote_continuation", 1)[1].split("def main", 1)[0]

    assert remote_body.index(
        "eval_metrics = validate_eval_metrics(trainer.evaluate())"
    ) < remote_body.index("if config.push_to_hub:")
    assert remote_body.index("behavioral_gate = run_coreference_behavioral_gate(") < (
        remote_body.index("if config.push_to_hub:")
    )
    assert "merge_and_unload" not in source
    assert "merged-fp16" not in source
    assert "CommitOperationAdd" in source
    assert "api.create_commit(" in source
    assert "api.upload_folder(" not in source


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
    result: dict[str, Any] = {
        "base_model": WORKER.BASE_MODEL,
        "base_revision": WORKER.BASE_REVISION,
        "source_adapter_repo": WORKER.ADAPTER_REPO,
        "source_adapter_revision": WORKER.DEFAULT_SOURCE_ADAPTER_REVISION,
        "dataset_identity": {"repository": WORKER.DATASET_REPO, "revision": "d" * 40},
        "steps": 250,
        "adapter_sha256": "a" * 64,
        "eval_metrics": {"eval_loss": 0.21, "eval_runtime": 12.0},
        "coreference_behavioral_gate": {
            "positive_tool_argument_accuracy": 1.0,
            "ambiguity_accuracy": 1.0,
            "pair_flip_accuracy": 1.0,
        },
        "pushed_to_hub": "example/new-remediation-adapter",
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    metadata_path.write_text("{}", encoding="utf-8")

    class FakeApi:
        def __init__(self) -> None:
            self.create_repo_call: dict[str, Any] | None = None
            self.create_commit_call: dict[str, Any] | None = None

        def create_repo(self, repo_id: str, **kwargs: Any) -> None:
            self.create_repo_call = {"repo_id": repo_id, **kwargs}

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
