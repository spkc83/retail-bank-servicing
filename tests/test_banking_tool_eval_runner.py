from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

RUNNER_PATH = Path("scripts/retail_bank/cloud_generate_tool_eval.py")
JOB_PATH = Path("scripts/retail_bank/hf_job_tool_eval.py")
LAUNCHER_PATH = Path("scripts/retail_bank/run_remote_tool_eval_job.sh")


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_module(RUNNER_PATH, "cloud_generate_tool_eval")


class TemplateTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    chat_template = "unit-test-granite-tool-eval-template"

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        return_tensors: str | None = None,
    ) -> str | dict[str, list[list[int]]]:
        assert tools is not None
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"]
        del return_tensors
        parts = []
        for message in messages:
            role = message["role"]
            if role == "assistant" and message.get("tool_calls"):
                parts.append(
                    "assistant:" + json.dumps({"tool_calls": message["tool_calls"]}, sort_keys=True)
                )
            elif role == "tool":
                parts.append(
                    f"tool {message['name']}[{message['tool_call_id']}]:{message['content']}"
                )
            else:
                parts.append(f"{role}:{message.get('content', '')}")
        if add_generation_prompt:
            parts.append("assistant:")
        rendered = "\n".join(parts)
        if tokenize:
            return {"input_ids": [[ord(char) for char in rendered]]}
        return rendered

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        **_: Any,
    ) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [ord(char) for char in text]}

    def decode(self, tokens: list[int], *, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(int(token)) for token in tokens if int(token) > 2)


class RecordingBackend:
    tokenizer = TemplateTokenizer()

    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []
        self.tools: list[list[dict[str, Any]] | None] = []

    def generate_text(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        assert max_new_tokens > 0
        self.calls.append([dict(message) for message in messages])
        self.tools.append(tools)
        if any(message.get("role") == "tool" for message in messages):
            return "Done. You have Main Checking ending in 1792."
        if messages[-1]["content"] == "What accounts do I have?":
            return '<tool_call>{"name":"list_accounts","arguments":{}}</tool_call>'
        return "Please provide the last four digits."


class OneThenFinalBackend:
    tokenizer = TemplateTokenizer()

    def generate_text(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        del max_new_tokens, tools
        if not any(message.get("role") == "tool" for message in messages):
            return '<tool_call>{"name":"list_accounts","arguments":{}}</tool_call>'
        return "Done. You have account ending in 1792."


class TwoStepBackend:
    tokenizer = TemplateTokenizer()

    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    def generate_text(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        del max_new_tokens, tools
        self.calls.append([dict(message) for message in messages])
        result_count = sum(1 for message in messages if message.get("role") == "tool")
        if result_count == 0:
            return '<tool_call>{"name":"list_accounts","arguments":{}}</tool_call>'
        if result_count == 1:
            return '<tool_call>{"name":"list_cards","arguments":{}}</tool_call>'
        return "Done. I checked your accounts and cards."


class ScreenshotFixtureBackend:
    tokenizer = TemplateTokenizer()

    def generate_text(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        del max_new_tokens, tools
        tool_messages = [message for message in messages if message.get("role") == "tool"]
        if tool_messages:
            return {
                "list_service_cases": "The address case was created recently.",
                "replace_card": "Replacement is pending for the card ending in 4821.",
                "list_transfers": "I found the recent transfer.",
                "list_transactions": "I found the requested transaction history.",
            }[tool_messages[-1]["name"]]
        current = str(messages[-1]["content"]).lower()
        if "weather" in current:
            return "I can only help with retail banking requests."
        if "replace" in current:
            return '<tool_call>{"name":"replace_card","arguments":{"last4":"4821"}}</tool_call>'
        if "money i sent" in current:
            return '<tool_call>{"name":"list_transfers","arguments":{}}</tool_call>'
        if "transactions" in current:
            return '<tool_call>{"name":"list_transactions","arguments":{"limit":5}}</tool_call>'
        return '<tool_call>{"name":"list_service_cases","arguments":{}}</tool_call>'


def _tool_record() -> dict[str, Any]:
    return {
        "schema_version": "banking-tool-sft/v1",
        "record_id": "tool_record",
        "messages": [
            {"role": "system", "content": "banking system", "loss": False},
            {"role": "user", "content": "What accounts do I have?", "loss": False},
            {
                "role": "assistant",
                "content": None,
                "loss": True,
                "tool_calls": [
                    {
                        "id": "call_accounts_0",
                        "index": 0,
                        "type": "function",
                        "function": {"name": "list_accounts", "arguments": {}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_accounts_0",
                "name": "list_accounts",
                "content": {"ok": True, "result": {"accounts": [{"last4": "1792"}]}},
                "loss": False,
            },
            {
                "role": "assistant",
                "content": "Done. You have Main Checking ending in 1792.",
                "loss": True,
            },
        ],
        "expected": {
            "requires_tool": True,
            "path": "tool_success",
            "tool_calls": [{"name": "list_accounts", "arguments": {}}],
            "grounding_facts": ["account.last4=1792"],
            "generation_contract": {
                "version": "banking-v7-route-to-generation/v1",
                "mode": "execute_tool",
                "entity_state": "not_required",
                "tool_names": ["list_accounts"],
                "argument_constraints": {},
            },
        },
    }


def _two_tool_record() -> dict[str, Any]:
    record = _tool_record()
    record["record_id"] = "two_tool_record"
    record["messages"][2]["tool_calls"].append(
        {
            "id": "call_cards_1",
            "index": 1,
            "type": "function",
            "function": {"name": "list_cards", "arguments": {}},
        }
    )
    record["messages"].insert(
        4,
        {
            "role": "tool",
            "tool_call_id": "call_cards_1",
            "name": "list_cards",
            "content": {"ok": True, "result": {"cards": [{"last4": "4821"}]}},
            "loss": False,
        },
    )
    record["expected"] = {
        "requires_tool": True,
        "path": "tool_success",
        "tool_calls": [
            {"name": "list_accounts", "arguments": {}},
            {"name": "list_cards", "arguments": {}},
        ],
        "grounding_facts": ["account.last4=1792"],
        "multi_tool": True,
    }
    return record


def _no_tool_record() -> dict[str, Any]:
    return {
        "schema_version": "banking-tool-sft/v1",
        "record_id": "no_tool_record",
        "messages": [
            {"role": "system", "content": "banking system", "loss": False},
            {"role": "user", "content": "Please replace my card", "loss": False},
            {"role": "assistant", "content": "Please provide the last four digits.", "loss": True},
        ],
        "expected": {
            "requires_tool": False,
            "path": "clarification",
            "tool_calls": [],
            "grounding_facts": ["missing_field=last4"],
            "generation_contract": {
                "version": "banking-v7-route-to-generation/v1",
                "mode": "clarify",
                "entity_state": "missing",
                "tool_names": [],
                "argument_constraints": {},
            },
        },
    }


def _write_manifest(tmp_path: Path) -> Path:
    data_path = tmp_path / "test.jsonl"
    data_path.write_text(
        json.dumps(_tool_record(), sort_keys=True)
        + "\n"
        + json.dumps(_no_tool_record(), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"tool_sft": [{"name": "test", "path": "test.jsonl"}]}),
        encoding="utf-8",
    )
    return manifest_path


def _perfect_fixture_prediction(record: dict[str, Any]) -> dict[str, Any]:
    expected_calls = record["expected"]["tool_calls"]
    final = next(
        message["content"]
        for message in reversed(record["messages"])
        if message["role"] == "assistant" and not message.get("tool_calls")
    )
    parsed = {"role": "assistant", "content": final, "tool_calls": []}
    return {
        "record_id": record["record_id"],
        "ordered_emitted_tool_calls": expected_calls,
        "appended_tool_results": [
            {"expected_index": index} for index, _call in enumerate(expected_calls)
        ],
        "pass_reports": [{"parse_error": None}],
        "first_assistant_parse_error": None,
        "first_assistant_parsed": parsed if not expected_calls else {"tool_calls": []},
        "grounded_final_parse_error": None,
        "grounded_final_parsed": parsed if expected_calls else None,
        "stop_reason": "final_answer",
    }


def _config(tmp_path: Path) -> Any:
    return runner.EvalConfig(
        model_repo="spkc83/retail-bank-agent-9b",
        model_revision="a" * 40,
        dataset_repo="spkc83/retail-bank-agent-sft",
        dataset_revision="b" * 40,
        manifest=_write_manifest(tmp_path),
        output_dir=tmp_path / "out",
        predictions_jsonl=None,
        metadata_json=None,
        split="test",
        family="granite",
        device="cpu",
        dtype="fp32",
        max_new_tokens_first=8,
        max_new_tokens_final=9,
        max_tool_passes=4,
        max_tool_calls=6,
        limit=None,
        trust_remote_code=False,
        push_to_hub=False,
        enforce_release_gates=False,
        token=None,
    )


def _single_record_config(tmp_path: Path, record: dict[str, Any]) -> Any:
    base = _config(tmp_path)
    manifest_path = base.manifest
    assert manifest_path is not None
    data_path = manifest_path.parent / "test.jsonl"
    data_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps({"tool_sft": [{"name": "test", "path": "test.jsonl"}]}),
        encoding="utf-8",
    )
    return base


def test_runner_generates_two_isolated_phases_and_metadata(tmp_path: Path) -> None:
    backend = RecordingBackend()
    metadata = runner.run_eval(_config(tmp_path), backend=backend)

    predictions_path = Path(metadata["outputs"]["predictions_jsonl"])
    rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line
    ]

    assert [row["record_id"] for row in rows] == ["tool_record", "no_tool_record"]
    assert rows[0]["first_assistant_parsed"]["tool_calls"][0]["function"]["name"] == "list_accounts"
    assert rows[0]["grounded_final_parsed"]["content"].endswith("1792.")
    assert rows[0]["raw_passes"] == [
        rows[0]["first_assistant_raw_output"],
        rows[0]["grounded_final_raw_output"],
    ]
    assert rows[0]["raw_output"] == "\n".join(rows[0]["raw_passes"])
    assert rows[0]["ordered_emitted_tool_calls"] == [{"name": "list_accounts", "arguments": {}}]
    assert rows[0]["appended_tool_results"][0]["name"] == "list_accounts"
    assert rows[0]["stop_reason"] == "final_answer"
    assert rows[1]["first_assistant_parsed"]["content"] == "Please provide the last four digits."
    assert rows[1]["grounded_final_raw_output"] is None
    assert rows[1]["raw_output"] == rows[1]["first_assistant_raw_output"]
    assert [message["role"] for message in backend.calls[0]] == ["system", "user"]
    assert [tool["name"] for tool in backend.tools[0] or []] == ["list_accounts"]
    assert backend.tools[2] == []
    assert [message["role"] for message in backend.calls[1]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert metadata["phases"] == {
        "first_assistant_records": 2,
        "grounded_final_records": 1,
    }
    assert metadata["read_only_contract"] == {
        "tool_execution": False,
        "deterministic_output_repair": False,
        "canonical_results_only_for_exact_emitted_calls": True,
        "teacher_forced_unseen_assistant_tool_calls": False,
    }
    assert metadata["oracle_contract_gate"] == {
        "contract": "banking-v7-granite-oracle-contract-gate/v1",
        "record_count": 2,
        "contracted_record_count": 2,
        "exact_one_or_no_tool_count": 2,
        "eligible": True,
        "predicted_e2e_gate_contract": "banking-v7-granite-predicted-e2e-gate/v1",
    }
    report = json.loads(Path(metadata["outputs"]["report_json"]).read_text(encoding="utf-8"))
    assert report["checkpoint_revision"] == "a" * 40
    assert report["metrics"]["tool_name_accuracy"]["score"] == 1.0
    assert report["metrics"]["grounded_final_factuality"]["score"] == 1.0


def test_eval_derives_v7_contract_for_frozen_record_without_mutating_it() -> None:
    record = _tool_record()
    record["expected"].pop("generation_contract")
    record["metadata"] = {"scenario_family": "accounts"}
    adapter = runner.ToolWireAdapter(
        TemplateTokenizer(),
        family="granite",
        public_tool_manifest=runner.PUBLIC_BANKING_TOOL_MANIFEST,
    )

    contract, tools = runner.evaluation_contract_and_tools(record, adapter)

    assert contract == {
        "version": "banking-v7-route-to-generation/v1",
        "mode": "execute_tool",
        "entity_state": "not_required",
        "tool_names": ["list_accounts"],
        "argument_constraints": {},
    }
    assert tools is not None and [tool["name"] for tool in tools] == ["list_accounts"]
    assert "generation_contract" not in record["expected"]


def test_runner_requires_model_to_emit_subsequent_tool_calls(tmp_path: Path) -> None:
    config = _single_record_config(tmp_path, _two_tool_record())

    metadata = runner.run_eval(config, backend=OneThenFinalBackend())
    row = json.loads(Path(metadata["outputs"]["predictions_jsonl"]).read_text(encoding="utf-8"))
    report = json.loads(Path(metadata["outputs"]["report_json"]).read_text(encoding="utf-8"))

    assert row["raw_passes"] == [
        '<tool_call>{"name":"list_accounts","arguments":{}}</tool_call>',
        "Done. You have account ending in 1792.",
    ]
    assert row["ordered_emitted_tool_calls"] == [{"name": "list_accounts", "arguments": {}}]
    assert "list_cards" not in row["raw_output"]
    assert row["matched_expected_tool_call_count"] == 1
    assert report["metrics"]["tool_name_accuracy"]["numerator"] == 0
    assert report["metrics"]["tool_name_accuracy"]["denominator"] == 2


def test_runner_allows_model_owned_followup_tool_calls(tmp_path: Path) -> None:
    backend = TwoStepBackend()
    config = _single_record_config(tmp_path, _two_tool_record())

    metadata = runner.run_eval(config, backend=backend)
    row = json.loads(Path(metadata["outputs"]["predictions_jsonl"]).read_text(encoding="utf-8"))

    assert row["ordered_emitted_tool_calls"] == [
        {"name": "list_accounts", "arguments": {}},
        {"name": "list_cards", "arguments": {}},
    ]
    assert row["matched_expected_tool_call_count"] == 2
    assert len(row["appended_tool_results"]) == 2
    assert [message["role"] for message in backend.calls[1]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert row["stop_reason"] == "final_answer"


def test_runner_resumes_existing_prediction_jsonl_without_duplicates(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first_metadata = runner.run_eval(config, backend=RecordingBackend())
    second_metadata = runner.run_eval(config, backend=RecordingBackend())

    rows = (
        Path(second_metadata["outputs"]["predictions_jsonl"])
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(rows) == 2
    assert first_metadata["outputs"]["new_rows_written"] == 2
    assert second_metadata["outputs"]["new_rows_written"] == 0


def test_counterfactual_manifest_is_validated_before_generation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.manifest is not None
    data_path = config.manifest.parent / "test.jsonl"
    config.manifest.write_text(
        json.dumps(
            {
                "contract": "banking-counterfactual-eval-manifest/v1",
                "schema_version": "banking-tool-sft/v1",
                "training_allowed": False,
                "allowed_use": ["counterfactual-evaluation"],
                "splits": {
                    "test": {
                        "path": "test.jsonl",
                        "record_count": 2,
                        "bytes": data_path.stat().st_size,
                        "sha256": "0" * 64,
                        "allowed_use": ["counterfactual-evaluation"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    backend = RecordingBackend()

    with pytest.raises(ValueError, match="digest mismatch"):
        runner.run_eval(config, backend=backend)

    assert backend.calls == []


def test_tool_phase_targets_tool_call_after_prior_multiturn_clarification() -> None:
    record = _tool_record()
    record["messages"][1:1] = [
        {"role": "user", "content": "I need help with an account.", "loss": False},
        {
            "role": "assistant",
            "content": "What would you like to know?",
            "loss": True,
        },
    ]

    selected = runner.first_phase_messages(record)

    assert [message["role"] for message in selected] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert selected[-1]["content"] == "What accounts do I have?"


def test_followup_evaluation_retains_context_only_tool_history() -> None:
    record = {
        "record_id": "followup_record",
        "messages": [
            {"role": "system", "content": "banking system", "loss": False},
            {"role": "user", "content": "Freeze my active card.", "loss": False},
            {
                "role": "assistant",
                "content": None,
                "loss": False,
                "tool_calls": [
                    {
                        "id": "context_followup_record_0",
                        "index": 0,
                        "type": "function",
                        "function": {"name": "list_cards", "arguments": {}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "context_followup_record_0",
                "name": "list_cards",
                "content": {"ok": True, "result": {"cards": [{"last4": "4821"}]}},
                "loss": False,
            },
            {
                "role": "assistant",
                "content": "I froze the active card ending in 4821.",
                "loss": False,
            },
            {"role": "user", "content": "What did you just do?", "loss": False},
            {
                "role": "assistant",
                "content": "I froze the active card ending in 4821.",
                "loss": True,
            },
        ],
        "expected": {
            "requires_tool": False,
            "path": "multi_turn",
            "tool_calls": [],
            "grounding_facts": ["card.last4=4821", "card.status=frozen"],
        },
    }

    selected = runner.first_phase_messages(record)

    assert [message["role"] for message in selected] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert runner.canonical_tool_results(record) == []


def test_tool_evaluation_ignores_prior_context_tool_results_for_current_calls() -> None:
    record = _tool_record()
    record["messages"][1:1] = [
        {"role": "user", "content": "Show my cards first.", "loss": False},
        {
            "role": "assistant",
            "content": None,
            "loss": False,
            "tool_calls": [
                {
                    "id": "context_tool_record_0",
                    "index": 0,
                    "type": "function",
                    "function": {"name": "list_cards", "arguments": {}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "context_tool_record_0",
            "name": "list_cards",
            "content": {"ok": True, "result": {"cards": [{"last4": "4821"}]}},
            "loss": False,
        },
        {
            "role": "assistant",
            "content": "Your card ending in 4821 is active.",
            "loss": False,
        },
    ]

    selected = runner.first_phase_messages(record)
    results = runner.canonical_tool_results(record)

    assert selected[-1]["content"] == "What accounts do I have?"
    assert [result["name"] for result in results] == ["list_accounts"]


def test_exact_revision_guard_rejects_branch_names(tmp_path: Path) -> None:
    config = runner.EvalConfig(**{**_config(tmp_path).__dict__, "model_revision": "main"})

    with pytest.raises(runner.ToolEvalGenerationError, match="exact 40-character"):
        runner.run_eval(config, backend=RecordingBackend())


def test_four_bit_loader_uses_nf4_double_quantization(tmp_path: Path) -> None:
    config = runner.EvalConfig(
        **{**_config(tmp_path).__dict__, "device": "cuda", "load_in_4bit": True}
    )
    captured: dict[str, Any] = {}

    class FakeTorch:
        bfloat16 = "bf16"
        float16 = "fp16"
        float32 = "fp32"

    def fake_quantization_config(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return dict(kwargs)

    kwargs = runner.model_load_kwargs(
        config,
        torch_module=FakeTorch,
        quantization_config_factory=fake_quantization_config,
    )

    assert kwargs["dtype"] == "fp32"
    assert kwargs["device_map"] == {"": 0}
    assert kwargs["quantization_config"] == captured
    assert captured == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "fp32",
    }


def test_four_bit_loader_rejects_cpu_device(tmp_path: Path) -> None:
    config = runner.EvalConfig(**{**_config(tmp_path).__dict__, "load_in_4bit": True})

    with pytest.raises(runner.ToolEvalGenerationError, match="requires --device cuda"):
        runner.validate_config(config)


def test_peft_eval_requires_complete_exact_base_adapter_identity(tmp_path: Path) -> None:
    base = _config(tmp_path)
    partial = runner.EvalConfig(**{**base.__dict__, "adapter_repo": "spkc83/adapter"})
    with pytest.raises(runner.ToolEvalGenerationError, match="requires"):
        runner.validate_config(partial)

    invalid = runner.EvalConfig(
        **{
            **base.__dict__,
            "base_model_repo": "ibm-granite/base",
            "base_model_revision": "main",
            "adapter_repo": "spkc83/adapter",
            "adapter_revision": "c" * 40,
        }
    )
    with pytest.raises(runner.ToolEvalGenerationError, match="base-model-revision"):
        runner.validate_config(invalid)


def test_peft_eval_uses_adapter_revision_as_checkpoint_identity(tmp_path: Path) -> None:
    config = runner.EvalConfig(
        **{
            **_config(tmp_path).__dict__,
            "base_model_repo": "ibm-granite/base",
            "base_model_revision": "c" * 40,
            "adapter_repo": "spkc83/adapter",
            "adapter_revision": "d" * 40,
        }
    )

    metadata = runner.run_eval(config, backend=RecordingBackend())
    report = json.loads(Path(metadata["outputs"]["report_json"]).read_text(encoding="utf-8"))

    assert report["checkpoint_revision"] == "d" * 40
    assert metadata["model"]["loading_mode"] == "peft_adapter"
    assert metadata["model"]["base"]["revision"] == "c" * 40
    assert metadata["model"]["adapter"]["revision"] == "d" * 40


def test_transformers_backend_attaches_pinned_adapter_without_merging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.EvalConfig(
        **{
            **_config(tmp_path).__dict__,
            "base_model_repo": "ibm-granite/base",
            "base_model_revision": "e" * 40,
            "adapter_repo": "spkc83/adapter",
            "adapter_revision": "f" * 40,
        }
    )
    calls: dict[str, tuple[Any, ...]] = {}

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTokenizer:
        pad_token_id = 0
        eos_token = "<eos>"
        padding_side = "right"

    class FakeModel:
        device = "cpu"

        def to(self, device: str) -> None:
            calls["device"] = (device,)

        def eval(self) -> None:
            calls["eval"] = ()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(repo: str, **kwargs: Any) -> FakeTokenizer:
            calls["tokenizer"] = (repo, kwargs)
            return FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(repo: str, **kwargs: Any) -> FakeModel:
            calls["base"] = (repo, kwargs)
            return FakeModel()

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model: FakeModel, repo: str, **kwargs: Any) -> FakeModel:
            calls["adapter"] = (model, repo, kwargs)
            return model

    class FakeBitsAndBytesConfig:
        pass

    torch = ModuleType("torch")
    torch.bfloat16 = "bf16"  # type: ignore[attr-defined]
    torch.float16 = "fp16"  # type: ignore[attr-defined]
    torch.float32 = "fp32"  # type: ignore[attr-defined]
    torch.cuda = FakeCuda()  # type: ignore[attr-defined]
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = FakeAutoTokenizer  # type: ignore[attr-defined]
    transformers.AutoModelForCausalLM = FakeAutoModel  # type: ignore[attr-defined]
    transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig  # type: ignore[attr-defined]
    peft = ModuleType("peft")
    peft.PeftModel = FakePeftModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)

    backend = runner.TransformersGenerationBackend(config)

    assert calls["tokenizer"][0] == "spkc83/adapter"
    assert calls["base"][0] == "ibm-granite/base"
    adapter_call = calls["adapter"]
    assert adapter_call[1:] == (
        "spkc83/adapter",
        {
            "revision": "f" * 40,
            "token": None,
            "autocast_adapter_dtype": False,
        },
    )
    assert backend.model is adapter_call[0]
    assert not hasattr(backend.model, "merge_and_unload")


def test_local_manifest_accepts_only_matching_sha256_identity(tmp_path: Path) -> None:
    base = _config(tmp_path)
    assert base.manifest is not None
    digest = hashlib.sha256(base.manifest.read_bytes()).hexdigest()
    config = runner.EvalConfig(**{**base.__dict__, "dataset_revision": f"sha256:{digest}"})

    runner.validate_config(config)

    mismatched = runner.EvalConfig(**{**base.__dict__, "dataset_revision": f"sha256:{'0' * 64}"})
    with pytest.raises(runner.ToolEvalGenerationError, match="does not match"):
        runner.validate_config(mismatched)


def test_named_non_trainable_fixtures_load_with_validated_identity() -> None:
    manifest = Path("data/banking-servicing-alignment-v5/manifest.json")

    shadow = runner.load_manifest_records(manifest, "granite-v7-shadow")
    screenshots = runner.load_manifest_records(manifest, "screenshot-regression")

    assert len(shadow) == 13
    assert len(screenshots) == 9
    assert all(row["metadata"]["trainable"] is False for row in [*shadow, *screenshots])
    assert all(row["expected"]["fixture_gate_contract"] for row in screenshots)
    card = next(row for row in screenshots if row["record_id"] == "screenshot-card-selection")
    assert card["expected"]["tool_calls"] == [
        {"name": "replace_card", "arguments": {"last4": "4821"}}
    ]
    assert [message["role"] for message in card["messages"][:4]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


def test_named_fixture_loader_rejects_tampered_manifest_contract(tmp_path: Path) -> None:
    source = Path("data/banking-servicing-alignment-v5")
    fixture = tmp_path / "granite-v7-shadow.jsonl"
    fixture.write_bytes((source / fixture.name).read_bytes())
    entry = json.loads((source / "manifest.json").read_text())["behavioral_gates"][1]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"behavioral_gates": [{**entry, "bytes": entry["bytes"] + 1}]}),
        encoding="utf-8",
    )

    with pytest.raises(runner.ToolEvalGenerationError, match="byte count mismatch"):
        runner.load_manifest_records(manifest, "granite-v7-shadow")


def test_run_eval_executes_and_enforces_screenshot_fixture_gate(tmp_path: Path) -> None:
    source = Path("data/banking-servicing-alignment-v5")
    rows = [
        json.loads(line)
        for line in (source / "screenshot-regression.jsonl").read_text().splitlines()
    ]
    fixture = tmp_path / "screenshot-regression.jsonl"
    fixture.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = tmp_path / "fixture-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "evaluation_fixtures": [
                    {
                        "name": "screenshot-regression",
                        "path": fixture.name,
                        "record_count": 9,
                        "bytes": fixture.stat().st_size,
                        "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                        "allowed_use": ["regression-evaluation"],
                        "trainable": False,
                        "gate_contract": "banking-v7-screenshot-regression/v1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    base = _config(tmp_path)
    config = runner.EvalConfig(
        **{
            **base.__dict__,
            "manifest": manifest,
            "split": "screenshot-regression",
            "enforce_release_gates": True,
        }
    )

    metadata = runner.run_eval(config, backend=ScreenshotFixtureBackend())

    assert metadata["predicted_e2e_gate"]["eligible"] is True
    assert metadata["release_gate"] == {
        "contract": "banking-v7-screenshot-regression/v1",
        "enforced": True,
        "eligible": True,
        "failures": [],
    }

    limited = runner.EvalConfig(**{**config.__dict__, "limit": 1})
    with pytest.raises(runner.ToolEvalGenerationError, match="cannot be limited"):
        runner.run_eval(limited, backend=ScreenshotFixtureBackend())


@pytest.mark.parametrize(
    ("target", "expected_count"),
    (("granite-v7-shadow", 13), ("screenshot-regression", 9)),
)
def test_predicted_e2e_gate_uses_fixture_record_count(
    target: str,
    expected_count: int,
) -> None:
    records = runner.load_manifest_records(
        Path("data/banking-servicing-alignment-v5/manifest.json"), target
    )
    predictions = [_perfect_fixture_prediction(record) for record in records]

    gate = runner.build_predicted_e2e_gate(records, predictions)

    assert gate["record_count"] == expected_count
    assert gate["metrics"] == {
        "exact_tool_or_no_tool": {"passed": expected_count, "total": expected_count},
        "exact_arguments": {"passed": expected_count, "total": expected_count},
        "parse_success": {"passed": expected_count, "total": expected_count},
        "executable": {"passed": expected_count, "total": expected_count},
        "grounded_final": {"passed": expected_count, "total": expected_count},
    }
    assert gate["eligible"] is True
    assert gate["failures"] == []

    predictions[0]["ordered_emitted_tool_calls"] = [{"name": "list_cards", "arguments": {}}]
    failed = runner.build_predicted_e2e_gate(records, predictions)
    assert failed["eligible"] is False
    assert failed["records"][records[0]["record_id"]]["passed"] is False


def test_hf_job_requires_exact_revisions_and_invokes_eval_runner() -> None:
    job = _load_module(JOB_PATH, "hf_job_tool_eval")
    source = JOB_PATH.read_text(encoding="utf-8")

    assert "# /// script" in source
    assert '"transformers==5.13.0"' in source
    assert job.MODEL_REPO == "spkc83/retail-bank-servicing-agent-9b-peft"
    assert job.DATASET_REPO == "spkc83/retail-bank-servicing-alignment-sft"
    assert "cloud_generate_tool_eval.py" in source
    assert 'parser.add_argument("--model-repo", default=MODEL_REPO)' in source
    assert 'parser.add_argument("--dataset-repo", default=DATASET_REPO)' in source
    assert 'parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")' in source
    assert '"peft==0.18.1"' in source
    assert 'parser.add_argument("--base-model-revision", default=BASE_MODEL_REVISION)' in source
    assert 'parser.add_argument("--adapter-revision", default=MODEL_REVISION)' in source
    assert "EVALUATION_TARGETS" in source
    assert "for evaluation_target in args.evaluation_targets:" in source
    assert '"--split",' in source
    assert '"--model-revision",' in source
    assert '"--dataset-revision",' in source
    assert '"--push-to-hub",' in source
    with pytest.raises(ValueError, match="exact 40-character"):
        job.validate_git_revision("feat/tool-use-sft-v3", field="--model-revision")


def test_hf_eval_launcher_uses_pinned_url_durable_volume_and_two_hour_cap() -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "--flavor rtx-pro-6000" in source
    assert "--timeout 2h" in source
    assert "--volume hf://buckets/spkc83/jobs-artifacts:/data" in source
    assert "/scripts/retail_bank/hf_job_tool_eval.py" in source
    assert "/scripts/banking_v2/hf_job_tool_eval.py" not in source
    assert 'script_url="$legacy_script_url"' not in source
    assert "hf_job_tool_eval.py" in source
    assert "--model-revision" in source
    assert "--dataset-revision" in source
    assert 'model_repo="${MODEL_REPO:-spkc83/retail-bank-servicing-agent-9b-peft}"' in source
    assert 'dataset_repo="${DATASET_REPO:-spkc83/retail-bank-servicing-alignment-sft}"' in source
    assert '--model-repo "$model_repo"' in source
    assert '--dataset-repo "$dataset_repo"' in source
    assert 'dtype="${4:-bf16}"' in source
    assert '--dtype "$dtype"' in source
    assert "1d56824995aa1adecfe20f62ca42fb1c0c443817" in source
    assert "cc95e446af2b5e1d8d9df2751a8192613ad386e3" in source
    assert '--base-model-revision "$base_model_revision"' in source
    assert '--adapter-revision "$adapter_revision"' in source
    assert "--evaluation-targets test granite-v7-shadow screenshot-regression" in source
