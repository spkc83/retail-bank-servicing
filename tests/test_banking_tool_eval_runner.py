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

    def generate_text(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int,
    ) -> str:
        assert max_new_tokens > 0
        self.calls.append([dict(message) for message in messages])
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
    ) -> str:
        del max_new_tokens
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
    ) -> str:
        del max_new_tokens
        self.calls.append([dict(message) for message in messages])
        result_count = sum(1 for message in messages if message.get("role") == "tool")
        if result_count == 0:
            return '<tool_call>{"name":"list_accounts","arguments":{}}</tool_call>'
        if result_count == 1:
            return '<tool_call>{"name":"list_cards","arguments":{}}</tool_call>'
        return "Done. I checked your accounts and cards."


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
    report = json.loads(Path(metadata["outputs"]["report_json"]).read_text(encoding="utf-8"))
    assert report["checkpoint_revision"] == "a" * 40
    assert report["metrics"]["tool_name_accuracy"]["score"] == 1.0
    assert report["metrics"]["grounded_final_factuality"]["score"] == 1.0


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


def test_local_manifest_accepts_only_matching_sha256_identity(tmp_path: Path) -> None:
    base = _config(tmp_path)
    assert base.manifest is not None
    digest = hashlib.sha256(base.manifest.read_bytes()).hexdigest()
    config = runner.EvalConfig(**{**base.__dict__, "dataset_revision": f"sha256:{digest}"})

    runner.validate_config(config)

    mismatched = runner.EvalConfig(**{**base.__dict__, "dataset_revision": f"sha256:{'0' * 64}"})
    with pytest.raises(runner.ToolEvalGenerationError, match="does not match"):
        runner.validate_config(mismatched)


def test_hf_job_requires_exact_revisions_and_invokes_eval_runner() -> None:
    job = _load_module(JOB_PATH, "hf_job_tool_eval")
    source = JOB_PATH.read_text(encoding="utf-8")

    assert "# /// script" in source
    assert '"transformers==5.13.0"' in source
    assert job.MODEL_REPO == "spkc83/retail-bank-servicing-agent-9b"
    assert job.DATASET_REPO == "spkc83/retail-bank-servicing-alignment-sft"
    assert "cloud_generate_tool_eval.py" in source
    assert 'parser.add_argument("--model-repo", default=MODEL_REPO)' in source
    assert 'parser.add_argument("--dataset-repo", default=DATASET_REPO)' in source
    assert 'parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")' in source
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
    assert 'model_repo="${MODEL_REPO:-spkc83/retail-bank-servicing-agent-9b}"' in source
    assert 'dataset_repo="${DATASET_REPO:-spkc83/retail-bank-servicing-alignment-sft}"' in source
    assert '--model-repo "$model_repo"' in source
    assert '--dataset-repo "$dataset_repo"' in source
    assert 'dtype="${4:-fp16}"' in source
    assert '--dtype "$dtype"' in source
