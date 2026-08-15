from __future__ import annotations

import copy
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from hello_slm.banking_counterfactual_eval_data import (
    COUNTERFACTUAL_MANIFEST_CONTRACT,
    CounterfactualEvalDataError,
    audit_counterfactual_records,
    build_counterfactual_records,
    counterfactual_gate_failures,
    validate_counterfactual_manifest,
    validate_counterfactual_records,
    write_counterfactual_benchmark,
)
from hello_slm.banking_tool_eval import (
    StaticPredictionModel,
    TaggedJsonToolAdapter,
    evaluate_records,
)

EVAL_RUNNER_PATH = Path("scripts/retail_bank/cloud_generate_tool_eval.py")


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_module(EVAL_RUNNER_PATH, "counterfactual_eval_runner")


def _canonical_results(record: dict[str, Any]) -> list[dict[str, Any]]:
    return runner.canonical_tool_results(record)


def test_records_are_deterministic_evaluation_only_and_cover_pairs() -> None:
    first = build_counterfactual_records()
    second = build_counterfactual_records()

    assert first == second
    assert len(first) >= 18
    assert all(record["metadata"]["split"] == "test" for record in first)
    assert all(record["metadata"]["trainable"] is False for record in first)
    assert all(record["validation"]["accepted"] is False for record in first)
    assert all(record["validation"]["replay_hash"] is None for record in first)
    assert all(record["validation"]["replay_verified"] is False for record in first)
    assert all(record["validation"]["final_state_verified"] is False for record in first)
    assert all(
        record["provenance"]["generator_version"] == "banking-counterfactual-eval/v1"
        for record in first
    )

    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in first:
        pair_id = record["metadata"].get("counterfactual_pair_id")
        if pair_id:
            pairs[str(pair_id)].append(record)

    assert len(pairs) >= 5
    for pair_id, variants in pairs.items():
        assert len(variants) == 2, pair_id
        assert {record["metadata"]["counterfactual_variant"] for record in variants} == {
            "a",
            "b",
        }
        assert runner.first_phase_messages(variants[0]) == runner.first_phase_messages(variants[1])
        assert _canonical_results(variants[0]) != _canonical_results(variants[1])
        first_prompt = json.dumps(runner.first_phase_messages(variants[0]), sort_keys=True)
        for record in variants:
            for fact in record["metadata"]["varied_facts"]:
                assert str(fact) not in first_prompt


def test_records_are_compatible_with_two_phase_evaluator() -> None:
    records = build_counterfactual_records()

    for record in records:
        first_phase = runner.first_phase_messages(record)
        assert first_phase
        if record["expected"]["requires_tool"]:
            assert len(_canonical_results(record)) == len(record["expected"]["tool_calls"])
            assert all(result["role"] == "tool" for result in _canonical_results(record))
        else:
            assert _canonical_results(record) == []


def test_writer_emits_strict_test_only_manifest_and_deterministic_files(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "counterfactual"
    first = write_counterfactual_benchmark(output_dir)
    first_bytes = {
        name: (output_dir / name).read_bytes()
        for name in ("manifest.json", "preparation-report.json", "test.jsonl", "README.md")
    }
    second = write_counterfactual_benchmark(output_dir)

    assert first == second
    assert first["contract"] == COUNTERFACTUAL_MANIFEST_CONTRACT
    assert first["training_allowed"] is False
    assert first["allowed_use"] == ["counterfactual-evaluation"]
    assert set(first["splits"]) == {"test"}
    assert first["splits"]["test"]["allowed_use"] == ["counterfactual-evaluation"]
    assert first_bytes == {name: (output_dir / name).read_bytes() for name in first_bytes}

    validated = validate_counterfactual_manifest(output_dir / "manifest.json")
    assert validated == first


def test_manifest_rejects_training_splits_and_trainable_records(tmp_path: Path) -> None:
    output_dir = tmp_path / "counterfactual"
    manifest = write_counterfactual_benchmark(output_dir)

    bad_split = copy.deepcopy(manifest)
    bad_split["splits"]["train"] = dict(bad_split["splits"]["test"])
    (output_dir / "manifest.json").write_text(json.dumps(bad_split), encoding="utf-8")
    with pytest.raises(CounterfactualEvalDataError, match="only the test split"):
        validate_counterfactual_manifest(output_dir / "manifest.json")

    manifest = write_counterfactual_benchmark(output_dir)
    records = [json.loads(line) for line in (output_dir / "test.jsonl").read_text().splitlines()]
    records[0]["metadata"]["trainable"] = True
    with pytest.raises(CounterfactualEvalDataError, match="must be non-trainable"):
        validate_counterfactual_records(records)


def test_contamination_gate_rejects_known_training_prompt_and_poc_fact() -> None:
    records = build_counterfactual_records()
    leaked_prompt = copy.deepcopy(records)
    leaked_prompt[0]["messages"][1]["content"] = "yo sup"

    with pytest.raises(CounterfactualEvalDataError, match="training user-text overlap"):
        validate_counterfactual_records(leaked_prompt)

    leaked_fact = copy.deepcopy(records)
    leaked_fact[0]["metadata"]["varied_facts"].append("4821")

    with pytest.raises(CounterfactualEvalDataError, match="POC fact overlap"):
        validate_counterfactual_records(leaked_fact)


def test_contamination_gate_matches_formatted_money_to_integer_cents(
    tmp_path: Path,
) -> None:
    records = build_counterfactual_records()
    training_row = {
        "record_id": "amount_only_training_row",
        "messages": [
            {"role": "user", "content": "A deliberately unrelated training prompt."},
            {"role": "assistant", "content": "A deliberately unrelated final answer."},
        ],
        "split_keys": {"template_id": "unrelated-training-template"},
        "hidden_state": {"amount_cents": 843127},
    }
    stage1 = tmp_path / "stage1.jsonl"
    stage2 = tmp_path / "stage2.jsonl"
    encoded = json.dumps(training_row) + "\n"
    stage1.write_text(encoded, encoding="utf-8")
    stage2.write_text(encoded, encoding="utf-8")
    poc_app = tmp_path / "app.py"
    poc_app.write_text("LABEL = 'unrelated poc label'\n", encoding="utf-8")
    poc_bank = tmp_path / "bank.json"
    poc_bank.write_text('{"customers": []}\n', encoding="utf-8")

    with pytest.raises(CounterfactualEvalDataError, match="training fact overlap"):
        audit_counterfactual_records(
            records,
            stage1_train=stage1,
            stage2_train=stage2,
            poc_app=poc_app,
            poc_bank=poc_bank,
        )


def test_pair_validator_rejects_fact_visible_before_tool_result() -> None:
    records = build_counterfactual_records()
    paired = next(record for record in records if record["metadata"].get("counterfactual_pair_id"))
    leaked = copy.deepcopy(records)
    target = next(record for record in leaked if record["record_id"] == paired["record_id"])
    fact = target["metadata"]["varied_facts"][0]
    target["messages"][1]["content"] += f" The hidden value is {fact}."

    with pytest.raises(CounterfactualEvalDataError, match="visible before its tool result"):
        validate_counterfactual_records(leaked)


def test_counterfactual_gate_accepts_reference_outputs_and_rejects_pair_swap() -> None:
    records = build_counterfactual_records()
    outputs = {record["record_id"]: _reference_output(record) for record in records}
    report = evaluate_records(
        records,
        model=StaticPredictionModel(outputs),
        adapter=TaggedJsonToolAdapter(),
        checkpoint_revision="a" * 40,
    )

    assert counterfactual_gate_failures(report, records) == []
    assert report["metrics"]["executable_tool_success"]["denominator"] == 0

    paired = [
        record
        for record in records
        if record["metadata"].get("counterfactual_pair_id") == "accounts-returned-facts"
    ]
    outputs[paired[0]["record_id"]] = _reference_output(paired[1])
    swapped = evaluate_records(
        records,
        model=StaticPredictionModel(outputs),
        adapter=TaggedJsonToolAdapter(),
        checkpoint_revision="a" * 40,
    )

    failures = counterfactual_gate_failures(swapped, records)
    assert any("grounded_final_factuality" in failure for failure in failures)
    assert any("counterfactual grounding failed" in failure for failure in failures)


def _reference_output(record: dict[str, Any]) -> str:
    calls = record["expected"]["tool_calls"]
    final = next(
        message["content"]
        for message in reversed(record["messages"])
        if message["role"] == "assistant" and isinstance(message.get("content"), str)
    )
    encoded_calls = "\n".join(
        "<tool_call>"
        + json.dumps(
            {"name": call["name"], "arguments": call.get("arguments", {})},
            separators=(",", ":"),
        )
        + "</tool_call>"
        for call in calls
    )
    return "\n".join(part for part in (encoded_calls, str(final)) if part)
