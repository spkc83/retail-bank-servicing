from __future__ import annotations

import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

from hello_slm.banking_servicing_alignment_data import (
    SCREENSHOT_HELDOUT_CURRENTS,
    build_servicing_alignment_splits,
    validate_servicing_alignment_splits,
    write_servicing_alignment_dataset,
)
from hello_slm.banking_tool_sft_data import prepare, validate_banking_tool_sft_manifest


def _last_user(record: dict[str, Any]) -> str:
    for message in reversed(record["messages"]):
        if message["role"] == "user":
            return str(message["content"])
    raise AssertionError("missing user message")


def _normalize(text: str) -> str:
    return " ".join(
        "".join(character.lower() if character.isalnum() else " " for character in text).split()
    )


def _load_preparation_module() -> ModuleType:
    path = Path("scripts/retail_bank/prepare_servicing_alignment_data.py")
    spec = importlib.util.spec_from_file_location(
        "prepare_servicing_alignment_data",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_servicing_alignment_records_validate_and_cover_failure_modes() -> None:
    splits, report = build_servicing_alignment_splits()

    validate_servicing_alignment_splits(splits)
    assert report["split_counts"] == {
        "train": 672,
        "validation": 168,
        "test": 35,
    }
    train_families = Counter(record["metadata"]["scenario_family"] for record in splits["train"])
    assert train_families == {
        "service_case_context": 64,
        "card_anaphora_action": 64,
        "clarification_answer": 64,
        "agent_repair": 64,
        "external_topic_shift": 32,
        "banking_topic_shift": 32,
        "policy_detour": 32,
        "policy_resume": 32,
        "history_entity_action": 128,
        "history_entity_ambiguity": 32,
        "tool_outcome_consistency": 128,
    }
    service_case_records = [
        record
        for record in splits["train"]
        if record["metadata"]["scenario_family"] == "service_case_context"
    ]
    assert service_case_records
    for record in service_case_records:
        assert record["expected"]["tool_calls"] == [{"name": "list_service_cases", "arguments": {}}]
        final_text = record["messages"][-1]["content"]
        assert "2026-06-18" in final_text
        assert "address_update" in final_text
        assert "Confirm mailing address update" in final_text

    created_at_records = [
        record
        for record in service_case_records
        if str(record["record_id"]).startswith("svc_case_created_")
    ]
    assert created_at_records
    assert all(
        record["expected"]["grounding_facts"] == ["case.created_at=2026-06-18T14:00:00Z"]
        for record in created_at_records
    )
    ood_records = [
        record
        for records in splits.values()
        for record in records
        if record["expected"]["path"] == "ood"
    ]
    assert ood_records
    assert all(record["expected"]["grounding_facts"] == [] for record in ood_records)


def test_remediation_examples_cover_coreference_ambiguity_and_tool_outcomes() -> None:
    splits, _report = build_servicing_alignment_splits()

    for split in ("train", "validation"):
        history_actions = [
            record
            for record in splits[split]
            if record["metadata"]["scenario_family"] == "history_entity_action"
        ]
        assert history_actions
        assert {
            call["name"] for row in history_actions for call in row["expected"]["tool_calls"]
        } >= {
            "replace_card",
            "freeze_card",
            "cancel_transfer",
            "dispute_transaction",
        }
        history_text = json.dumps(history_actions, ensure_ascii=False).lower()
        assert "bright meadow" not in history_text
        assert "4821" not in history_text

        ambiguous = [
            record
            for record in splits[split]
            if record["metadata"]["scenario_family"] == "history_entity_ambiguity"
        ]
        assert ambiguous
        assert all(row["expected"]["path"] == "clarification" for row in ambiguous)
        assert all(not row["expected"]["tool_calls"] for row in ambiguous)
        assert all(
            "last four digits" in str(row["messages"][-1]["content"]).lower() for row in ambiguous
        )

        outcomes = [
            record
            for record in splits[split]
            if record["metadata"]["scenario_family"] == "tool_outcome_consistency"
        ]
        assert outcomes
        assert {row["expected"]["path"] for row in outcomes} == {"tool_success", "tool_error"}
        for row in outcomes:
            envelopes = [
                message["content"] for message in row["messages"] if message["role"] == "tool"
            ]
            assert envelopes
            final = str(row["messages"][-1]["content"]).lower()
            if row["expected"]["path"] == "tool_error":
                assert all(envelope["ok"] is False for envelope in envelopes)
                assert any(token in final for token in ("could not", "was not"))
                assert any(token in final for token in ("no ", "not ", "unchanged"))
            else:
                assert all(envelope["ok"] is True for envelope in envelopes)

    train_remediation = [
        row
        for row in splits["train"]
        if row["metadata"]["scenario_family"].startswith(("history_entity", "tool_outcome"))
    ]
    validation_remediation = [
        row
        for row in splits["validation"]
        if row["metadata"]["scenario_family"].startswith(("history_entity", "tool_outcome"))
    ]
    train_last4 = set(re.findall(r"\b\d{4}\b", json.dumps(train_remediation)))
    validation_last4 = set(re.findall(r"\b\d{4}\b", json.dumps(validation_remediation)))
    assert train_last4.isdisjoint(validation_last4)


def test_exact_screenshot_currents_are_held_out_from_training() -> None:
    splits, _report = build_servicing_alignment_splits()
    heldout = {_normalize(text) for text in SCREENSHOT_HELDOUT_CURRENTS}
    train_currents = {_normalize(_last_user(record)) for record in splits["train"]}
    test_currents = {_normalize(_last_user(record)) for record in splits["test"]}

    assert not train_currents & heldout
    assert {
        "when was that created",
        "ok thats the one i want to replace",
        "what about the weather there",
    } <= test_currents


def test_v5_alignment_adds_policy_detour_resume_and_unique_targets() -> None:
    splits, _report = build_servicing_alignment_splits()
    train = splits["train"]
    families = {record["metadata"]["scenario_family"] for record in train}
    assert {"policy_detour", "policy_resume"} <= families

    detour = next(
        record for record in train if record["metadata"]["scenario_family"] == "policy_detour"
    )
    resume = next(
        record for record in train if record["metadata"]["scenario_family"] == "policy_resume"
    )
    assert detour["expected"]["path"] == "retrieval_grounded_policy"
    assert detour["expected"]["policy_citations"]
    assert "[" in detour["messages"][-1]["content"]
    assert any(
        "continue" in str(message["content"]).lower()
        for message in resume["messages"]
        if message["role"] == "user"
    )
    assert resume["expected"]["requires_tool"] is True

    finals = [
        _normalize(str(record["messages"][-1]["content"]))
        for records in splits.values()
        for record in records
    ]
    assert len(finals) == len(set(finals))


def test_writer_outputs_manifest_and_schema_valid_splits(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    prepare(output_dir=base_dir, pilot_count=120)
    manifest = write_servicing_alignment_dataset(tmp_path / "alignment", base_sft_dir=base_dir)

    assert manifest["name"] == "retail-bank-servicing-alignment-v5"
    assert manifest["schema_version"] == "banking-tool-sft/v1"
    assert manifest["report"]["alignment_split_counts"] == {
        "train": 672,
        "validation": 168,
        "test": 35,
    }
    base_counts = manifest["report"]["base_split_counts"]
    assert sum(base_counts.values()) == 120
    assert manifest["report"]["split_counts"] == {
        split: base_counts[split] + manifest["report"]["alignment_split_counts"][split]
        for split in ("train", "validation", "test")
    }
    assert {entry["name"] for entry in manifest["tool_sft"]} == {
        "train",
        "validation",
        "test",
    }
    for entry in manifest["tool_sft"]:
        path = tmp_path / "alignment" / entry["path"]
        assert path.is_file()
        assert path.stat().st_size == entry["bytes"]
        assert len(path.read_text(encoding="utf-8").splitlines()) == entry["record_count"]
    validate_banking_tool_sft_manifest(tmp_path / "alignment" / "manifest.json")

    report = json.loads((tmp_path / "alignment" / "preparation-report.json").read_text())
    assert report["pii_matches"] == 0
    assert report["heldout_exact_currents_in_train"] == []


def test_release_lock_detects_split_drift(tmp_path: Path) -> None:
    preparation = _load_preparation_module()
    base_dir = tmp_path / "base"
    prepare(output_dir=base_dir, pilot_count=120)
    manifest = write_servicing_alignment_dataset(tmp_path / "dataset", base_sft_dir=base_dir)
    lock = {
        "base_manifest_sha256": manifest["report"]["base_manifest_sha256"],
        "prepared_split_sha256": {entry["name"]: entry["sha256"] for entry in manifest["tool_sft"]},
    }
    good_lock = tmp_path / "good.lock.json"
    good_lock.write_text(json.dumps(lock), encoding="utf-8")
    preparation.verify_release_lock(manifest, good_lock)
    lock["prepared_split_sha256"]["train"] = "0" * 64
    bad_lock = tmp_path / "bad.lock.json"
    bad_lock.write_text(json.dumps(lock), encoding="utf-8")

    try:
        preparation.verify_release_lock(manifest, bad_lock)
    except ValueError as error:
        assert "split digests drifted" in str(error)
    else:
        raise AssertionError("drifted lock should fail")
