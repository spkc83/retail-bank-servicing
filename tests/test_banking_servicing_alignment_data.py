from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import hello_slm.banking_servicing_alignment_data as alignment_data
import hello_slm.banking_tool_sft_data as tool_sft_data
from hello_slm.banking_servicing_alignment_data import (
    SCREENSHOT_HELDOUT_CURRENTS,
    build_coreference_shadow_gate,
    build_screenshot_regression_fixture,
    build_servicing_alignment_splits,
    load_base_sft_splits,
    validate_servicing_alignment_splits,
    write_servicing_alignment_dataset,
)
from hello_slm.banking_tool_sft_data import (
    BankingToolSftDataError,
    prepare,
    validate_banking_tool_sft_manifest,
    validate_records,
)


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
        "train": 1930,
        "validation": 218,
        "test": 35,
    }
    assert report["coreference_pair_counts"] == {
        "train": 608,
        "validation": 16,
        "test": 0,
    }
    assert report["duplicate_current_policy"] == (
        "only declared two-record coreference pairs with opposite targets; "
        "a normalized current may group multiple pairs only across distinct history forms"
    )
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
        "deictic_replace_action": 608,
        "deictic_replace_ambiguity": 608,
        "natural_social_style": 12,
        "missing_entity_clarification": 1,
        "v7_natural_greeting": 1,
        "v7_mortgage_policy_detour": 1,
        "v7_list_transfers": 1,
        "v7_grounded_selector": 1,
        "v7_selector_clarification": 3,
        "v7_tool_outcome": 2,
        "v7_list_transactions_limit": 20,
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


def test_v7_generation_contract_matches_runtime_tool_exposure_and_social_style() -> None:
    splits, report = build_servicing_alignment_splits()

    for split in ("train", "validation"):
        for row in splits[split]:
            contract = row["expected"]["generation_contract"]
            assert contract["version"] == "banking-v7-route-to-generation/v1"
            calls = row["expected"]["tool_calls"]
            if contract["mode"] == "execute_tool":
                assert contract["tool_names"] == [calls[0]["name"]]
                assert len({call["name"] for call in calls}) == 1
                assert contract["argument_constraints"] == {
                    name: {"const": value} for name, value in calls[0]["arguments"].items()
                }
            else:
                assert contract["tool_names"] == []
                assert contract["argument_constraints"] == {}
                assert calls == []

    entity_states = {
        row["expected"]["generation_contract"]["entity_state"]
        for split in ("train", "validation")
        for row in splits[split]
    }
    assert {"resolved", "missing", "ambiguous", "not_required"} <= entity_states

    social = [
        row
        for split in ("train", "validation")
        for row in splits[split]
        if row["metadata"]["scenario_family"] == "natural_social_style"
    ]
    assert len(social) == 16
    assert all(row["expected"]["generation_contract"]["mode"] == "converse" for row in social)
    assert all(not row["expected"]["generation_contract"]["tool_names"] for row in social)
    assert all("sorry" not in str(row["messages"][-1]["content"]).lower() for row in social)
    assert all(
        not str(row["messages"][-1]["content"]).startswith(
            (
                "Here is the current update:",
                "I reviewed the conversation",
                "For clarity,",
            )
        )
        for row in social
    )
    assert report["scenario_family_counts"]["train"]["natural_social_style"] == 12


def test_v7_split_isolated_granite_examples_cover_routing_and_arguments() -> None:
    splits, _report = build_servicing_alignment_splits()
    granite = {
        split: [row for row in rows if str(row["metadata"]["scenario_family"]).startswith("v7_")]
        for split, rows in splits.items()
    }

    assert granite["train"] and granite["validation"]
    assert not granite["test"]
    assert {row["metadata"]["scenario_family"] for row in granite["train"]} >= {
        "v7_natural_greeting",
        "v7_mortgage_policy_detour",
        "v7_list_transfers",
        "v7_list_transactions_limit",
        "v7_grounded_selector",
        "v7_selector_clarification",
        "v7_tool_outcome",
    }
    limits = {
        row["expected"]["tool_calls"][0]["arguments"]["limit"]
        for row in granite["train"]
        if row["metadata"]["scenario_family"] == "v7_list_transactions_limit"
    }
    assert limits == set(range(1, 21))
    prompts = "\n".join(_last_user(row).lower() for row in granite["train"])
    assert "one" in prompts and "20" in prompts
    assert any("mortgage" in _last_user(row).lower() for row in granite["train"])
    assert any(
        row["expected"]["tool_calls"] == [{"name": "list_transfers", "arguments": {}}]
        for row in granite["train"]
    )
    clarification_states = {
        row["expected"]["generation_contract"]["entity_state"]
        for row in granite["train"]
        if row["metadata"]["scenario_family"] == "v7_selector_clarification"
    }
    assert clarification_states == {"missing", "ambiguous", "ineligible"}
    train_text = {_normalize(_last_user(row)) for row in granite["train"]}
    validation_text = {_normalize(_last_user(row)) for row in granite["validation"]}
    assert train_text.isdisjoint(validation_text)


def test_v7_shadow_is_untouched_and_frozen_test_count_stays_215(tmp_path: Path) -> None:
    prepare(output_dir=tmp_path / "base", pilot_count=1200, split_seed=711)
    manifest = write_servicing_alignment_dataset(
        tmp_path / "alignment",
        base_sft_dir=tmp_path / "base",
    )

    test_entry = next(entry for entry in manifest["tool_sft"] if entry["name"] == "test")
    assert test_entry["record_count"] == 215
    gate = next(
        item for item in manifest["behavioral_gates"] if item["name"] == "granite-v7-shadow"
    )
    assert gate["trainable"] is False
    assert gate["allowed_use"] == ["checkpoint-selection", "generalization-evaluation"]
    shadow_rows = [
        json.loads(line)
        for line in (tmp_path / "alignment" / gate["path"]).read_text().splitlines()
    ]
    assert shadow_rows
    assert all(row["metadata"]["trainable"] is False for row in shadow_rows)
    governed = {
        _normalize(_last_user(row))
        for split, rows in build_servicing_alignment_splits()[0].items()
        if split in {"train", "validation"}
        for row in rows
    }
    assert governed.isdisjoint({_normalize(_last_user(row)) for row in shadow_rows})


def test_v7_screenshot_regression_fixture_has_nine_complete_isolated_cases(
    tmp_path: Path,
) -> None:
    fixture = build_screenshot_regression_fixture()
    assert len(fixture) == 9
    assert {row["metadata"]["trainable"] for row in fixture} == {False}
    assert {row["metadata"]["regression_only"] for row in fixture} == {True}
    assert {_normalize(row["current"]) for row in fixture} == {
        _normalize(current) for current in SCREENSHOT_HELDOUT_CURRENTS
    }
    assert all(row["history"] is not None for row in fixture)
    assert all(
        set(row["expected"])
        == {
            "route",
            "effective_action",
            "entity_state",
            "tool_name",
            "argument_constraints",
            "response_properties",
        }
        for row in fixture
    )

    prepare(output_dir=tmp_path / "base", pilot_count=1200, split_seed=711)
    manifest = write_servicing_alignment_dataset(
        tmp_path / "alignment", base_sft_dir=tmp_path / "base"
    )
    entry = manifest["evaluation_fixtures"][0]
    assert entry["name"] == "screenshot-regression"
    assert entry["record_count"] == 9
    assert entry["trainable"] is False
    assert entry["allowed_use"] == ["regression-evaluation"]


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
    assert _normalize("ok, thats the one i want to replace") not in train_currents
    assert _report["heldout_long_ngram_leaks_in_train"] == []


def test_coreference_curriculum_is_diverse_matched_and_split_disjoint() -> None:
    splits, _report = build_servicing_alignment_splits()
    curricula = {
        split: [
            row
            for row in splits[split]
            if row["metadata"]["scenario_family"].startswith("deictic_replace_")
        ]
        for split in ("train", "validation")
    }

    assert len(curricula["train"]) == 1216
    assert len(curricula["validation"]) == 32
    for split, rows in curricula.items():
        action = [row for row in rows if row["expected"]["path"] == "multi_turn"]
        ambiguous = [row for row in rows if row["expected"]["path"] == "clarification"]
        assert len(action) == len(ambiguous)
        assert len({row["metadata"]["coreference_phrase_family"] for row in action}) >= (
            12 if split == "train" else 4
        )
        assert {row["metadata"]["coreference_phrase_family"] for row in action} == {
            row["metadata"]["coreference_phrase_family"] for row in ambiguous
        }
        assert all(row["expected"]["tool_calls"][0]["name"] == "replace_card" for row in action)
        assert all(not row["expected"]["tool_calls"] for row in ambiguous)
        assert all(row["messages"][-1]["loss"] is True for row in action)
        assert all(
            next(
                message
                for message in row["messages"]
                if message["role"] == "assistant" and message.get("tool_calls")
            )["loss"]
            is True
            for row in action
        )
        assert all(
            "last four digits" in str(row["messages"][-1]["content"]).lower() for row in ambiguous
        )
        action_histories = {
            str(row["messages"][2]["content"])
            for row in action
            if row["messages"][2]["role"] == "assistant"
        }
        assert any(history.startswith("You have an active ") for history in action_histories)
        assert any(history.startswith("Cards on this profile: ") for history in action_histories)
        assert any(history.startswith("I found ") for history in action_histories)
        assert any(history.startswith("Card results: ") for history in action_histories)

        pairs: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            pairs.setdefault(row["metadata"]["coreference_pair_id"], []).append(row)
        assert len(pairs) == (608 if split == "train" else 16)
        for pair_rows in pairs.values():
            assert len(pair_rows) == 2
            assert {_last_user(row) for row in pair_rows} == {_last_user(pair_rows[0])}
            assert {row["metadata"]["coreference_target"] for row in pair_rows} == {
                "replace_card",
                "clarification",
            }
            assert {row["metadata"]["actionable_card_count"] for row in pair_rows} == {1, 2}

        assert "selected" not in json.dumps(rows, ensure_ascii=False).lower()
        pair_text = json.dumps(
            [
                {
                    "messages": row["messages"],
                    "coreference_entity_keys": row["metadata"]["coreference_entity_keys"],
                }
                for row in rows
            ],
            ensure_ascii=False,
        ).lower()
        assert "one active" not in pair_text
        assert "two active" not in pair_text
        assert "4821" not in pair_text
        assert "everyday visa debit" not in pair_text

        prompts = {str(row["metadata"]["coreference_prompt"]) for row in rows}
        assert any(prompt.startswith("i would like ") for prompt in prompts)
        assert any(
            " is the card to " in prompt or " is what i need" in prompt for prompt in prompts
        )
        assert any(_last_user(row) == row["metadata"]["coreference_prompt"] for row in rows)
        action_rows = [row for row in rows if row["expected"]["path"] == "multi_turn"]
        for prompt_form in range(4):
            same_form = [
                row
                for row in action_rows
                if row["metadata"]["coreference_prompt_form"] == prompt_form
            ]
            assert len({row["metadata"]["coreference_history_form"] for row in same_form}) == 4
            assert len({row["metadata"]["coreference_tier"] for row in same_form}) == 4
        for family in {row["metadata"]["coreference_phrase_family"] for row in action_rows}:
            same_family = [
                row for row in action_rows if row["metadata"]["coreference_phrase_family"] == family
            ]
            if split == "train":
                assert {
                    (
                        row["metadata"]["coreference_prompt_form"],
                        row["metadata"]["coreference_history_form"],
                    )
                    for row in same_family
                } == {(prompt, history) for prompt in range(4) for history in range(4)}
                assert len({row["metadata"]["coreference_product"] for row in same_family}) >= 8
                assert len({row["metadata"]["coreference_tier"] for row in same_family}) == 4
            else:
                assert len({row["metadata"]["coreference_product"] for row in same_family}) == 4

    train_entities = {
        entity
        for row in curricula["train"]
        for entity in row["metadata"]["coreference_entity_keys"]
    }
    validation_entities = {
        entity
        for row in curricula["validation"]
        for entity in row["metadata"]["coreference_entity_keys"]
    }
    assert train_entities.isdisjoint(validation_entities)
    train_phrases = {row["metadata"]["coreference_prompt"] for row in curricula["train"]}
    validation_phrases = {row["metadata"]["coreference_prompt"] for row in curricula["validation"]}
    assert train_phrases.isdisjoint(validation_phrases)


def test_duplicate_current_requires_one_declared_opposite_target_pair() -> None:
    splits, _report = build_servicing_alignment_splits()
    pair = [
        deepcopy(row)
        for row in splits["train"]
        if row["metadata"].get("coreference_pair_id") == "coreference-train-that-card-0-0"
    ]
    assert len(pair) == 2
    validate_records(pair)
    pair[1]["metadata"].pop("coreference_pair_id")

    with pytest.raises(ValueError, match="duplicate current user text"):
        alignment_data._assert_no_duplicate_current(pair, split="train")
    with pytest.raises(BankingToolSftDataError, match="duplicates normalized user text"):
        validate_records(pair)


def test_grouped_counterfactual_rejects_forged_history_metadata() -> None:
    splits, _report = build_servicing_alignment_splits()
    pair = [
        deepcopy(row)
        for row in splits["train"]
        if row["metadata"].get("coreference_pair_id") == "coreference-train-that-card-0-0"
    ]
    forged = deepcopy(pair)
    for index, row in enumerate(forged):
        row["record_id"] = f"forged-history-{index}"
        row["metadata"]["coreference_pair_id"] = "forged-history-pair"
        row["metadata"]["coreference_history_form"] = 1

    assert tool_sft_data._is_governed_counterfactual_group([*pair, *forged]) is False


def test_candidate5_preserves_all_215_test_behavior_fields_byte_equivalent() -> None:
    _base_manifest, base_splits = load_base_sft_splits()
    alignment_splits, _report = build_servicing_alignment_splits()
    rows = [*base_splits["test"], *alignment_splits["test"]]
    behavioral_fields = ("messages", "expected", "split_keys", "metadata")
    payload = [{field: row[field] for field in behavioral_fields} for row in rows]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    assert len(rows) == 215
    assert hashlib.sha256(encoded).hexdigest() == (
        "4ac64ad9177273edb19c0752f94b71da51337b388cffe8cd03b5a9d9718c186e"
    )


def test_alignment_rows_quarantine_unverified_replay_claims() -> None:
    splits, _report = build_servicing_alignment_splits()

    for record in (row for rows in splits.values() for row in rows):
        validation = record["validation"]
        assert validation["accepted"] is False
        assert validation["schema_accepted"] is True
        assert validation["replay_hash"] is None
        assert validation["replay_verified"] is False
        assert validation["final_state_verified"] is False


def test_candidate5_shadow_gate_is_untouched_and_isolated() -> None:
    splits, _report = build_servicing_alignment_splits()
    shadow = build_coreference_shadow_gate()

    assert len(shadow) == 32
    assert len({row["metadata"]["coreference_pair_id"] for row in shadow}) == 16
    assert {row["metadata"]["coreference_target"] for row in shadow} == {
        "replace_card",
        "clarification",
    }
    assert all(row["metadata"]["trainable"] is False for row in shadow)
    shadow_entities = {
        entity for row in shadow for entity in row["metadata"]["coreference_entity_keys"]
    }
    governed_entities = {
        entity
        for split in ("train", "validation")
        for row in splits[split]
        for entity in row["metadata"].get("coreference_entity_keys", ())
    }
    assert shadow_entities.isdisjoint(governed_entities)
    shadow_families = {row["metadata"]["coreference_phrase_family"] for row in shadow}
    governed_families = {
        row["metadata"]["coreference_phrase_family"]
        for split in ("train", "validation")
        for row in splits[split]
        if "coreference_phrase_family" in row["metadata"]
    }
    assert shadow_families.isdisjoint(governed_families)


def test_coreference_shadow_carries_runtime_generation_contracts() -> None:
    shadow = build_coreference_shadow_gate()

    for row in shadow:
        contract = row["expected"]["generation_contract"]
        if row["metadata"]["coreference_target"] == "replace_card":
            assert contract["mode"] == "execute_tool"
            assert contract["entity_state"] == "resolved"
            assert contract["tool_names"] == ["replace_card"]
            assert set(contract["argument_constraints"]) == {"last4"}
        else:
            assert contract["mode"] == "clarify"
            assert contract["entity_state"] == "ambiguous"
            assert contract["tool_names"] == []
            assert contract["argument_constraints"] == {}


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


def test_writer_outputs_manifest_and_governed_splits(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    prepare(output_dir=base_dir, pilot_count=120)
    manifest = write_servicing_alignment_dataset(tmp_path / "alignment", base_sft_dir=base_dir)

    assert manifest["name"] == "retail-bank-servicing-alignment-v5"
    assert manifest["schema_version"] == "banking-tool-sft/v1"
    assert manifest["generation_contract_version"] == "banking-v7-route-to-generation/v1"
    assert manifest["generation_contract_model_inputs"] == (
        "compatible tool schemas only; routing metadata is not rendered"
    )
    assert manifest["report"]["generation_contract_counts"]["test"] == {}
    assert manifest["report"]["alignment_split_counts"] == {
        "train": 1930,
        "validation": 218,
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
    assert manifest["behavioral_gates"][0] == {
        "name": "coreference-shadow",
        "path": "coreference-shadow.jsonl",
        "record_count": 32,
        "pair_count": 16,
        "sha256": manifest["behavioral_gates"][0]["sha256"],
        "bytes": manifest["behavioral_gates"][0]["bytes"],
        "allowed_use": ["post-selection-evaluation-once"],
        "trainable": False,
    }
    assert manifest["behavioral_gates"][1] == {
        "name": "granite-v7-shadow",
        "path": "granite-v7-shadow.jsonl",
        "record_count": 13,
        "sha256": manifest["behavioral_gates"][1]["sha256"],
        "bytes": manifest["behavioral_gates"][1]["bytes"],
        "allowed_use": ["checkpoint-selection", "generalization-evaluation"],
        "trainable": False,
        "gate_contract": "banking-v7-granite-predicted-e2e-gate/v1",
    }
    for entry in manifest["tool_sft"]:
        path = tmp_path / "alignment" / entry["path"]
        assert path.is_file()
        assert path.stat().st_size == entry["bytes"]
        assert len(path.read_text(encoding="utf-8").splitlines()) == entry["record_count"]
    report = json.loads((tmp_path / "alignment" / "preparation-report.json").read_text())
    assert report["pii_matches"] == 0
    assert report["heldout_exact_currents_in_train"] == []
    assert validate_banking_tool_sft_manifest(tmp_path / "alignment" / "manifest.json") == manifest

    train_path = tmp_path / "alignment" / "train.jsonl"
    original_train = train_path.read_bytes()
    lines = original_train.splitlines(keepends=True)
    lines[0], lines[1] = lines[1], lines[0]
    train_path.write_bytes(b"".join(lines))
    with pytest.raises(BankingToolSftDataError, match="train sha256 mismatch"):
        validate_banking_tool_sft_manifest(tmp_path / "alignment" / "manifest.json")
    mutated_train = bytearray(original_train)
    mutated_train[0] = ord("[") if mutated_train[0] != ord("[") else ord("{")
    train_path.write_bytes(mutated_train)
    with pytest.raises(BankingToolSftDataError, match="train sha256 mismatch"):
        validate_banking_tool_sft_manifest(tmp_path / "alignment" / "manifest.json")


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
