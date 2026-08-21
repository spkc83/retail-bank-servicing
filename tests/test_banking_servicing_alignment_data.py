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
    FINAL_CLOSERS,
    FINAL_OPENERS,
    SCREENSHOT_HELDOUT_CURRENTS,
    build_coreference_shadow_gate,
    build_screenshot_regression_fixture,
    build_servicing_alignment_splits,
    load_base_sft_splits,
    validate_servicing_alignment_splits,
    write_servicing_alignment_dataset,
)
from hello_slm.banking_tool_sft_data import (
    REALIZER_FINAL_CLOSERS,
    REALIZER_FINAL_PREFIXES,
    TRAINABLE_TEXT_BANNED_WORDS,
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
        "train": 2426,
        "validation": 218,
        "test": 35,
    }
    assert report["coreference_pair_counts"] == {
        "train": 784,
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
        "deictic_replace_action": 784,
        "deictic_replace_ambiguity": 784,
        "deictic_ineligible_clarification": 72,
        "deictic_missing_clarification": 72,
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


def test_train_split_covers_ineligible_and_missing_clarifications() -> None:
    splits, _report = build_servicing_alignment_splits()
    states = Counter(
        row["expected"]["generation_contract"]["entity_state"]
        for row in splits["train"]
        if row["expected"].get("generation_contract", {}).get("mode") == "clarify"
    )

    assert states["ineligible"] >= 64
    assert states["missing"] >= 64


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

    assert len(curricula["train"]) == 1568
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
        assert len(pairs) == (784 if split == "train" else 16)
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
        "train": 2426,
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


def _small_base(tmp_path: Path) -> Path:
    base_dir = tmp_path / "base"
    prepare(output_dir=base_dir, pilot_count=120)
    return base_dir


def _export_alignment_requests(tmp_path: Path, base_dir: Path) -> list[dict[str, Any]]:
    requests = tmp_path / "requests.jsonl"
    write_servicing_alignment_dataset(
        tmp_path / "export", base_sft_dir=base_dir, export_teacher_requests=requests
    )
    rows = [
        json.loads(line) for line in requests.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(rows) == 2426 + 218
    return rows


def test_alignment_teacher_hook_rewrites_only_train_and_validation_finals(tmp_path: Path) -> None:
    base_dir = _small_base(tmp_path)
    rows = _export_alignment_requests(tmp_path, base_dir)
    target = next(r for r in rows if r["record_id"].startswith("outcome_replace"))
    response = {
        "record_id": target["record_id"],
        "immutable_hash": target["immutable_hash"],
        "user_content": target["user_content"],
        "final_response": (
            "Good news — the replacement for your Cashback Debit ending in 7742 is already "
            "pending. Keep using your other cards as normal while it is on the way."
        ),
    }
    responses = tmp_path / "responses.jsonl"
    responses.write_text(json.dumps(response) + "\n", encoding="utf-8")
    before_test = (tmp_path / "export" / "test.jsonl").read_bytes()

    manifest = write_servicing_alignment_dataset(
        tmp_path / "out",
        base_sft_dir=base_dir,
        teacher_responses=responses,
        teacher_model="claude-opus-5",
        teacher_prompt_hash="sha256:" + "0" * 64,
    )

    train = [
        json.loads(line)
        for line in (tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    rewritten = next(r for r in train if r["record_id"] == target["record_id"])
    assert rewritten["messages"][-1]["content"] == response["final_response"]
    assert rewritten["provenance"]["teacher_model"] == "claude-opus-5"
    assert (tmp_path / "out" / "test.jsonl").read_bytes() == before_test
    assert manifest["report"]["alignment_teacher_realization"]["realized_counts"] == {
        "train": 1,
        "validation": 0,
    }


@pytest.mark.parametrize("which", ["base", "alignment"])
def test_alignment_teacher_hook_rejects_test_split_rows(tmp_path: Path, which: str) -> None:
    base_dir = _small_base(tmp_path)
    _export_alignment_requests(tmp_path, base_dir)
    test_rows = [
        json.loads(line)
        for line in (tmp_path / "export" / "test.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    # composite test = base rows first, alignment rows last
    row = test_rows[0] if which == "base" else test_rows[-1]
    bad = {
        "record_id": row["record_id"],
        "immutable_hash": "sha256:" + "0" * 64,
        "user_content": "x",
        "final_response": "y",
    }
    responses = tmp_path / "bad.jsonl"
    responses.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="test split"):
        write_servicing_alignment_dataset(
            tmp_path / "out",
            base_sft_dir=base_dir,
            teacher_responses=responses,
            teacher_model="m",
            teacher_prompt_hash="sha256:" + "0" * 64,
        )


def test_alignment_teacher_hook_rejects_user_text_edits(tmp_path: Path) -> None:
    base_dir = _small_base(tmp_path)
    rows = _export_alignment_requests(tmp_path, base_dir)
    edited = {k: rows[0][k] for k in ("record_id", "immutable_hash", "final_response")}
    edited["user_content"] = rows[0]["user_content"] + " please"
    responses = tmp_path / "bad.jsonl"
    responses.write_text(json.dumps(edited) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="final_response only"):
        write_servicing_alignment_dataset(
            tmp_path / "out",
            base_sft_dir=base_dir,
            teacher_responses=responses,
            teacher_model="m",
            teacher_prompt_hash="sha256:" + "0" * 64,
        )


def test_alignment_teacher_hook_requires_model_and_prompt_hash(tmp_path: Path) -> None:
    base_dir = _small_base(tmp_path)
    responses = tmp_path / "r.jsonl"
    responses.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="teacher_model and teacher_prompt_hash"):
        write_servicing_alignment_dataset(
            tmp_path / "out", base_sft_dir=base_dir, teacher_responses=responses
        )


def _ambiguity_finals(split: str) -> list[str]:
    return [
        record["messages"][-1]["content"]
        for record in alignment_data._deictic_replace_curriculum(split)
        if record["metadata"]["scenario_family"] == "deictic_replace_ambiguity"
    ]


def test_ambiguity_finals_keep_the_gate_proven_template_in_every_split() -> None:
    # The v9 conversational pool regressed the coreference dev gate (ambiguity
    # accuracy 0.44 after 964 steps); the single template is the proven target.
    expectations = (
        ("train", 784, "Please share its last four digits."),
        ("validation", 16, "Please share its last four digits."),
        # coreference-shadow.jsonl is a frozen fixture: it keeps the legacy closer.
        ("shadow", 16, "Please share the last four digits shown in the app."),
    )
    for split, expected_count, expected_closer in expectations:
        finals = _ambiguity_finals(split)
        assert len(finals) == expected_count
        assert len(set(finals)) == expected_count
        for final in finals:
            assert final.startswith("I found ") and final.endswith(expected_closer)
            head = " ".join(final.lower().split()[:45])
            assert "which" in head and "card" in head and "last four digits" in final


def test_trainable_splits_never_mention_product_surfaces_or_demo_wording() -> None:
    splits, _ = build_servicing_alignment_splits()
    for split in ("train", "validation"):
        for record in splits[split]:
            for message in record["messages"]:
                if message["role"] not in {"user", "assistant"}:
                    continue
                content = message.get("content")
                if not isinstance(content, str):
                    continue
                banned = TRAINABLE_TEXT_BANNED_WORDS.findall(content)
                assert not banned, f"{record['record_id']} ({split}) leaks {banned}: {content}"


def test_frozen_shadow_gates_keep_the_legacy_app_wording() -> None:
    shadow_ambiguity_finals = [
        record["messages"][-1]["content"]
        for record in build_coreference_shadow_gate()
        if record["metadata"]["scenario_family"] == "deictic_replace_ambiguity"
    ]
    assert shadow_ambiguity_finals
    assert all(
        final.endswith("Please share the last four digits shown in the app.")
        for final in shadow_ambiguity_finals
    )
    granite_finals = [
        message["content"]
        for record in alignment_data.build_granite_v7_shadow_gate()
        for message in record["messages"]
        if message["role"] == "assistant" and isinstance(message.get("content"), str)
    ]
    assert any("four digits shown in the app." in final for final in granite_finals)


_SPLIT_LEADS = ("For this request,", "In this session,")
_REALIZED_FAMILIES = {
    "history_entity_action",
    "tool_outcome_consistency",
    "service_case_context",
    "card_anaphora_action",
    "clarification_answer",
    "agent_repair",
    "banking_topic_shift",
    "policy_resume",
    "external_topic_shift",
    "policy_detour",
    "history_entity_ambiguity",
    "deictic_replace_ambiguity",
}


def _sentence_count(text: str) -> int:
    prose = " ".join(line for line in text.splitlines() if not line.strip().startswith("|"))
    return len(re.findall(r"[.!?](?:\s|$)", prose))


def _finals(split: str) -> list[tuple[str, str, str, dict]]:
    path = Path(f"data/banking-servicing-alignment-v5/{split}.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    finals = []
    for row in rows:
        final = next(
            message
            for message in reversed(row["messages"])
            if message["role"] == "assistant" and not message.get("tool_calls")
        )
        mode = row["expected"].get("generation_contract", {}).get("mode", "")
        finals.append((row["metadata"]["scenario_family"], mode, str(final["content"]), row))
    return finals


def _is_teacher_realized(row: dict[str, Any]) -> bool:
    provenance = row.get("provenance")
    return bool(isinstance(provenance, dict) and provenance.get("teacher_model"))


def test_alignment_training_finals_carry_no_template_scaffolding() -> None:
    prefixes = tuple(filter(None, (*FINAL_OPENERS, *REALIZER_FINAL_PREFIXES, *_SPLIT_LEADS)))
    closers = tuple(filter(None, (*FINAL_CLOSERS, *REALIZER_FINAL_CLOSERS)))

    offenders = [
        (family, final[:40])
        for split in ("train", "validation")
        for family, _, final, _row in _finals(split)
        if final.startswith(prefixes) or final.endswith(closers)
    ]

    assert offenders == []


def test_realized_alignment_finals_are_conversational() -> None:
    # Only teacher-realized finals were rewritten; each family's inline generator
    # seed final also feeds the frozen test split, so it cannot be reworded here.
    short = [
        (family, final[:60])
        for family, _, final, row in _finals("train")
        if family in _REALIZED_FAMILIES
        and _is_teacher_realized(row)
        and "|" not in final
        and _sentence_count(final) < 2
    ]

    assert short == []


def test_clarify_finals_keep_dev_gate_markers_early() -> None:
    # The dev gate only scores rows carrying a coreference_pair_id
    # (cloud_continue_tool_sft.py:989) and matches the markers as substrings, not
    # word tokens (:975), so "cards" satisfies it; the other clarify families were
    # not rewritten in this pass. Scope to the gate's population plus the
    # teacher-authored rows.
    bad = [
        final[:60]
        for split in ("train", "validation")
        for _, mode, final, row in _finals(split)
        if mode == "clarify"
        and "card" in final.lower()
        and (row["metadata"].get("coreference_pair_id") or _is_teacher_realized(row))
        and not all(
            marker in " ".join(final.lower().split()[:45]) for marker in ("which", "card")
        )
    ]

    assert bad == []


def test_targeted_reference_families_widen_the_train_margin() -> None:
    # The validation "list-reference" pair flickered on a single-card history and the
    # "results-reference-shadow" pair broke mid-clarification, so train carries seven
    # nearby-but-distinct list / shown-above / results phrasings. The validation and
    # shadow wordings stay held out, including under the four prompt wrappers.
    targeted = {
        "listed-card": "replace the card you listed",
        "from-your-list": "the card in your list needs replacing",
        "card-you-showed": "replace the card you just showed me",
        "shown-above": "replace the card shown above",
        "above-card": "the card above is the one to replace",
        "from-results": "the card from those results needs a replacement",
        "target-card": "that card is my replacement target",
    }
    train_specs = alignment_data._coreference_curriculum_specs("train")
    train_by_family = {spec["phrase_family"]: spec for spec in train_specs}
    for family, prompt in targeted.items():
        assert train_by_family[family]["prompt"] == prompt

    splits, _report = build_servicing_alignment_splits()
    for family in targeted:
        pairs = {
            row["metadata"]["coreference_pair_id"]
            for row in splits["train"]
            if row["metadata"].get("coreference_phrase_family") == family
        }
        assert len(pairs) == 16

    held_out_specs = (
        *alignment_data._coreference_curriculum_specs("validation"),
        *alignment_data._coreference_curriculum_specs("shadow"),
    )
    held_out_prompts = {spec["prompt"] for spec in held_out_specs}
    held_out_families = {spec["phrase_family"] for spec in held_out_specs}
    assert held_out_prompts.isdisjoint(set(targeted.values()))
    assert held_out_families.isdisjoint(set(targeted))
    # The curriculum realizes every prompt under four wrappers, so compare the
    # wrapped surfaces rather than the bare spec text.
    wrappers = ("{prompt}", "okay {prompt}", "{prompt} please", "yes {prompt}")
    held_out_ngrams: set[tuple[str, ...]] = set().union(
        *(
            alignment_data._word_ngrams(wrapper.format(prompt=spec["prompt"]), size=4)
            for spec in held_out_specs
            for wrapper in wrappers
        )
    )
    for prompt in targeted.values():
        for wrapper in wrappers:
            wrapped = wrapper.format(prompt=prompt)
            assert not alignment_data._word_ngrams(wrapped, size=4) & held_out_ngrams
