from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import hello_slm.banking_tool_sft_data as banking_tool_sft_data
from hello_slm.banking_tool_sft_data import (
    BANKING_TOOL_SFT_CONTRACT,
    TRAINABLE_TEXT_BANNED_WORDS,
    BankingToolSftDataError,
    export_teacher_realization_requests,
    generate_records,
    import_teacher_realizations,
    load_canonical_policy_corpus,
    main,
    normalized_user_text,
    prepare,
    public_tool_manifest,
    validate_banking_tool_sft_manifest,
    validate_records,
)

POC_PRESETS = {
    "Hello, how are you?",
    "yo, sup?",
    "Show my account balances.",
    "What happened with the money I sent recently?",
    "Show my five most recent transactions.",
    "What is the status of my debit card?",
    "My card was stolen. Freeze it.",
    "Please replace my debit card.",
    "I did not make the North Harbor Market purchase. Dispute it.",
    "Cancel the pending transfer to River Consulting.",
    "When was my mailing address changed?",
    "Can you help me open a mortgage account?",
    "What is the weather tomorrow?",
}


def _final(record: dict) -> str:
    return str(record["messages"][-1]["content"])


def _final_text(record: dict) -> str:
    for message in reversed(record["messages"]):
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            return str(message["content"])
    raise AssertionError(f"{record.get('record_id')} has no final assistant message")


def _sentence_count(text: str) -> int:
    prose = " ".join(line for line in text.splitlines() if not line.strip().startswith("|"))
    return len(re.findall(r"[.!?](?:\s|$)", prose))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _fuzzy_record(record_id: str, final: str) -> dict:
    return {
        "record_id": record_id,
        "metadata": {"scenario_family": "fuzzy-boundary"},
        "messages": [{"role": "assistant", "content": final}],
    }


def test_fuzzy_final_duplicate_upper_bound_preserves_threshold_behavior() -> None:
    with pytest.raises(BankingToolSftDataError, match="fuzzily duplicates"):
        banking_tool_sft_data._assert_no_fuzzy_final_duplicates(
            [_fuzzy_record("short", "a" * 198), _fuzzy_record("long", "a" * 199)]
        )
    with pytest.raises(BankingToolSftDataError, match="fuzzily duplicates"):
        banking_tool_sft_data._assert_no_fuzzy_final_duplicates(
            [
                _fuzzy_record("equal-a", ("a" * 199) + "b"),
                _fuzzy_record("equal-b", ("a" * 199) + "c"),
            ]
        )

    banking_tool_sft_data._assert_no_fuzzy_final_duplicates(
        [
            _fuzzy_record("below-a", ("a" * 198) + "b"),
            _fuzzy_record("below-b", ("a" * 198) + "c"),
        ]
    )


def test_generate_records_cover_tool_and_non_tool_contracts() -> None:
    records = generate_records(pilot_count=36, split_seed=9001)
    validate_records(records)

    assert len(records) == 36
    assert {record["schema_version"] for record in records} == {BANKING_TOOL_SFT_CONTRACT}

    manifest_names = {tool["function"]["name"] for tool in public_tool_manifest()}
    called_names = {
        call["function"]["name"]
        for record in records
        for message in record["messages"]
        for call in message.get("tool_calls", [])
    }
    assert called_names == manifest_names

    tool_messages = [
        message for record in records for message in record["messages"] if message["role"] == "tool"
    ]
    assert any(
        message["content"]["ok"] is True and "result" in message["content"]
        for message in tool_messages
    )
    assert any(
        message["content"]["ok"] is False and "error" in message["content"]
        for message in tool_messages
    )

    cases = {record["expected"]["path"] for record in records}
    assert {
        "tool_success",
        "tool_error",
        "clarification",
        "retrieval_grounded_policy",
        "ood",
        "hard_negative",
        "multi_turn",
    } <= cases

    assert any(
        any(
            message.get("loss") is False and message.get("tool_calls")
            for message in record["messages"]
        )
        and len(record["expected"]["ordered_calls"]) == 1
        for record in records
    )
    assert any(
        len([m for m in record["messages"] if m["role"] == "user"]) > 1 for record in records
    )


def test_v5_customer_facing_contract_is_grounded_varied_and_preset_free() -> None:
    records = generate_records(pilot_count=1200, split_seed=711)
    forbidden = ("demo", "synthetic", "mock", "test")

    assert "Harborlight Bank" in records[0]["messages"][0]["content"]
    assert "Harbor" in records[0]["messages"][0]["content"]
    assert not any(term in records[0]["messages"][0]["content"].lower() for term in forbidden)
    assert not any(
        term in tool["function"]["description"].lower()
        for tool in public_tool_manifest()
        for term in forbidden
    )
    assert not any(term in _final(record).lower() for record in records for term in forbidden)

    preset_keys = {normalized_user_text(prompt) for prompt in POC_PRESETS}
    train_users = {
        normalized_user_text(
            next(
                message["content"]
                for message in reversed(record["messages"])
                if message["role"] == "user"
            )
        )
        for record in records
        if record["metadata"]["split"] == "train"
    }
    assert not train_users & preset_keys

    policy_rows = [
        record for record in records if record["expected"]["path"] == "retrieval_grounded_policy"
    ]
    assert policy_rows
    for record in policy_rows:
        chunk_ids = record["expected"]["policy_citations"]
        assert len(chunk_ids) == 1
        assert f"[Policy: {chunk_ids[0]}]" in _final(record)
        assert any(
            message["role"] == "system" and chunk_ids[0] in str(message["content"])
            for message in record["messages"][1:-1]
        )

    normalized_finals = [normalized_user_text(_final(record)) for record in records]
    assert len(normalized_finals) == len(set(normalized_finals))
    assert len({_final(row) for row in policy_rows if row["metadata"]["split"] == "train"}) > 8

    assert any("| Account |" in _final(record) for record in records)
    assert any("| Date |" in _final(record) for record in records)
    emergency = next(row for row in records if row["record_id"] == "emergency_card_freeze")
    assert "sorry" in _final(emergency).lower()


def test_v7_generation_contract_has_exact_argument_constraints() -> None:
    records = generate_records(pilot_count=1200, split_seed=711)

    contracted = [row for row in records if "generation_contract" in row["expected"]]
    assert contracted
    for row in contracted:
        contract = row["expected"]["generation_contract"]
        calls = row["expected"]["tool_calls"]
        assert contract["version"] == "banking-v7-route-to-generation/v1"
        if contract["mode"] == "execute_tool":
            assert len(contract["tool_names"]) == 1
            assert {call["name"] for call in calls} == set(contract["tool_names"])
            assert contract["argument_constraints"] == {
                name: {"const": value} for name, value in calls[0]["arguments"].items()
            }
        else:
            assert contract["tool_names"] == []
            assert contract["argument_constraints"] == {}
            assert calls == []


def test_v7_social_targets_are_natural_and_multistage_is_single_tool() -> None:
    records = generate_records(pilot_count=1200, split_seed=711)
    train_and_validation = [
        row for row in records if row["metadata"]["split"] in {"train", "validation"}
    ]

    social = [
        row
        for row in records
        if row["metadata"]["split"] in {"train", "validation"}
        if row["metadata"]["scenario_family"]
        in {"small_talk_greeting", "small_talk_checkin", "conversational_thanks"}
    ]
    assert social
    assert all("sorry" not in _final(row).lower() for row in social)
    assert all(
        not _final(row).startswith(banking_tool_sft_data.REALIZER_FINAL_PREFIXES[1:])
        for row in social
    )

    multistage = [
        row for row in train_and_validation if "v6_multistage_alignment" in row["metadata"]
    ]
    assert multistage
    for row in multistage:
        assert row["expected"]["generation_contract"]["tool_names"] == ["freeze_card"]
        assert row["expected"]["generation_contract"]["argument_constraints"] == {
            "last4": {"const": row["expected"]["tool_calls"][0]["arguments"]["last4"]}
        }
        assistant_calls = [message for message in row["messages"] if message.get("tool_calls")]
        assert [message["loss"] for message in assistant_calls] == [False, True]
        assert assistant_calls[-1]["tool_calls"][0]["function"]["name"] == "freeze_card"


def test_v5_policy_rows_use_the_canonical_runtime_corpus_contract() -> None:
    corpus = load_canonical_policy_corpus()
    chunks = {chunk["chunk_id"]: chunk for chunk in corpus["chunks"]}
    records = generate_records(pilot_count=1200, split_seed=711)
    policy_rows = [
        record for record in records if record["expected"]["path"] == "retrieval_grounded_policy"
    ]

    assert {
        citation for row in policy_rows for citation in row["expected"]["policy_citations"]
    } == {
        "mortgage.opening.us.v1",
        "deposit.opening.us.v1",
        "deposit.overdraft.us.v1",
        "savings.interest.us.v1",
        "card.dispute.us.v1",
        "card.replacement.us.v1",
        "card.fraud.us.v1",
    }
    assert "mortgage.eligibility.age.us.v1" not in chunks
    for record in policy_rows:
        chunk_id = record["expected"]["policy_citations"][0]
        chunk = chunks[chunk_id]
        assert record["expected"]["policy_corpus_revision"] == corpus["corpus_revision"]
        assert record["expected"]["grounding_facts"] == chunk["required_claims"]
        assert record["expected"]["forbidden_facts"] == chunk["forbidden_claims"]
        assert chunk["answer"] in _final(record)


def test_tool_calls_have_stable_ids_typed_args_and_replay_hashes() -> None:
    first = generate_records(pilot_count=30, split_seed=711)
    second = generate_records(pilot_count=30, split_seed=711)

    assert second == first
    for record in first:
        ordered_calls = record["expected"]["ordered_calls"]
        call_ids = [
            call["id"]
            for message in record["messages"]
            if message.get("loss") is True
            for call in message.get("tool_calls", [])
        ]
        assert call_ids == ordered_calls
        for index, call_id in enumerate(call_ids):
            assert call_id == f"call_{record['record_id']}_{index}"
        for message in record["messages"]:
            for call in message.get("tool_calls", []):
                assert isinstance(call["function"]["arguments"], dict)
                assert call["index"] == 0
        assert record["validation"]["accepted"] is True
        assert record["validation"]["replay_verified"] is True
        assert record["validation"]["tool_manifest_hash"].startswith("sha256:")
        assert record["validation"]["replay_hash"].startswith("sha256:")
        if ordered_calls:
            assert record["validation"]["final_state_verified"] is True
            assert record["expected"]["final_state_hash"].startswith("sha256:")
        else:
            assert record["validation"]["final_state_verified"] is False


def test_default_split_seed_matches_the_published_release() -> None:
    assert generate_records(pilot_count=30) == generate_records(
        pilot_count=30,
        split_seed=711,
    )


def test_multi_call_plan_is_serialized_as_causal_tool_steps() -> None:
    record = next(
        record
        for record in generate_records(pilot_count=30, split_seed=711)
        if record["record_id"] == "multi_tool_freeze"
    )

    assert record["expected"]["ordered_calls"] == [
        "call_multi_tool_freeze_0",
        "call_multi_tool_freeze_1",
    ]
    assert [call["name"] for call in record["expected"]["tool_calls"]] == [
        "list_cards",
        "freeze_card",
    ]
    assert [message["role"] for message in record["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]

    first_call_message = record["messages"][2]
    first_tool_result = record["messages"][3]
    second_call_message = record["messages"][4]
    second_tool_result = record["messages"][5]
    assert first_call_message["tool_calls"][0]["index"] == 0
    assert second_call_message["tool_calls"][0]["index"] == 0

    call_counts = [
        len(message["tool_calls"]) for message in (first_call_message, second_call_message)
    ]
    assert call_counts == [1, 1]
    assert first_call_message["tool_calls"][0]["function"]["name"] == "list_cards"
    assert first_tool_result["tool_call_id"] == first_call_message["tool_calls"][0]["id"]
    second_function = second_call_message["tool_calls"][0]["function"]
    assert second_function["name"] == "freeze_card"
    assert second_function == record["expected"]["tool_calls"][1]
    assert second_tool_result["tool_call_id"] == second_call_message["tool_calls"][0]["id"]


def test_second_tool_call_arguments_are_observable_from_prior_tool_result() -> None:
    record = next(
        record
        for record in generate_records(pilot_count=30, split_seed=711)
        if record["record_id"] == "multi_tool_freeze"
    )
    first_tool_result = record["messages"][3]["content"]
    second_call = record["messages"][4]["tool_calls"][0]

    assert second_call["function"]["name"] == "freeze_card"
    second_last4 = second_call["function"]["arguments"]["last4"]
    assert isinstance(second_last4, str)
    assert second_last4 in json.dumps(first_tool_result, sort_keys=True)


def test_emergency_freeze_and_followups_cover_real_conversation_shapes() -> None:
    records = generate_records(pilot_count=64, split_seed=711)
    by_id = {record["record_id"]: record for record in records}

    emergency = by_id["emergency_card_freeze"]
    assert "stolen" in emergency["messages"][1]["content"].lower()
    assert [call["name"] for call in emergency["expected"]["tool_calls"]] == ["freeze_card"]
    assert emergency["messages"][2]["loss"] is False
    assert emergency["messages"][2]["tool_calls"][0]["function"]["name"] == "list_cards"
    discovered_last4 = emergency["messages"][5]["tool_calls"][0]["function"]["arguments"]["last4"]
    assert discovered_last4 in json.dumps(emergency["messages"][3]["content"])
    assert discovered_last4 in emergency["messages"][4]["content"]

    summary = by_id["action_summary_followup"]
    assert [message["role"] for message in summary["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "user",
        "assistant",
    ]
    assert summary["messages"][2]["loss"] is False
    assert summary["messages"][2]["tool_calls"][0]["function"]["name"] == "list_cards"
    assert summary["messages"][4]["loss"] is False
    assert summary["messages"][4]["tool_calls"][0]["function"]["name"] == "freeze_card"
    assert summary["messages"][4]["tool_calls"][0]["function"]["arguments"]["last4"] in (
        json.dumps(summary["messages"][3]["content"])
    )
    assert summary["messages"][6]["loss"] is False
    assert summary["messages"][-1]["loss"] is True
    assert summary["expected"]["requires_tool"] is False
    assert "what did you just do" in summary["messages"][-2]["content"].lower()
    assert "froze" in summary["messages"][-1]["content"].lower()

    no_action = by_id["no_action_followup"]
    assert no_action["expected"]["requires_tool"] is False
    assert "no card change was made" in no_action["messages"][-1]["content"].lower()


def test_faq_and_conversation_templates_cover_out_of_template_live_prompts() -> None:
    records = generate_records(pilot_count=64, split_seed=711)
    text = "\n".join(
        str(message["content"])
        for record in records
        for message in record["messages"]
        if isinstance(message.get("content"), str)
    ).lower()

    for marker in (
        "mortgage",
        "underwriting",
        "open a new savings account",
        "savings interest",
        "hello harbor",
        "thanks for the help",
    ):
        assert marker in text
    assert "based on the banking result" not in text
    assert "i checked that" not in text


def test_servicing_targets_include_requested_balances_without_canned_prefixes() -> None:
    records = generate_records(pilot_count=96, split_seed=711)
    account_rows = [
        record for record in records if record["metadata"]["scenario_family"] == "read_accounts"
    ]
    assert account_rows
    for record in account_rows:
        response = record["messages"][-1]["content"]
        facts = record["expected"]["grounding_facts"]
        assert "available" in response.lower()
        assert "current" in response.lower()
        assert sum(fact.startswith("account.balance=") for fact in facts) == 2
        assert all(
            fact.split("=", 1)[1] in response
            for fact in facts
            if fact.startswith("account.balance=")
        )
        assert not response.startswith(
            (
                "Done",
                "Here’s what I found",
                "Here is the current status",
                "I completed that",
                "Here’s the update",
            )
        )

    clarification_rows = [
        record
        for record in records
        if record["metadata"]["scenario_family"] == "clarification_card"
    ]
    assert clarification_rows
    assert all(
        "no other identifier" not in record["messages"][-1]["content"].lower()
        for record in clarification_rows
    )


def test_each_sequential_assistant_emission_restarts_tool_index_at_zero() -> None:
    record = next(
        record
        for record in generate_records(pilot_count=30, split_seed=711)
        if record["record_id"] == "multi_tool_freeze"
    )
    invalid = json.loads(json.dumps(record))
    invalid["messages"][4]["tool_calls"][0]["index"] = 1

    with pytest.raises(
        BankingToolSftDataError,
        match="tool call index must restart at zero",
    ):
        validate_records([invalid])


def test_prepare_writes_manifest_report_and_is_split_isolated(tmp_path: Path) -> None:
    report = prepare(output_dir=tmp_path / "tool-sft", pilot_count=120, split_seed=1234)
    second_report = prepare(output_dir=tmp_path / "tool-sft", pilot_count=120, split_seed=1234)

    assert second_report == report
    assert report["summary"]["total_records"] == 120
    assert report["checks"]["accepted_records"] == 120
    assert report["checks"]["tool_names_covered"] == sorted(
        tool["function"]["name"] for tool in public_tool_manifest()
    )
    assert report["source"] == {
        "name": "self-authored-synthetic",
        "license": "MIT",
        "trainable": True,
    }
    data_card = (tmp_path / "tool-sft" / "README.md").read_text(encoding="utf-8")
    assert "Retail Bank Agent Tool-Use SFT" in data_card
    assert "120 deterministic" in data_card
    assert "self-authored synthetic data" in data_card
    assert "never enter" in data_card
    assert "generative splits" in data_card

    manifest = validate_banking_tool_sft_manifest(tmp_path / "tool-sft" / "manifest.json")
    assert manifest["contract"] == "banking-tool-sft-manifest"
    assert manifest["generation_contract_version"] == "banking-v7-route-to-generation/v1"
    assert manifest["generation_contract_model_inputs"] == (
        "compatible tool schemas only; routing metadata is not rendered"
    )
    assert report["checks"]["generation_contract_records"] > 0
    assert {"execute_tool", "converse", "retrieve_policy", "refuse_ood"} <= set(
        report["checks"]["generation_modes"]
    )
    assert [entry["name"] for entry in manifest["tool_sft"]] == [
        "train",
        "validation",
        "test",
    ]

    semantic_split_by_group: dict[tuple[str, str, str, str], str] = {}
    for entry in manifest["tool_sft"]:
        assert entry["path"] == f"{entry['name']}.jsonl"
        rows = _read_jsonl(tmp_path / "tool-sft" / entry["path"])
        assert rows
        assert entry["record_count"] == len(rows)
        for row in rows:
            semantic_group = tuple(
                row["split_keys"][key]
                for key in (
                    "scenario_family",
                    "state_seed",
                    "customer_id",
                    "template_id",
                )
            )
            assert row["metadata"]["split_group"] == "|".join(semantic_group)
            assigned_split = semantic_split_by_group.setdefault(semantic_group, entry["name"])
            assert assigned_split == entry["name"]

    realized_groups = {
        tuple(
            row["split_keys"][key]
            for key in (
                "scenario_family",
                "state_seed",
                "customer_id",
                "template_id",
            )
        )
        for entry in manifest["tool_sft"]
        for row in _read_jsonl(tmp_path / "tool-sft" / entry["path"])
        if row["split_keys"]["realization_seed"] != "realization-000"
    }
    assert realized_groups


def test_pilot_realizer_uses_natural_text_and_varied_state_slots() -> None:
    records = generate_records(pilot_count=1800, split_seed=4321)
    user_keys = [
        normalized_user_text(
            next(
                message["content"]
                for message in reversed(record["messages"])
                if message["role"] == "user"
            )
        )
        for record in records
    ]

    assert len(user_keys) == 1800
    assert len(set(user_keys)) == 1800
    serialized_users = "\n".join(
        message["content"]
        for record in records
        for message in record["messages"]
        if message["role"] == "user"
    )
    assert "Use the " not in serialized_users
    assert " phrasing from the " not in serialized_users

    assert len({record["split_keys"]["state_seed"] for record in records}) >= 100
    write_values: dict[tuple[str, str], set[str]] = {
        ("freeze_card", "last4"): set(),
        ("replace_card", "last4"): set(),
        ("dispute_transaction", "description"): set(),
        ("cancel_transfer", "recipient"): set(),
    }
    for record in records:
        for message in record["messages"]:
            for call in message.get("tool_calls", []):
                name = call["function"]["name"]
                arguments = call["function"]["arguments"]
                for key in list(write_values):
                    if key[0] == name and key[1] in arguments:
                        write_values[key].add(arguments[key[1]])

    assert all(len(values) >= 20 for values in write_values.values())

    final_responses = [str(record["messages"][-1]["content"]).strip() for record in records]
    assert all(len(normalized_user_text(response).split()) >= 7 for response in final_responses)
    by_path = {
        path: [
            response
            for record, response in zip(records, final_responses, strict=True)
            if record["expected"]["path"] == path
        ]
        for path in {
            "clarification",
            "retrieval_grounded_policy",
            "ood",
            "hard_negative",
        }
    }
    assert all("last four digits" in response.lower() for response in by_path["clarification"])
    assert all(
        all(
            forbidden not in response.lower()
            for forbidden in ("account number", "customer id", "password", "pin")
        )
        for response in by_path["clarification"]
    )
    faq_template_markers = {
        "faq-overdraft-v1": "overdraft",
        "faq-mortgage-opening-v1": "mortgage",
        "faq-card-dispute-v1": "dispute",
        "faq-card-replacement-v1": "card",
        "faq-card-fraud-v1": "card",
        "faq-deposit-opening-v1": "account",
        "faq-savings-interest-v1": "interest",
    }
    for record, response in zip(records, final_responses, strict=True):
        if record["expected"]["path"] == "retrieval_grounded_policy":
            assert faq_template_markers[record["split_keys"]["template_id"]] in response.lower()
    assert all("retail banking" in response.lower() for response in by_path["ood"])
    assert all(
        "account number" in response.lower() and "customer id" in response.lower()
        for response in by_path["hard_negative"]
    )


def test_teacher_realization_round_trip_allows_only_wording_changes(tmp_path: Path) -> None:
    records = generate_records(pilot_count=30, split_seed=711)
    request_path = tmp_path / "teacher-requests.jsonl"
    response_path = tmp_path / "teacher-responses.jsonl"

    export_teacher_realization_requests(records, request_path)
    rows = _read_jsonl(request_path)
    rows[0]["user_content"] = "Please check the accounts on this signed-in profile."
    rows[0]["final_response"] = "I checked your profile and found your account summary."
    with response_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    realized = import_teacher_realizations(
        records,
        response_path,
        teacher_model="teacher-test",
        teacher_prompt_hash="sha256:" + "1" * 64,
    )

    assert realized[0]["messages"][1]["content"] == rows[0]["user_content"]
    assert realized[0]["messages"][-1]["content"] == rows[0]["final_response"]
    assert realized[0]["messages"][2:] != []
    assert realized[0]["expected"] == records[0]["expected"]
    assert realized[0]["provenance"]["teacher_model"] == "teacher-test"
    validate_records(realized)

    rows[0]["immutable_hash"] = "sha256:" + "0" * 64
    with response_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with pytest.raises(BankingToolSftDataError, match="teacher request hash mismatch"):
        import_teacher_realizations(
            records,
            response_path,
            teacher_model="teacher-test",
            teacher_prompt_hash="sha256:" + "1" * 64,
        )


def test_cli_exports_and_applies_teacher_realizations(tmp_path: Path) -> None:
    output_dir = tmp_path / "tool-sft"
    request_path = tmp_path / "teacher-requests.jsonl"
    response_path = tmp_path / "teacher-responses.jsonl"

    assert (
        main(
            [
                "--output-dir",
                str(output_dir),
                "--pilot-count",
                "30",
                "--export-teacher-requests",
                str(request_path),
            ]
        )
        == 0
    )
    rows = _read_jsonl(request_path)
    rows[0]["user_content"] = "Please summarize the accounts for this signed-in profile."
    rows[0]["final_response"] = "I checked the signed-in profile and found the account summary."
    with response_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    assert (
        main(
            [
                "--output-dir",
                str(output_dir),
                "--pilot-count",
                "30",
                "--teacher-responses",
                str(response_path),
                "--teacher-model",
                "teacher-cli-test",
                "--teacher-prompt-hash",
                "sha256:" + "2" * 64,
            ]
        )
        == 0
    )
    manifest = validate_banking_tool_sft_manifest(output_dir / "manifest.json")
    split_paths = [entry["path"] for entry in manifest["tool_sft"]]
    assert split_paths == ["train.jsonl", "validation.jsonl", "test.jsonl"]
    all_rows = [
        row for entry in manifest["tool_sft"] for row in _read_jsonl(output_dir / entry["path"])
    ]
    realized = next(row for row in all_rows if row["record_id"] == rows[0]["record_id"])
    assert realized["provenance"]["teacher_model"] == "teacher-cli-test"
    assert realized["messages"][-1]["content"] == rows[0]["final_response"]


def test_validator_rejects_private_or_unknown_tool_arguments() -> None:
    record = generate_records(pilot_count=30)[0]
    assistant = next(message for message in record["messages"] if message.get("tool_calls"))
    assistant["tool_calls"][0]["function"]["arguments"]["customer_id"] = "cust_alex"

    with pytest.raises(BankingToolSftDataError, match="unsupported arguments"):
        validate_records([record])


def test_validator_rejects_semantically_empty_final_response() -> None:
    record = generate_records(pilot_count=30)[0]
    record["messages"][-1]["content"] = "Done."

    with pytest.raises(BankingToolSftDataError, match="missing semantic content"):
        validate_records([record])


def test_validator_rejects_untrainable_final_assistant_response() -> None:
    record = generate_records(pilot_count=30)[0]
    record["messages"][-1]["loss"] = False

    with pytest.raises(BankingToolSftDataError, match="must be trainable"):
        validate_records([record])


def test_validator_rejects_internal_language_in_customer_facing_messages() -> None:
    record = generate_records(pilot_count=30)[0]
    record["messages"][0]["content"] = "You are an assistant for a demo bank."
    with pytest.raises(BankingToolSftDataError, match="system prompt leaks"):
        validate_records([record])

    record = generate_records(pilot_count=30)[0]
    record["messages"][-1]["content"] = (
        "The backend completed this request using an internal tool call successfully."
    )
    with pytest.raises(BankingToolSftDataError, match="final assistant response leaks"):
        validate_records([record])


def test_validator_rejects_product_wording_only_in_trainable_splits() -> None:
    def _train_record() -> dict:
        return next(
            record
            for record in generate_records(pilot_count=30)
            if record["metadata"]["split"] == "train"
        )

    record = _train_record()
    record["messages"][-1]["content"] += " Please share the last four digits shown in the app."
    with pytest.raises(BankingToolSftDataError, match="banned product wording"):
        validate_records([record])

    # The frozen evaluation splits keep their legacy wording, so they stay exempt.
    exempt = _train_record()
    exempt["messages"][-1]["content"] += " Please share the last four digits shown in the app."
    exempt["metadata"]["split"] = "test"
    validate_records([exempt])


def test_trainable_generated_records_carry_no_product_or_demo_wording() -> None:
    for record in generate_records(pilot_count=120):
        if record["metadata"]["split"] not in {"train", "validation"}:
            continue
        for message in record["messages"]:
            if message["role"] not in {"user", "assistant"}:
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            banned = TRAINABLE_TEXT_BANNED_WORDS.findall(content)
            assert not banned, f"{record['record_id']} leaks {banned}: {content}"


def test_training_finals_carry_no_realizer_scaffolding_after_teacher_pass() -> None:
    rows = _read_jsonl(Path("data/banking-v5-tool-sft/train.jsonl"))
    prefixes = tuple(filter(None, banking_tool_sft_data.REALIZER_FINAL_PREFIXES))
    closers = tuple(filter(None, banking_tool_sft_data.REALIZER_FINAL_CLOSERS))

    offenders = [
        row["record_id"]
        for row in rows
        for final in [_final_text(row)]
        if final.startswith(prefixes) or final.endswith(closers)
    ]

    assert offenders == []


def test_training_finals_are_diverse_without_scaffolding() -> None:
    rows = _read_jsonl(Path("data/banking-v5-tool-sft/train.jsonl"))
    finals = [_final_text(row) for row in rows]

    assert len(set(finals)) == len(finals)


def test_every_customer_can_answer_the_five_most_recent_preset() -> None:
    payload = json.loads(
        Path("poc/retail-bank-customer-service-poc/synthetic_bank.json").read_text()
    )

    for customer in payload["customers"]:
        assert len(customer["transactions"]) >= 5


def test_training_execute_tool_finals_are_conversational() -> None:
    rows = _read_jsonl(Path("data/banking-v5-tool-sft/train.jsonl"))

    short: list[tuple[str, str]] = []
    trailing: list[tuple[str, str]] = []
    for row in rows:
        if row["expected"].get("generation_contract", {}).get("mode") != "execute_tool":
            continue
        final = _final_text(row)
        if "|" not in final:
            if _sentence_count(final) < 2:
                short.append((row["record_id"], final[:60]))
            continue
        if _sentence_count(final.split("|", 1)[0]) < 1:
            short.append((row["record_id"], final[:60]))
        lines = final.splitlines()
        last_row = max(index for index, line in enumerate(lines) if line.strip().startswith("|"))
        if any(line.strip() for line in lines[last_row + 1 :]):
            trailing.append((row["record_id"], final[:60]))

    assert short == []
    assert trailing == []
