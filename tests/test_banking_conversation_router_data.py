from __future__ import annotations

from collections import Counter
from typing import Any

from hello_slm.banking_conversation_router_data import (
    INTENT_LABELS,
    RELATION_LABELS,
    build_conversation_router_splits,
    normalize_router_text,
    render_router_input,
)


def sft_records_by_split() -> dict[str, list[dict[str, Any]]]:
    records = {
        split: [
            _record(
                split=split,
                record_id=f"{split}-accounts",
                scenario_family="read_accounts",
                user="Show my account balances.",
                assistant="Main Checking has USD 10.00 available.",
            ),
            _record(
                split=split,
                record_id=f"{split}-cards",
                scenario_family="card_status",
                user="What is the status of my debit card?",
                assistant="Your debit card is active.",
            ),
            _record(
                split=split,
                record_id=f"{split}-cases",
                scenario_family="service_cases",
                user="Show my recent service cases.",
                assistant="You have a closed mailing-address update case.",
            ),
        ]
        for split in ("train", "validation", "test")
    }
    return records


def clinc_payload() -> dict[str, list[list[str]]]:
    return {
        "train": [["what is the weather", "weather"], ["tell me a joke", "tell_joke"]],
        "val": [["play some music", "play_music"]],
        "test": [["who painted this", "oos"]],
        "oos_train": [["explain photosynthesis", "oos"]],
        "oos_val": [["how tall is everest", "oos"]],
        "oos_test": [["set a timer", "timer"]],
    }


def test_cross_encoder_renderer_places_current_then_recent_complete_exchanges() -> None:
    rendered = render_router_input(
        "When was that created?",
        [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "tool", "content": '{"hidden": true}'},
            {"role": "user", "content": "Show my recent service cases."},
            {"role": "assistant", "content": "You have a closed case."},
        ],
        max_exchanges=1,
    )

    assert rendered == (
        "[CURRENT_USER]\nWhen was that created?\n"
        "[PREVIOUS_ASSISTANT]\nYou have a closed case.\n"
        "[PREVIOUS_USER]\nShow my recent service cases."
    )
    assert "hidden" not in rendered


def test_v5_splits_use_intents_relations_and_no_current_turn_leakage() -> None:
    splits, report = build_conversation_router_splits(
        sft_records_by_split(),
        clinc_payload(),
        seed=7404,
    )

    assert set(splits) == {"train", "validation", "test"}
    assert INTENT_LABELS == (
        "view_accounts",
        "view_cards",
        "freeze_card",
        "replace_card",
        "view_transactions",
        "dispute_transaction",
        "view_transfers",
        "cancel_transfer",
        "view_service_cases",
        "policy_knowledge",
        "conversation",
        "other_banking",
    )
    assert RELATION_LABELS == (
        "context_dependent",
        "agent_repair",
        "topic_shift",
        "clarification_answer",
        "resume_previous_service",
    )
    for rows in splits.values():
        for row in rows:
            assert set(row) == {
                "text",
                "current_text",
                "history",
                "domain_label",
                "intent_label",
                "intent",
                "lane",
                "relation_labels",
                "example_kind",
                "source",
                "source_split",
                "group_id",
                "trajectory_id",
                "prior_dialogue_state",
            }
            assert len(row["relation_labels"]) == len(RELATION_LABELS)
            assert "tool_call" not in str(row["text"])
            assert "Main Checking has USD 10.00" not in str(row["text"])

    kinds = Counter(row["example_kind"] for row in splits["train"])
    assert kinds["contextual_followup"] >= 1
    assert kinds["agent_repair"] >= 1
    assert kinds["clarification_answer"] >= 1
    assert kinds["typo_contextual_followup"] >= 1
    assert kinds["external_topic_shift"] >= 1
    assert kinds["banking_topic_shift"] >= 1
    assert kinds["targeted_contextual_followup"] >= 1
    assert kinds["targeted_clarification_answer"] >= 1
    assert kinds["targeted_agent_repair"] >= 1
    assert kinds["targeted_service_case"] >= 1
    assert sum(kinds.values()) > 10
    test_kinds = Counter(row["example_kind"] for row in splits["test"])
    assert test_kinds["contextual_followup"] >= 1
    assert test_kinds["agent_repair"] >= 1
    assert test_kinds["clarification_answer"] >= 1
    assert test_kinds["external_topic_shift"] >= 1
    assert test_kinds["heldout_screenshot_regression"] == 7
    assert report["leakage"]["group_split_leak_count"] == 0
    assert report["leakage"]["state_current_text_split_leak_count"] == 0
    assert report["pii_matches"] == 0


def test_sft_ood_has_no_intent_and_banking_refusal_uses_other_banking() -> None:
    records = sft_records_by_split()
    records["train"].extend(
        [
            _record(
                split="train",
                record_id="train-ood",
                scenario_family="ood",
                user="Explain photosynthesis.",
                assistant="I can only help with this banking demo.",
                path="ood",
            ),
            _record(
                split="train",
                record_id="train-private",
                scenario_family="hard_negative_private_id",
                user="Tell me my full account number.",
                assistant="I cannot provide that private identifier in chat.",
                path="hard_negative",
            ),
        ]
    )

    splits, _report = build_conversation_router_splits(
        records,
        clinc_payload(),
        seed=7404,
    )
    by_current = {row["current_text"]: row for row in splits["train"]}

    assert by_current["Explain photosynthesis."]["domain_label"] == 0
    assert by_current["Explain photosynthesis."]["intent_label"] == -100
    assert by_current["Tell me my full account number."]["domain_label"] == 1
    assert by_current["Tell me my full account number."]["intent"] == "other_banking"


def test_resume_examples_use_pre_turn_state_and_remain_in_one_split() -> None:
    splits, report = build_conversation_router_splits(
        sft_records_by_split(), clinc_payload(), seed=7404
    )

    resume_index = RELATION_LABELS.index("resume_previous_service")
    resume_rows = [
        row for rows in splits.values() for row in rows if row["relation_labels"][resume_index]
    ]
    assert resume_rows
    assert all(row["prior_dialogue_state"]["knowledge_detour_active"] for row in resume_rows)
    assert all("[PRIOR_DIALOGUE_STATE]" in row["text"] for row in resume_rows)
    assert report["leakage"]["trajectory_split_leak_count"] == 0


def test_state_conditioned_negatives_cover_switch_ood_policy_social_and_orphan() -> None:
    splits, _report = build_conversation_router_splits(
        sft_records_by_split(), clinc_payload(), seed=7404
    )
    resume_index = RELATION_LABELS.index("resume_previous_service")
    for split, rows in splits.items():
        required_kinds = {
            "state_intent_switch",
            "state_ood_detour",
            "state_orphan_resume",
            (
                "state_policy_followup"
                if split == "train"
                else "heldout_policy_followup_generalization"
            ),
            ("state_social_detour" if split == "train" else "heldout_social_generalization"),
        }
        selected = [row for row in rows if row["example_kind"] in required_kinds]
        assert {row["example_kind"] for row in selected} == required_kinds
        assert all(row["relation_labels"][resume_index] == 0 for row in selected)
        assert any(
            row["example_kind"] == "state_intent_switch" and row["intent"] == "freeze_card"
            for row in selected
        )
        assert any(
            row["example_kind"] == "state_ood_detour" and row["domain_label"] == 0
            for row in selected
        )


def test_state_social_generalization_spans_all_intents_and_policy_histories() -> None:
    splits, report = build_conversation_router_splits(
        sft_records_by_split(), clinc_payload(), seed=7404
    )
    expected_intents = {
        "view_accounts",
        "view_cards",
        "freeze_card",
        "replace_card",
        "view_transactions",
        "dispute_transaction",
        "view_transfers",
        "cancel_transfer",
        "view_service_cases",
    }
    social_kinds = {"state_social_detour", "heldout_social_generalization"}
    current_by_split: dict[str, set[str]] = {}
    for split, rows in splits.items():
        social = [row for row in rows if row["example_kind"] in social_kinds]
        current_by_split[split] = {
            normalize_router_text(str(row["current_text"])) for row in social
        }
        assert {row["prior_dialogue_state"]["pending_servicing"]["intent"] for row in social} == (
            expected_intents
        )
        assert len({row["history"][2]["content"] for row in social}) == 7
        assert all(row["intent"] == "conversation" for row in social)
        assert all(row["relation_labels"] == [0, 0, 0, 0, 0] for row in social)

    assert current_by_split["train"].isdisjoint(current_by_split["validation"])
    assert current_by_split["train"].isdisjoint(current_by_split["test"])
    assert current_by_split["validation"].isdisjoint(current_by_split["test"])
    assert current_by_split["test"] == {
        normalize_router_text(text)
        for text in (
            "Thanks",
            "Thank you",
            "Okay, thanks",
            "Got it",
            "That helps",
            "Hello",
            "How are you?",
            "Never mind",
        )
    }
    assert report["leakage"]["state_current_text_split_leak_count"] == 0


def test_implicit_policy_followups_span_all_canonical_topics_and_act_families() -> None:
    splits, _report = build_conversation_router_splits(
        sft_records_by_split(), clinc_payload(), seed=7404
    )
    policy_kinds = {
        "state_policy_followup",
        "heldout_policy_followup_generalization",
    }
    current_by_split: dict[str, set[str]] = {}
    for split, rows in splits.items():
        policy = [row for row in rows if row["example_kind"] in policy_kinds]
        current_by_split[split] = {
            normalize_router_text(str(row["current_text"])) for row in policy
        }
        assert len({row["history"][2]["content"] for row in policy}) == 7
        assert len(current_by_split[split]) == (10 if split == "train" else 5)
        assert all(row["intent"] == "policy_knowledge" for row in policy)
        assert all(row["relation_labels"][0] == 1 for row in policy)

    assert current_by_split["train"].isdisjoint(current_by_split["validation"])
    assert current_by_split["train"].isdisjoint(current_by_split["test"])
    assert current_by_split["validation"].isdisjoint(current_by_split["test"])
    assert normalize_router_text("What documents might you need?") in current_by_split["test"]


def test_policy_history_rows_preserve_switch_ood_and_resume_behavior() -> None:
    splits, _report = build_conversation_router_splits(
        sft_records_by_split(), clinc_payload(), seed=7404
    )
    resume_index = RELATION_LABELS.index("resume_previous_service")
    for rows in splits.values():
        switches = [
            row for row in rows if str(row["trajectory_id"]).startswith("state-switch-policy|")
        ]
        ood = [row for row in rows if str(row["trajectory_id"]).startswith("state-ood-policy|")]
        resumes = [
            row for row in rows if str(row["trajectory_id"]).startswith("state-resume-policy|")
        ]
        assert len({row["history"][2]["content"] for row in switches}) == 7
        assert len({row["history"][2]["content"] for row in ood}) == 7
        assert len({row["history"][2]["content"] for row in resumes}) == 7
        assert all(row["example_kind"] == "state_intent_switch" for row in switches)
        assert all(row["domain_label"] == 0 for row in ood)
        assert all(row["relation_labels"][resume_index] == 1 for row in resumes)


def test_explicit_trajectory_cannot_cross_splits() -> None:
    records = sft_records_by_split()
    records["train"][0]["metadata"]["trajectory_id"] = "shared-trajectory"
    records["test"][0]["metadata"]["trajectory_id"] = "shared-trajectory"

    import pytest

    with pytest.raises(ValueError, match="trajectory .* appears in both"):
        build_conversation_router_splits(records, clinc_payload(), seed=7404)


def test_held_out_screenshot_regressions_are_test_only() -> None:
    splits, _report = build_conversation_router_splits(
        sft_records_by_split(),
        clinc_payload(),
        seed=7404,
    )

    heldout_by_split = {
        split: [row for row in rows if row["example_kind"] == "heldout_screenshot_regression"]
        for split, rows in splits.items()
    }
    assert heldout_by_split["train"] == []
    assert heldout_by_split["validation"] == []
    heldout_current = {
        normalize_router_text(str(row["current_text"])) for row in heldout_by_split["test"]
    }
    assert heldout_current == {
        "i didn t ask about mortgage",
        "ok thats the one i want to replace",
        "was the mailing address updated recently",
        "when was that created",
        "what is that all about when was it created",
        "what about the weather there",
        "why are you repeating yourself",
    }


def _record(
    *,
    split: str,
    record_id: str,
    scenario_family: str,
    user: str,
    assistant: str,
    path: str = "tool_success",
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "messages": [
            {"role": "system", "content": "system prompt", "loss": False},
            {"role": "user", "content": user, "loss": False},
            {"role": "assistant", "content": None, "loss": True, "tool_calls": []},
            {"role": "tool", "name": "list_accounts", "content": {"ok": True}, "loss": False},
            {"role": "assistant", "content": assistant, "loss": True},
        ],
        "expected": {"tool_calls": [{"name": "list_accounts", "arguments": {}}]},
        "metadata": {
            "scenario_family": scenario_family,
            "path": path,
            "split": split,
            "split_group": f"{split}|{record_id}",
        },
        "split_keys": {
            "scenario_family": scenario_family,
            "state_seed": f"{split}-{record_id}",
            "customer_id": f"cust-{split}-{record_id}",
            "template_id": "template",
        },
    }
