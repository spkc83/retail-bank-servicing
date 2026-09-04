from __future__ import annotations

import copy
from collections import Counter
from typing import Any

import pytest

from hello_slm.banking_conversation_router_data import (
    INTENT_LABELS,
    RELATION_LABELS,
    build_conversation_router_splits,
    normalize_router_text,
    render_router_input,
    render_router_input_with_context,
)
from hello_slm.banking_domain_taxonomy import (
    ACTION_LABELS,
    DOMAIN_LABELS,
    ENTITY_RESOLUTION_LABELS,
    FAMILY_LABELS,
    LANE_LABELS,
    validate_hierarchical_labels,
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


def test_cross_encoder_renderer_omits_semantically_empty_dialogue_state() -> None:
    rendered, context_applied = render_router_input_with_context(
        "What happened with the money I sent recently?",
        prior_dialogue_state={
            "knowledge_detour_active": False,
            "pending_servicing": None,
            "version": 1,
        },
    )

    assert rendered == "[CURRENT_USER]\nWhat happened with the money I sent recently?"
    assert context_applied is False


def test_v5_splits_use_hierarchy_actions_relations_and_no_current_turn_leakage() -> None:
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
                "domain_name",
                "domain_index",
                "intent_label",
                "intent",
                "lane",
                "lane_name",
                "lane_index",
                "family_name",
                "family_index",
                "action_name",
                "action_index",
                "entity_resolution_name",
                "entity_resolution_index",
                "relation_labels",
                "example_kind",
                "source",
                "source_split",
                "group_id",
                "trajectory_id",
                "prior_dialogue_state",
                "counterfactual_pair_id",
                "counterfactual_target",
                "counterfactual_phrase_family",
                "actionable_entity_count",
            }
            assert len(row["relation_labels"]) == len(RELATION_LABELS)
            assert row["domain_index"] == DOMAIN_LABELS.index(row["domain_name"])
            assert row["lane_index"] == LANE_LABELS.index(row["lane_name"])
            assert row["family_index"] == FAMILY_LABELS.index(row["family_name"])
            assert row["action_index"] == ACTION_LABELS.index(row["action_name"])
            assert row["entity_resolution_index"] == ENTITY_RESOLUTION_LABELS.index(
                row["entity_resolution_name"]
            )
            validate_hierarchical_labels(row)
            assert "tool_call" not in str(row["text"])
            assert "Main Checking has USD 10.00" not in str(row["text"])

    kinds = Counter(row["example_kind"] for row in splits["train"])
    ungrounded_generic_context = [
        row
        for row in splits["train"]
        if row["source"] == "self-authored-router-v5-synthetic"
        and row["example_kind"]
        in {
            "contextual_followup",
            "agent_repair",
            "clarification_answer",
            "typo_contextual_followup",
        }
    ]
    assert ungrounded_generic_context == []
    assert kinds["external_topic_shift"] >= 1
    assert kinds["banking_topic_shift"] >= 1
    assert kinds["targeted_contextual_followup"] >= 1
    assert kinds["targeted_clarification_answer"] >= 1
    assert kinds["targeted_agent_repair"] >= 1
    assert kinds["targeted_service_case"] >= 1
    assert sum(kinds.values()) > 10
    test_kinds = Counter(row["example_kind"] for row in splits["test"])
    assert test_kinds["external_topic_shift"] >= 1
    assert test_kinds["heldout_screenshot_regression"] == 9
    assert report["leakage"]["group_split_leak_count"] == 0
    assert report["leakage"]["state_current_text_split_leak_count"] == 0
    assert report["leakage"]["counterfactual_pair_split_leak_count"] == 0
    assert report["pii_matches"] == 0


def test_synthetic_topic_shift_preserves_governed_action_and_entity_labels() -> None:
    clarification = _record(
        split="train",
        record_id="train-replace-clarification",
        scenario_family="clarification_card",
        user="Please replace a card.",
        assistant="Which card should I replace?",
        path="clarification",
    )
    clarification["metadata"]["explicit_entity_resolution"] = "missing"
    records = sft_records_by_split()
    records["train"] = [clarification]

    splits, _report = build_conversation_router_splits(records, clinc_payload(), seed=7404)
    resumed = [
        row
        for row in splits["train"]
        if row["example_kind"] == "banking_topic_shift"
        and row["current_text"] == "Please replace a card."
    ]

    assert resumed
    assert {(row["action_name"], row["entity_resolution_name"]) for row in resumed} == {
        ("clarify", "missing")
    }
    detours = [
        row
        for row in splits["train"]
        if row["example_kind"] == "external_topic_shift"
        and row["history"][0]["content"] == "Please replace a card."
    ]
    assert detours
    assert all(row["history"][1]["content"] == "Which card should I replace?" for row in detours)


def test_governed_policy_path_takes_precedence_over_mutation_keywords() -> None:
    records = sft_records_by_split()
    policy = _record(
        split="train",
        record_id="replacement-policy",
        scenario_family="faq_card_replacement",
        user="How long does it take to replace a damaged card?",
        assistant="Replacement timing is disclosed when the request is made.",
        path="retrieval_grounded_policy",
    )
    policy["expected"]["tool_calls"] = []
    records["train"].append(policy)

    splits, _report = build_conversation_router_splits(records, clinc_payload(), seed=7404)
    row = next(
        item
        for item in splits["train"]
        if item["current_text"] == "How long does it take to replace a damaged card?"
        and item["example_kind"] == "retrieval_grounded_policy"
    )

    assert row["intent"] == "policy_knowledge"
    assert row["lane_name"] == "policy"
    assert row["action_name"] == "retrieve_policy"
    assert row["entity_resolution_name"] == "not_required"


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
    assert by_current["Explain photosynthesis."]["domain_name"] == "out_of_domain"
    assert by_current["Explain photosynthesis."]["intent_label"] == -100
    assert by_current["Explain photosynthesis."]["action_name"] == "refuse_ood"
    assert by_current["Tell me my full account number."]["domain_label"] == 1
    assert by_current["Tell me my full account number."]["domain_name"] == "banking"
    assert by_current["Tell me my full account number."]["intent"] == "other_banking"


def test_sft_coreference_pairs_retain_action_entity_supervision_without_output_leakage() -> None:
    records = sft_records_by_split()
    action = _record(
        split="train",
        record_id="coref-action",
        scenario_family="deictic_replace_action",
        user="replace the card from that list",
        assistant="The replacement is pending.",
        path="multi_turn",
    )
    action["metadata"].update(
        {
            "coreference_pair_id": "pair-1",
            "coreference_phrase_family": "that-list",
            "coreference_target": "replace_card",
            "actionable_card_count": 1,
        }
    )
    action["expected"]["tool_calls"] = [{"name": "replace_card", "arguments": {"last4": "4821"}}]
    action["messages"][1:1] = [
        {"role": "user", "content": "List my cards.", "loss": False},
        {
            "role": "assistant",
            "content": "The active card ends in 4821.",
            "loss": False,
        },
    ]
    ambiguous = _record(
        split="train",
        record_id="coref-clarify",
        scenario_family="deictic_replace_ambiguity",
        user="replace the card from that list",
        assistant="Which card should I replace?",
        path="clarification",
    )
    ambiguous["metadata"].update(
        {
            "coreference_pair_id": "pair-1",
            "coreference_phrase_family": "that-list",
            "coreference_target": "clarification",
            "actionable_card_count": 2,
        }
    )
    ambiguous["expected"]["tool_calls"] = []
    ambiguous["messages"][1:1] = [
        {"role": "user", "content": "List my cards.", "loss": False},
        {
            "role": "assistant",
            "content": "Active cards end in 4821 and 7319.",
            "loss": False,
        },
    ]
    records["train"].extend([action, ambiguous])

    splits, report = build_conversation_router_splits(records, clinc_payload(), seed=7404)
    pair_rows = [row for row in splits["train"] if row["counterfactual_pair_id"] == "pair-1"]

    assert {(row["action_name"], row["entity_resolution_name"]) for row in pair_rows} == {
        ("execute_tool", "resolved"),
        ("clarify", "ambiguous"),
    }
    assert {row["counterfactual_target"] for row in pair_rows} == {
        "replace_card",
        "clarification",
    }
    assert {row["actionable_entity_count"] for row in pair_rows} == {1, 2}
    assert all("replacement is pending" not in row["text"].casefold() for row in pair_rows)
    assert all("which card should i replace" not in row["text"].casefold() for row in pair_rows)
    assert report["leakage"]["counterfactual_pair_split_leak_count"] == 0


def test_counterfactual_pair_cannot_cross_splits() -> None:
    records = sft_records_by_split()
    records["train"][0]["metadata"]["coreference_pair_id"] = "shared-pair"
    records["test"][0]["metadata"]["coreference_pair_id"] = "shared-pair"

    import pytest

    with pytest.raises(ValueError, match="counterfactual pair .* appears in both"):
        build_conversation_router_splits(records, clinc_payload(), seed=7404)


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
    assert sum(row["example_kind"] == "state_orphan_resume" for row in splits["train"]) == 23
    assert sum(row["example_kind"] == "state_orphan_resume" for row in splits["validation"]) == 6


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

    train_wellbeing = {
        normalize_router_text(str(row["current_text"]))
        for row in splits["train"]
        if row["example_kind"] == "state_social_detour"
        and "wellbeing_train" in str(row["group_id"])
    }
    assert normalize_router_text("How have you been?") in train_wellbeing
    assert normalize_router_text("Are you doing okay?") in train_wellbeing
    assert normalize_router_text("How are you?") not in train_wellbeing
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


def test_transfer_transaction_contrasts_resist_social_and_balance_history() -> None:
    splits, report = build_conversation_router_splits(
        sft_records_by_split(), clinc_payload(), seed=7404
    )
    expected_counts = {"train": 96, "validation": 32, "test": 16}
    markdown_counts = {"train": 24, "validation": 8, "test": 4}
    split_products = {
        "train": {"Harbor Everyday", "Cedar Reserve"},
        "validation": {"Pioneer Checking", "Meadow Savings"},
        "test": {"Lakeside Spend", "Summit Savings"},
    }
    current_by_split: dict[str, set[str]] = {}

    for split, rows in splits.items():
        contrast = [
            row for row in rows if row["example_kind"] == "transfer_transaction_semantic_contrast"
        ]
        assert len(contrast) == expected_counts[split]
        assert {row["intent"] for row in contrast} == {
            "view_transfers",
            "view_transactions",
        }
        assert all(row["action_name"] == "execute_tool" for row in contrast)
        assert all(row["relation_labels"][RELATION_LABELS.index("topic_shift")] for row in contrast)
        assert all(row["history"] for row in contrast)
        assert all(
            "balance" in str(row["history"]).casefold()
            or any(
                greeting in str(row["history"]).casefold()
                for greeting in ("morning", "hello", "afternoon", "well", "speak")
            )
            for row in contrast
        )
        markdown = [
            row
            for row in contrast
            if "## Accounts\n\n| Name | Type | Last 4 | Available | Current | Status |"
            in str(row["history"][-1]["content"])
        ]
        assert len(markdown) == markdown_counts[split]
        assert all(
            "| --- | --- | --- | --- | --- | --- |" in row["history"][-1]["content"]
            for row in markdown
        )
        assert all(
            all(product in row["history"][-1]["content"] for product in split_products[split])
            for row in markdown
        )
        assert all(
            product not in str(row["text"])
            for other_split, products in split_products.items()
            if other_split != split
            for product in products
            for row in contrast
        )
        grouped: dict[str, set[str]] = {}
        for row in contrast:
            grouped.setdefault(row["group_id"], set()).add(row["intent"])
        assert grouped
        assert set(map(frozenset, grouped.values())) == {
            frozenset({"view_transfers", "view_transactions"})
        }
        current_by_split[split] = {
            normalize_router_text(str(row["current_text"])) for row in contrast
        }

    assert current_by_split["train"].isdisjoint(current_by_split["validation"])
    assert current_by_split["train"].isdisjoint(current_by_split["test"])
    assert current_by_split["validation"].isdisjoint(current_by_split["test"])
    assert report["leakage"]["group_split_leak_count"] == 0
    assert report["leakage"]["trajectory_split_leak_count"] == 0
    assert report["leakage"]["state_current_text_split_leak_count"] == 0
    assert report["leakage"]["state_paraphrase_family_split_leak_count"] == 0


def test_v7_servicing_to_policy_topic_shifts_are_split_isolated() -> None:
    splits, report = build_conversation_router_splits(
        sft_records_by_split(), clinc_payload(), seed=7404
    )
    expected_counts = {"train": 16, "validation": 8, "test": 8}
    current_by_split: dict[str, set[str]] = {}

    for split, rows in splits.items():
        shifts = [row for row in rows if row["example_kind"] == "servicing_policy_topic_shift"]
        assert len(shifts) == expected_counts[split]
        assert all(row["intent"] == "policy_knowledge" for row in shifts)
        assert all(row["action_name"] == "retrieve_policy" for row in shifts)
        assert all(
            row["relation_labels"][RELATION_LABELS.index("topic_shift")] == 1 for row in shifts
        )
        assert all(row["history"] for row in shifts)
        assert any(
            "## Service cases\n\n| Case | Type | Status | Created |"
            in str(row["history"][-1]["content"])
            for row in shifts
        )
        current_by_split[split] = {
            normalize_router_text(str(row["current_text"])) for row in shifts
        }

    assert current_by_split["train"].isdisjoint(current_by_split["validation"])
    assert current_by_split["train"].isdisjoint(current_by_split["test"])
    assert current_by_split["validation"].isdisjoint(current_by_split["test"])
    assert report["leakage"]["state_current_text_split_leak_count"] == 0
    assert report["leakage"]["state_paraphrase_family_split_leak_count"] == 0


def test_ineligible_entity_supervision_is_present_in_every_split() -> None:
    splits, _report = build_conversation_router_splits(
        sft_records_by_split(),
        clinc_payload(),
        seed=7404,
    )

    for _split, rows in splits.items():
        ineligible = [row for row in rows if row["example_kind"] == "state_ineligible_entity"]
        assert {row["intent"] for row in ineligible} == {
            "freeze_card",
            "replace_card",
            "dispute_transaction",
            "cancel_transfer",
        }
        assert all(row["action_name"] == "converse" for row in ineligible)
        assert all(row["entity_resolution_name"] == "ineligible" for row in ineligible)


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
        "what happened with the money i sent recently",
        "was the mailing address updated recently",
        "when was that created",
        "what is that all about when was it created",
        "what about the weather there",
        "why are you repeating yourself",
        "show my five most recent transactions",
    }
    transfer_regression = next(
        row
        for row in heldout_by_split["test"]
        if row["group_id"] == "heldout|recent-sent-money-status"
    )
    rendered_balance = transfer_regression["history"][-1]["content"]
    assert rendered_balance.startswith(
        "## Accounts\n\n| Name | Type | Last 4 | Available | Current | Status |"
    )
    assert "| --- | --- | --- | --- | --- | --- |" in rendered_balance
    assert "| Everyday Checking | checking | 1042 | USD 3,245.67 |" in rendered_balance
    assert "| Goal Saver | savings | 8831 | USD 12,500.00 |" in rendered_balance
    for split in ("train", "validation"):
        all_current = {normalize_router_text(str(row["current_text"])) for row in splits[split]}
        assert heldout_current.isdisjoint(all_current)
        normalized_inputs = [normalize_router_text(str(row["text"])) for row in splits[split]]
        assert all(
            heldout_prompt not in rendered
            for heldout_prompt in heldout_current
            for rendered in normalized_inputs
        )
    assert _report["leakage"]["heldout_exact_leaks_in_train_validation"] == []
    assert _report["leakage"]["heldout_long_ngram_leaks_in_train_validation"] == []


def test_coreference_shadow_suite_remains_untouched_and_test_only() -> None:
    records = sft_records_by_split()
    shadow = _record(
        split="test",
        record_id="shadow-v7-history",
        scenario_family="clarification_card",
        user="Use the card identified in that earlier list.",
        assistant="I need you to select one eligible card.",
        path="clarification",
    )
    shadow["metadata"].update(
        {
            "trainable": False,
            "coreference_pair_id": "shadow-v7-pair",
            "coreference_target": "clarification",
            "coreference_phrase_family": "shadow-v7-history-family",
        }
    )
    records["test"].append(shadow)
    before = copy.deepcopy(records)

    splits, _report = build_conversation_router_splits(records, clinc_payload(), seed=7404)

    assert records == before
    assert all(
        row["source"] != "self-authored-coreference-shadow"
        for split in ("train", "validation")
        for row in splits[split]
    )
    shadow_rows = [
        row for row in splits["test"] if row["source"] == "self-authored-coreference-shadow"
    ]
    assert len(shadow_rows) == 1
    assert shadow_rows[0]["counterfactual_pair_id"] == "shadow-v7-pair"
    assert shadow_rows[0]["counterfactual_phrase_family"] == "shadow-v7-history-family"


def test_sft_row_with_live_regression_prompt_in_history_is_excluded() -> None:
    records = sft_records_by_split()
    contaminated = _record(
        split="train",
        record_id="history-contaminated",
        scenario_family="service_cases",
        user="Continue with a different service request.",
        assistant="I found the service request.",
    )
    contaminated["messages"][1:1] = [
        {"role": "user", "content": "What happened with the money I sent recently?"},
        {"role": "assistant", "content": "I could not provide the transfer details."},
    ]
    records["train"].append(contaminated)

    splits, _report = build_conversation_router_splits(records, clinc_payload(), seed=7404)

    assert all(
        row["current_text"] != "Continue with a different service request."
        for row in splits["train"]
    )


def _record(
    *,
    split: str,
    record_id: str,
    scenario_family: str,
    user: str,
    assistant: str,
    path: str = "tool_success",
) -> dict[str, Any]:
    tool_name = {
        "card_status": "list_cards",
        "service_cases": "list_service_cases",
    }.get(scenario_family, "list_accounts")
    return {
        "record_id": record_id,
        "messages": [
            {"role": "system", "content": "system prompt", "loss": False},
            {"role": "user", "content": user, "loss": False},
            {"role": "assistant", "content": None, "loss": True, "tool_calls": []},
            {"role": "tool", "name": "list_accounts", "content": {"ok": True}, "loss": False},
            {"role": "assistant", "content": assistant, "loss": True},
        ],
        "expected": {"tool_calls": [{"name": tool_name, "arguments": {}}]},
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


def test_first_turn_mutation_openers_supervise_execute_without_touching_test() -> None:
    from hello_slm.banking_conversation_router_data import _first_turn_mutation_opener_rows

    assert _first_turn_mutation_opener_rows("test") == []

    train_rows = _first_turn_mutation_opener_rows("train")
    validation_rows = _first_turn_mutation_opener_rows("validation")
    assert train_rows and validation_rows

    probe = normalize_router_text("My card was stolen. Freeze it.")
    paraphrase_probe = normalize_router_text(
        "My debit card just got stolen, please freeze it right away."
    )
    for rows in (train_rows, validation_rows):
        for row in rows:
            assert row["history"] == []
            assert row["prior_dialogue_state"] == {}
            assert row["example_kind"] == "first_turn_mutation_opener"
            assert row["source"] == "self-authored-router-v8-first-turn-mutation-openers"
            validate_hierarchical_labels(row, intent=row["intent"])
            normalized = normalize_router_text(row["current_text"])
            assert normalized != probe
            assert normalized != paraphrase_probe
            assert "i want to replace" not in normalized

    by_intent = {
        intent: [row for row in train_rows if row["intent"] == intent]
        for intent in (
            "freeze_card",
            "replace_card",
            "dispute_transaction",
            "cancel_transfer",
            "policy_knowledge",
        )
    }
    assert all(by_intent.values())

    assert {
        (row["action_name"], row["entity_resolution_name"]) for row in by_intent["freeze_card"]
    } == {("execute_tool", "resolved")}
    assert {
        (row["action_name"], row["entity_resolution_name"]) for row in by_intent["replace_card"]
    } == {("clarify", "missing")}
    assert {
        (row["action_name"], row["entity_resolution_name"])
        for row in by_intent["dispute_transaction"]
    } == {("execute_tool", "resolved")}
    assert {
        (row["action_name"], row["entity_resolution_name"]) for row in by_intent["cancel_transfer"]
    } == {("execute_tool", "resolved")}
    assert {
        (row["action_name"], row["entity_resolution_name"]) for row in by_intent["policy_knowledge"]
    } == {("retrieve_policy", "not_required")}

    stolen_freeze = [
        row
        for row in by_intent["freeze_card"]
        if "stolen" in normalize_router_text(row["current_text"])
    ]
    assert len(stolen_freeze) >= 8

    train_texts = {normalize_router_text(row["current_text"]) for row in train_rows}
    validation_texts = {normalize_router_text(row["current_text"]) for row in validation_rows}
    assert train_texts.isdisjoint(validation_texts)


def test_first_turn_mutation_openers_are_wired_into_train_and_validation_only() -> None:
    splits, _report = build_conversation_router_splits(
        sft_records_by_split(),
        clinc_payload(),
        seed=7404,
    )
    kinds = {
        split: [row for row in rows if row["example_kind"] == "first_turn_mutation_opener"]
        for split, rows in splits.items()
    }
    assert kinds["train"] and kinds["validation"]
    assert kinds["test"] == []


def test_retrospective_status_questions_stay_converse_and_never_touch_test() -> None:
    from hello_slm.banking_conversation_router_data import _retrospective_status_rows

    assert _retrospective_status_rows("test") == []
    train_rows = _retrospective_status_rows("train")
    validation_rows = _retrospective_status_rows("validation")
    assert train_rows and validation_rows
    for rows in (train_rows, validation_rows):
        for row in rows:
            assert row["history"]
            assert row["example_kind"] == "retrospective_mutation_status"
            assert row["action_name"] == "converse"
            assert row["entity_resolution_name"] == "not_required"
            validate_hierarchical_labels(row, intent=row["intent"])
    assert {row["intent"] for row in train_rows} == {
        "freeze_card",
        "replace_card",
        "dispute_transaction",
        "cancel_transfer",
    }


# The tool-SFT realizer used to stack a request opener on a stem that was already a
# question, producing "Can you what information is needed for a card dispute". Train
# stopped carrying that shape on 2026-08-20, but the frozen test splits still do, so a
# router trained on the current corpus refuses them as out-of-domain and the false-refusal
# gate measures the retired template instead of the router.
@pytest.mark.parametrize(
    "text",
    [
        "Can you how does a card purchase dispute work so I can finish this banking task",
        "Can you what information is needed for a card dispute",
        "I need you to how do I safely report card fraud while I am checking the mobile app",
        "Would you what should I do if I see card fraud",
        "Could you how is delivery handled for a replacement card",
        "Help me what are the usual steps to open a bank account so I can decide what to do next",
        "Help me can you open another checking account for me while I am checking the mobile app",
        "Could you can I add a new checking account in this chat",
        "Can you can you help me open a mortgage account because I noticed something unexpected",
        "Please what do banks usually require for a new account",
        "Please how do I apply for a mortgage because I am reviewing my monthly budget",
        "Would you how does interest on a savings account work",
    ],
)
def test_retired_realizer_shape_is_detected(text: str) -> None:
    from hello_slm.banking_conversation_router_data import is_retired_realizer_prompt

    assert is_retired_realizer_prompt(text)


@pytest.mark.parametrize(
    "text",
    [
        "Can you tell me about replacement card timing while I am checking the app",
        "Could you mark Bright Meadow Electronics for dispute",
        "Please can you help me open a mortgage account",
        "Please can this demo create a new deposit account",
        "Help me choose the debit card to replace.",
        "What information is needed for a card dispute?",
        "Can you pull up my account list with balances?",
        "hello can you help",
        "why are you repeating yourself",
        "Could you, when you have a moment, list my cards?",
    ],
)
def test_well_formed_openers_are_not_mistaken_for_the_retired_shape(text: str) -> None:
    from hello_slm.banking_conversation_router_data import is_retired_realizer_prompt

    assert not is_retired_realizer_prompt(text)


def test_retired_realizer_rows_never_reach_any_router_split() -> None:
    records = sft_records_by_split()
    for split in ("train", "validation", "test"):
        records[split].append(
            _record(
                split=split,
                record_id=f"{split}-mangled",
                scenario_family="faq_card_dispute",
                user="Can you what information is needed for a card dispute",
                assistant="A dispute needs the merchant, amount, and date.",
                path="retrieval_grounded_policy",
            )
        )
        carried = _record(
            split=split,
            record_id=f"{split}-mangled-history",
            scenario_family="service_cases",
            user="And when was that case opened?",
            assistant="It was opened last month.",
        )
        carried["messages"][1:1] = [
            {"role": "user", "content": "Help me how do I safely report card fraud"},
            {"role": "assistant", "content": "Report it through the fraud line."},
        ]
        records[split].append(carried)

    splits, report = build_conversation_router_splits(records, clinc_payload(), seed=7404)

    mangled = normalize_router_text("Can you what information is needed for a card dispute")
    for split, rows in splits.items():
        assert all(normalize_router_text(row["current_text"]) != mangled for row in rows), split
        assert all(row["current_text"] != "And when was that case opened?" for row in rows), split
    assert report["retired_realizer_rows_excluded"] == 6


def test_targeted_repair_rows_vary_the_wrong_topic_and_the_assistant_voice() -> None:
    """A router that only ever saw four mortgage sentences as the wrong answer
    routed a short repair after a savings or fee line out of domain. Train now
    rotates through many wrong answers; the user prompts are unchanged, the row
    count is unchanged, and validation/test keep their original two histories."""
    from hello_slm.banking_conversation_router_data import _targeted_use_case_rows

    def repair_rows(split: str) -> list[dict[str, Any]]:
        return [
            row
            for row in _targeted_use_case_rows(split)
            if row["example_kind"] in {"targeted_agent_repair", "targeted_wrong_topic_repair"}
        ]

    train = repair_rows("train")
    assistant_lines = {row["history"][-1]["content"] for row in train}
    assert len(assistant_lines) >= 12
    assert "Mortgage applicants are typically at least 18." not in assistant_lines
    topics = {
        topic
        for line in assistant_lines
        for topic in ("mortgage", "loan", "savings", "overdraft", "card")
        if topic in line.lower()
    }
    assert {"savings", "overdraft", "card"} <= topics
    conversational = ("Great question", "Sure thing", "Happy to", "Of course")
    assert any(line.startswith(conversational) for line in assistant_lines)
    assert sum(row["example_kind"] == "targeted_agent_repair" for row in train) == 512
    assert sum(row["example_kind"] == "targeted_wrong_topic_repair" for row in train) == 512
    line_counts = Counter(row["history"][-1]["content"] for row in train)
    assert max(line_counts.values()) - min(line_counts.values()) <= 16
    for split in ("validation", "test"):
        held = {row["history"][-1]["content"] for row in repair_rows(split)}
        assert held == {
            "Mortgage applications usually require an eligibility review.",
            "Home-loan rates vary with the selected product.",
        }


def test_first_turn_phrasing_rows_cover_every_servicing_intent_in_question_and_modal_form() -> None:
    """The plain first ask -- "What is my balance?", "Could you pull up my transfers?",
    "Hi, can you freeze my card?" -- was the neglected case across the servicing
    lane: first-turn wh-questions were zero for six intents and modal requests
    numbered ten to twenty-six, while CLINC supplies thousands of "can you set an
    alarm" out-of-domain rows. These rows are train/validation only and never test."""
    from hello_slm.banking_conversation_router_data import (
        _contains_screenshot_regression_ngram,
        _first_turn_phrasing_rows,
        _is_screenshot_regression_text,
    )
    from hello_slm.banking_corpus_coverage import phrasing_form

    assert _first_turn_phrasing_rows("test") == []
    train = _first_turn_phrasing_rows("train")
    validation = _first_turn_phrasing_rows("validation")
    assert train and validation
    assert {row["current_text"] for row in train}.isdisjoint(
        row["current_text"] for row in validation
    )
    servicing = {
        "view_accounts",
        "view_cards",
        "view_transactions",
        "view_transfers",
        "view_service_cases",
        "freeze_card",
        "replace_card",
        "dispute_transaction",
        "cancel_transfer",
    }
    forms: Counter[tuple[str, str]] = Counter()
    for row in train + validation:
        assert row["history"] == []
        assert row["domain_label"] == 1
        assert row["example_kind"] == "first_turn_phrasing"
        assert row["source"] == "self-authored-router-v9-first-turn-phrasing"
        assert not _is_screenshot_regression_text(row["current_text"])
        assert not _contains_screenshot_regression_ngram(row["current_text"])
        forms[(row["intent"], phrasing_form(row["current_text"]))] += 1
    reads = {name for name in servicing if name.startswith("view_")}
    for intent in servicing:
        # "Can I get a replacement card?" reads as a modal request, so the
        # mutation intents carry fewer pure wh-questions than the reads.
        assert forms[(intent, "wh_question")] >= (6 if intent in reads else 2), intent
        assert forms[(intent, "modal_request")] >= 30, intent
    assert forms[("policy_knowledge", "modal_request")] >= 20
    by_intent = {row["intent"]: row for row in train}
    assert (
        by_intent["view_accounts"]["action_name"],
        by_intent["view_accounts"]["entity_resolution_name"],
    ) == ("execute_tool", "not_required")
    assert (
        by_intent["replace_card"]["action_name"],
        by_intent["replace_card"]["entity_resolution_name"],
    ) == ("clarify", "missing")
    assert (
        by_intent["dispute_transaction"]["action_name"],
        by_intent["dispute_transaction"]["entity_resolution_name"],
    ) == ("execute_tool", "resolved")
    assert (
        by_intent["policy_knowledge"]["action_name"],
        by_intent["policy_knowledge"]["entity_resolution_name"],
    ) == ("retrieve_policy", "not_required")


def test_first_turn_phrasing_rows_reach_the_built_splits_but_not_test() -> None:
    splits, report = build_conversation_router_splits(
        sft_records_by_split(), clinc_payload(), seed=7404
    )
    assert report["kind_counts"]["train"]["first_turn_phrasing"] >= 400
    assert report["kind_counts"]["validation"]["first_turn_phrasing"] >= 60
    assert "first_turn_phrasing" not in report["kind_counts"]["test"]


def test_terminal_punctuation_carries_no_domain_signal() -> None:
    """Teacher-written banking prompts end in "?" or "."; CLINC lines are bare. With
    an uncased encoder that mark was the domain label: the same words scored 0.01
    banking bare and 1.00 punctuated. Train and validation now carry both surface
    forms on both sides of the domain boundary; the test split is never rewritten."""
    from hello_slm.banking_conversation_router_data import (
        _bare_surface_form,
        _punctuated_surface_form,
    )

    assert _bare_surface_form("Could you mark Bright Meadow Electronics for dispute?") == (
        "could you mark Bright Meadow Electronics for dispute"
    )
    assert _punctuated_surface_form("can you set an alarm") == "Can you set an alarm?"
    assert _punctuated_surface_form("tell me a joke") == "Tell me a joke."
    assert _punctuated_surface_form("Explain photosynthesis.") == "Explain photosynthesis."

    records = sft_records_by_split()
    for split in ("train", "validation"):
        for index in range(40):
            records[split].append(
                _record(
                    split=split,
                    record_id=f"{split}-surface-{index}",
                    scenario_family="read_accounts",
                    user=f"Show my balances for account number {index} please.",
                    assistant="Main Checking has USD 10.00 available.",
                )
            )
    clinc = clinc_payload()
    clinc["train"] = [[f"tell me a joke about number {index}", "tell_joke"] for index in range(40)]
    clinc["val"] = [[f"play song number {index}", "play_music"] for index in range(40)]
    splits, _report = build_conversation_router_splits(records, clinc, seed=7404)

    def punctuated_share(rows: list[dict[str, Any]]) -> float:
        return sum(row["current_text"].endswith((".", "?", "!")) for row in rows) / len(rows)

    for split in ("train", "validation"):
        banking = [row for row in splits[split] if row["domain_label"] == 1 and not row["history"]]
        external = [row for row in splits[split] if row["example_kind"] == "clinc_external_ood"]
        assert 0.15 <= 1 - punctuated_share(banking) <= 0.6, split
        assert 0.15 <= punctuated_share(external) <= 0.6, split
        for row in splits[split]:
            assert (
                row["text"].split("\n")[1] == row["current_text"]
                or "[PRIOR_DIALOGUE_STATE]" in row["text"]
            )
    assert all(
        not row["current_text"].endswith((".", "?", "!"))
        for row in splits["test"]
        if row["example_kind"] == "clinc_external_ood"
    )
