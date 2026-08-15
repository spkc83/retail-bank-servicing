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
    assert sum(row["example_kind"] == "state_orphan_resume" for row in splits["train"]) == 18
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
    expected_counts = {"train": 32, "validation": 16, "test": 16}
    markdown_counts = {"train": 8, "validation": 4, "test": 4}
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
