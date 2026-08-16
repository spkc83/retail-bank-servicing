from __future__ import annotations

from typing import Any

import pytest

from entity_grounding import (
    EFFECTIVE_DECISION_CONTRACT,
    ground_servicing_decision,
    public_selectors_from_message,
)


def route(intent: str, *, entity_resolution: str = "resolved") -> dict[str, Any]:
    return {
        "route": "in_domain",
        "action": "execute_tool",
        "fine_intent": intent,
        "entity_resolution": entity_resolution,
    }


def snapshot(*, cards: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "cards": cards if cards is not None else [],
        "transactions": [],
        "transfers": [],
    }


def test_explicit_current_selector_becomes_the_only_effective_constraint() -> None:
    grounded = ground_servicing_decision(
        route("replace_card"),
        current_user_values={"last4": "4821"},
        live_snapshot=snapshot(
            cards=[
                {"last4": "4821", "status": "active"},
                {"last4": "7319", "status": "active"},
            ]
        ),
    )

    assert grounded["effective_decision_contract"] == EFFECTIVE_DECISION_CONTRACT
    assert grounded["decision_accepted"] is True
    assert grounded["learned_action"] == "execute_tool"
    assert grounded["learned_entity_resolution"] == "resolved"
    assert grounded["action"] == "execute_tool"
    assert grounded["entity_resolution"] == "resolved"
    assert grounded["argument_constraints"] == {"last4": "4821"}
    assert grounded["entity_grounding_source"] == "current_user"
    assert grounded["entity_candidate_count"] == 1


def test_ineligible_explicit_selector_never_falls_back_to_another_live_candidate() -> None:
    grounded = ground_servicing_decision(
        route("replace_card"),
        current_user_values={"last4": "4821"},
        live_snapshot=snapshot(
            cards=[
                {"last4": "4821", "status": "replacement_pending"},
                {"last4": "7319", "status": "active"},
            ]
        ),
    )

    assert grounded["decision_accepted"] is True
    assert grounded["tool_execution_allowed"] is False
    assert grounded["action"] == "clarify"
    assert grounded["entity_resolution"] == "ineligible"
    assert grounded["argument_constraints"] == {}
    assert grounded["entity_grounding_source"] == "current_user"
    assert grounded["entity_candidate_count"] == 0


def test_unique_trusted_prior_tool_value_is_rechecked_against_live_eligibility() -> None:
    grounded = ground_servicing_decision(
        route("cancel_transfer"),
        trusted_tool_results=[
            {
                "role": "tool",
                "content": '{"ok":true,"result":{"transfers":['
                '{"recipient":"River Consulting","status":"pending"},'
                '{"recipient":"Jamie Lee","status":"completed"}]}}',
            }
        ],
        live_snapshot={
            "cards": [],
            "transactions": [],
            "transfers": [
                {"recipient": "River Consulting", "status": "pending"},
                {"recipient": "Jamie Lee", "status": "completed"},
            ],
        },
    )

    assert grounded["decision_accepted"] is True
    assert grounded["argument_constraints"] == {"recipient": "River Consulting"}
    assert grounded["entity_grounding_source"] == "trusted_tool_result"
    assert grounded["entity_candidate_count"] == 1


def test_zero_or_multiple_eligible_live_candidates_force_clarification() -> None:
    none = ground_servicing_decision(
        route("freeze_card"),
        live_snapshot=snapshot(cards=[{"last4": "4821", "status": "closed"}]),
    )
    multiple = ground_servicing_decision(
        route("freeze_card"),
        live_snapshot=snapshot(
            cards=[
                {"last4": "4821", "status": "active"},
                {"last4": "7319", "status": "active"},
            ]
        ),
    )

    assert (none["action"], none["entity_resolution"]) == ("clarify", "ineligible")
    assert none["entity_candidate_count"] == 0
    assert (multiple["action"], multiple["entity_resolution"]) == (
        "clarify",
        "ambiguous",
    )
    assert multiple["entity_candidate_count"] == 2
    assert multiple["argument_constraints"] == {}

    missing = ground_servicing_decision(route("freeze_card"), live_snapshot=snapshot())
    assert (missing["action"], missing["entity_resolution"]) == ("clarify", "missing")


def test_exactly_one_eligible_live_candidate_can_ground_the_write() -> None:
    grounded = ground_servicing_decision(
        route("dispute_transaction"),
        live_snapshot={
            "cards": [],
            "transactions": [
                {
                    "description": "North Harbor Market",
                    "amount_cents": -8624,
                    "status": "posted",
                    "disputed": False,
                },
                {
                    "description": "CloudStream",
                    "amount_cents": -1499,
                    "status": "reversed",
                    "disputed": False,
                },
            ],
            "transfers": [],
        },
    )

    assert grounded["argument_constraints"] == {"description": "North Harbor Market"}
    assert grounded["entity_grounding_source"] == "live_candidate"
    assert grounded["entity_candidate_count"] == 1


def test_public_selector_extraction_is_literal_and_candidate_bounded() -> None:
    live = {
        "cards": [{"last4": "4821", "status": "active"}],
        "transactions": [{"description": "North Harbor Market", "status": "posted"}],
        "transfers": [{"recipient": "River Consulting", "status": "pending"}],
    }

    assert public_selectors_from_message("Freeze card ending 4821.", "freeze_card") == {
        "last4": "4821"
    }
    assert public_selectors_from_message(
        "Cancel the River Consulting transfer.",
        "cancel_transfer",
        live_snapshot=live,
    ) == {"recipient": "River Consulting"}
    assert public_selectors_from_message(
        "Dispute North Harbor Market.",
        "dispute_transaction",
        live_snapshot=live,
    ) == {"description": "North Harbor Market"}
    assert public_selectors_from_message("Cancel the transfer.", "cancel_transfer") == {}


@pytest.mark.parametrize("intent", ["view_accounts", "view_cards", "view_transactions"])
def test_selector_free_read_intents_keep_the_effective_execute_decision(intent: str) -> None:
    grounded = ground_servicing_decision(
        route(intent, entity_resolution="not_required"),
        live_snapshot=snapshot(),
    )

    assert grounded["decision_accepted"] is True
    assert grounded["action"] == "execute_tool"
    assert grounded["entity_resolution"] == "not_required"
    assert grounded["argument_constraints"] == {}
    assert grounded["entity_candidate_count"] == 0


@pytest.mark.parametrize(
    ("intent", "lane", "action"),
    [
        ("policy_knowledge", "policy", "retrieve_policy"),
        ("conversation", "conversation", "converse"),
        ("other_banking", "other_banking", "converse"),
    ],
)
def test_non_servicing_joint_decisions_remain_accepted(
    intent: str,
    lane: str,
    action: str,
) -> None:
    grounded = ground_servicing_decision(
        {
            "route": "in_domain",
            "domain": "banking",
            "lane": lane,
            "intent": intent,
            "action": action,
            "entity_resolution": "not_required",
            "joint_decision_accepted": True,
        },
        live_snapshot=snapshot(),
    )

    assert grounded["decision_accepted"] is True
    assert grounded["action"] == action
    assert grounded["entity_resolution"] == "not_required"


def test_legacy_route_remains_on_legacy_confidence_contract() -> None:
    legacy = {
        "route": "in_domain",
        "intent": "view_accounts",
        "intent_confidence": 0.9,
    }

    grounded = ground_servicing_decision(legacy, live_snapshot=snapshot())

    assert grounded == legacy
    assert "effective_decision_contract" not in grounded
