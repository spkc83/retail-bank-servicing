from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from dialogue_state import (
    DialogueState,
    DialogueStateRegistry,
    PendingServicing,
    begin_turn,
    commit_operations,
    finish_turn,
)


def route(
    intent: str,
    *,
    confidence: float = 0.99,
    relations: tuple[str, ...] = (),
    route_name: str = "in_domain",
) -> dict[str, object]:
    return {
        "intent": intent,
        "intent_confidence": confidence,
        "active_relations": list(relations),
        "route": route_name,
    }


def pending_state(intent: str = "dispute_transaction") -> DialogueState:
    return DialogueState(
        pending_servicing=PendingServicing(
            intent=intent,
            anchor_user_message="I need to dispute a purchase.",
            anchor_assistant_message="Which transaction should I look for?",
        )
    )


def test_confident_servicing_turn_starts_and_finishes_an_anchored_task() -> None:
    transition = begin_turn(
        DialogueState(),
        route("dispute_transaction"),
        "I need to dispute a purchase.",
    )

    assert transition.lane == "servicing"
    assert transition.state.pending_servicing == PendingServicing(
        intent="dispute_transaction",
        anchor_user_message="I need to dispute a purchase.",
        anchor_assistant_message="",
        phase="awaiting_assistant",
    )

    state = finish_turn(transition.state, "Which transaction?", (), ())

    assert state.pending_servicing == PendingServicing(
        intent="dispute_transaction",
        anchor_user_message="I need to dispute a purchase.",
        anchor_assistant_message="Which transaction?",
    )


def test_same_intent_follow_up_keeps_original_anchor() -> None:
    state = pending_state()

    transition = begin_turn(state, route("dispute_transaction"), "The coffee shop charge.")

    assert transition.state == state
    assert transition.previous_pending is None


def test_policy_detour_preserves_task_and_resume_returns_anchor_exchange() -> None:
    state = pending_state()
    detour = begin_turn(state, route("policy_knowledge"), "How long do disputes take?")

    assert detour.lane == "policy"
    assert detour.state.pending_servicing == state.pending_servicing
    assert detour.state.knowledge_detour_active is True

    resumed = begin_turn(
        detour.state,
        route("conversation", relations=("resume_previous_service",)),
        "Let's continue.",
    )

    assert resumed.resumed is True
    assert resumed.state.knowledge_detour_active is False
    assert resumed.previous_pending == state.pending_servicing
    assert resumed.anchor_exchange == (
        {"role": "user", "content": "I need to dispute a purchase."},
        {"role": "assistant", "content": "Which transaction should I look for?"},
    )


def test_same_servicing_intent_implicitly_resumes_policy_detour() -> None:
    state = DialogueState(
        pending_servicing=pending_state("replace_card").pending_servicing,
        knowledge_detour_active=True,
    )

    resumed = begin_turn(
        state,
        route("replace_card"),
        "Okay, replace the card ending in 4821.",
    )

    assert resumed.lane == "servicing"
    assert resumed.resumed is True
    assert resumed.state.knowledge_detour_active is False
    assert resumed.anchor_exchange[0]["content"] == "I need to dispute a purchase."


def test_different_servicing_intent_replaces_old_task_without_stale_resume() -> None:
    detour = DialogueState(
        pending_servicing=pending_state().pending_servicing,
        knowledge_detour_active=True,
    )

    switched = begin_turn(detour, route("freeze_card"), "Freeze my card.")

    assert switched.state.pending_servicing is not None
    assert switched.state.pending_servicing.intent == "freeze_card"
    assert switched.state.knowledge_detour_active is False
    assert switched.previous_pending is None

    resumed = begin_turn(
        switched.state,
        route("conversation", relations=("resume_previous_service",)),
        "Continue the dispute.",
    )
    assert resumed.resumed is False
    assert resumed.state == switched.state


def test_ood_uncertain_social_and_classifier_failure_do_not_mutate_state() -> None:
    state = DialogueState(
        pending_servicing=pending_state().pending_servicing,
        knowledge_detour_active=True,
    )
    routes = (
        route("freeze_card", route_name="out_of_domain"),
        route("freeze_card", route_name="uncertain"),
        route("conversation"),
        {"route": "classifier_failure"},
    )

    for observation in routes:
        assert begin_turn(state, observation, "message").state == state


def test_uncertain_resume_observation_cannot_resume_pending_service() -> None:
    state = DialogueState(
        pending_servicing=pending_state().pending_servicing,
        knowledge_detour_active=True,
    )

    transition = begin_turn(
        state,
        route(
            "dispute_transaction",
            route_name="uncertain",
            relations=("resume_previous_service",),
        ),
        "Continue the dispute.",
    )

    assert transition.state == state
    assert transition.resumed is False
    assert transition.lane == "unchanged"


def test_low_confidence_intent_does_not_mutate_state() -> None:
    state = pending_state()

    result = begin_turn(state, route("freeze_card", confidence=0.49), "Freeze it.")

    assert result.state == state
    assert result.lane == "unchanged"


def test_accepted_v7_joint_decision_replaces_stale_state_without_raw_head_threshold() -> None:
    state = pending_state("view_service_cases")
    observation = {
        **route("view_transfers", confidence=0.329),
        "lane": "servicing",
        "action": "execute_tool",
        "entity_resolution": "not_required",
        "effective_decision_contract": "retail-bank-effective-turn-decision/v1",
        "decision_accepted": True,
    }

    result = begin_turn(
        state,
        observation,
        "What became of the transfer I sent a short while ago?",
    )

    assert result.lane == "servicing"
    assert result.state.pending_servicing is not None
    assert result.state.pending_servicing.intent == "view_transfers"


def test_unaccepted_v7_joint_decision_does_not_mutate_state() -> None:
    state = pending_state("view_service_cases")
    observation = {
        **route("view_transfers", confidence=0.99),
        "lane": "servicing",
        "effective_decision_contract": "retail-bank-effective-turn-decision/v1",
        "decision_accepted": False,
    }

    result = begin_turn(state, observation, "Check the transfer.")

    assert result.state == state
    assert result.lane == "unchanged"


def test_only_successful_matching_tool_completion_clears_pending_task() -> None:
    state = pending_state()

    wrong_tool = finish_turn(
        state,
        "Here are your cards.",
        ("list_cards",),
        ({"cards": []},),
    )
    failed_tool = finish_turn(
        state,
        "I couldn't submit it.",
        ("dispute_transaction",),
        ({"error": "not found"},),
    )
    completed = finish_turn(
        state,
        "Your dispute was submitted.",
        ("dispute_transaction",),
        ({"transaction": {"disputed": True}},),
    )

    assert wrong_tool.pending_servicing is not None
    assert failed_tool.pending_servicing is not None
    assert completed == DialogueState()


def test_successful_matching_read_completes_view_task() -> None:
    state = pending_state("view_accounts")

    completed = finish_turn(
        state,
        "Here are your accounts.",
        ("list_accounts",),
        ({"accounts": []},),
    )

    assert completed == DialogueState()


def test_successful_read_is_not_committed_before_response_delivery() -> None:
    state = pending_state("view_accounts")

    committed = commit_operations(
        state,
        ("list_accounts",),
        ({"accounts": []},),
    )

    assert committed == state


def test_successful_mutation_is_committed_before_response_delivery() -> None:
    state = pending_state("freeze_card")

    committed = commit_operations(
        state,
        ("freeze_card",),
        ({"card": {"status": "frozen"}},),
    )

    assert committed == DialogueState()


def test_registry_isolated_by_username_and_session_and_serializes_diagnostics() -> None:
    registry = DialogueStateRegistry()
    state = pending_state()

    registry.set("alex", "browser-a", state)

    assert registry.get("alex", "browser-a") == state
    assert registry.get("alex", "browser-b") == DialogueState()
    assert registry.get("sam", "browser-a") == DialogueState()
    assert registry.as_dict("alex", "browser-a") == {
        "version": 1,
        "pending_servicing": {
            "intent": "dispute_transaction",
            "anchor_user_message": "I need to dispute a purchase.",
            "anchor_assistant_message": "Which transaction should I look for?",
            "phase": "awaiting_user",
        },
        "knowledge_detour_active": False,
    }

    registry.reset("alex", "browser-a")
    assert registry.get("alex", "browser-a") == DialogueState()


def test_registry_begin_turn_is_atomic_for_independent_sessions() -> None:
    registry = DialogueStateRegistry()

    def start(index: int) -> None:
        registry.begin_turn(
            "alex",
            f"session-{index}",
            route("view_accounts"),
            f"Show accounts {index}",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(start, range(50)))

    assert all(
        registry.get("alex", f"session-{index}").pending_servicing is not None
        for index in range(50)
    )


def test_dialogue_state_round_trips_through_ui_payload() -> None:
    state = DialogueState(
        pending_servicing=pending_state("replace_card").pending_servicing,
        knowledge_detour_active=True,
    )

    assert DialogueState.from_dict(state.as_dict()) == state


def test_dialogue_state_rejects_unknown_pending_intent() -> None:
    payload = {
        "version": 1,
        "pending_servicing": {
            "intent": "invented_action",
            "anchor_user_message": "Do something",
        },
    }

    with pytest.raises(ValueError, match="invalid pending servicing intent"):
        DialogueState.from_dict(payload)
