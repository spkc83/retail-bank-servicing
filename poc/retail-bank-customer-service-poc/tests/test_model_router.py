from __future__ import annotations

from typing import Any

import pytest
from model_router import (
    ModelConversationRouter,
    enforce_legality,
    parse_decision,
)
from model_service import _generation_plan


class ScriptedModel:
    def __init__(self, *outputs: str) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def generate(self, messages, tools, max_new_tokens, *, sample=False, prefill=""):
        self.calls.append({"messages": messages, "tools": tools})
        return self.outputs.pop(0)


class ExplodingModel:
    def generate(self, *_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("the GPU worker went away")


def route_for(output: str) -> dict[str, Any]:
    return ModelConversationRouter(ScriptedModel(output)).classify("hi", [])


def test_a_clean_decision_becomes_the_routers_own_shape() -> None:
    result = route_for(
        '{"domain": "banking", "intent": "view_cards", '
        '"action": "execute_tool", "entity_resolution": "not_required"}'
    )

    assert result["route"] == "in_domain"
    assert (result["action"], result["fine_intent"]) == ("execute_tool", "view_cards")
    assert result["lane"] == "servicing"
    assert result["classifier"] == "model"


@pytest.mark.parametrize(
    "raw",
    [
        "I can help you with that!",  # no JSON at all
        '{"domain": "banking", "intent": "view_cards"}',  # missing fields
        '{"domain": "banking", "intent": "buy_stocks", '
        '"action": "execute_tool", "entity_resolution": "resolved"}',  # unknown intent
        '{"domain": "banking", "intent": "view_cards", '
        '"action": "do_it", "entity_resolution": "resolved"}',  # unknown action
        "{not json at all}",
    ],
)
def test_anything_unparsable_fails_closed(raw: str) -> None:
    """An unusable classification must reach the same branch a router outage reaches."""
    result = route_for(raw)

    assert "action" not in result
    assert "tool_authority" not in result, "a failed classification must not unlock every tool"
    _system, tools = _generation_plan(result)
    assert tools == []


def test_a_classifier_outage_fails_closed_rather_than_propagating() -> None:
    result = ModelConversationRouter(ExplodingModel()).classify("freeze my card", [])

    assert "action" not in result
    _system, tools = _generation_plan(result)
    assert tools == []


def test_an_unknown_label_is_rejected_not_snapped_to_the_nearest_legal_one() -> None:
    assert parse_decision('{"domain": "bank", "intent": "view_cards", '
                          '"action": "execute_tool", "entity_resolution": "resolved"}') is None


def test_a_mutation_may_not_be_conversed_through() -> None:
    """The constraint the learned router enforces, enforced identically here."""
    decision, notes = enforce_legality(
        {
            "domain": "banking",
            "intent": "freeze_card",
            "action": "converse",
            "entity_resolution": "not_required",
        }
    )

    assert decision["action"] == "clarify"
    assert decision["entity_resolution"] != "not_required"
    assert "constraint:mutation-intent-cannot-converse" in notes


def test_an_unresolved_target_cannot_execute() -> None:
    decision, notes = enforce_legality(
        {
            "domain": "banking",
            "intent": "replace_card",
            "action": "execute_tool",
            "entity_resolution": "ambiguous",
        }
    )

    assert decision["action"] == "clarify"
    assert "constraint:unresolved-target-cannot-execute" in notes
    _system, tools = _generation_plan({**decision, "fine_intent": decision["intent"]})
    assert tools == [], "an ambiguous target must expose no tool"


def test_a_policy_question_cannot_be_routed_to_a_tool() -> None:
    decision, notes = enforce_legality(
        {
            "domain": "banking",
            "intent": "policy_knowledge",
            "action": "execute_tool",
            "entity_resolution": "resolved",
        }
    )

    assert decision["action"] == "retrieve_policy"
    assert "constraint:intent-exposes-no-tool" in notes


def test_out_of_domain_must_refuse_whatever_action_was_proposed() -> None:
    decision, notes = enforce_legality(
        {
            "domain": "out_of_domain",
            "intent": "view_cards",
            "action": "execute_tool",
            "entity_resolution": "resolved",
        }
    )

    assert decision["action"] == "refuse_ood"
    assert "constraint:out-of-domain-must-refuse" in notes


def test_corrections_are_reported_rather_than_silently_applied() -> None:
    """Illegal-tuple rate is a headline difference; it must stay measurable."""
    result = route_for(
        '{"domain": "banking", "intent": "freeze_card", '
        '"action": "converse", "entity_resolution": "not_required"}'
    )

    assert result["proposed"]["action"] == "converse"
    assert result["action"] == "clarify"
    assert "constraint:mutation-intent-cannot-converse" in result["constraint_diagnostics"]


def test_the_routing_pass_is_given_no_tools() -> None:
    """Classification must not be able to execute anything by itself."""
    model = ScriptedModel(
        '{"domain": "banking", "intent": "view_cards", '
        '"action": "execute_tool", "entity_resolution": "not_required"}'
    )

    ModelConversationRouter(model).classify("show my cards", [])

    assert model.calls[0]["tools"] is None


def test_both_classifiers_reach_the_same_tool_surface_for_the_same_decision() -> None:
    """The comparison is only meaningful if the harness treats both alike.

    A model decision and a learned-router decision carrying the same tuple must
    produce the same exposed tools, or the experiment measures the harness
    rather than the classifier.
    """
    tuples = [
        ("view_cards", "execute_tool", "not_required"),
        ("replace_card", "execute_tool", "resolved"),
        ("replace_card", "clarify", "ambiguous"),
        ("policy_knowledge", "retrieve_policy", "not_required"),
        ("conversation", "converse", "not_required"),
    ]

    for intent, action, entity in tuples:
        learned = {
            "route": "in_domain",
            "domain": "banking",
            "lane": "servicing",
            "fine_intent": intent,
            "intent": intent,
            "action": action,
            "entity_resolution": entity,
        }
        model = ModelConversationRouter(
            ScriptedModel(
                '{"domain": "banking", "intent": "%s", "action": "%s", '
                '"entity_resolution": "%s"}' % (intent, action, entity)
            )
        ).classify("x", [])

        assert model["action"] == action, f"{intent}: legality moved a legal tuple"
        _s1, learned_tools = _generation_plan(learned)
        _s2, model_tools = _generation_plan(model)
        assert learned_tools == model_tools, f"{intent}/{action}/{entity} gates differently"
