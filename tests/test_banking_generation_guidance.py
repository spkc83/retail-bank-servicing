from __future__ import annotations

from typing import Any

import pytest
from model_service import (  # type: ignore[import-not-found]
    _render_turn_guidance as render_poc_turn_guidance,
)

from hello_slm.banking_generation_guidance import (
    messages_with_turn_guidance,
    render_turn_guidance,
)


@pytest.mark.parametrize(
    ("contract", "expected_fragment"),
    [
        (
            {
                "mode": "execute_tool",
                "entity_state": "resolved",
                "tool_names": ["freeze_card"],
                "argument_constraints": {"last4": {"const": "4821"}},
            },
            "Use only freeze_card for this turn",
        ),
        (
            {
                "mode": "clarify",
                "entity_state": "ambiguous",
                "tool_names": [],
                "argument_constraints": {},
            },
            "distinguish which banking item or event they mean",
        ),
        (
            {
                "mode": "converse",
                "entity_state": "not_required",
                "tool_names": [],
                "argument_constraints": {},
            },
            "Respond naturally and concisely",
        ),
        (
            {
                "mode": "retrieve_policy",
                "entity_state": "not_required",
                "tool_names": [],
                "argument_constraints": {},
            },
            "Answer the banking policy question",
        ),
        (
            {
                "mode": "refuse_ood",
                "entity_state": "not_required",
                "tool_names": [],
                "argument_constraints": {},
            },
            "only help with retail banking and financial-services questions",
        ),
    ],
)
def test_shared_and_standalone_poc_guidance_are_contract_equivalent(
    contract: dict[str, Any], expected_fragment: str
) -> None:
    shared = render_turn_guidance(contract)

    assert shared == render_poc_turn_guidance(contract)
    assert expected_fragment in shared
    assert all(term not in shared.lower() for term in ("classifier", "logit", "label", "router"))


def test_messages_with_turn_guidance_appends_to_system_without_mutating_input() -> None:
    messages = [
        {"role": "system", "content": "Bank system", "loss": False},
        {"role": "user", "content": "Freeze card 4821", "loss": False},
    ]
    contract = {
        "mode": "execute_tool",
        "entity_state": "resolved",
        "tool_names": ["freeze_card"],
        "argument_constraints": {"last4": {"const": "4821"}},
        "action_logits": [99.0],
        "classifier_label": "private-internal-label",
    }

    guided = messages_with_turn_guidance(messages, contract)

    assert messages[0]["content"] == "Bank system"
    assert guided[0]["content"].startswith("Bank system\n\nTURN GUIDANCE: ")
    assert "exactly as constrained by the exposed schema" in guided[0]["content"]
    assert "99.0" not in guided[0]["content"]
    assert "private-internal-label" not in guided[0]["content"]
    assert guided[0]["loss"] is False


def test_contractless_messages_preserve_legacy_prompt() -> None:
    messages = [{"role": "system", "content": "Legacy system", "loss": False}]

    assert messages_with_turn_guidance(messages, None) == messages


def test_poc_realizer_filler_copies_match_the_training_realizer_pools() -> None:
    from response_policy import (  # type: ignore[import-not-found]
        REALIZER_FILLER_CLOSERS,
        REALIZER_FILLER_PREFIXES,
    )

    from hello_slm.banking_tool_sft_data import (
        REALIZER_FINAL_CLOSERS,
        REALIZER_FINAL_PREFIXES,
    )

    assert set(REALIZER_FILLER_PREFIXES) == set(filter(None, REALIZER_FINAL_PREFIXES))
    assert set(REALIZER_FILLER_CLOSERS) == set(filter(None, REALIZER_FINAL_CLOSERS))
