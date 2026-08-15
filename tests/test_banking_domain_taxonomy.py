from __future__ import annotations

import pytest

from hello_slm.banking_domain_taxonomy import (
    hierarchy_for_intent,
    labels_for_example,
    validate_hierarchical_labels,
)


def test_taxonomy_maps_servicing_policy_social_and_ood_examples() -> None:
    assert hierarchy_for_intent("replace_card") == ("banking", "servicing", "cards")
    assert hierarchy_for_intent("policy_knowledge") == ("banking", "policy", "policy")
    assert hierarchy_for_intent("conversation") == ("social", "conversation", "social")
    assert hierarchy_for_intent(None) == ("out_of_domain", "out_of_domain", "external")


def test_taxonomy_rejects_incompatible_hierarchy_action_and_entity_combinations() -> None:
    valid = labels_for_example(
        intent="replace_card",
        action="execute_tool",
        entity_resolution="resolved",
        tool_names=("replace_card",),
    )
    validate_hierarchical_labels(valid, tool_names=("replace_card",))

    with pytest.raises(ValueError, match="intent replace_card requires family cards"):
        validate_hierarchical_labels({**valid, "family_name": "accounts"})
    with pytest.raises(ValueError, match="execute_tool requires at least one compatible tool"):
        validate_hierarchical_labels(valid, tool_names=())
    with pytest.raises(ValueError, match="tool .* is incompatible with intent replace_card"):
        validate_hierarchical_labels(valid, tool_names=("cancel_transfer",))
    with pytest.raises(ValueError, match="clarify requires missing or ambiguous entity resolution"):
        validate_hierarchical_labels(
            labels_for_example(
                intent="replace_card",
                action="clarify",
                entity_resolution="resolved",
            )
        )


@pytest.mark.parametrize("resolution", ["missing", "ambiguous"])
def test_explicit_unresolved_entity_requires_clarification(resolution: str) -> None:
    labels = labels_for_example(
        intent="replace_card",
        explicit_entity_resolution=resolution,
    )

    assert labels["action_name"] == "clarify"
    assert labels["entity_resolution_name"] == resolution
