"""Canonical hierarchy and label policy for the conversation router."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

DOMAIN_LABELS = ("out_of_domain", "banking", "social")
LANE_LABELS = ("out_of_domain", "servicing", "policy", "conversation", "other_banking")
FAMILY_LABELS = (
    "external",
    "accounts",
    "cards",
    "transactions",
    "transfers",
    "service_cases",
    "policy",
    "social",
    "other_banking",
)
INTENT_LABELS = (
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
ACTION_LABELS = ("refuse_ood", "execute_tool", "clarify", "retrieve_policy", "converse")
ENTITY_RESOLUTION_LABELS = ("not_required", "resolved", "missing", "ambiguous", "ineligible")

_INDEXES = {
    "domain": {name: index for index, name in enumerate(DOMAIN_LABELS)},
    "lane": {name: index for index, name in enumerate(LANE_LABELS)},
    "family": {name: index for index, name in enumerate(FAMILY_LABELS)},
    "action": {name: index for index, name in enumerate(ACTION_LABELS)},
    "entity_resolution": {name: index for index, name in enumerate(ENTITY_RESOLUTION_LABELS)},
}

_INTENT_HIERARCHY = {
    "view_accounts": ("banking", "servicing", "accounts"),
    "view_cards": ("banking", "servicing", "cards"),
    "freeze_card": ("banking", "servicing", "cards"),
    "replace_card": ("banking", "servicing", "cards"),
    "view_transactions": ("banking", "servicing", "transactions"),
    "dispute_transaction": ("banking", "servicing", "transactions"),
    "view_transfers": ("banking", "servicing", "transfers"),
    "cancel_transfer": ("banking", "servicing", "transfers"),
    "view_service_cases": ("banking", "servicing", "service_cases"),
    "policy_knowledge": ("banking", "policy", "policy"),
    "conversation": ("social", "conversation", "social"),
    "other_banking": ("banking", "other_banking", "other_banking"),
}

INTENT_TOOL_COMPATIBILITY = {
    "view_accounts": frozenset({"list_accounts"}),
    "view_cards": frozenset({"list_cards"}),
    "freeze_card": frozenset({"list_cards", "freeze_card"}),
    "replace_card": frozenset({"list_cards", "replace_card"}),
    "view_transactions": frozenset({"list_transactions"}),
    "dispute_transaction": frozenset({"list_transactions", "dispute_transaction"}),
    "view_transfers": frozenset({"list_transfers"}),
    "cancel_transfer": frozenset({"list_transfers", "cancel_transfer"}),
    "view_service_cases": frozenset({"list_service_cases"}),
    "policy_knowledge": frozenset(),
    "conversation": frozenset(),
    "other_banking": frozenset(),
}

_ENTITY_REQUIRED_INTENTS = frozenset(
    {"freeze_card", "replace_card", "dispute_transaction", "cancel_transfer"}
)


def hierarchy_for_intent(intent: str | None) -> tuple[str, str, str]:
    if intent is None:
        return "out_of_domain", "out_of_domain", "external"
    try:
        return _INTENT_HIERARCHY[intent]
    except KeyError as error:
        raise ValueError(f"unsupported intent: {intent}") from error


def lane_for_intent(intent: str) -> str:
    """Return the canonical lane while accepting legacy V2 capability names."""
    legacy_servicing_capabilities = {
        "accounts",
        "cards",
        "card_actions",
        "transactions",
        "transfers",
        "service_cases",
    }
    if intent in legacy_servicing_capabilities:
        return "servicing"
    if intent == "faq":
        return "policy"
    return hierarchy_for_intent(intent)[1]


def derive_action_entity(
    *,
    intent: str | None,
    path: str = "",
    tool_names: Sequence[str] | None = None,
    coreference_target: str = "",
    actionable_entity_count: int | None = None,
    explicit_entity_resolution: str = "",
) -> tuple[str, str]:
    """Derive action/entity supervision from pre-existing governed annotations."""
    if intent is None:
        return "refuse_ood", "not_required"
    if intent == "policy_knowledge":
        return "retrieve_policy", "not_required"
    if intent in {"conversation", "other_banking"}:
        return "converse", "not_required"

    if explicit_entity_resolution:
        if explicit_entity_resolution not in ENTITY_RESOLUTION_LABELS:
            raise ValueError(f"unsupported entity resolution: {explicit_entity_resolution}")
        if explicit_entity_resolution in {"missing", "ambiguous"}:
            return "clarify", explicit_entity_resolution
        if explicit_entity_resolution == "ineligible":
            return "converse", "ineligible"

    normalized_path = path.casefold()
    normalized_target = coreference_target.casefold()
    if normalized_target == "clarification" or normalized_path == "clarification":
        resolution = (
            "ambiguous" if actionable_entity_count and actionable_entity_count > 1 else "missing"
        )
        return "clarify", resolution

    if (
        normalized_target == intent
        and actionable_entity_count == 1
        and intent in (tool_names or ())
    ):
        return "execute_tool", "resolved"

    if tool_names is not None and not tool_names:
        return "converse", "not_required"

    entity_resolution = "resolved" if intent in _ENTITY_REQUIRED_INTENTS else "not_required"
    return "execute_tool", entity_resolution


def labels_for_example(
    *,
    intent: str | None,
    action: str | None = None,
    entity_resolution: str | None = None,
    tool_names: Sequence[str] | None = None,
    path: str = "",
    coreference_target: str = "",
    actionable_entity_count: int | None = None,
    explicit_entity_resolution: str = "",
) -> dict[str, Any]:
    domain, lane, family = hierarchy_for_intent(intent)
    default_action, default_entity = derive_action_entity(
        intent=intent,
        path=path,
        tool_names=tool_names,
        coreference_target=coreference_target,
        actionable_entity_count=actionable_entity_count,
        explicit_entity_resolution=explicit_entity_resolution,
    )
    action = action or default_action
    entity_resolution = entity_resolution or default_entity
    labels: dict[str, Any] = {
        "intent": intent,
        "domain_name": domain,
        "domain_index": _INDEXES["domain"][domain],
        "lane_name": lane,
        "lane_index": _INDEXES["lane"][lane],
        "family_name": family,
        "family_index": _INDEXES["family"][family],
        "action_name": action,
        "action_index": _index("action", action),
        "entity_resolution_name": entity_resolution,
        "entity_resolution_index": _index("entity_resolution", entity_resolution),
    }
    validate_hierarchical_labels(labels, intent=intent, tool_names=tool_names)
    return labels


def validate_hierarchical_labels(
    labels: Mapping[str, Any],
    *,
    intent: str | None = None,
    tool_names: Sequence[str] | None = None,
) -> None:
    intent = intent if intent is not None else _optional_string(labels.get("intent"))
    expected_domain, expected_lane, expected_family = hierarchy_for_intent(intent)
    actual_domain = str(labels.get("domain_name", ""))
    actual_lane = str(labels.get("lane_name", labels.get("lane", "")))
    actual_family = str(labels.get("family_name", ""))
    for dimension, actual, expected in (
        ("domain", actual_domain, expected_domain),
        ("lane", actual_lane, expected_lane),
        ("family", actual_family, expected_family),
    ):
        if actual != expected:
            subject = f"intent {intent}" if intent is not None else "out-of-domain examples"
            raise ValueError(f"{subject} requires {dimension} {expected}, got {actual}")
        index = labels.get(f"{dimension}_index")
        if index is not None and int(index) != _INDEXES[dimension][actual]:
            raise ValueError(f"{dimension} index does not match {actual}")

    action = str(labels.get("action_name", ""))
    resolution = str(labels.get("entity_resolution_name", ""))
    _index("action", action)
    _index("entity_resolution", resolution)
    if (
        labels.get("action_index") is not None
        and int(labels["action_index"]) != _INDEXES["action"][action]
    ):
        raise ValueError(f"action index does not match {action}")
    if (
        labels.get("entity_resolution_index") is not None
        and int(labels["entity_resolution_index"]) != _INDEXES["entity_resolution"][resolution]
    ):
        raise ValueError(f"entity resolution index does not match {resolution}")

    if action == "refuse_ood" and intent is not None:
        raise ValueError("refuse_ood is only valid for out-of-domain examples")
    if intent is None and (action != "refuse_ood" or resolution != "not_required"):
        raise ValueError("out-of-domain examples require refuse_ood and not_required")
    if action == "retrieve_policy" and intent != "policy_knowledge":
        raise ValueError("retrieve_policy requires policy_knowledge intent")
    if action == "clarify" and resolution not in {"missing", "ambiguous"}:
        raise ValueError("clarify requires missing or ambiguous entity resolution")
    if action == "execute_tool":
        if expected_lane != "servicing":
            raise ValueError("execute_tool requires a servicing intent")
        if resolution in {"missing", "ambiguous", "ineligible"}:
            raise ValueError("execute_tool requires a usable entity resolution")
        if tool_names is not None:
            if not tool_names:
                raise ValueError("execute_tool requires at least one compatible tool")
            compatible = INTENT_TOOL_COMPATIBILITY[intent or ""]
            for tool_name in tool_names:
                if tool_name not in compatible:
                    raise ValueError(f"tool {tool_name} is incompatible with intent {intent}")


def _index(dimension: str, name: str) -> int:
    try:
        return _INDEXES[dimension][name]
    except KeyError as error:
        raise ValueError(f"unsupported {dimension} label: {name}") from error


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
