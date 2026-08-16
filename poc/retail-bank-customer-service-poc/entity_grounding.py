from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from dialogue_state import EFFECTIVE_DECISION_CONTRACT, MUTATION_INTENTS, SERVICING_TOOLS

_SELECTORS = {
    "freeze_card": ("last4", "cards"),
    "replace_card": ("last4", "cards"),
    "dispute_transaction": ("description", "transactions"),
    "cancel_transfer": ("recipient", "transfers"),
}


def public_selectors_from_message(
    message: str,
    intent: str,
    *,
    live_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Extract only literal public selectors stated in the current user turn."""

    if not isinstance(message, str) or intent not in _SELECTORS:
        return {}
    selector, collection = _SELECTORS[intent]
    if selector == "last4":
        matches = tuple(dict.fromkeys(re.findall(r"(?<!\d)(\d{4})(?!\d)", message)))
        return {selector: matches[0]} if len(matches) == 1 else {}
    records = _records(live_snapshot, collection)
    candidates = _unique_values(records, selector)
    mentioned = [value for value in candidates if _literal_mentioned(message, value)]
    return {selector: mentioned[0]} if len(mentioned) == 1 else {}


def ground_servicing_decision(
    route: Mapping[str, Any],
    *,
    message: str | None = None,
    current_user_values: Mapping[str, Any] | None = None,
    trusted_tool_results: Sequence[Any] | Mapping[str, Any] = (),
    live_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a flat V7 effective decision without trusting learned entity guesses."""

    if "action" not in route:
        return dict(route)
    learned_action = route.get("action")
    learned_resolution = route.get("entity_resolution")
    effective = {
        **route,
        "effective_decision_contract": EFFECTIVE_DECISION_CONTRACT,
        "decision_accepted": False,
        "tool_execution_allowed": False,
        "learned_action": learned_action,
        "learned_entity_resolution": learned_resolution,
        "argument_constraints": {},
        "entity_grounding_source": "not_applicable",
        "entity_candidate_count": 0,
    }
    intent = route.get("fine_intent") or route.get("intent")
    if (
        route.get("route") != "in_domain"
        or route.get(
            "joint_decision_accepted",
            True,
        )
        is not True
    ):
        return effective
    if learned_action != "execute_tool":
        effective["decision_accepted"] = True
        return effective
    if not isinstance(intent, str) or intent not in SERVICING_TOOLS:
        return effective
    if intent not in MUTATION_INTENTS:
        effective["decision_accepted"] = True
        effective["tool_execution_allowed"] = True
        return effective

    selector, collection = _SELECTORS[intent]
    live_records = _records(live_snapshot, collection)
    eligible = [record for record in live_records if _eligible(intent, record)]
    explicit = _clean_selector(current_user_values, selector)
    if explicit is None and message is not None:
        explicit = public_selectors_from_message(
            message,
            intent,
            live_snapshot=live_snapshot,
        ).get(selector)
    if explicit is not None:
        matches = _matching_records(eligible, selector, explicit)
        return _resolved_or_clarify(
            effective,
            selector=selector,
            value=explicit,
            matches=matches,
            source="current_user",
            ineligible=_value_exists(live_records, selector, explicit),
        )

    trusted_values = _trusted_selector_values(trusted_tool_results, selector)
    if trusted_values:
        matches = [
            record
            for record in eligible
            if any(_same_public_value(record.get(selector), value) for value in trusted_values)
        ]
        if len(matches) == 1:
            value = str(matches[0][selector])
            return _accepted(effective, selector, value, "trusted_tool_result", 1)
        return _clarify(
            effective,
            "ambiguous" if len(matches) > 1 else "ineligible",
            "trusted_tool_result",
            len(matches),
        )

    if len(eligible) == 1:
        value = str(eligible[0][selector])
        return _accepted(effective, selector, value, "live_candidate", 1)
    resolution = "ambiguous" if len(eligible) > 1 else "ineligible" if live_records else "missing"
    return _clarify(effective, resolution, "live_candidate", len(eligible))


def eligible_public_candidates(
    intent: str,
    live_snapshot: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], ...]:
    """Expose the same eligibility view used by effective grounding."""

    spec = _SELECTORS.get(intent)
    if spec is None:
        return ()
    return tuple(record for record in _records(live_snapshot, spec[1]) if _eligible(intent, record))


def _resolved_or_clarify(
    effective: dict[str, Any],
    *,
    selector: str,
    value: str,
    matches: list[Mapping[str, Any]],
    source: str,
    ineligible: bool,
) -> dict[str, Any]:
    if len(matches) == 1:
        canonical = str(matches[0][selector])
        return _accepted(effective, selector, canonical, source, 1)
    resolution = "ambiguous" if len(matches) > 1 else "ineligible" if ineligible else "missing"
    return _clarify(effective, resolution, source, len(matches))


def _accepted(
    effective: dict[str, Any],
    selector: str,
    value: str,
    source: str,
    count: int,
) -> dict[str, Any]:
    effective.update(
        decision_accepted=True,
        tool_execution_allowed=True,
        action="execute_tool",
        entity_resolution="resolved",
        argument_constraints={selector: value},
        entity_grounding_source=source,
        entity_candidate_count=count,
    )
    return effective


def _clarify(
    effective: dict[str, Any],
    resolution: str,
    source: str,
    count: int,
) -> dict[str, Any]:
    effective.update(
        decision_accepted=True,
        tool_execution_allowed=False,
        action="clarify",
        entity_resolution=resolution,
        argument_constraints={},
        entity_grounding_source=source,
        entity_candidate_count=count,
    )
    return effective


def _records(
    snapshot: Mapping[str, Any] | None,
    collection: str,
) -> list[Mapping[str, Any]]:
    if not isinstance(snapshot, Mapping):
        return []
    value = snapshot.get(collection)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return [record for record in value if isinstance(record, Mapping)]


def _eligible(intent: str, record: Mapping[str, Any]) -> bool:
    if intent == "freeze_card":
        return record.get("status") == "active"
    if intent == "replace_card":
        return record.get("status") in {"active", "frozen"}
    if intent == "dispute_transaction":
        amount = record.get("amount_cents")
        return (
            isinstance(amount, int)
            and not isinstance(amount, bool)
            and amount < 0
            and record.get("status") == "posted"
            and record.get("disputed") is False
        )
    if intent == "cancel_transfer":
        return record.get("status") == "pending"
    return False


def _clean_selector(values: Mapping[str, Any] | None, selector: str) -> str | None:
    if not isinstance(values, Mapping):
        return None
    value = values.get(selector)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _matching_records(
    records: Sequence[Mapping[str, Any]],
    selector: str,
    value: str,
) -> list[Mapping[str, Any]]:
    return [record for record in records if _same_public_value(record.get(selector), value)]


def _value_exists(records: Sequence[Mapping[str, Any]], selector: str, value: str) -> bool:
    return any(_same_public_value(record.get(selector), value) for record in records)


def _same_public_value(left: Any, right: Any) -> bool:
    return isinstance(left, str) and isinstance(right, str) and left.casefold() == right.casefold()


def _unique_values(records: Sequence[Mapping[str, Any]], selector: str) -> list[str]:
    values: dict[str, str] = {}
    for record in records:
        value = record.get(selector)
        if isinstance(value, str) and value:
            values.setdefault(value.casefold(), value)
    return list(values.values())


def _literal_mentioned(message: str, value: str) -> bool:
    return (
        re.search(
            rf"(?<!\w){re.escape(value)}(?!\w)",
            message,
            flags=re.IGNORECASE,
        )
        is not None
    )


def _trusted_selector_values(payload: Any, selector: str) -> list[str]:
    values: dict[str, str] = {}

    def visit(value: Any, *, trusted: bool) -> None:
        if isinstance(value, str) and trusted:
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                return
            visit(decoded, trusted=True)
            return
        if isinstance(value, Mapping):
            role = value.get("role")
            if isinstance(role, str) and role != "tool":
                return
            next_trusted = trusted or role == "tool"
            if next_trusted:
                candidate = value.get(selector)
                if isinstance(candidate, str) and candidate:
                    values.setdefault(candidate.casefold(), candidate)
            for child in value.values():
                visit(child, trusted=next_trusted)
            return
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            for child in value:
                visit(child, trusted=trusted)

    visit(payload, trusted=True)
    return list(values.values())
