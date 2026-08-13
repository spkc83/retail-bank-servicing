from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

READ_VIEWS = {
    "list_accounts": ("Accounts", "accounts"),
    "list_cards": ("Cards", "cards"),
    "list_service_cases": ("Service cases", "service_cases"),
    "list_transactions": ("Recent transactions", "transactions"),
    "list_transfers": ("Transfers", "transfers"),
}


@dataclass(frozen=True)
class GroundingValidation:
    valid: bool
    errors: tuple[str, ...]


def render_read_tool_results(
    calls: Sequence[Any],
    results: Sequence[Mapping[str, Any]],
) -> str | None:
    """Render successful read-only tool results without asking the model to copy facts."""

    if not calls or len(calls) != len(results):
        return None
    names = [_call_name(call) for call in calls]
    if any(name not in READ_VIEWS for name in names):
        return None
    if any(result.get("ok") is not True for result in results):
        return None

    sections: list[str] = []
    for name, envelope in zip(names, results, strict=True):
        title, collection_name = READ_VIEWS[name]
        result = envelope.get("result")
        if not isinstance(result, Mapping):
            return None
        rows = result.get(collection_name)
        if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
            return None
        sections.append(f"## {title}\n\n{_render_view(name, rows)}")
    return "\n\n".join(sections)


def validate_grounded_answer(
    answer: str,
    calls: Sequence[Any],
    results: Sequence[Mapping[str, Any]],
) -> GroundingValidation:
    """Check action answers against the small set of facts that must be stated exactly."""

    if not isinstance(answer, str) or not answer.strip():
        return GroundingValidation(False, ("final answer is empty",))
    if len(calls) != len(results):
        return GroundingValidation(False, ("tool calls and results are not aligned",))

    normalized = answer.casefold()
    errors: list[str] = []
    for call, envelope in zip(calls, results, strict=True):
        name = _call_name(call)
        for private_id in _private_identifiers(envelope):
            if private_id.casefold() in normalized:
                errors.append("answer exposes a private backend identifier")
        if envelope.get("ok") is not True:
            failure_markers = ("could not", "couldn't", "unable", "not found")
            if not any(marker in normalized for marker in failure_markers):
                errors.append(f"{name} failed but the answer does not disclose the failure")
            continue
        payload = envelope.get("result")
        if not isinstance(payload, Mapping):
            errors.append(f"{name} returned an invalid result envelope")
            continue
        if name in {"freeze_card", "replace_card"}:
            card = payload.get("card")
            if not isinstance(card, Mapping):
                errors.append(f"{name} result is missing the card object")
                continue
            _require_value(answer, card.get("last4"), "card ending digits", errors)
            if name == "freeze_card":
                if "froze" not in normalized and "frozen" not in normalized:
                    errors.append("answer is missing the freeze_card outcome")
            else:
                _require_marker(normalized, "replacement", f"{name} outcome", errors)
        elif name == "dispute_transaction":
            transaction = payload.get("transaction")
            if not isinstance(transaction, Mapping):
                errors.append("dispute_transaction result is missing the transaction object")
                continue
            _require_value(
                answer,
                transaction.get("description"),
                "transaction description",
                errors,
            )
            _require_marker(normalized, "dispute", "dispute outcome", errors)
        elif name == "cancel_transfer":
            transfer = payload.get("transfer")
            if not isinstance(transfer, Mapping):
                errors.append("cancel_transfer result is missing the transfer object")
                continue
            _require_value(answer, transfer.get("recipient"), "transfer recipient", errors)
            if "cancelled" not in normalized and "canceled" not in normalized:
                errors.append("answer is missing the cancel_transfer outcome")
    return GroundingValidation(not errors, tuple(errors))


def build_final_repair_messages(
    *,
    user_message: str,
    draft: str,
    calls: Sequence[Any],
    results: Sequence[Mapping[str, Any]],
    errors: Sequence[str],
) -> list[dict[str, str]]:
    events = [
        {
            "tool": _call_name(call),
            "arguments": _call_arguments(call),
            "result": _without_private_identifiers(result),
        }
        for call, result in zip(calls, results, strict=True)
    ]
    payload = {
        "customer_request": user_message,
        "draft_answer": draft,
        "validation_errors": list(errors),
        "authoritative_tool_events": events,
    }
    return [
        {
            "role": "system",
            "content": (
                "Rewrite a retail-bank assistant answer using only the authoritative tool "
                "events. Correct every validation error. Do not call tools, expose private "
                "internal IDs, or add facts. Return only the concise final answer."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        },
    ]


def _render_view(name: str, rows: list[Mapping[str, Any]]) -> str:
    if not rows:
        return "No records found."
    headers: Sequence[str]
    values: list[Sequence[Any]]
    if name == "list_accounts":
        headers = ("Name", "Type", "Last 4", "Available", "Current", "Status")
        values = [
            (
                row.get("name"),
                row.get("type"),
                row.get("last4"),
                _money(row.get("available_balance_cents"), row.get("currency")),
                _money(row.get("current_balance_cents"), row.get("currency")),
                row.get("status"),
            )
            for row in rows
        ]
    elif name == "list_cards":
        headers = ("Name", "Last 4", "Status", "Digital wallet")
        values = [
            (row.get("name"), row.get("last4"), row.get("status"), row.get("wallet_status"))
            for row in rows
        ]
    elif name == "list_transactions":
        headers = ("Date", "Description", "Amount", "Status", "Category", "Disputed")
        values = [
            (
                _timestamp(row.get("posted_at")),
                row.get("description"),
                _money(row.get("amount_cents"), row.get("currency")),
                row.get("status"),
                row.get("category"),
                "Yes" if row.get("disputed") else "No",
            )
            for row in rows
        ]
    elif name == "list_transfers":
        headers = ("Recipient", "Amount", "Status", "Created")
        values = [
            (
                row.get("recipient"),
                _money(row.get("amount_cents"), row.get("currency")),
                row.get("status"),
                _timestamp(row.get("created_at")),
            )
            for row in rows
        ]
    else:
        headers = ("Created", "Type", "Subject", "Status")
        values = [
            (
                _timestamp(row.get("created_at")),
                row.get("case_type"),
                row.get("subject"),
                row.get("status"),
            )
            for row in rows
        ]
    return _markdown_table(headers, values)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_cell(value) for value in row) + " |" for row in rows]
    return "\n".join((header, divider, *body))


def _cell(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _money(cents: Any, currency: Any) -> str:
    if not isinstance(cents, int) or isinstance(cents, bool):
        return "—"
    sign = "-" if cents < 0 else ""
    return f"{sign}{str(currency or 'USD').upper()} {abs(cents) / 100:,.2f}"


def _timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return "—"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    if parsed.utcoffset() is not None:
        parsed = parsed.astimezone(UTC)
        suffix = " UTC"
    else:
        suffix = ""
    return parsed.strftime("%Y-%m-%d %H:%M") + suffix


def _call_name(call: Any) -> str:
    if isinstance(call, Mapping):
        return str(call.get("name", ""))
    return str(getattr(call, "name", ""))


def _call_arguments(call: Any) -> Mapping[str, Any]:
    value = (
        call.get("arguments")
        if isinstance(call, Mapping)
        else getattr(call, "arguments", None)
    )
    return value if isinstance(value, Mapping) else {}


def _require_value(answer: str, value: Any, label: str, errors: list[str]) -> None:
    if value is not None and str(value).casefold() not in answer.casefold():
        errors.append(f"answer is missing authoritative {label}: {value}")


def _require_marker(normalized: str, marker: str, label: str, errors: list[str]) -> None:
    if not re.search(rf"\b{re.escape(marker)}\w*\b", normalized):
        errors.append(f"answer is missing the {label}")


def _private_identifiers(value: Any) -> set[str]:
    identifiers: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).endswith("_id") and isinstance(item, str) and item:
                identifiers.add(item)
            identifiers.update(_private_identifiers(item))
    elif isinstance(value, list):
        for item in value:
            identifiers.update(_private_identifiers(item))
    return identifiers


def _without_private_identifiers(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_private_identifiers(item)
            for key, item in value.items()
            if not str(key).endswith("_id")
        }
    if isinstance(value, list):
        return [_without_private_identifiers(item) for item in value]
    return value
