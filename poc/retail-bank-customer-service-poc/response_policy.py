from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

READ_VIEWS = {
    "list_accounts": ("Accounts", "accounts"),
    "list_cards": ("Cards", "cards"),
    "list_service_cases": ("Service cases", "service_cases"),
    "list_transactions": ("Recent transactions", "transactions"),
    "list_transfers": ("Transfers", "transfers"),
}

INTERNAL_LANGUAGE_PATTERNS = {
    "synthetic": re.compile(r"\bsynthetic\b", re.IGNORECASE),
    "demo": re.compile(r"\bdemo(?:nstration)?\b", re.IGNORECASE),
    "mock": re.compile(r"\bmock\b", re.IGNORECASE),
    "test": re.compile(r"\btest(?:ing| data)?\b", re.IGNORECASE),
    "backend": re.compile(r"\bback[- ]?end\b", re.IGNORECASE),
    "model": re.compile(r"\b(?:language )?model\b|\b9b\b", re.IGNORECASE),
    "router": re.compile(r"\b(?:classifier|router)\b", re.IGNORECASE),
    "compute": re.compile(r"\b(?:gpu|cpu|cuda|zerogpu)\b", re.IGNORECASE),
    "tool": re.compile(r"\btool(?: call| result|ing|s)?\b", re.IGNORECASE),
}
POLICY_CITATION = re.compile(r"\[Policy:\s*([^\]]+?)\s*\]", re.IGNORECASE)
FACTUAL_NUMBER = re.compile(
    r"(?<![\w.])(?P<currency>\$)?"
    r"(?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"(?P<percent>%)?(?![\w.])"
)
NUMBER_WORD_VALUES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
FACTUAL_NUMBER_WORD = re.compile(
    rf"\b(?:{'|'.join(NUMBER_WORD_VALUES)})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GroundingValidation:
    valid: bool
    errors: tuple[str, ...]


def validate_customer_facing_answer(answer: str) -> GroundingValidation:
    """Reject implementation vocabulary that belongs in diagnostics, not chat."""

    if not isinstance(answer, str) or not answer.strip():
        return GroundingValidation(False, ("customer-facing answer is empty",))
    errors = tuple(
        f"answer contains internal term: {label}"
        for label, pattern in INTERNAL_LANGUAGE_PATTERNS.items()
        if pattern.search(answer)
    )
    return GroundingValidation(not errors, errors)


def validate_policy_answer(
    answer: str,
    matches: Sequence[Mapping[str, Any]],
) -> GroundingValidation:
    """Require citations to the exact policy chunks supplied to generation."""

    customer_validation = validate_customer_facing_answer(answer)
    errors = list(customer_validation.errors)
    allowed = {
        str(match.get("chunk_id", "")).strip()
        for match in matches
        if isinstance(match.get("chunk_id"), str) and str(match.get("chunk_id")).strip()
    }
    citations = {value.strip() for value in POLICY_CITATION.findall(answer)}
    if not allowed:
        errors.append("policy generation received no authoritative chunks")
    if not citations:
        errors.append("policy answer is missing a policy citation")
    invented = citations - allowed
    if invented:
        errors.append(f"policy answer cites unreturned chunks: {sorted(invented)}")
    if citations and not citations & allowed:
        errors.append("policy answer does not cite a returned chunk")
    evidence_text = " ".join(
        str(match.get(field, ""))
        for match in matches
        for field in ("title", "text", "effective_from", "effective_to")
    )
    answer_without_citations = POLICY_CITATION.sub("", answer)
    evidence_quantities = {normalized for _, normalized in _quantities(evidence_text)}
    unsupported_numbers = {
        displayed
        for displayed, normalized in _quantities(answer_without_citations)
        if normalized not in evidence_quantities
    }
    if unsupported_numbers:
        errors.append(
            f"policy answer contains unsupported numeric claims: {sorted(unsupported_numbers)}"
        )
    return GroundingValidation(not errors, tuple(errors))


def _quantities(text: str) -> set[tuple[str, str]]:
    """Return displayed and canonical quantities without substring matching."""

    quantities = {
        (match.group(0), _normalize_numeric_quantity(match))
        for match in FACTUAL_NUMBER.finditer(text)
    }
    quantities.update(
        (match.group(0), str(NUMBER_WORD_VALUES[match.group(0).casefold()]))
        for match in FACTUAL_NUMBER_WORD.finditer(text)
    )
    return quantities


def _normalize_numeric_quantity(match: re.Match[str]) -> str:
    raw_number = match.group("number").replace(",", "")
    try:
        number = format(Decimal(raw_number).normalize(), "f")
    except InvalidOperation:
        number = raw_number
    return f"{match.group('currency') or ''}{number}{match.group('percent') or ''}"


def build_customer_experience_repair_messages(
    *,
    user_message: str,
    draft: str,
    errors: Sequence[str],
    authoritative_evidence: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, str]]:
    """Build one tools-disabled repair request without losing grounding evidence."""

    payload = {
        "customer_request": user_message,
        "draft_answer": draft,
        "validation_errors": list(errors),
        "authoritative_evidence": [
            _without_private_identifiers(item, public_id_keys={"chunk_id"})
            for item in authoritative_evidence
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "Rewrite the answer as Harbor, the Harborlight Bank assistant. "
                "Use only the authoritative evidence when it is present. Correct every "
                "validation error, preserve valid policy citations exactly, and keep the "
                "answer warm and concise. Do not mention prototypes, demos, synthetic "
                "data, models, classifiers, tools, compute infrastructure, or internal "
                "identifiers. Return only the final customer-facing answer."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        },
    ]


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


# Standalone copy of the training realizer pools. The corpus needs this scaffolding to
# keep every final answer unique, so the model learned to emit it; customers should not
# see it. Kept in sync with hello_slm.banking_tool_sft_data by a parity test.
REALIZER_FILLER_PREFIXES = (
    "Here’s the requested update:",
    "I reviewed the relevant details.",
    "For clarity,",
    "The current information shows this:",
    "Here is the concise result:",
    "I checked the available information.",
    "This is the current result:",
    "I found the following details:",
    "The account information supports this answer:",
    "Here’s what applies to your request:",
    "I can confirm the following:",
    "The relevant banking details are:",
    "Your requested summary is below:",
    "I reviewed this carefully.",
    "The available record shows this:",
)
REALIZER_FILLER_CLOSERS = (
    "I can help with the next banking step.",
    "This reflects the information available in this session.",
    "I’ve kept the result focused on your request.",
    "You can use this summary to decide what to do next.",
    "Let me know if you need another related detail.",
    "That covers the banking request you made.",
    "I can also explain any item in this result.",
)


COMPLETED_ACTION_CLAIMS = re.compile(
    r"\b(?:froze|has been frozen|is now frozen|replacement is pending|"
    r"I(?:'ve| have)? (?:frozen|replaced|cancelled|canceled|disputed))\b",
    re.IGNORECASE,
)


def validate_no_unsupported_action_claims(
    answer: str,
    results: Sequence[Mapping[str, Any]],
    conversation: Sequence[Mapping[str, Any]] = (),
) -> GroundingValidation:
    """Zero-tool turns must never assert a completed banking action or state change.

    Evidence is either a current-turn tool result in ``results`` or a prior
    ``role == "tool"`` message anywhere in ``conversation`` — the same trusted
    channel ``ground_servicing_decision`` consumes (entity_grounding.py).
    """

    has_prior_tool_evidence = any(
        isinstance(message, Mapping) and message.get("role") == "tool"
        for message in conversation
    )
    if results or has_prior_tool_evidence:
        return GroundingValidation(True, ())
    if not isinstance(answer, str):
        return GroundingValidation(False, ("final answer is not text",))
    match = COMPLETED_ACTION_CLAIMS.search(answer)
    if match is None:
        return GroundingValidation(True, ())
    return GroundingValidation(
        False,
        (f"answer claims a completed action ({match.group(0)!r}) without tool evidence",),
    )


def strip_realizer_filler(text: str) -> str:
    """Remove learned realizer scaffolding, keeping the banking substance.

    Returns an empty string when the draft was nothing but filler; callers decide
    whether to fall back to the original text or drop the sentence entirely.
    """

    if not isinstance(text, str):
        return text
    stripped = text.strip()
    for prefix in sorted(REALIZER_FILLER_PREFIXES, key=len, reverse=True):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :].strip()
            break
    for closer in sorted(REALIZER_FILLER_CLOSERS, key=len, reverse=True):
        if stripped.endswith(closer):
            stripped = stripped[: -len(closer)].strip()
            break
    return stripped


def leading_prose(draft: str) -> str:
    """Return the sentence a draft opens with, dropping any table the model wrote itself."""

    if not isinstance(draft, str):
        return ""
    return draft.split("|", 1)[0].strip()


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
    value = call.get("arguments") if isinstance(call, Mapping) else getattr(call, "arguments", None)
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


def _without_private_identifiers(
    value: Any,
    *,
    public_id_keys: set[str] | None = None,
) -> Any:
    public_ids = public_id_keys or set()
    if isinstance(value, Mapping):
        return {
            str(key): _without_private_identifiers(
                item,
                public_id_keys=public_ids,
            )
            for key, item in value.items()
            if not str(key).endswith("_id") or str(key) in public_ids
        }
    if isinstance(value, list):
        return [_without_private_identifiers(item, public_id_keys=public_ids) for item in value]
    return value


POLICY_FALLBACK_NOTE = "fallback:policy-no-match-grounded-converse"

_QUESTION_STOPWORDS = frozenset(
    {
        "about",
        "and",
        "are",
        "can",
        "could",
        "did",
        "does",
        "explain",
        "for",
        "from",
        "have",
        "how",
        "into",
        "our",
        "please",
        "she",
        "tell",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "they",
        "this",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)

_POLICY_QUESTION_DISQUALIFIERS = frozenset(
    {
        "charge",
        "charges",
        "fee",
        "fees",
        "limit",
        "limits",
        "policies",
        "policy",
        "rate",
        "rates",
        "rule",
        "rules",
        "terms",
    }
)


def _tool_payload_value_tokens(content: str) -> set[str]:
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return set()
    tokens: set[str] = set()
    stack: list[Any] = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        elif isinstance(value, str):
            tokens.update(re.findall(r"[a-z0-9]+", value.lower()))
        elif isinstance(value, int | float) and not isinstance(value, bool):
            tokens.update(re.findall(r"[a-z0-9]+", str(value).lower()))
    return tokens


def tool_evidence_answers_question(
    question: str,
    conversation: Sequence[Mapping[str, Any]],
    *,
    window: int = 8,
) -> bool:
    """Report whether recent tool results plausibly answer the question.

    Every salient question token must appear among the tool-result VALUES within
    the last ``window`` messages, so an entity follow-up such as a merchant
    question after a transaction table qualifies. Questions asking for policy
    terms (fees, limits, rates, rules) never qualify: an unmatched policy
    question keeps the stock response instead of an ungrounded generation.
    """

    raw_tokens = re.findall(r"[a-z0-9]+", str(question).lower())
    if any(token in _POLICY_QUESTION_DISQUALIFIERS for token in raw_tokens):
        return False
    tokens = [
        token for token in raw_tokens if len(token) >= 3 and token not in _QUESTION_STOPWORDS
    ]
    if not tokens:
        return False
    recent = list(conversation)[-window:] if window > 0 else []
    evidence_tokens: set[str] = set()
    for message in recent:
        if isinstance(message, Mapping) and str(message.get("role", "")) == "tool":
            evidence_tokens.update(_tool_payload_value_tokens(str(message.get("content", ""))))
    if not evidence_tokens:
        return False
    return all(token in evidence_tokens for token in tokens)


def policy_context_fallback_route(route: Mapping[str, Any]) -> dict[str, Any]:
    """Amend a policy route into a grounded converse turn over prior tool results.

    The learned decision is preserved under ``learned_*`` keys so diagnostics keep
    reporting what the router actually chose; only the effective action changes.
    """

    amended: dict[str, Any] = dict(route)
    amended.setdefault("learned_action", str(route.get("action", "retrieve_policy")))
    amended.setdefault(
        "learned_entity_resolution", str(route.get("entity_resolution", "not_required"))
    )
    amended["action"] = "converse"
    amended["entity_resolution"] = "not_required"
    amended["constraint_diagnostics"] = (
        *tuple(amended.get("constraint_diagnostics") or ()),
        POLICY_FALLBACK_NOTE,
    )
    return amended
