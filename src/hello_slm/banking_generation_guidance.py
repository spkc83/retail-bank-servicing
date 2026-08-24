"""Prompt guidance derived only from the public V7 generation contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

GENERATION_MODES = frozenset(
    {"execute_tool", "clarify", "converse", "retrieve_policy", "refuse_ood"}
)
ENTITY_STATES = frozenset({"resolved", "missing", "ambiguous", "ineligible", "not_required"})


def render_turn_guidance(contract: Mapping[str, Any]) -> str:
    """Render model-facing prose without leaking classifier internals."""

    mode = contract.get("mode")
    entity_state = contract.get("entity_state")
    tool_names = contract.get("tool_names")
    constraints = contract.get("argument_constraints")
    if mode not in GENERATION_MODES:
        raise ValueError(f"unsupported generation mode: {mode!r}")
    if entity_state not in ENTITY_STATES:
        raise ValueError(f"unsupported entity state: {entity_state!r}")
    if not isinstance(tool_names, Sequence) or isinstance(tool_names, str | bytes):
        raise ValueError("generation contract tool_names must be a sequence")
    names = tuple(str(name) for name in tool_names)
    if len(names) != len(set(names)) or len(names) > 1:
        raise ValueError("generation contract must expose exactly one or no tools")
    if not isinstance(constraints, Mapping):
        raise ValueError("generation contract argument_constraints must be an object")

    if mode == "execute_tool":
        if len(names) != 1 or entity_state not in {"resolved", "not_required"}:
            raise ValueError("execute_tool requires one tool and an executable entity state")
        argument_instruction = (
            "Emit every required argument exactly as constrained by the exposed schema; "
            "do not omit, infer, or alter it."
            if constraints
            else "Choose every argument from the conversation; this guidance supplies no "
            "tool arguments."
        )
        return (
            f"Use only {names[0]} for this turn. Call it when the conversation supplies "
            "the selectors its schema requires; otherwise ask one concise, natural "
            f"clarification question. {argument_instruction}"
        )

    if names or constraints:
        raise ValueError(f"{mode} cannot expose tools or argument constraints")
    if mode == "clarify":
        detail = {
            "missing": "supply the missing banking detail",
            "ambiguous": "distinguish which banking item or event they mean",
            "ineligible": "choose an eligible banking item",
        }.get(str(entity_state))
        if detail is None:
            raise ValueError("clarify requires a missing, ambiguous, or ineligible entity state")
        return (
            f"Ask exactly one concise, natural clarification question that helps the customer "
            f"{detail}. Do not claim that an action was completed. Never state the status of an "
            f"account, card, payment, dispute or request you have not been shown."
        )
    if entity_state != "not_required":
        raise ValueError(f"{mode} requires entity_state='not_required'")
    if mode == "converse":
        return (
            "Respond naturally and concisely without looking up customer records or performing "
            "a banking action. Never infer distress, trouble, or a failed banking event from a "
            "neutral greeting or social message. If the customer asked for something this "
            "assistant cannot do, say so plainly and name what you can help with instead; never "
            "promise to do it or ask for details you cannot use. Never state the status of an "
            "account, card, payment, dispute or request you have not been shown."
        )
    if mode == "retrieve_policy":
        return (
            "Answer the banking policy question naturally and concisely without calling a "
            "customer-record tool. After the answer, resume the prior servicing task only when "
            "the conversation supports it."
        )
    return (
        "Explain concisely that you can only help with retail banking and financial-services "
        "questions. Do not call a banking tool or claim that an action was completed."
    )


def messages_with_turn_guidance(
    messages: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Copy messages and append V7 guidance to the first system message."""

    copied = [dict(message) for message in messages]
    if contract is None:
        return copied
    guidance = render_turn_guidance(contract)
    for message in copied:
        if message.get("role") == "system":
            content = message.get("content")
            if not isinstance(content, str) or not content:
                raise ValueError("system message content must be non-empty text")
            message["content"] = f"{content}\n\nTURN GUIDANCE: {guidance}"
            return copied
    raise ValueError("contracted record is missing a system message")


def messages_with_record_turn_guidance(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Resolve messages and the optional explicit contract from one SFT/eval record."""

    messages = record.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, str | bytes):
        raise ValueError("record is missing messages")
    expected = record.get("expected")
    contract = expected.get("generation_contract") if isinstance(expected, Mapping) else None
    if contract is not None and not isinstance(contract, Mapping):
        raise ValueError("generation_contract must be an object")
    return messages_with_turn_guidance(messages, contract)
