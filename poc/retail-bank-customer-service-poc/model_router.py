"""Routing decided by the fine-tuned SLM instead of the DistilBERT cross-encoder.

This exists to answer one question honestly: *who should classify the turn?*
It is deliberately not "run the harness without a router". Removing the router
also removes single-schema exposure, entity gating and turn guidance, so a
comparison built that way changes who decides and what is enforced at the same
time and can attribute the harness's contribution to the classifier.

So this class produces the **same decision shape** the learned router produces,
and everything downstream -- ``_generation_plan``, the one-tool contract, the
claim guards, the validators -- runs unchanged. What varies between the two
configurations is the classifier and nothing else.

Two rules make the comparison fair rather than flattering:

* **Unknown means rejected.** A label outside the canonical set is a parse
  failure, not something to coerce to the nearest legal value. Coercion would
  quietly repair the model's mistakes and count them as successes.
* **Illegal tuples are corrected, and the correction is recorded.** The learned
  router cannot emit an illegal tuple because its joint decoder enumerates the
  legal ones; holding that constant means applying the same legality here. Every
  correction is reported in ``constraint_diagnostics`` so "how often did the
  classifier propose something illegal" stays a measurable difference rather
  than a hidden repair.

A failure to parse resolves to an uncertain route with **no** ``tool_authority``,
which reaches the same fail-closed branch a router outage reaches: clarify, no
tools.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from dialogue_state import MUTATION_INTENTS, SERVICING_TOOLS
from taxonomy import ACTION_LABELS, DOMAIN_LABELS, ENTITY_RESOLUTION_LABELS, INTENT_LABELS

ROUTING_CONTRACT = "retail-bank-model-routing/v1"

#: Entity states that make a target unusable; kept identical to the set
#: ``_generation_plan`` blocks on, so the two configurations gate alike.
_BLOCKED_ENTITY_STATES = {"missing", "ambiguous", "ineligible"}

#: Intents that never take a servicing tool, whatever the model proposes.
_NON_SERVICING_INTENTS = {"policy_knowledge", "conversation", "other_banking"}

_JSON_BLOCK = re.compile(r"\{.*?\}", re.DOTALL)

_INSTRUCTIONS = (
    "You are the routing classifier for a retail-bank assistant. Read the "
    "customer's latest turn and the recent conversation, then reply with ONE "
    "JSON object and nothing else.\n\n"
    "Fields and their only permitted values:\n"
    f"  domain: {' | '.join(DOMAIN_LABELS)}\n"
    f"  intent: {' | '.join(INTENT_LABELS)}\n"
    f"  action: {' | '.join(ACTION_LABELS)}\n"
    f"  entity_resolution: {' | '.join(ENTITY_RESOLUTION_LABELS)}\n\n"
    "Guidance:\n"
    "- entity_resolution describes the action's target: resolved when the turn "
    "or the conversation identifies exactly one, ambiguous when several fit, "
    "missing when none is identified, not_required when the action needs none.\n"
    "- action is execute_tool only when a specific banking record should be "
    "read or changed and its target is resolved; clarify when you must ask "
    "first; retrieve_policy for questions about rules, fees or timelines; "
    "converse for greetings and small talk; refuse_ood for anything outside "
    "retail banking.\n"
    'Reply with exactly: {"domain": "...", "intent": "...", "action": "...", '
    '"entity_resolution": "..."}'
)


class _Generator(Protocol):
    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_new_tokens: int,
        *,
        sample: bool = False,
        prefill: str = "",
    ) -> str: ...


def _lane_for(intent: str, action: str) -> str:
    if action == "refuse_ood":
        return "out_of_domain"
    if intent == "policy_knowledge":
        return "policy"
    if intent == "conversation":
        return "conversation"
    if intent == "other_banking":
        return "other_banking"
    return "servicing"


def unroutable(reason: str) -> dict[str, Any]:
    """The fail-closed decision: no action, and no unrouted tool authority."""
    return {
        "route": "uncertain",
        "routing_contract": ROUTING_CONTRACT,
        "classifier": "model",
        "reason": reason,
        "constraint_diagnostics": ("model-routing:unparsable",),
    }


def parse_decision(raw: str) -> dict[str, str] | None:
    """Strictly read one routing object, or return None.

    Unknown labels fail rather than snap to the nearest legal value: a
    classifier that says something outside the taxonomy has not classified.
    """
    match = _JSON_BLOCK.search(raw or "")
    if match is None:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    fields = {
        "domain": DOMAIN_LABELS,
        "intent": INTENT_LABELS,
        "action": ACTION_LABELS,
        "entity_resolution": ENTITY_RESOLUTION_LABELS,
    }
    decision: dict[str, str] = {}
    for name, permitted in fields.items():
        value = payload.get(name)
        if not isinstance(value, str) or value not in permitted:
            return None
        decision[name] = value
    return decision


def enforce_legality(decision: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Apply the constraints the joint decoder applies, and say what moved."""
    intent, action = decision["intent"], decision["action"]
    entity = decision["entity_resolution"]
    notes: list[str] = []

    if decision["domain"] == "out_of_domain" and action != "refuse_ood":
        action, entity = "refuse_ood", "not_required"
        notes.append("constraint:out-of-domain-must-refuse")

    # A mutation the model wants to chat through is the failure mode the router
    # constraint exists for: it must ask, not act and not converse.
    if intent in MUTATION_INTENTS and action == "converse":
        action, entity = "clarify", "missing"
        notes.append("constraint:mutation-intent-cannot-converse")

    if action == "execute_tool":
        if intent in _NON_SERVICING_INTENTS or intent not in SERVICING_TOOLS:
            action = "retrieve_policy" if intent == "policy_knowledge" else "converse"
            entity = "not_required"
            notes.append("constraint:intent-exposes-no-tool")
        elif entity in _BLOCKED_ENTITY_STATES:
            # Mirrors _generation_plan's own definition: "not_required" is a
            # legitimate execute_tool state for the list operations, which have
            # no target to resolve. Only an unusable target blocks execution.
            action = "clarify"
            notes.append("constraint:unresolved-target-cannot-execute")

    if action == "clarify" and entity == "not_required":
        entity = "missing"
        notes.append("constraint:clarify-needs-an-unresolved-target")

    return {**decision, "action": action, "entity_resolution": entity}, notes


class ModelConversationRouter:
    """Classifies with the SLM, then hands the harness the router's own shape."""

    def __init__(
        self,
        model: _Generator,
        *,
        revision: str = "",
        max_new_tokens: int = 96,
    ) -> None:
        self.model = model
        self.revision = revision
        self.max_new_tokens = max_new_tokens

    def _messages(
        self, message: str, history: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]]:
        recent = [
            {"role": str(turn.get("role", "")), "content": str(turn.get("content", ""))}
            for turn in (history or [])[-6:]
            if turn.get("role") in {"user", "assistant"}
        ]
        return [
            {"role": "system", "content": _INSTRUCTIONS},
            *recent,
            {"role": "user", "content": message},
        ]

    def classify(
        self,
        message: str,
        history: list[dict[str, Any]] | None,
        *,
        dialogue_state: Any = None,
    ) -> dict[str, Any]:
        try:
            raw = self.model.generate(
                self._messages(message, history), None, self.max_new_tokens
            )
        except Exception as error:  # noqa: BLE001 - a classifier outage must fail closed
            return unroutable(f"model routing call failed: {type(error).__name__}")

        decision = parse_decision(raw)
        if decision is None:
            return unroutable("model did not return a usable routing object")

        legal, notes = enforce_legality(decision)
        action = legal["action"]
        return {
            "route": "out_of_domain" if action == "refuse_ood" else "in_domain",
            "routing_contract": ROUTING_CONTRACT,
            "classifier": "model",
            "domain": legal["domain"],
            "lane": _lane_for(legal["intent"], action),
            "intent": legal["intent"],
            "fine_intent": legal["intent"],
            "action": action,
            "entity_resolution": legal["entity_resolution"],
            "proposed": decision,
            "constraint_diagnostics": tuple(notes),
            "relation_probabilities": {},
            "active_relations": [],
            "router_revision": self.revision or "model-routing",
            "raw_output": raw,
        }
