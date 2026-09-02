"""Measure a corpus against the use cases a customer actually brings.

The 2026-08-29 audit checked that every mechanism does what it claims; it never
asked whether the corpus contains what people say. The gap that shipped was
exactly that kind: among 12,394 router training rows, none started with "what"
and mentioned "balance", so the most common question in retail banking routed
to the policy lane and was refused. Every demo preset was imperative, which is
why nothing showed.

This module classifies each row along two axes and counts the cells:

* **category** -- multi-label, from the row's own metadata where it exists
  (domain, relations, counterfactual pairing, history length) and from wording
  where it does not (adversarial, multi-intent);
* **phrasing form** -- one label per row: ``interrogative``, ``imperative``,
  ``elliptical`` or ``deictic``.

A cell is ``(intent, form, first_turn | multi_turn)``. The cell that shipped
empty was ``(view_accounts, interrogative, first_turn)``.

Detection is by metadata first and by regex only where nothing better exists.
Every detector has a test that plants a row and proves it fires, because a
coverage gate whose detectors cannot fire certifies nothing.
"""

from __future__ import annotations

import re
import tomllib
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Index order of the router's relation vector, as ``RELATION_LABELS`` declares it.
RELATION_INDEX = {
    "context_dependent": 0,
    "agent_repair": 1,
    "topic_shift": 2,
    "clarification_answer": 3,
    "resume_previous_service": 4,
}

LONG_RUNNING_TURNS = 6

#: ``wh_question`` and ``modal_request`` are kept apart on purpose. "Could you
#: pull up my accounts?" is an imperative in polite clothing and routes like
#: one; "What is my balance?" is a question about an amount, and it is that
#: shape -- not questions in general -- that the router had never seen for
#: view_accounts and sent to the policy lane.
FORMS = ("imperative", "wh_question", "modal_request", "elliptical", "deictic")

CATEGORIES = (
    "in_domain",
    "social",
    "out_of_domain",
    "first_turn",
    "multi_turn",
    "long_running",
    "counterfactual",
    "policy_question",
    "intent_drift",
    "loop_back",
    "agent_repair",
    "clarification_answer",
    "adversarial",
    "multi_intent",
)

_WH_OPENER = re.compile(
    r"^\s*(?:\w+,?\s+)?(what|what's|whats|how|how's|when|where|which|who|why|"
    r"is|are|am|was|were|do|does|did|have|has)\b",
    re.IGNORECASE,
)
_MODAL_OPENER = re.compile(
    r"^\s*(?:\w+,?\s+)?(can|could|would|will|should|may|might)\s+(?:you|u|we|i)\b",
    re.IGNORECASE,
)
_DEICTIC = re.compile(
    r"\b(that one|this one|the one (?:to|from|for|with|above|we|you)|those|"
    r"th(?:at|is|e) (?:card|transfer|transaction|case|account|payment|purchase|charge|request|item)"
    r" (?:we|you|above|from|i|that|in the list|shown)|"
    r"the (?:card|transfer|transaction|case|account) "
    r"(?:we were|you (?:listed|showed|mentioned)))\b",
    re.IGNORECASE,
)
#: Text that tries to override instructions or extract a credential.
_ADVERSARIAL = re.compile(
    r"(ignore (?:your|all|the) (?:previous|prior|earlier) instructions|"
    r"disregard (?:your|the) (?:rules|instructions|guidelines)|"
    r"you are now|pretend (?:you are|to be)|jailbreak|"
    r"(?:print|show|tell|give|read)(?: me)? (?:the |my )?(?:full |entire |complete )?"
    r"(?:card number|pin|password|cvv|security code|ssn|social security)|"
    r"what(?:'s| is) my (?:pin|password|cvv))",
    re.IGNORECASE,
)
_VERB = (
    r"(?:show|list|pull|bring|display|freeze|cancel|replace|dispute|open|start|stop|"
    r"check|tell|give|send|report|block|find|get)"
)
_OBJECT = (
    r"(?:card|cards|transfer|transfers|transaction|transactions|purchase|charge|"
    r"account|accounts|balance|balances|dispute|case|cases|request|payment|statement|"
    r"one|it|them)"
)
#: Two asks in one turn: two verb-plus-banking-object clauses joined by a
#: conjunction or punctuation, optionally with a sequencer. Both clauses need a
#: banking object: "stop that task and cancel the transfer" is one ask wrapped
#: in a topic-switch scaffold, and it appeared sixty times in the first count.
_MULTI_INTENT = re.compile(
    rf"\b{_VERB}\b[^.?!;,]*?\b{_OBJECT}\b[^.?!;]*?"
    rf"(?:\b(?:and|then)\s+|[;,]\s*)(?:(?:then|also|after that|next),?\s+)?"
    rf"{_VERB}\b[^.?!;]*?\b{_OBJECT}\b",
    re.IGNORECASE,
)


def phrasing_form(text: str) -> str:
    """One form per turn. Deixis wins over the others: it is what makes the
    turn hard, whatever its grammatical mood."""
    stripped = (text or "").strip()
    if not stripped:
        return "elliptical"
    if _DEICTIC.search(stripped):
        return "deictic"
    if _MODAL_OPENER.match(stripped):
        return "modal_request"
    if _WH_OPENER.match(stripped) or stripped.endswith("?"):
        return "wh_question"
    if len(stripped.split()) <= 3:
        return "elliptical"
    return "imperative"


def _relations(row: Mapping[str, Any]) -> dict[str, bool]:
    labels = row.get("relation_labels")
    if isinstance(labels, list) and len(labels) == len(RELATION_INDEX):
        return {name: bool(labels[index]) for name, index in RELATION_INDEX.items()}
    active = row.get("active_relations")
    if isinstance(active, list):
        return {name: name in active for name in RELATION_INDEX}
    return {name: False for name in RELATION_INDEX}


#: Alignment rows carry no relation vector; their scenario family says what
#: the row exercises. This is the declared mapping from family to category.
FAMILY_CATEGORIES = {
    "policy_detour": "intent_drift",
    "banking_topic_shift": "intent_drift",
    "external_topic_shift": "intent_drift",
    "policy_resume": "loop_back",
    "agent_repair": "agent_repair",
    "clarification_answer": "clarification_answer",
    "credential_hygiene": "adversarial",
    "hard_negative_private_id": "adversarial",
    "scope_refusal": "out_of_domain",
    "capability_boundary": "out_of_domain",
}
_PAIRED_FAMILY = re.compile(r"^deictic_.*_(action|ambiguity)$")
_POLICY_FAMILY = re.compile(r"^(faq_|no_tool_banking_faq|policy_)")


def _alignment_categories(row: Mapping[str, Any], found: set[str]) -> None:
    messages = row.get("messages") or []
    user_turns = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "user")
    prior = max(user_turns - 1, 0) * 2  # user+assistant exchanges before the current turn
    found.add("multi_turn" if prior else "first_turn")
    if prior >= LONG_RUNNING_TURNS:
        found.add("long_running")
    metadata = row.get("metadata") or {}
    path = metadata.get("path")
    family = str(metadata.get("scenario_family") or "")
    if path == "ood" or FAMILY_CATEGORIES.get(family) == "out_of_domain":
        found.add("out_of_domain")
    elif path == "conversation":
        found.add("social")
    else:
        found.add("in_domain")
    if _PAIRED_FAMILY.match(family):
        found.add("counterfactual")
    if path == "retrieval_grounded_policy" or _POLICY_FAMILY.match(family):
        found.add("policy_question")
    mapped = FAMILY_CATEGORIES.get(family)
    if mapped and mapped != "out_of_domain":
        found.add(mapped)


def categories_for(row: Mapping[str, Any], text: str) -> frozenset[str]:
    """Every category the row belongs to. Metadata first; regex only where the
    corpus records nothing."""
    found: set[str] = set()
    if "messages" in row and "history" not in row:
        _alignment_categories(row, found)
    else:
        domain = row.get("domain_name") or row.get("domain")
        if domain == "banking":
            found.add("in_domain")
        elif domain == "social":
            found.add("social")
        elif domain == "out_of_domain":
            found.add("out_of_domain")

        history = row.get("history") or []
        turns = len(history) if isinstance(history, list) else 0
        found.add("multi_turn" if turns else "first_turn")
        if turns >= LONG_RUNNING_TURNS:
            found.add("long_running")

        if row.get("counterfactual_pair_id"):
            found.add("counterfactual")
        if (row.get("intent") or row.get("fine_intent")) == "policy_knowledge":
            found.add("policy_question")

        relations = _relations(row)
        if relations["topic_shift"]:
            found.add("intent_drift")
        if relations["resume_previous_service"]:
            found.add("loop_back")
        if relations["agent_repair"]:
            found.add("agent_repair")
        if relations["clarification_answer"]:
            found.add("clarification_answer")

    if _ADVERSARIAL.search(text or ""):
        found.add("adversarial")
    if _MULTI_INTENT.search(text or ""):
        found.add("multi_intent")
    return frozenset(found)


def row_text(row: Mapping[str, Any]) -> str:
    """The customer's current turn, whichever corpus shape the row has."""
    for key in ("current_text", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content") or "")
    return ""


def row_intent(row: Mapping[str, Any]) -> str:
    intent = row.get("intent") or row.get("fine_intent")
    if isinstance(intent, str) and intent:
        return intent
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("scenario_family"), str):
        return metadata["scenario_family"]
    domain = row.get("domain_name") or row.get("domain")
    if domain == "out_of_domain":
        return "out_of_domain"
    return "unlabelled"


@dataclass
class CoverageReport:
    rows: int = 0
    category_counts: Counter[str] = field(default_factory=Counter)
    form_counts: Counter[str] = field(default_factory=Counter)
    #: (intent, form, first_turn|multi_turn) -> rows
    cells: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    intents: set[str] = field(default_factory=set)

    def cell(self, intent: str, form: str, turn: str) -> int:
        return self.cells[(intent, form, turn)]


def measure(rows: Iterable[Mapping[str, Any]]) -> CoverageReport:
    report = CoverageReport()
    for row in rows:
        text = row_text(row)
        found = categories_for(row, text)
        form = phrasing_form(text)
        intent = row_intent(row)
        turn = "multi_turn" if "multi_turn" in found else "first_turn"
        report.rows += 1
        report.category_counts.update(found)
        report.form_counts[form] += 1
        report.cells[(intent, form, turn)] += 1
        report.intents.add(intent)
    return report


# --- the declared expectation -------------------------------------------------


@dataclass(frozen=True)
class CellSpec:
    intent: str
    form: str
    turn: str
    minimum: int
    target: int

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.intent, self.form, self.turn)


@dataclass(frozen=True)
class CoverageSpec:
    corpus: str
    category_minimums: Mapping[str, int]
    cells: tuple[CellSpec, ...]


def load_spec(path: Path, corpus: str) -> CoverageSpec:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    section = payload.get(corpus)
    if not isinstance(section, dict):
        raise ValueError(f"{path} declares no [{corpus}] section")
    cells = []
    for entry in section.get("cell", []):
        cells.append(
            CellSpec(
                intent=str(entry["intent"]),
                form=str(entry["form"]),
                turn=str(entry.get("turn", "first_turn")),
                minimum=int(entry.get("minimum", 0)),
                target=int(entry.get("target", entry.get("minimum", 0))),
            )
        )
    minimums = {str(k): int(v) for k, v in (section.get("categories") or {}).items()}
    return CoverageSpec(corpus=corpus, category_minimums=minimums, cells=tuple(cells))


@dataclass(frozen=True)
class Shortfall:
    what: str
    have: int
    minimum: int
    target: int

    @property
    def blocks(self) -> bool:
        return self.have < self.minimum


def evaluate(report: CoverageReport, spec: CoverageSpec) -> list[Shortfall]:
    """Every declared cell or category below its target, worst first.

    A shortfall below ``minimum`` blocks the build; below ``target`` it is
    reported so the empty cells set the authoring order.
    """
    found: list[Shortfall] = []
    for name, minimum in spec.category_minimums.items():
        have = report.category_counts[name]
        if have < minimum:
            found.append(Shortfall(f"category:{name}", have, minimum, minimum))
    for cell in spec.cells:
        have = report.cell(*cell.key)
        if have < cell.target:
            label = f"{cell.intent} × {cell.form} × {cell.turn}"
            found.append(Shortfall(label, have, cell.minimum, cell.target))
    found.sort(key=lambda s: (not s.blocks, -(s.target - s.have)))
    return found
