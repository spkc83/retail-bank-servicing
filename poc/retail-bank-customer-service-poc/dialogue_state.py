from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from threading import RLock
from typing import Any

STATE_VERSION = 1
DEFAULT_INTENT_CONFIDENCE = 0.5
EFFECTIVE_DECISION_CONTRACT = "retail-bank-effective-turn-decision/v1"

SERVICING_TOOLS = {
    "view_accounts": "list_accounts",
    "view_cards": "list_cards",
    "freeze_card": "freeze_card",
    "replace_card": "replace_card",
    "view_transactions": "list_transactions",
    "dispute_transaction": "dispute_transaction",
    "view_transfers": "list_transfers",
    "cancel_transfer": "cancel_transfer",
    "view_service_cases": "list_service_cases",
}
MUTATION_INTENTS = {
    "freeze_card",
    "replace_card",
    "dispute_transaction",
    "cancel_transfer",
}


@dataclass(frozen=True)
class PendingServicing:
    intent: str
    anchor_user_message: str
    anchor_assistant_message: str = ""
    phase: str = "awaiting_user"


@dataclass(frozen=True)
class DialogueState:
    version: int = STATE_VERSION
    pending_servicing: PendingServicing | None = None
    knowledge_detour_active: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> DialogueState:
        """Rehydrate trusted UI session state without accepting extra fields."""
        if not payload:
            return cls()
        if int(payload.get("version", STATE_VERSION)) != STATE_VERSION:
            raise ValueError("unsupported dialogue-state version")
        pending_payload = payload.get("pending_servicing")
        pending = None
        if isinstance(pending_payload, Mapping):
            intent = pending_payload.get("intent")
            anchor_user = pending_payload.get("anchor_user_message")
            if not isinstance(intent, str) or intent not in SERVICING_TOOLS:
                raise ValueError("invalid pending servicing intent")
            if not isinstance(anchor_user, str) or not anchor_user.strip():
                raise ValueError("invalid pending servicing anchor")
            pending = PendingServicing(
                intent=intent,
                anchor_user_message=anchor_user.strip(),
                anchor_assistant_message=str(
                    pending_payload.get("anchor_assistant_message", "")
                ).strip(),
                phase=str(pending_payload.get("phase", "awaiting_user")),
            )
        return cls(
            pending_servicing=pending,
            knowledge_detour_active=bool(payload.get("knowledge_detour_active", False))
            and pending is not None,
        )


@dataclass(frozen=True)
class TransitionResult:
    state: DialogueState
    lane: str
    resumed: bool = False
    previous_pending: PendingServicing | None = None

    @property
    def anchor_exchange(self) -> tuple[dict[str, str], ...]:
        pending = self.previous_pending
        if pending is None:
            return ()
        return (
            {"role": "user", "content": pending.anchor_user_message},
            {"role": "assistant", "content": pending.anchor_assistant_message},
        )


def begin_turn(
    state: DialogueState,
    route: Mapping[str, Any],
    user_message: str,
    *,
    confidence_threshold: float = DEFAULT_INTENT_CONFIDENCE,
) -> TransitionResult:
    """Apply router observations without performing or authorizing an action."""
    route_name = route.get("route")
    intent = route.get("intent", route.get("fine_intent"))
    confidence = route.get("intent_confidence", route.get("confidence"))
    relations = route.get("active_relations", route.get("relations", ()))
    is_effective_decision = route.get("effective_decision_contract") == EFFECTIVE_DECISION_CONTRACT
    decision_accepted = route.get("decision_accepted") is True

    if (
        route_name != "in_domain"
        or not isinstance(intent, str)
        or (is_effective_decision and not decision_accepted)
        or (
            not is_effective_decision
            and (
                isinstance(confidence, bool)
                or not isinstance(confidence, int | float)
                or float(confidence) < confidence_threshold
            )
        )
    ):
        return TransitionResult(state=state, lane="unchanged")

    active_relations = _relation_names(relations)
    if (
        "resume_previous_service" in active_relations
        and state.knowledge_detour_active
        and state.pending_servicing is not None
    ):
        return TransitionResult(
            state=replace(state, knowledge_detour_active=False),
            lane="servicing",
            resumed=True,
            previous_pending=state.pending_servicing,
        )

    if intent in SERVICING_TOOLS:
        pending = state.pending_servicing
        if pending is not None and pending.intent == intent:
            if state.knowledge_detour_active:
                return TransitionResult(
                    state=replace(state, knowledge_detour_active=False),
                    lane="servicing",
                    resumed=True,
                    previous_pending=pending,
                )
            return TransitionResult(state=state, lane="servicing")
        return TransitionResult(
            state=DialogueState(
                pending_servicing=PendingServicing(
                    intent=intent,
                    anchor_user_message=user_message,
                    phase="awaiting_assistant",
                )
            ),
            lane="servicing",
        )

    if intent == "policy_knowledge":
        return TransitionResult(
            state=replace(
                state,
                knowledge_detour_active=state.pending_servicing is not None,
            ),
            lane="policy",
        )
    if intent == "conversation":
        return TransitionResult(state=state, lane="conversation")
    if intent == "other_banking":
        return TransitionResult(state=state, lane="other_banking")
    return TransitionResult(state=state, lane="unchanged")


def finish_turn(
    state: DialogueState,
    assistant_message: str,
    executed_tool_names: Sequence[str],
    tool_results: Mapping[str, Any] | Sequence[Any],
) -> DialogueState:
    """Commit customer-visible delivery and record the first assistant anchor."""
    state = commit_operations(state, executed_tool_names, tool_results)
    pending = state.pending_servicing
    if pending is None:
        return state

    expected_tool = SERVICING_TOOLS[pending.intent]
    if _tool_succeeded(expected_tool, executed_tool_names, tool_results):
        return DialogueState()

    if not pending.anchor_assistant_message and assistant_message:
        return replace(
            state,
            pending_servicing=replace(
                pending,
                anchor_assistant_message=assistant_message,
                phase="awaiting_user",
            ),
        )
    return state


def commit_operations(
    state: DialogueState,
    executed_tool_names: Sequence[str],
    tool_results: Mapping[str, Any] | Sequence[Any],
) -> DialogueState:
    """Commit successful mutations independently of response delivery."""
    pending = state.pending_servicing
    if pending is None or pending.intent not in MUTATION_INTENTS:
        return state
    expected_tool = SERVICING_TOOLS[pending.intent]
    if _tool_succeeded(expected_tool, executed_tool_names, tool_results):
        return DialogueState()
    return state


class DialogueStateRegistry:
    def __init__(self) -> None:
        self._states: dict[tuple[str, str], DialogueState] = {}
        self._lock = RLock()

    def get(self, username: str, session_id: str) -> DialogueState:
        with self._lock:
            return self._states.get((username, session_id), DialogueState())

    def set(self, username: str, session_id: str, state: DialogueState) -> None:
        with self._lock:
            self._states[(username, session_id)] = state

    def begin_turn(
        self,
        username: str,
        session_id: str,
        route: Mapping[str, Any],
        user_message: str,
        *,
        confidence_threshold: float = DEFAULT_INTENT_CONFIDENCE,
    ) -> TransitionResult:
        key = (username, session_id)
        with self._lock:
            result = begin_turn(
                self._states.get(key, DialogueState()),
                route,
                user_message,
                confidence_threshold=confidence_threshold,
            )
            self._states[key] = result.state
            return result

    def finish_turn(
        self,
        username: str,
        session_id: str,
        assistant_message: str,
        executed_tool_names: Sequence[str],
        tool_results: Mapping[str, Any] | Sequence[Any],
    ) -> DialogueState:
        key = (username, session_id)
        with self._lock:
            state = finish_turn(
                self._states.get(key, DialogueState()),
                assistant_message,
                executed_tool_names,
                tool_results,
            )
            self._states[key] = state
            return state

    def commit_operations(
        self,
        username: str,
        session_id: str,
        executed_tool_names: Sequence[str],
        tool_results: Mapping[str, Any] | Sequence[Any],
    ) -> DialogueState:
        key = (username, session_id)
        with self._lock:
            state = commit_operations(
                self._states.get(key, DialogueState()),
                executed_tool_names,
                tool_results,
            )
            self._states[key] = state
            return state

    def reset(self, username: str, session_id: str) -> None:
        with self._lock:
            self._states.pop((username, session_id), None)

    def as_dict(self, username: str, session_id: str) -> dict[str, Any]:
        return self.get(username, session_id).as_dict()


def _relation_names(relations: Any) -> set[str]:
    if isinstance(relations, Mapping):
        return {str(name) for name, active in relations.items() if active}
    if isinstance(relations, Sequence) and not isinstance(relations, str | bytes):
        return {str(name) for name in relations}
    return set()


def _tool_succeeded(
    expected_tool: str,
    executed_tool_names: Sequence[str],
    tool_results: Mapping[str, Any] | Sequence[Any],
) -> bool:
    matching_indexes = [
        index for index, name in enumerate(executed_tool_names) if name == expected_tool
    ]
    if not matching_indexes:
        return False

    if isinstance(tool_results, Mapping):
        result = tool_results.get(expected_tool, tool_results)
        return _successful_result(result)

    return any(
        index < len(tool_results) and _successful_result(tool_results[index])
        for index in matching_indexes
    )


def _successful_result(result: Any) -> bool:
    return (
        isinstance(result, Mapping)
        and not result.get("error")
        and result.get("success", True) is not False
    )
