from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from diagnostics import diagnostic_summary, error_model_passes
from dialogue_state import DialogueStateRegistry
from entity_grounding import ground_servicing_decision
from local_gpu_runtime import MODEL_ID, MODEL_REVISION
from mock_bank import SessionBankRegistry
from model_service import (
    AgentExecutionError,
    AgentProtocolError,
    ConversationalBankingAgent,
    ModelPassTrace,
    ModelRuntime,
    ToolCall,
    canonical_conversation,
    router_diagnostic_fields,
)
from policy_retrieval import DEFAULT_POLICY_PATH, PolicyKnowledgeBase
from response_policy import policy_context_fallback_route, tool_evidence_answers_question
from responses import (
    MODEL_FAILURE_RESPONSE,
    OOD_RESPONSE,
    POLICY_NOT_FOUND_RESPONSE,
    render_policy_citations,
)
from responses import policy_chunk_lookup as build_policy_chunk_lookup
from router import ROUTER_REVISION


@dataclass(frozen=True)
class LocalTurnResult:
    response: str
    conversation: list[dict[str, Any]]
    snapshot: dict[str, Any]
    route: dict[str, Any]
    tool_calls: tuple[ToolCall, ...]
    tool_results: tuple[dict[str, Any], ...]
    model_passes: tuple[ModelPassTrace, ...]
    response_path: str
    activity: str
    diagnostics: str
    policy_sources: tuple[str, ...] = ()
    dialogue_state: dict[str, Any] | None = None

    @property
    def model_call_count(self) -> int:
        return len(self.model_passes)


class LocalBankingController:
    """UI-independent local chat boundary shared by Streamlit reruns."""

    def __init__(
        self,
        *,
        bank: SessionBankRegistry,
        runtime: ModelRuntime,
        router: Any | None,
        policy_knowledge: Any | None = None,
        dialogue_states: DialogueStateRegistry | None = None,
    ) -> None:
        self.bank = bank
        self.runtime = runtime
        self.router = router
        self.policy_knowledge = policy_knowledge or PolicyKnowledgeBase.from_json(
            DEFAULT_POLICY_PATH
        )
        self.dialogue_states = dialogue_states or DialogueStateRegistry()
        # Seeded from the full corpus when available; each policy turn also learns
        # its retrieved matches so citation rendering works even against a policy
        # knowledge stand-in that only implements ``lookup`` (as tests do).
        self._policy_chunk_lookup: dict[str, dict[str, Any]] = build_policy_chunk_lookup(
            getattr(self.policy_knowledge, "chunks", ())
        )

    @property
    def policy_chunk_lookup(self) -> dict[str, dict[str, Any]]:
        return dict(self._policy_chunk_lookup)

    def run_turn(
        self,
        *,
        username: str,
        session_hash: str,
        message: str,
        conversation: list[dict[str, Any]],
    ) -> LocalTurnResult:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        canonical = canonical_conversation(conversation)
        prior_state = self.dialogue_states.as_dict(username, session_hash)
        learned_route = self._route(message.strip(), canonical, prior_state)
        route = ground_servicing_decision(
            learned_route,
            message=message.strip(),
            trusted_tool_results=canonical,
            live_snapshot=self.bank.snapshot(username, session_hash),
        )
        transition = self.dialogue_states.begin_turn(
            username,
            session_hash,
            route,
            message.strip(),
        )
        if route["route"] == "classifier_error":
            return self._direct_result(
                username=username,
                session_hash=session_hash,
                message=message,
                response=MODEL_FAILURE_RESPONSE,
                conversation=canonical,
                route=route,
                response_path="classifier failure",
                activity="The conversation classifier failed; Granite was not invoked.",
            )
        if route["route"] == "out_of_domain":
            return self._direct_result(
                username=username,
                session_hash=session_hash,
                message=message,
                response=OOD_RESPONSE,
                conversation=canonical,
                route=route,
                response_path="OOD stock response",
                activity="High-confidence OOD route; Granite was not invoked.",
            )

        agent = ConversationalBankingAgent(bank=self.bank, model=self.runtime)
        try:
            pinned_exchange = list(transition.anchor_exchange) or None
            if (route.get("lane") or transition.lane) == "policy":
                lookup = self.policy_knowledge.lookup(message.strip())
                if not lookup.matched:
                    if not tool_evidence_answers_question(message.strip(), canonical):
                        return self._direct_result(
                            username=username,
                            session_hash=session_hash,
                            message=message,
                            response=POLICY_NOT_FOUND_RESPONSE,
                            conversation=canonical,
                            route=route,
                            response_path="policy_no_match",
                            activity="No approved current policy passage matched the question.",
                        )
                    route = policy_context_fallback_route(route)
                    turn = agent.run_turn(
                        username=username,
                        session_hash=session_hash,
                        message=message,
                        conversation=canonical,
                        router_result=route,
                        pinned_exchange=pinned_exchange,
                    )
                else:
                    policy_matches = tuple(match.as_dict() for match in lookup.matches)
                    self._policy_chunk_lookup.update(build_policy_chunk_lookup(policy_matches))
                    turn = agent.run_policy_turn(
                        username=username,
                        session_hash=session_hash,
                        message=message,
                        conversation=canonical,
                        policy_matches=policy_matches,
                        corpus_revision=lookup.corpus_revision,
                        pinned_exchange=pinned_exchange,
                    )
            else:
                turn = agent.run_turn(
                    username=username,
                    session_hash=session_hash,
                    message=message,
                    conversation=canonical,
                    router_result=route,
                    pinned_exchange=pinned_exchange,
                )
        except AgentExecutionError as error:
            route = {
                **route,
                "failure_type": type(error.__cause__).__name__,
            }
            dialogue_state = self.dialogue_states.commit_operations(
                username,
                session_hash,
                tuple(call.name for call in error.tool_calls),
                error.tool_results,
            ).as_dict()
            failed_conversation = [
                *error.conversation,
                {"role": "assistant", "content": MODEL_FAILURE_RESPONSE},
            ]
            return self._result(
                response=MODEL_FAILURE_RESPONSE,
                conversation=failed_conversation,
                snapshot=error.snapshot,
                route=route,
                tool_calls=error.tool_calls,
                tool_results=error.tool_results,
                model_passes=error.model_passes,
                response_path="local model/tool execution failure",
                activity=(
                    "Granite failed during the recorded generation/tool sequence; "
                    "diagnostics retain its raw output and any tool calls."
                ),
                dialogue_state=dialogue_state,
                diagnostic_error=error,
            )
        except (AgentProtocolError, RuntimeError, TypeError, ValueError) as error:
            route = {**route, "failure_type": type(error).__name__}
            failed_conversation = [
                *canonical,
                {"role": "user", "content": message.strip()},
                {"role": "assistant", "content": MODEL_FAILURE_RESPONSE},
            ]
            return self._result(
                response=MODEL_FAILURE_RESPONSE,
                conversation=failed_conversation,
                snapshot=self.bank.snapshot(username, session_hash),
                route=route,
                tool_calls=(),
                tool_results=(),
                model_passes=error_model_passes(error),
                response_path="local model failure",
                activity=(
                    "Granite generation failed; no CPU-authored servicing answer was substituted."
                ),
                dialogue_state=transition.state.as_dict(),
                diagnostic_error=error,
            )

        dialogue_state = self.dialogue_states.finish_turn(
            username,
            session_hash,
            turn.response,
            tuple(call.name for call in turn.tool_calls),
            turn.tool_results,
        ).as_dict()
        if turn.policy_sources:
            activity = "Granite authored the answer from the cited policy passages."
        elif turn.response_path.endswith("_rendered"):
            activity = "Granite selected the data request; verified results were rendered."
        elif turn.tool_calls:
            activity = "Granite selected the recorded banking actions and authored the answer."
        else:
            activity = "Granite authored the response directly."
        return self._result(
            response=turn.response,
            conversation=turn.conversation,
            snapshot=turn.snapshot,
            route=route,
            tool_calls=turn.tool_calls,
            tool_results=turn.tool_results,
            model_passes=turn.model_passes,
            response_path=turn.response_path,
            activity=activity,
            policy_sources=turn.policy_sources,
            dialogue_state=dialogue_state,
        )

    def snapshot(self, username: str, session_hash: str) -> dict[str, Any]:
        return self.bank.snapshot(username, session_hash)

    def reset(self, username: str, session_hash: str) -> dict[str, Any]:
        self.dialogue_states.reset(username, session_hash)
        return self.bank.reset(username, session_hash)

    def dialogue_state(self, username: str, session_hash: str) -> dict[str, Any]:
        return self.dialogue_states.as_dict(username, session_hash)

    def runtime_metadata(self) -> dict[str, str]:
        provider = getattr(self.runtime, "runtime_metadata", None)
        return provider() if callable(provider) else {}

    def _route(
        self,
        message: str,
        conversation: list[dict[str, Any]],
        dialogue_state: dict[str, Any],
    ) -> dict[str, Any]:
        if self.router is None:
            return _uncertain_route("local router unavailable; delegated to Granite")
        try:
            parameters = inspect.signature(self.router.classify).parameters.values()
            supports_dialogue_state = any(
                parameter.name == "dialogue_state"
                or parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
            if supports_dialogue_state:
                result = self.router.classify(
                    message,
                    conversation,
                    dialogue_state=dialogue_state,
                )
            else:
                result = self.router.classify(message, conversation)
            return dict(result)
        except (RuntimeError, TypeError, ValueError) as error:
            return {
                **_uncertain_route("classifier failed; Granite was not invoked"),
                "route": "classifier_error",
                "failure_type": type(error).__name__,
            }

    def _direct_result(
        self,
        *,
        username: str,
        session_hash: str,
        message: str,
        response: str,
        conversation: list[dict[str, Any]],
        route: dict[str, Any],
        response_path: str,
        activity: str,
    ) -> LocalTurnResult:
        completed = [
            *conversation,
            {"role": "user", "content": message.strip()},
            {"role": "assistant", "content": response},
        ]
        return self._result(
            response=response,
            conversation=completed,
            snapshot=self.bank.snapshot(username, session_hash),
            route=route,
            tool_calls=(),
            tool_results=(),
            model_passes=(),
            response_path=response_path,
            activity=activity,
            dialogue_state=self.dialogue_states.as_dict(username, session_hash),
        )

    def _result(
        self,
        *,
        response: str,
        conversation: list[dict[str, Any]],
        snapshot: dict[str, Any],
        route: dict[str, Any],
        tool_calls: tuple[ToolCall, ...],
        tool_results: tuple[dict[str, Any], ...],
        model_passes: tuple[ModelPassTrace, ...],
        response_path: str,
        activity: str,
        policy_sources: tuple[str, ...] = (),
        dialogue_state: dict[str, Any] | None = None,
        diagnostic_error: object | None = None,
    ) -> LocalTurnResult:
        state_payload = dialogue_state or {}
        return LocalTurnResult(
            response=response,
            conversation=conversation,
            snapshot=snapshot,
            route=route,
            tool_calls=tool_calls,
            tool_results=tool_results,
            model_passes=model_passes,
            response_path=response_path,
            activity=activity,
            policy_sources=policy_sources,
            dialogue_state=state_payload,
            diagnostics=render_local_diagnostics(
                route=route,
                calls=tool_calls,
                results=tool_results,
                response_path=response_path,
                model_passes=model_passes,
                visible_response=response,
                runtime_metadata=self.runtime_metadata(),
                policy_sources=policy_sources,
                dialogue_state=state_payload,
                diagnostic_error=diagnostic_error,
            ),
        )


def visible_conversation(
    conversation: list[dict[str, Any]] | None,
    policy_chunk_lookup: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Render displayable turns, replacing policy citation tags with numbered markers.

    Each entry carries ``policy_references`` (empty unless the message cited a
    policy chunk) alongside ``role``/``content``; ``content`` is display text
    only — the canonical conversation passed in keeps its raw ``[Policy: id]``
    tags untouched.
    """
    visible: list[dict[str, Any]] = []
    for item in conversation or []:
        if not (
            isinstance(item, dict)
            and item.get("role") in {"user", "assistant"}
            and isinstance(item.get("content"), str)
            and str(item["content"]).strip()
            and not item.get("tool_calls")
        ):
            continue
        role = str(item["role"])
        content = str(item["content"])
        if role == "assistant":
            display_text, references = render_policy_citations(content, policy_chunk_lookup)
            visible.append({"role": role, "content": display_text, "policy_references": references})
        else:
            visible.append({"role": role, "content": content, "policy_references": []})
    return visible


def _rendered_prefill(prefill: str) -> str:
    """Show text injected before generation, so a trace never credits it to the model.

    Tolerates a trace object without the field: this runs on the error path, where a
    renderer crash would replace a real diagnostic with a rendering failure.
    """

    if not prefill:
        return ""
    return f"prefilled with:\n\n```text\n{prefill}\n```\n\n"


def render_local_diagnostics(
    *,
    route: dict[str, Any],
    calls: tuple[ToolCall, ...],
    results: tuple[dict[str, Any], ...],
    response_path: str,
    model_passes: tuple[ModelPassTrace, ...],
    visible_response: str,
    runtime_metadata: dict[str, str],
    policy_sources: tuple[str, ...] = (),
    dialogue_state: dict[str, Any] | None = None,
    diagnostic_error: object | None = None,
) -> str:
    decision = router_diagnostic_fields(route)
    candidates = route.get("capability_candidates", [])
    candidate_text = (
        "\n".join(
            f"- `{item.get('capability')}`: {float(item.get('probability', 0)):.3f}"
            for item in candidates
            if isinstance(item, dict)
        )
        or "- None"
    )
    call_text = (
        "\n".join(
            f"- `{call.id}` `{call.name}` `{json.dumps(call.arguments, sort_keys=True)}`"
            for call in calls
        )
        or "- None"
    )
    result_text = (
        "\n".join(
            f"- `{call.id}` `{call.name}`: {'success' if result.get('ok') else 'error'}"
            for call, result in zip(calls, results, strict=False)
        )
        or "- None"
    )
    pass_text = (
        "\n".join(
            (
                f"- `{trace.label}` — input tokens `{trace.input_tokens}`, prompt SHA-256 "
                f"`{trace.prompt_sha256}`, output SHA-256 `{trace.output_sha256}`, "
                f"runtime `{trace.runtime_device}`, CUDA `{trace.cuda_device_name}`\n\n"
                + _rendered_prefill(getattr(trace, "prefill", ""))
                + f"```text\n{trace.raw_output}\n```"
            )
            for trace in model_passes
        )
        or "- None; Granite was not invoked."
    )
    visible_hash = hashlib.sha256(visible_response.encode("utf-8")).hexdigest()
    summary = diagnostic_summary(
        route=route,
        calls=calls,
        results=results,
        response_path=response_path,
        model_passes=model_passes,
        policy_sources=policy_sources,
        error=diagnostic_error,
    )
    return (
        f"{summary}\n\n### Full technical details\n\n"
        f"- Route: `{route.get('route')}`\n"
        f"- In-domain probability: `{route.get('banking_probability')}`\n"
        f"- OOD probability: `{route.get('ood_probability')}`\n"
        f"- Context applied: `{route.get('context_applied', False)}`\n"
        f"- V6 domain: `{decision['domain']}`\n"
        f"- V6 lane: `{decision['lane']}`\n"
        f"- V6 family: `{decision['family']}`\n"
        f"- V6 intent: `{decision['intent']}`\n"
        f"- V6 action: `{decision['action']}`\n"
        f"- V6 entity resolution: `{decision['entity_resolution']}`\n"
        f"- Exposed tools: `{json.dumps(decision['exposed_tools'])}`\n"
        f"- Relation probabilities: "
        f"`{json.dumps(route.get('relation_probabilities', {}), sort_keys=True)}`\n"
        f"- Router reason: `{route.get('reason', 'not provided')}`\n"
        f"- Router failure type: `{route.get('failure_type', 'none')}`\n"
        f"- Constraint notes: "
        f"`{json.dumps(list(route.get('constraint_diagnostics') or []))}`\n"
        f"- Response path: `{response_path}`\n\n"
        f"- Policy sources: `{json.dumps(policy_sources)}`\n"
        f"- Dialogue state: `{json.dumps(dialogue_state or {}, sort_keys=True)}`\n\n"
        f"**Diagnostic capabilities**\n{candidate_text}\n\n"
        f"**Generated tool calls**\n{call_text}\n\n"
        f"**Tool results**\n{result_text}\n\n"
        f"**Model passes**\n{pass_text}\n\n"
        f"- Generation calls: `{len(model_passes)}`\n"
        f"- Model: `{runtime_metadata.get('model_id', MODEL_ID)}`\n"
        f"- Exact model revision: "
        f"`{runtime_metadata.get('model_revision', MODEL_REVISION)}`\n"
        f"- Exact router revision: `{ROUTER_REVISION}`\n"
        f"- Runtime device: `{runtime_metadata.get('runtime_device', 'unavailable')}`\n"
        f"- CUDA device: `{runtime_metadata.get('cuda_device_name', 'unavailable')}`\n"
        f"- Weight quantization: "
        f"`{runtime_metadata.get('weight_quantization', 'unavailable')}`\n"
        f"- CUDA memory allocated GiB: "
        f"`{runtime_metadata.get('cuda_memory_allocated_gib', 'unavailable')}`\n"
        "- Registered execution boundary: `Local CUDA / NF4`\n"
        f"- Canonical response SHA-256: `{visible_hash}`"
    )


def _uncertain_route(reason: str) -> dict[str, Any]:
    return {
        "route": "uncertain",
        "banking_probability": None,
        "ood_probability": None,
        "confidence": None,
        "capability": None,
        "capability_confidence": None,
        "capability_candidates": [],
        "intent": None,
        "intent_confidence": None,
        "intent_candidates": [],
        "lane": None,
        "relation_probabilities": {},
        "context_applied": False,
        "router_revision": ROUTER_REVISION,
        "reason": reason,
    }
