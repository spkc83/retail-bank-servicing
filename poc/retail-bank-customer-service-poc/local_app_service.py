from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

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
)
from responses import MODEL_FAILURE_RESPONSE, OOD_RESPONSE
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
    ) -> None:
        self.bank = bank
        self.runtime = runtime
        self.router = router

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
        route = self._route(message.strip(), canonical)
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
            turn = agent.run_turn(
                username=username,
                session_hash=session_hash,
                message=message,
                conversation=canonical,
                router_result=route,
            )
        except AgentExecutionError as error:
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
                response_path="local model/tool follow-up failure",
                activity=(
                    "Granite failed after the recorded tool calls; no CPU-authored "
                    "servicing answer was substituted."
                ),
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
                model_passes=(),
                response_path="local model failure",
                activity=(
                    "Granite generation failed; no CPU-authored servicing answer "
                    "was substituted."
                ),
            )

        activity = (
            "Granite selected and executed the recorded synthetic tools, then "
            "authored the final response."
            if turn.tool_calls
            else "Granite authored the response directly."
        )
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
        )

    def snapshot(self, username: str, session_hash: str) -> dict[str, Any]:
        return self.bank.snapshot(username, session_hash)

    def reset(self, username: str, session_hash: str) -> dict[str, Any]:
        return self.bank.reset(username, session_hash)

    def runtime_metadata(self) -> dict[str, str]:
        provider = getattr(self.runtime, "runtime_metadata", None)
        return provider() if callable(provider) else {}

    def _route(
        self,
        message: str,
        conversation: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.router is None:
            return _uncertain_route("local router unavailable; delegated to Granite")
        try:
            return dict(self.router.classify(message, conversation))
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
    ) -> LocalTurnResult:
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
            diagnostics=render_local_diagnostics(
                route=route,
                calls=tool_calls,
                results=tool_results,
                response_path=response_path,
                model_passes=model_passes,
                visible_response=response,
                runtime_metadata=self.runtime_metadata(),
            ),
        )


def visible_conversation(
    conversation: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    return [
        {"role": str(item["role"]), "content": str(item["content"])}
        for item in (conversation or [])
        if isinstance(item, dict)
        and item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
        and str(item["content"]).strip()
        and not item.get("tool_calls")
    ]


def render_local_diagnostics(
    *,
    route: dict[str, Any],
    calls: tuple[ToolCall, ...],
    results: tuple[dict[str, Any], ...],
    response_path: str,
    model_passes: tuple[ModelPassTrace, ...],
    visible_response: str,
    runtime_metadata: dict[str, str],
) -> str:
    candidates = route.get("capability_candidates", [])
    candidate_text = "\n".join(
        f"- `{item.get('capability')}`: {float(item.get('probability', 0)):.3f}"
        for item in candidates
        if isinstance(item, dict)
    ) or "- None"
    call_text = "\n".join(
        f"- `{call.id}` `{call.name}` `{json.dumps(call.arguments, sort_keys=True)}`"
        for call in calls
    ) or "- None"
    result_text = "\n".join(
        f"- `{call.id}` `{call.name}`: {'success' if result.get('ok') else 'error'}"
        for call, result in zip(calls, results, strict=False)
    ) or "- None"
    pass_text = "\n".join(
        (
            f"- `{trace.label}` — input tokens `{trace.input_tokens}`, prompt SHA-256 "
            f"`{trace.prompt_sha256}`, output SHA-256 `{trace.output_sha256}`, "
            f"runtime `{trace.runtime_device}`, CUDA `{trace.cuda_device_name}`\n\n"
            f"```text\n{trace.raw_output}\n```"
        )
        for trace in model_passes
    ) or "- None; Granite was not invoked."
    visible_hash = hashlib.sha256(visible_response.encode("utf-8")).hexdigest()
    return (
        "### Local experiment diagnostics\n\n"
        f"- Route: `{route.get('route')}`\n"
        f"- In-domain probability: `{route.get('banking_probability')}`\n"
        f"- OOD probability: `{route.get('ood_probability')}`\n"
        f"- Context applied: `{route.get('context_applied', False)}`\n"
        f"- Relation probabilities: "
        f"`{json.dumps(route.get('relation_probabilities', {}), sort_keys=True)}`\n"
        f"- Response path: `{response_path}`\n\n"
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
        f"- Visible response SHA-256: `{visible_hash}`"
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
        "relation_probabilities": {},
        "context_applied": False,
        "router_revision": ROUTER_REVISION,
        "reason": reason,
    }
