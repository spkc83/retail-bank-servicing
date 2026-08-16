from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from dialogue_state import SERVICING_TOOLS
from mock_bank import SessionBankRegistry
from response_policy import (
    build_customer_experience_repair_messages,
    build_final_repair_messages,
    render_read_tool_results,
    validate_customer_facing_answer,
    validate_grounded_answer,
    validate_policy_answer,
)

INPUT_TOKEN_BUDGET = 8192
MAX_NEW_TOKENS = 512
MAX_TOOL_CALLS = 8

AGENT_SYSTEM_PROMPT = (
    "You are Harbor, the conversational customer-service assistant for Harborlight "
    "Bank. The customer is already authenticated. Use the supplied tools for "
    "customer-specific banking records or actions, use tool results for final answers, "
    "call dependent tools one at a time so each later call can use the earlier result, "
    "and never ask for account numbers, customer IDs, passwords, PINs, or private IDs."
    " Respond warmly and concisely, acknowledge distress only when the customer "
    "explicitly expresses it, never infer distress from a neutral greeting or request, "
    "name banking products clearly, and never mention prototypes, demos, synthetic "
    "data, models, "
    "routers, tools, GPUs, CPUs, or implementation details."
)

MODEL_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_accounts",
            "description": "List the signed-in customer's accounts and balances.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_cards",
            "description": "List the signed-in customer's cards and statuses.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_service_cases",
            "description": "List the signed-in customer's recent service cases.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_transactions",
            "description": "List the signed-in customer's recent account transactions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_transfers",
            "description": "List the signed-in customer's transfers and statuses.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "freeze_card",
            "description": "Freeze a card, optionally selected by last four digits.",
            "parameters": {
                "type": "object",
                "properties": {"last4": {"type": ["string", "null"]}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_card",
            "description": "Request replacement of a card.",
            "parameters": {
                "type": "object",
                "properties": {"last4": {"type": ["string", "null"]}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dispute_transaction",
            "description": "Dispute a transaction by description.",
            "parameters": {
                "type": "object",
                "properties": {"description": {"type": ["string", "null"]}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_transfer",
            "description": "Cancel a pending transfer by recipient.",
            "parameters": {
                "type": "object",
                "properties": {"recipient": {"type": ["string", "null"]}},
                "additionalProperties": False,
            },
        },
    },
]

_TOOL_CALL_BLOCK = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    flags=re.DOTALL,
)


@dataclass(frozen=True)
class ModelPassTrace:
    label: str
    input_tokens: int
    prompt_sha256: str
    output_chars: int
    output_sha256: str
    raw_output: str
    runtime_device: str
    cuda_device_name: str


class AgentProtocolError(ValueError):
    pass


class AgentExecutionError(AgentProtocolError):
    def __init__(
        self,
        message: str,
        *,
        conversation: list[dict[str, Any]],
        tool_calls: tuple[ToolCall, ...],
        tool_results: tuple[dict[str, Any], ...],
        snapshot: dict[str, Any],
        model_passes: tuple[ModelPassTrace, ...],
    ) -> None:
        super().__init__(message)
        self.conversation = conversation
        self.tool_calls = tool_calls
        self.tool_results = tool_results
        self.snapshot = snapshot
        self.model_passes = model_passes


class ModelRuntime(Protocol):
    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_new_tokens: int,
    ) -> str: ...

    def count_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> int: ...


@dataclass(frozen=True)
class ToolCall:
    id: str
    index: int
    name: str
    arguments: dict[str, Any]

    def as_message_call(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "index": self.index,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


class ToolSyntaxAdapter(Protocol):
    family: str

    def render_tools(
        self,
        public_tool_manifest: list[dict[str, Any]],
    ) -> list[dict[str, Any]]: ...

    def parse_assistant(
        self,
        output: str,
        *,
        turn_key: str | None = None,
    ) -> tuple[ToolCall, ...]: ...

    def render_assistant_tool_calls(
        self,
        calls: tuple[ToolCall, ...],
    ) -> dict[str, Any]: ...

    def render_tool_result(
        self,
        call: ToolCall,
        content: dict[str, Any],
    ) -> dict[str, Any]: ...


class TaggedJsonToolSyntaxAdapter:
    family = "tagged-json"

    def render_tools(
        self,
        public_tool_manifest: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return public_tool_manifest

    def parse_assistant(
        self,
        output: str,
        *,
        turn_key: str | None = None,
    ) -> tuple[ToolCall, ...]:
        return _parse_tagged_json_tool_calls(output, turn_key=turn_key)

    def render_assistant_tool_calls(
        self,
        calls: tuple[ToolCall, ...],
    ) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [call.as_message_call() for call in calls],
        }

    def render_tool_result(
        self,
        call: ToolCall,
        content: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.name,
            "content": json.dumps(
                content,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        }


class GraniteToolSyntaxAdapter(TaggedJsonToolSyntaxAdapter):
    family = "granite"


@dataclass(frozen=True)
class AgentTurnResult:
    response: str
    conversation: list[dict[str, Any]]
    tool_calls: tuple[ToolCall, ...]
    tool_results: tuple[dict[str, Any], ...]
    snapshot: dict[str, Any]
    response_path: str
    model_passes: tuple[ModelPassTrace, ...]
    policy_sources: tuple[str, ...] = ()


class ConversationalBankingAgent:
    def __init__(
        self,
        *,
        bank: SessionBankRegistry,
        model: ModelRuntime,
        tool_adapter: ToolSyntaxAdapter | None = None,
        input_budget: int = INPUT_TOKEN_BUDGET,
    ) -> None:
        self.bank = bank
        self.model = model
        self.tool_adapter = tool_adapter or GraniteToolSyntaxAdapter()
        self.input_budget = input_budget

    def run_turn(
        self,
        *,
        username: str,
        session_hash: str,
        message: str,
        conversation: list[dict[str, Any]],
        router_result: dict[str, Any],
        pinned_exchange: list[dict[str, Any]] | None = None,
    ) -> AgentTurnResult:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        canonical = canonical_conversation(conversation)
        current = [*canonical, {"role": "user", "content": message.strip()}]
        system, public_tools = _generation_plan(router_result)
        serving_tools = self.tool_adapter.render_tools(public_tools) if public_tools else None
        routed_single_call = _requires_tool_call(router_result, public_tools)
        first_context = select_token_budgeted_context(
            system,
            current,
            tools=serving_tools,
            token_counter=self.model.count_tokens,
            input_budget=self.input_budget,
            pinned_exchange=pinned_exchange,
        )
        model_passes: list[ModelPassTrace] = []
        first_output, first_trace = self._generate_pass(
            "base",
            first_context,
            serving_tools,
        )
        model_passes.append(first_trace)
        with_tools = [*current]
        all_calls: list[ToolCall] = []
        results: list[dict[str, Any]] = []
        try:
            if not first_output:
                raise AgentProtocolError("model returned an empty first response")
            calls = self.tool_adapter.parse_assistant(
                first_output,
                turn_key=first_trace.prompt_sha256,
            )
            if not calls and routed_single_call:
                retry_system = _required_tool_retry_system(system, public_tools)
                retry_context = select_token_budgeted_context(
                    retry_system,
                    current,
                    tools=serving_tools,
                    token_counter=self.model.count_tokens,
                    input_budget=self.input_budget,
                    pinned_exchange=pinned_exchange,
                )
                retry_output, retry_trace = self._generate_pass(
                    "required_tool_retry_1",
                    retry_context,
                    serving_tools,
                )
                model_passes.append(retry_trace)
                if not retry_output:
                    raise AgentProtocolError(
                        "model returned an empty response when a tool call was required"
                    )
                calls = self.tool_adapter.parse_assistant(
                    retry_output,
                    turn_key=retry_trace.prompt_sha256,
                )
                if not calls:
                    raise AgentProtocolError(
                        "model did not return a required tool call after one retry"
                    )
            if routed_single_call and len(calls) != 1:
                raise AgentProtocolError("routed servicing turns require exactly one tool call")
            if not calls:
                return self._complete_without_tools(
                    username=username,
                    session_hash=session_hash,
                    current=current,
                    first_output=first_output,
                    response_path="direct_answer",
                    model_passes=model_passes,
                )
            response_path = "base_tool"
            _validate_tool_calls(calls, allowed_tools=public_tools)
            pending_calls = calls
            post_tool_passes = 0
            while True:
                _validate_tool_calls(pending_calls, allowed_tools=public_tools)
                if len(all_calls) + len(pending_calls) > MAX_TOOL_CALLS:
                    raise AgentProtocolError(
                        f"model attempted more than {MAX_TOOL_CALLS} total tool calls"
                    )
                call_message = self.tool_adapter.render_assistant_tool_calls(pending_calls)
                with_tools.append(call_message)
                all_calls.extend(pending_calls)
                for call in pending_calls:
                    result = self._execute_tool(username, session_hash, call)
                    results.append(result)
                    with_tools.append(self.tool_adapter.render_tool_result(call, result))

                post_tool_passes += 1
                followup_system = _grounded_final_system(system) if routed_single_call else system
                followup_tools = None if routed_single_call else serving_tools
                followup_context = select_token_budgeted_context(
                    followup_system,
                    with_tools,
                    tools=followup_tools,
                    token_counter=self.model.count_tokens,
                    input_budget=self.input_budget,
                    pinned_exchange=pinned_exchange,
                )
                pass_label = (
                    "grounded_final"
                    if post_tool_passes == 1
                    else f"tool_followup_{post_tool_passes}"
                )
                followup_output, followup_trace = self._generate_pass(
                    pass_label,
                    followup_context,
                    followup_tools,
                )
                model_passes.append(followup_trace)
                if not followup_output:
                    raise AgentProtocolError("model returned an empty follow-up response")
                next_calls = self.tool_adapter.parse_assistant(
                    followup_output,
                    turn_key=followup_trace.prompt_sha256,
                )
                if routed_single_call and next_calls:
                    raise AgentProtocolError(
                        "grounded-final response attempted another routed tool call"
                    )
                if not next_calls:
                    final_output = followup_output
                    break
                pending_calls = next_calls
                if not response_path.endswith("_chain"):
                    response_path = f"{response_path}_chain"
            rendered = render_read_tool_results(all_calls, results)
            if rendered is not None:
                final_output = rendered
                response_path = f"{response_path}_rendered"
            else:
                validation = validate_grounded_answer(final_output, all_calls, results)
                if not validation.valid:
                    repair_messages = build_final_repair_messages(
                        user_message=message.strip(),
                        draft=final_output,
                        calls=all_calls,
                        results=results,
                        errors=validation.errors,
                    )
                    repaired, repair_trace = self._generate_pass(
                        "final_repair_1",
                        repair_messages,
                        None,
                    )
                    model_passes.append(repair_trace)
                    if "<tool_call" in repaired:
                        raise AgentProtocolError("final-answer repair attempted a tool call")
                    repaired_validation = validate_grounded_answer(
                        repaired,
                        all_calls,
                        results,
                    )
                    if not repaired_validation.valid:
                        raise AgentProtocolError(
                            "final-answer repair failed grounding validation: "
                            + "; ".join(repaired_validation.errors)
                        )
                    final_output = repaired
                    response_path = f"{response_path}_repaired"
            final_output, response_path = self._ensure_customer_facing(
                user_message=message.strip(),
                draft=final_output,
                response_path=response_path,
                model_passes=model_passes,
                authoritative_evidence=results,
            )
        except (AgentProtocolError, RuntimeError, TypeError, ValueError) as error:
            raise AgentExecutionError(
                str(error),
                conversation=with_tools,
                tool_calls=tuple(all_calls),
                tool_results=tuple(results),
                snapshot=self.bank.snapshot(username, session_hash),
                model_passes=tuple(model_passes),
            ) from error
        completed = [*with_tools, {"role": "assistant", "content": final_output}]
        return AgentTurnResult(
            response=final_output,
            conversation=completed,
            tool_calls=tuple(all_calls),
            tool_results=tuple(results),
            snapshot=self.bank.snapshot(username, session_hash),
            response_path=response_path,
            model_passes=tuple(model_passes),
        )

    def run_policy_turn(
        self,
        *,
        username: str,
        session_hash: str,
        message: str,
        conversation: list[dict[str, Any]],
        policy_matches: tuple[dict[str, Any], ...],
        corpus_revision: str,
        pinned_exchange: list[dict[str, Any]] | None = None,
    ) -> AgentTurnResult:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        if not policy_matches:
            raise AgentProtocolError("policy generation requires retrieved evidence")
        canonical = canonical_conversation(conversation)
        current = [*canonical, {"role": "user", "content": message.strip()}]
        system = _policy_system_message(policy_matches, corpus_revision)
        context = select_token_budgeted_context(
            system,
            current,
            tools=None,
            token_counter=self.model.count_tokens,
            input_budget=self.input_budget,
            pinned_exchange=pinned_exchange,
        )
        output, trace = self._generate_pass("policy_grounded", context, None)
        model_passes = [trace]
        validation = validate_policy_answer(output, policy_matches)
        response_path = "policy_grounded"
        if not validation.valid:
            repair_messages = build_customer_experience_repair_messages(
                user_message=message.strip(),
                draft=output,
                errors=validation.errors,
                authoritative_evidence=policy_matches,
            )
            output, repair_trace = self._generate_pass("policy_repair_1", repair_messages, None)
            model_passes.append(repair_trace)
            repaired_validation = validate_policy_answer(output, policy_matches)
            if not repaired_validation.valid:
                raise AgentProtocolError(
                    "policy-answer repair failed validation: "
                    + "; ".join(repaired_validation.errors)
                )
            response_path = "policy_grounded_repaired"
        sources = tuple(str(item["chunk_id"]) for item in policy_matches)
        return AgentTurnResult(
            response=output,
            conversation=[*current, {"role": "assistant", "content": output}],
            tool_calls=(),
            tool_results=(),
            snapshot=self.bank.snapshot(username, session_hash),
            response_path=response_path,
            model_passes=tuple(model_passes),
            policy_sources=sources,
        )

    def _complete_without_tools(
        self,
        *,
        username: str,
        session_hash: str,
        current: list[dict[str, Any]],
        first_output: str,
        response_path: str,
        model_passes: list[ModelPassTrace],
    ) -> AgentTurnResult:
        final_output, final_path = self._ensure_customer_facing(
            user_message=str(current[-1]["content"]),
            draft=first_output,
            response_path=response_path,
            model_passes=model_passes,
        )
        completed = [*current, {"role": "assistant", "content": final_output}]
        return AgentTurnResult(
            response=final_output,
            conversation=completed,
            tool_calls=(),
            tool_results=(),
            snapshot=self.bank.snapshot(username, session_hash),
            response_path=final_path,
            model_passes=tuple(model_passes),
        )

    def _ensure_customer_facing(
        self,
        *,
        user_message: str,
        draft: str,
        response_path: str,
        model_passes: list[ModelPassTrace],
        authoritative_evidence: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
    ) -> tuple[str, str]:
        validation = validate_customer_facing_answer(draft)
        if validation.valid:
            return draft, response_path
        repair_messages = build_customer_experience_repair_messages(
            user_message=user_message,
            draft=draft,
            errors=validation.errors,
            authoritative_evidence=authoritative_evidence,
        )
        repaired, repair_trace = self._generate_pass(
            "customer_experience_repair_1", repair_messages, None
        )
        model_passes.append(repair_trace)
        repaired_validation = validate_customer_facing_answer(repaired)
        if not repaired_validation.valid:
            raise AgentProtocolError(
                "customer-experience repair failed validation: "
                + "; ".join(repaired_validation.errors)
            )
        return repaired, f"{response_path}_customer_repaired"

    def _generate_pass(
        self,
        label: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[str, ModelPassTrace]:
        input_tokens = self.model.count_tokens(messages, tools)
        prompt_payload = json.dumps(
            {"messages": messages, "tools": tools},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        output = self.model.generate(
            messages,
            tools,
            MAX_NEW_TOKENS,
        ).strip()
        metadata_provider = getattr(self.model, "runtime_metadata", None)
        metadata = metadata_provider() if callable(metadata_provider) else {}
        return output, ModelPassTrace(
            label=label,
            input_tokens=input_tokens,
            prompt_sha256=hashlib.sha256(prompt_payload.encode("utf-8")).hexdigest(),
            output_chars=len(output),
            output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
            raw_output=output,
            runtime_device=str(metadata.get("runtime_device", "unavailable")),
            cuda_device_name=str(metadata.get("cuda_device_name", "unavailable")),
        )

    def _execute_tool(
        self,
        username: str,
        session_hash: str,
        call: ToolCall,
    ) -> dict[str, Any]:
        try:
            result = self.bank.execute(
                username,
                session_hash,
                call.name,
                call.arguments,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            return {
                "ok": False,
                "error": _safe_tool_error(error),
            }
        return {
            "ok": True,
            "result": result,
        }


def parse_tool_calls(output: str) -> tuple[ToolCall, ...]:
    return GraniteToolSyntaxAdapter().parse_assistant(output)


def _parse_tagged_json_tool_calls(
    output: str,
    *,
    turn_key: str | None = None,
) -> tuple[ToolCall, ...]:
    if not isinstance(output, str):
        raise AgentProtocolError("model output must be text")
    blocks = _TOOL_CALL_BLOCK.findall(output)
    has_protocol_marker = "<tool_call" in output or "</tool_call>" in output
    if has_protocol_marker and not blocks:
        raise AgentProtocolError("model returned a malformed tool-call block")
    if len(blocks) > MAX_TOOL_CALLS:
        raise AgentProtocolError(f"model returned more than {MAX_TOOL_CALLS} tool calls")
    calls: list[ToolCall] = []
    for index, block in enumerate(blocks):
        try:
            payload = json.loads(block)
        except json.JSONDecodeError as error:
            raise AgentProtocolError("model tool call is not valid JSON") from error
        if not isinstance(payload, dict):
            raise AgentProtocolError("model tool call must be a JSON object")
        name = payload.get("name")
        arguments = payload.get("arguments")
        if not isinstance(name, str) or not name.strip():
            raise AgentProtocolError("model tool call requires a function name")
        if not isinstance(arguments, dict):
            raise AgentProtocolError("model tool-call arguments must be an object")
        explicit_id = payload.get("id")
        call_id = (
            explicit_id.strip()
            if isinstance(explicit_id, str) and explicit_id.strip()
            else _stable_tool_call_id(index, name, output, turn_key)
        )
        explicit_index = payload.get("index")
        if explicit_index is not None and (
            not isinstance(explicit_index, int) or isinstance(explicit_index, bool)
        ):
            raise AgentProtocolError("model tool-call index must be an integer")
        call_index = explicit_index if explicit_index is not None else index
        calls.append(
            ToolCall(
                id=call_id,
                index=call_index,
                name=name.strip(),
                arguments=arguments,
            )
        )
    return tuple(calls)


def _stable_tool_call_id(
    index: int,
    name: str,
    output: str,
    turn_key: str | None,
) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip()).strip("_") or "tool"
    digest_payload = f"{turn_key or ''}\0{output}".encode()
    digest = hashlib.sha256(digest_payload).hexdigest()[:10]
    return f"call_{digest}_{index}_{slug}"


def _validate_tool_calls(
    calls: tuple[ToolCall, ...],
    *,
    allowed_tools: list[dict[str, Any]] = MODEL_TOOLS,
) -> None:
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in allowed_tools}
    allowed_names = {tool["function"]["name"] for tool in allowed_tools}
    supported_names = {tool["function"]["name"] for tool in MODEL_TOOLS}
    seen_ids: set[str] = set()
    for expected_index, call in enumerate(calls):
        if call.index != expected_index:
            raise AgentProtocolError("model tool-call indexes must be ordered from zero")
        if call.id in seen_ids:
            raise AgentProtocolError("model tool-call IDs must be unique")
        seen_ids.add(call.id)
        if call.name not in supported_names:
            raise AgentProtocolError(f"model selected unsupported tool: {call.name}")
        if call.name not in allowed_names:
            raise AgentProtocolError(f"model selected unexposed tool: {call.name}")
        schema = schemas[call.name]
        _validate_arguments(call.name, call.arguments, schema)


def _validate_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    properties = schema.get("properties")
    allowed = properties if isinstance(properties, dict) else {}
    extras = set(arguments) - set(allowed)
    if extras and schema.get("additionalProperties") is False:
        raise AgentProtocolError(
            f"model supplied unsupported arguments for {tool_name}: {sorted(extras)}"
        )
    required = schema.get("required")
    if isinstance(required, list):
        missing = [name for name in required if name not in arguments]
        if missing:
            raise AgentProtocolError(f"model omitted required argument for {tool_name}: {missing}")
    for name, value in arguments.items():
        subschema = allowed.get(name)
        if not isinstance(subschema, dict):
            continue
        expected = subschema.get("type")
        expected_types = expected if isinstance(expected, list) else [expected]
        if not _value_matches_json_types(value, expected_types):
            raise AgentProtocolError(f"model supplied invalid type for {tool_name}.{name}")
        if "const" in subschema and value != subschema["const"]:
            raise AgentProtocolError(f"model supplied value outside const for {tool_name}.{name}")
        enum = subschema.get("enum")
        if isinstance(enum, list) and value not in enum:
            raise AgentProtocolError(f"model supplied value outside enum for {tool_name}.{name}")
        pattern = subschema.get("pattern")
        if isinstance(pattern, str) and (
            not isinstance(value, str) or re.fullmatch(pattern, value) is None
        ):
            raise AgentProtocolError(f"model supplied value outside pattern for {tool_name}.{name}")
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and isinstance(subschema.get("minimum"), int)
            and value < subschema["minimum"]
        ):
            raise AgentProtocolError(f"model supplied value below minimum for {tool_name}.{name}")
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and isinstance(subschema.get("maximum"), int)
            and value > subschema["maximum"]
        ):
            raise AgentProtocolError(f"model supplied value above maximum for {tool_name}.{name}")


def _value_matches_json_types(value: Any, expected_types: list[Any]) -> bool:
    for expected in expected_types:
        if expected == "null" and value is None:
            return True
        if expected == "string" and isinstance(value, str):
            return True
        if expected == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if expected == "number" and isinstance(value, int | float) and not isinstance(value, bool):
            return True
        if expected == "boolean" and isinstance(value, bool):
            return True
        if expected == "object" and isinstance(value, dict):
            return True
        if expected == "array" and isinstance(value, list):
            return True
    return False


def _safe_tool_error(error: Exception) -> dict[str, str]:
    message = str(error) or error.__class__.__name__
    if message.startswith("unsupported tool:"):
        code = "unsupported_tool"
        safe_message = "The requested tool is not available."
    elif message.startswith("unsupported arguments"):
        code = "invalid_arguments"
        safe_message = "The tool arguments are not valid for this tool."
    elif message.startswith("select a transaction by ID or description"):
        code = "ambiguous_transaction_selector"
        safe_message = "Select a transaction by either ID or description, not both."
    elif message.startswith("select a transfer by ID or recipient"):
        code = "ambiguous_transfer_selector"
        safe_message = "Select a transfer by either ID or recipient, not both."
    elif message.startswith("expected exactly one matching synthetic"):
        code = "record_match_count"
        safe_message = "The request did not match exactly one banking record."
    elif message == "no matching synthetic banking record":
        code = "record_not_found"
        safe_message = "No matching banking record was found."
    else:
        code = "backend_error"
        safe_message = "The banking service could not complete the request."
    return {
        "code": code,
        "message": safe_message,
    }


def select_token_budgeted_context(
    system_message: dict[str, Any],
    conversation: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    token_counter: Any,
    input_budget: int = INPUT_TOKEN_BUDGET,
    pinned_exchange: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if input_budget < 1:
        raise ValueError("input budget must be positive")
    canonical = canonical_conversation(conversation)
    groups = _conversation_groups(canonical)
    if not groups:
        selected = [system_message]
        if token_counter(selected, tools) > input_budget:
            raise AgentProtocolError("system prompt exceeds the model input budget")
        return selected

    required_indexes = {len(groups) - 1}
    canonical_pin = canonical_conversation(pinned_exchange)
    if canonical_pin:
        pin_index = next(
            (index for index, group in enumerate(groups) if group == canonical_pin),
            None,
        )
        if pin_index is None:
            raise AgentProtocolError("pinned servicing exchange is not in conversation")
        required_indexes.add(pin_index)
    selected_indexes = set(required_indexes)
    selected = [
        system_message,
        *(message for index in sorted(selected_indexes) for message in groups[index]),
    ]
    if token_counter(selected, tools) > input_budget:
        label = "pinned servicing exchange" if canonical_pin else "latest conversation turn"
        raise AgentProtocolError(f"{label} exceeds the model input budget")
    for index in reversed(range(len(groups) - 1)):
        if index in selected_indexes:
            continue
        proposal_indexes = {*selected_indexes, index}
        proposal = [
            system_message,
            *(
                message
                for group_index in sorted(proposal_indexes)
                for message in groups[group_index]
            ),
        ]
        if token_counter(proposal, tools) <= input_budget:
            selected_indexes = proposal_indexes
    return [
        system_message,
        *(message for group_index in sorted(selected_indexes) for message in groups[group_index]),
    ]


def canonical_conversation(
    conversation: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not isinstance(conversation, list):
        return []
    canonical: list[dict[str, Any]] = []
    for item in conversation:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role == "user" and isinstance(content, str) and content.strip():
            canonical.append({"role": "user", "content": content.strip()})
        elif role == "assistant" and isinstance(content, str):
            tool_calls = item.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                canonical.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    }
                )
            elif content.strip():
                canonical.append({"role": "assistant", "content": content.strip()})
        elif role == "tool" and isinstance(item.get("name"), str) and isinstance(content, str):
            canonical_tool = {
                "role": "tool",
                "name": str(item["name"]),
                "content": content,
            }
            tool_call_id = item.get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id.strip():
                canonical_tool["tool_call_id"] = tool_call_id.strip()
            canonical.append(canonical_tool)
    return canonical


def _conversation_groups(
    conversation: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for item in conversation:
        if item["role"] == "user":
            groups.append([item])
        elif groups:
            groups[-1].append(item)
    return groups


def _system_message(
    _router_result: dict[str, Any],
) -> dict[str, str]:
    return {
        "role": "system",
        "content": AGENT_SYSTEM_PROMPT,
    }


def _generation_plan(
    router_result: dict[str, Any],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    system = _system_message(router_result)
    if "action" not in router_result:
        return system, MODEL_TOOLS

    if router_result.get("route") != "in_domain":
        return {
            "role": "system",
            "content": (
                f"{system['content']}\n\nTURN GUIDANCE: Ask one concise, natural "
                "clarification question. Do not call a banking tool or claim that an "
                "action was completed because the classifier abstained on this turn."
            ),
        }, []

    action = router_result.get("action")
    entity_resolution = router_result.get("entity_resolution")
    intent = router_result.get("fine_intent", router_result.get("intent"))
    blocked_entity = entity_resolution in {"missing", "ambiguous", "ineligible"}
    instruction: str
    tools: list[dict[str, Any]] = []

    if blocked_entity or action == "clarify":
        instruction = (
            "Ask exactly one concise, natural clarification question that helps the "
            "customer supply or distinguish the missing banking detail. Do not claim "
            "that an action was completed."
        )
    elif action == "execute_tool" and isinstance(intent, str):
        tool_name = SERVICING_TOOLS.get(intent)
        if tool_name is None:
            instruction = (
                "Ask one concise, natural clarification question because no supported "
                "banking action is available for the predicted request."
            )
        else:
            tools = [tool for tool in MODEL_TOOLS if tool["function"]["name"] == tool_name]
            constraints = router_result.get("argument_constraints")
            if isinstance(constraints, dict) and constraints:
                tools = [_narrow_tool_schema(tools[0], constraints)]
                argument_instruction = (
                    "Emit every required argument exactly as constrained by the exposed "
                    "schema; do not omit, infer, or alter it."
                )
            else:
                argument_instruction = (
                    "Choose every argument from the conversation; this guidance supplies "
                    "no tool arguments."
                )
            instruction = (
                f"Use only {tool_name} for this turn. Call it when the conversation "
                "supplies the selectors its schema requires; otherwise ask one concise, "
                f"natural clarification question. {argument_instruction}"
            )
    elif action == "converse":
        instruction = (
            "Respond naturally and concisely without looking up customer records or "
            "performing a banking action. Never infer distress, trouble, or a failed "
            "banking event from a neutral greeting or social message."
        )
    else:
        instruction = (
            "Respond naturally without calling a banking tool. Ask one concise "
            "clarification question if the customer's request is not yet clear."
        )

    return {
        "role": "system",
        "content": f"{system['content']}\n\nTURN GUIDANCE: {instruction}",
    }, tools


def _narrow_tool_schema(
    tool: dict[str, Any],
    constraints: dict[str, Any],
) -> dict[str, Any]:
    function = tool["function"]
    parameters = function["parameters"]
    properties = parameters.get("properties", {})
    if set(constraints) - set(properties):
        raise AgentProtocolError("effective decision constrained an unsupported tool argument")
    narrowed_properties = dict(properties)
    for name, value in constraints.items():
        if not isinstance(value, str) or not value:
            raise AgentProtocolError("effective decision requires non-empty string constraints")
        narrowed_properties[name] = {
            "type": "string",
            "pattern": f"^{re.escape(value)}$",
            "const": value,
            "enum": [value],
        }
    return {
        **tool,
        "function": {
            **function,
            "parameters": {
                **parameters,
                "properties": narrowed_properties,
                "required": list(constraints),
            },
        },
    }


def _requires_tool_call(
    router_result: dict[str, Any],
    public_tools: list[dict[str, Any]],
) -> bool:
    return (
        "action" in router_result
        and router_result.get("route") == "in_domain"
        and router_result.get("action") == "execute_tool"
        and router_result.get("entity_resolution") in {"resolved", "not_required"}
        and len(public_tools) == 1
    )


def _required_tool_retry_system(
    system: dict[str, str],
    public_tools: list[dict[str, Any]],
) -> dict[str, str]:
    tool_name = str(public_tools[0]["function"]["name"])
    return {
        "role": "system",
        "content": (
            f"{system['content']}\n\nTOOL CALL REQUIRED: A tool call is required before "
            f"answering this customer-specific request. Call {tool_name}, the only "
            "available tool. Do not describe, simulate, or claim the result before "
            "the application returns the tool result."
        ),
    }


def _grounded_final_system(system: dict[str, str]) -> dict[str, str]:
    return {
        "role": "system",
        "content": (
            f"{system['content']}\n\nGROUNDED FINAL REQUIRED: The routed banking tool "
            "has already run. Do not call any tool again. Answer only from the returned "
            "tool result and do not claim anything the result does not establish."
        ),
    }


def router_diagnostic_fields(router_result: dict[str, Any]) -> dict[str, Any]:
    """Return safe route metadata and the exact tool surface shown to Granite."""
    compatibility_value = "not available (V3 compatibility)"
    _system, tools = _generation_plan(router_result)
    tool_names = [
        str(tool["function"]["name"])
        for tool in tools
        if isinstance(tool.get("function"), dict) and isinstance(tool["function"].get("name"), str)
    ]
    return {
        "domain": router_result.get("domain", compatibility_value),
        "lane": router_result.get("lane", compatibility_value),
        "family": router_result.get(
            "family",
            router_result.get("capability", compatibility_value),
        ),
        "intent": router_result.get(
            "fine_intent",
            router_result.get("intent", compatibility_value),
        ),
        "action": router_result.get("action", compatibility_value),
        "entity_resolution": router_result.get(
            "entity_resolution",
            compatibility_value,
        ),
        "effective_decision_contract": router_result.get(
            "effective_decision_contract",
            compatibility_value,
        ),
        "decision_accepted": router_result.get("decision_accepted", compatibility_value),
        "learned_action": router_result.get(
            "learned_action",
            router_result.get("action", compatibility_value),
        ),
        "effective_action": router_result.get("action", compatibility_value),
        "learned_entity_resolution": router_result.get(
            "learned_entity_resolution",
            router_result.get("entity_resolution", compatibility_value),
        ),
        "effective_entity_resolution": router_result.get(
            "entity_resolution",
            compatibility_value,
        ),
        "argument_constraints": router_result.get("argument_constraints", {}),
        "entity_grounding_source": router_result.get(
            "entity_grounding_source",
            compatibility_value,
        ),
        "entity_candidate_count": router_result.get(
            "entity_candidate_count",
            compatibility_value,
        ),
        "exposed_tools": tool_names,
    }


def _policy_system_message(
    matches: tuple[dict[str, Any], ...],
    corpus_revision: str,
) -> dict[str, str]:
    evidence = json.dumps(
        {
            "corpus_revision": corpus_revision,
            "policy_chunks": list(matches),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        "role": "system",
        "content": (
            "You are Harbor, the customer-service assistant for Harborlight Bank. "
            "Answer the customer's policy question using only APPROVED_POLICY_CONTEXT. "
            "Cite every material answer with one or more exact returned chunk IDs in "
            "the form [Policy: chunk_id]. If the context does not answer the question, "
            "say that you could not find the current policy. Do not perform account "
            "actions, invent terms, or mention implementation details. Be warm and "
            f"concise.\n\nAPPROVED_POLICY_CONTEXT={evidence}"
        ),
    }
