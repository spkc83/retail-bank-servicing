from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from mock_bank import SessionBankRegistry
from response_policy import (
    build_final_repair_messages,
    render_read_tool_results,
    validate_grounded_answer,
)

INPUT_TOKEN_BUDGET = 8192
MAX_NEW_TOKENS = 512
MAX_TOOL_CALLS = 8

AGENT_SYSTEM_PROMPT = (
    "You are the conversational customer-service agent for a fictional retail-bank "
    "demonstration. The customer is already authenticated. Use the supplied tools for "
    "customer-specific banking records or actions, use tool results for final answers, "
    "call dependent tools one at a time so each later call can use the earlier result, "
    "and never ask for account numbers, customer IDs, passwords, PINs, or private IDs."
)

MODEL_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_accounts",
            "description": "List the signed-in synthetic customer's accounts and balances.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_cards",
            "description": "List the signed-in synthetic customer's cards and statuses.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_service_cases",
            "description": "List recent synthetic service cases.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_transactions",
            "description": "List recent synthetic account transactions.",
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
            "description": "List the signed-in synthetic customer's transfers and statuses.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "freeze_card",
            "description": "Freeze a synthetic card, optionally selected by last four digits.",
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
            "description": "Request replacement of a synthetic card.",
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
            "description": "Dispute a synthetic transaction by description.",
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
            "description": "Cancel a synthetic pending transfer by recipient.",
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
    ) -> str:
        ...

    def count_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> int:
        ...


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
    ) -> list[dict[str, Any]]:
        ...

    def parse_assistant(
        self,
        output: str,
        *,
        turn_key: str | None = None,
    ) -> tuple[ToolCall, ...]:
        ...

    def render_assistant_tool_calls(
        self,
        calls: tuple[ToolCall, ...],
    ) -> dict[str, Any]:
        ...

    def render_tool_result(
        self,
        call: ToolCall,
        content: dict[str, Any],
    ) -> dict[str, Any]:
        ...


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
    ) -> AgentTurnResult:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        canonical = canonical_conversation(conversation)
        current = [*canonical, {"role": "user", "content": message.strip()}]
        system = _system_message(router_result)
        first_context = select_token_budgeted_context(
            system,
            current,
            tools=self.tool_adapter.render_tools(MODEL_TOOLS),
            token_counter=self.model.count_tokens,
            input_budget=self.input_budget,
        )
        model_passes: list[ModelPassTrace] = []
        first_output, first_trace = self._generate_pass(
            "base",
            first_context,
            self.tool_adapter.render_tools(MODEL_TOOLS),
        )
        model_passes.append(first_trace)
        if not first_output:
            raise AgentProtocolError("model returned an empty first response")
        calls = self.tool_adapter.parse_assistant(
            first_output,
            turn_key=first_trace.prompt_sha256,
        )
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

        _validate_tool_calls(calls)
        with_tools = [*current]
        all_calls: list[ToolCall] = []
        results: list[dict[str, Any]] = []
        pending_calls = calls
        post_tool_passes = 0
        serving_tools = self.tool_adapter.render_tools(MODEL_TOOLS)
        try:
            while True:
                _validate_tool_calls(pending_calls)
                if len(all_calls) + len(pending_calls) > MAX_TOOL_CALLS:
                    raise AgentProtocolError(
                        f"model attempted more than {MAX_TOOL_CALLS} total tool calls"
                    )
                call_message = self.tool_adapter.render_assistant_tool_calls(
                    pending_calls
                )
                with_tools.append(call_message)
                all_calls.extend(pending_calls)
                for call in pending_calls:
                    result = self._execute_tool(username, session_hash, call)
                    results.append(result)
                    with_tools.append(
                        self.tool_adapter.render_tool_result(call, result)
                    )

                post_tool_passes += 1
                followup_context = select_token_budgeted_context(
                    system,
                    with_tools,
                    tools=serving_tools,
                    token_counter=self.model.count_tokens,
                    input_budget=self.input_budget,
                )
                pass_label = (
                    "grounded_final"
                    if post_tool_passes == 1
                    else f"tool_followup_{post_tool_passes}"
                )
                followup_output, followup_trace = self._generate_pass(
                    pass_label,
                    followup_context,
                    serving_tools,
                )
                model_passes.append(followup_trace)
                if not followup_output:
                    raise AgentProtocolError("model returned an empty follow-up response")
                next_calls = self.tool_adapter.parse_assistant(
                    followup_output,
                    turn_key=followup_trace.prompt_sha256,
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
        completed = [*current, {"role": "assistant", "content": first_output}]
        return AgentTurnResult(
            response=first_output,
            conversation=completed,
            tool_calls=(),
            tool_results=(),
            snapshot=self.bank.snapshot(username, session_hash),
            response_path=response_path,
            model_passes=tuple(model_passes),
        )

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
            prompt_sha256=hashlib.sha256(
                prompt_payload.encode("utf-8")
            ).hexdigest(),
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


def _validate_tool_calls(calls: tuple[ToolCall, ...]) -> None:
    schemas = {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in MODEL_TOOLS
    }
    seen_ids: set[str] = set()
    for expected_index, call in enumerate(calls):
        if call.index != expected_index:
            raise AgentProtocolError("model tool-call indexes must be ordered from zero")
        if call.id in seen_ids:
            raise AgentProtocolError("model tool-call IDs must be unique")
        seen_ids.add(call.id)
        schema = schemas.get(call.name)
        if schema is None:
            raise AgentProtocolError(f"model selected unsupported tool: {call.name}")
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
    for name, value in arguments.items():
        subschema = allowed.get(name)
        if not isinstance(subschema, dict):
            continue
        expected = subschema.get("type")
        expected_types = expected if isinstance(expected, list) else [expected]
        if not _value_matches_json_types(value, expected_types):
            raise AgentProtocolError(
                f"model supplied invalid type for {tool_name}.{name}"
            )
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and isinstance(subschema.get("minimum"), int)
            and value < subschema["minimum"]
        ):
            raise AgentProtocolError(
                f"model supplied value below minimum for {tool_name}.{name}"
            )
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and isinstance(subschema.get("maximum"), int)
            and value > subschema["maximum"]
        ):
            raise AgentProtocolError(
                f"model supplied value above maximum for {tool_name}.{name}"
            )


def _value_matches_json_types(value: Any, expected_types: list[Any]) -> bool:
    for expected in expected_types:
        if expected == "null" and value is None:
            return True
        if expected == "string" and isinstance(value, str):
            return True
        if (
            expected == "integer"
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            return True
        if (
            expected == "number"
            and isinstance(value, int | float)
            and not isinstance(value, bool)
        ):
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
        safe_message = "The request did not match exactly one synthetic banking record."
    elif message == "no matching synthetic banking record":
        code = "record_not_found"
        safe_message = "No matching synthetic banking record was found."
    else:
        code = "backend_error"
        safe_message = "The synthetic banking backend could not complete the request."
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

    retained = [groups[-1]]
    selected = [system_message, *groups[-1]]
    if token_counter(selected, tools) > input_budget:
        raise AgentProtocolError("latest conversation turn exceeds the model input budget")
    for group in reversed(groups[:-1]):
        proposal_groups = [group, *retained]
        proposal = [
            system_message,
            *(message for item in proposal_groups for message in item),
        ]
        if token_counter(proposal, tools) <= input_budget:
            retained = proposal_groups
    return [
        system_message,
        *(message for item in retained for message in item),
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
        elif (
            role == "tool"
            and isinstance(item.get("name"), str)
            and isinstance(content, str)
        ):
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
