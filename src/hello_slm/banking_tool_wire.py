"""Banking-v3 syntax-only tool wire adapters.

The adapter owns only family chat-template syntax, token spans, and JSON parsing.
It intentionally does not infer intents, repair malformed calls, rename tools, or
fill missing arguments.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

ToolFamily = Literal["granite"]

IGNORED_LABEL = -100
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


@dataclass(frozen=True)
class ToolSpan:
    message_index: int
    role: str
    start: int
    end: int
    label: bool
    tool_call_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RenderedTrainingExample:
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor
    span_map: tuple[ToolSpan, ...]
    template_hash: str
    rendered_text: str
    messages_used: tuple[Mapping[str, Any], ...]


def template_hash(tokenizer: Any, family: str) -> str:
    template = getattr(tokenizer, "chat_template", None)
    payload = {
        "family": family,
        "chat_template": str(template) if template is not None else None,
        "tokenizer_class": tokenizer.__class__.__name__,
    }
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _normalize_family(family: str) -> ToolFamily:
    normalized = family.lower().replace("-", "").replace("_", "")
    if normalized not in {"granite", "ibmgranite"}:
        raise ValueError(f"the active tool wire only supports the Granite family, got {family!r}")
    return "granite"


def _copy_message(message: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(message)
    if "tool_calls" in copied and copied["tool_calls"] is not None:
        copied["tool_calls"] = [dict(call) for call in copied["tool_calls"]]
    return copied


def _normalize_manifest_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    if "function" in tool and isinstance(tool["function"], Mapping):
        function = tool["function"]
        return {
            "name": str(function["name"]),
            "description": str(function.get("description", "")),
            "parameters": dict(function.get("parameters", {"type": "object"})),
        }
    return {
        "name": str(tool["name"]),
        "description": str(tool.get("description", "")),
        "parameters": dict(tool.get("parameters", {"type": "object"})),
    }


class ToolWireAdapter:
    """Render and parse canonical banking tool-use records for one model family."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        family: str,
        public_tool_manifest: Sequence[Mapping[str, Any]] = (),
        pad_to_max_length: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.family: ToolFamily = _normalize_family(family)
        self.public_tool_manifest = tuple(
            _normalize_manifest_tool(tool) for tool in public_tool_manifest
        )
        self.pad_to_max_length = pad_to_max_length
        self._tool_names = frozenset(str(tool["name"]) for tool in self.public_tool_manifest)
        self._parameters_by_tool = {
            str(tool["name"]): dict(tool.get("parameters", {"type": "object"}))
            for tool in self.public_tool_manifest
        }

    @property
    def template_hash(self) -> str:
        return template_hash(self.tokenizer, self.family)

    def render_tools(
        self, public_tool_manifest: Sequence[Mapping[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        manifest = (
            self.public_tool_manifest
            if public_tool_manifest is None
            else tuple(_normalize_manifest_tool(tool) for tool in public_tool_manifest)
        )
        return [
            {
                "type": "function",
                "function": {
                    "name": str(tool["name"]),
                    "description": str(tool.get("description", "")),
                    "parameters": dict(tool.get("parameters", {"type": "object"})),
                },
            }
            for tool in manifest
        ]

    def render_training(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_seq_len: int,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> RenderedTrainingExample:
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        selected, offset = self._select_whole_chain_suffix(
            messages,
            max_seq_len=max_seq_len,
            tools=tools,
        )
        self._validate_message_tool_correlation(selected)
        rendered = self._render_messages(
            selected,
            add_generation_prompt=False,
            tools=tools,
        )
        full_ids = self._encode(rendered)
        labels = [IGNORED_LABEL] * len(full_ids)
        spans: list[ToolSpan] = []
        for local_index, message in enumerate(selected):
            if message.get("role") != "assistant" or message.get("loss", True) is False:
                continue
            prefix = self._render_messages(
                selected[:local_index],
                add_generation_prompt=True,
                tools=tools,
            )
            through = self._render_messages(
                selected[: local_index + 1],
                add_generation_prompt=False,
                tools=tools,
            )
            prefix_ids = self._encode(prefix)
            through_ids = self._encode(through)
            if (
                full_ids[: len(prefix_ids)] != prefix_ids
                or full_ids[: len(through_ids)] != through_ids
                or len(through_ids) <= len(prefix_ids)
            ):
                if not rendered.startswith(prefix):
                    raise ValueError(
                        "chat template is not prefix-stable; assistant-only loss cannot be proven"
                    )
                end_char = _assistant_span_end(rendered, len(prefix))
                through_ids = self._encode(rendered[:end_char])
                if (
                    full_ids[: len(prefix_ids)] != prefix_ids
                    or full_ids[: len(through_ids)] != through_ids
                    or len(through_ids) <= len(prefix_ids)
                ):
                    raise ValueError(
                        "chat template is not prefix-stable; assistant-only loss cannot be proven"
                    )
                through = rendered[:end_char]
            if "tool_calls" in message and message.get("tool_calls") and "<tool_call>" in rendered:
                labeled_fragment = self.tokenizer.decode(
                    full_ids[len(prefix_ids) : len(through_ids)],
                    skip_special_tokens=True,
                )
                if "<tool_call>" not in labeled_fragment:
                    raise ValueError("assistant tool-call span did not include tool_call syntax")
            labels[len(prefix_ids) : len(through_ids)] = full_ids[
                len(prefix_ids) : len(through_ids)
            ]
            spans.append(
                ToolSpan(
                    message_index=offset + local_index,
                    role="assistant",
                    start=len(prefix_ids),
                    end=len(through_ids),
                    label=True,
                    tool_call_ids=tuple(
                        str(call["id"]) for call in message.get("tool_calls", ()) or ()
                    ),
                )
            )

        if len(full_ids) > max_seq_len:
            raise ValueError("selected whole tool chain exceeds max_seq_len")
        if not any(label != IGNORED_LABEL for label in labels):
            raise ValueError("record has no assistant target")

        attention = [1] * len(full_ids)
        if self.pad_to_max_length:
            pad_id = int(getattr(self.tokenizer, "pad_token_id", 0))
            pad = max_seq_len - len(full_ids)
            full_ids = full_ids + ([pad_id] * pad)
            labels = labels + ([IGNORED_LABEL] * pad)
            attention = attention + ([0] * pad)

        return RenderedTrainingExample(
            input_ids=torch.tensor(full_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
            span_map=tuple(spans),
            template_hash=self.template_hash,
            rendered_text=rendered,
            messages_used=tuple(selected),
        )

    def render_generation(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> Mapping[str, Any]:
        rendered_tools = self.render_tools(tools) if tools is not None else self.render_tools()
        template_messages = [self._to_template_message(message) for message in messages]
        if hasattr(self.tokenizer, "apply_chat_template"):
            values = self.tokenizer.apply_chat_template(
                template_messages,
                tools=rendered_tools,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if isinstance(values, Tensor):
                return {"input_ids": values, "tools": rendered_tools}
            if isinstance(values, list):
                return {
                    "input_ids": torch.tensor([values], dtype=torch.long),
                    "tools": rendered_tools,
                }
            return dict(values) | {"tools": rendered_tools}
        text = self._render_messages(messages, add_generation_prompt=True)
        return {
            "input_ids": torch.tensor([self._encode(text)], dtype=torch.long),
            "tools": rendered_tools,
        }

    def render_tool_result(self, canonical_tool_message: Mapping[str, Any]) -> dict[str, Any]:
        if canonical_tool_message.get("role") != "tool":
            raise ValueError("tool result message must have role='tool'")
        return self._to_template_message(canonical_tool_message)

    def parse_assistant(self, generated_tokens: str | Sequence[int] | Tensor) -> dict[str, Any]:
        text = self._decode(generated_tokens).strip()
        message = self._parse_family_assistant(text)
        self._validate_assistant_message(message)
        return message

    def _select_whole_chain_suffix(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_seq_len: int,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        copied = [_copy_message(message) for message in messages]
        full_length = len(
            self._encode(
                self._render_messages(
                    copied,
                    add_generation_prompt=False,
                    tools=tools,
                )
            )
        )
        if full_length <= max_seq_len:
            return copied, 0
        system_prefix = []
        first_non_system = 0
        for index, message in enumerate(copied):
            if message.get("role") != "system":
                first_non_system = index
                break
            system_prefix.append(message)
        chain_starts = [
            index
            for index, message in enumerate(copied[first_non_system:], start=first_non_system)
            if message.get("role") == "user"
        ]
        for start in chain_starts:
            suffix = [*system_prefix, *copied[start:]]
            if not _has_assistant_loss(suffix):
                continue
            suffix_length = len(
                self._encode(
                    self._render_messages(
                        suffix,
                        add_generation_prompt=False,
                        tools=tools,
                    )
                )
            )
            if suffix_length <= max_seq_len:
                return suffix, start - len(system_prefix)
        raise ValueError("no complete user-to-final-assistant tool chain fits max_seq_len")

    def _render_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        template_messages = [self._to_template_message(message) for message in messages]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return str(
                self.tokenizer.apply_chat_template(
                    template_messages,
                    tools=self.render_tools(tools),
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            )
        rendered = "\n".join(_fallback_render_message(message) for message in template_messages)
        return rendered + ("\nassistant:" if add_generation_prompt else "")

    def _to_template_message(self, message: Mapping[str, Any]) -> dict[str, Any]:
        role = str(message["role"])
        rendered: dict[str, Any] = {"role": role}
        if role == "assistant" and message.get("tool_calls"):
            rendered["content"] = message.get("content")
            rendered["tool_calls"] = [
                _normalize_tool_call(call) for call in message.get("tool_calls", ())
            ]
        elif role == "tool":
            rendered["tool_call_id"] = str(message["tool_call_id"])
            rendered["name"] = str(message["name"])
            content = message.get("content")
            rendered["content"] = content if isinstance(content, str) else _stable_json(content)
        else:
            content = message.get("content")
            rendered["content"] = "" if content is None else str(content)
        return rendered

    def _encode(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False)
        values = encoded["input_ids"]
        if isinstance(values, Tensor):
            values = values.tolist()
        if values and isinstance(values[0], list):
            values = values[0]
        return [int(value) for value in values]

    def _decode(self, generated_tokens: str | Sequence[int] | Tensor) -> str:
        if isinstance(generated_tokens, str):
            return generated_tokens
        if isinstance(generated_tokens, Tensor):
            generated_tokens = generated_tokens.detach().cpu().tolist()
        if hasattr(self.tokenizer, "decode"):
            return str(self.tokenizer.decode(generated_tokens, skip_special_tokens=True))
        return "".join(chr(int(token)) for token in generated_tokens)

    def _parse_family_assistant(self, text: str) -> dict[str, Any]:
        if not text:
            raise ValueError("assistant output is empty")
        matches = list(TOOL_CALL_PATTERN.finditer(text))
        if matches:
            if text[: matches[0].start()].strip() or text[matches[-1].end() :].strip():
                raise ValueError("tool-call output must not contain prose outside tool_call blocks")
            calls = []
            for index, match in enumerate(matches):
                payload = self._parse_payload(match.group(1))
                if not isinstance(payload, Mapping):
                    raise ValueError("tool_call block must contain a JSON object")
                calls.append(
                    _payload_to_tool_call(
                        payload,
                        index=index,
                        raw_output=text,
                    )
                )
            return {"role": "assistant", "content": None, "tool_calls": calls}
        if "<tool_call" in text or "</tool_call>" in text:
            raise ValueError("assistant output contains malformed tool_call block")
        if text.startswith("{"):
            payload = self._parse_payload(text)
            return self._payload_to_assistant_message(payload, raw_output=text)
        return {"role": "assistant", "content": text}

    def _parse_payload(self, text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("assistant output is not parseable JSON") from exc

    def _payload_to_assistant_message(
        self,
        payload: Any,
        *,
        raw_output: str,
    ) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise ValueError("assistant payload must be a JSON object")
        if "tool_calls" in payload:
            return {
                "role": "assistant",
                "content": payload.get("content"),
                "tool_calls": [
                    _normalize_tool_call(call) for call in payload.get("tool_calls", ())
                ],
            }
        if "name" in payload and "arguments" in payload:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _payload_to_tool_call(
                        payload,
                        index=int(payload.get("index", 0)),
                        raw_output=raw_output,
                    )
                ],
            }
        if "content" in payload:
            return {"role": "assistant", "content": str(payload["content"])}
        raise ValueError("assistant payload contains neither tool_calls nor content")

    def _validate_assistant_message(self, message: Mapping[str, Any]) -> None:
        seen_ids: set[str] = set()
        for expected_index, call in enumerate(message.get("tool_calls", ()) or ()):
            call_id = str(call["id"])
            if call_id in seen_ids:
                raise ValueError(f"duplicate tool call id: {call_id}")
            seen_ids.add(call_id)
            if int(call.get("index", expected_index)) != expected_index:
                raise ValueError("tool call indexes must be ordered from zero")
            function = call["function"]
            name = str(function["name"])
            if name not in self._tool_names:
                raise ValueError(f"unknown tool name: {name}")
            arguments = function["arguments"]
            if not isinstance(arguments, Mapping):
                raise ValueError(f"tool arguments for {name} must be a JSON object")
            _validate_json_object(name, arguments, self._parameters_by_tool[name])

    def _validate_message_tool_correlation(self, messages: Sequence[Mapping[str, Any]]) -> None:
        pending: dict[str, str] = {}
        for message in messages:
            role = message.get("role")
            if role == "assistant" and message.get("tool_calls"):
                normalized_calls = [
                    _normalize_tool_call(call) for call in message.get("tool_calls", ())
                ]
                normalized = {
                    "role": "assistant",
                    "tool_calls": normalized_calls,
                }
                self._validate_assistant_message(normalized)
                for call in normalized_calls:
                    pending[str(call["id"])] = str(call["function"]["name"])
            elif role == "tool":
                call_id = str(message.get("tool_call_id", ""))
                name = str(message.get("name", ""))
                if call_id not in pending:
                    raise ValueError(f"tool result without preceding call: {call_id}")
                if pending[call_id] != name:
                    raise ValueError(
                        f"tool result name mismatch for {call_id}: "
                        f"expected {pending[call_id]}, got {name}"
                    )
                del pending[call_id]
            elif role == "assistant" and pending:
                raise ValueError(f"assistant final before tool results: {sorted(pending)}")
        if pending:
            raise ValueError(f"missing tool results for calls: {sorted(pending)}")


def _has_assistant_loss(messages: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        message.get("role") == "assistant" and message.get("loss", True) is not False
        for message in messages
    )


def _normalize_tool_call(call: Mapping[str, Any]) -> dict[str, Any]:
    function = call.get("function")
    if not isinstance(function, Mapping):
        raise ValueError("tool call missing function object")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, Mapping):
        raise ValueError("tool call arguments must be a JSON object")
    return {
        "id": str(call["id"]),
        "index": int(call.get("index", 0)),
        "type": str(call.get("type", "function")),
        "function": {
            "name": str(function["name"]),
            "arguments": dict(arguments),
        },
    }


def _payload_to_tool_call(
    payload: Mapping[str, Any],
    *,
    index: int,
    raw_output: str,
) -> dict[str, Any]:
    name = payload.get("name")
    arguments = payload.get("arguments")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool call requires a function name")
    if not isinstance(arguments, Mapping):
        raise ValueError("tool call arguments must be a JSON object")
    call_id = payload.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        call_id = _stable_generated_call_id(raw_output, index=index, name=name)
    return {
        "id": call_id,
        "index": int(payload.get("index", index)),
        "type": "function",
        "function": {"name": name, "arguments": dict(arguments)},
    }


def _stable_generated_call_id(raw_output: str, *, index: int, name: str) -> str:
    digest = hashlib.sha256(f"{raw_output}\n{index}\n{name}".encode()).hexdigest()[:16]
    return f"call_generated_{index}_{digest}"


def _assistant_span_end(rendered: str, start: int) -> int:
    markers = ["<|im_end|>", "<|end_of_role|>", "\nuser:", "\ntool "]
    positions = [rendered.find(marker, start) for marker in markers]
    positions = [position for position in positions if position >= start]
    if not positions:
        raise ValueError("could not locate assistant span end in chat template")
    marker_start = min(positions)
    marker = next(marker for marker in markers if rendered.startswith(marker, marker_start))
    return marker_start + len(marker)


def _validate_json_object(
    tool_name: str,
    arguments: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> None:
    if schema.get("type", "object") != "object":
        raise ValueError(f"tool schema for {tool_name} must be an object")
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise ValueError(f"tool schema for {tool_name} has invalid properties")
    required = schema.get("required", ())
    if not isinstance(required, Sequence) or isinstance(required, str):
        raise ValueError(f"tool schema for {tool_name} has invalid required list")
    missing = [name for name in required if name not in arguments]
    if missing:
        raise ValueError(f"missing required arguments for {tool_name}: {missing}")
    extra = set(str(key) for key in arguments) - set(str(key) for key in properties)
    if extra and schema.get("additionalProperties", True) is False:
        raise ValueError(f"unsupported arguments for {tool_name}: {sorted(extra)}")
    for name, value in arguments.items():
        if name not in properties:
            continue
        _validate_json_value(tool_name, str(name), value, properties[name])


def _validate_json_value(
    tool_name: str,
    argument_name: str,
    value: Any,
    schema: Any,
) -> None:
    if not isinstance(schema, Mapping):
        return
    expected_types = schema.get("type")
    if isinstance(expected_types, str):
        expected = {expected_types}
    elif isinstance(expected_types, Sequence):
        expected = {str(item) for item in expected_types}
    else:
        expected = set()
    if value is None:
        actual = "null"
    elif isinstance(value, bool):
        actual = "boolean"
    elif isinstance(value, int) and not isinstance(value, bool):
        actual = "integer"
    elif isinstance(value, float):
        actual = "number"
    elif isinstance(value, str):
        actual = "string"
    elif isinstance(value, list):
        actual = "array"
    elif isinstance(value, Mapping):
        actual = "object"
    else:
        actual = type(value).__name__
    number_matches_integer = actual == "integer" and "number" in expected
    if expected and actual not in expected and not number_matches_integer:
        raise ValueError(
            f"invalid type for {tool_name}.{argument_name}: expected "
            f"{sorted(expected)}, got {actual}"
        )
    if actual in {"integer", "number"}:
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            raise ValueError(f"value below minimum for {tool_name}.{argument_name}")
        if maximum is not None and value > maximum:
            raise ValueError(f"value above maximum for {tool_name}.{argument_name}")
    enum = schema.get("enum")
    if enum is not None and value not in enum:
        raise ValueError(f"value not in enum for {tool_name}.{argument_name}")


def _fallback_render_message(message: Mapping[str, Any]) -> str:
    role = str(message["role"])
    if role == "assistant" and message.get("tool_calls"):
        return f"assistant:{_stable_json({'tool_calls': message['tool_calls']})}"
    if role == "tool":
        return f"tool {message['name']}[{message['tool_call_id']}]:{message.get('content', '')}"
    return f"{role}:{message.get('content', '')}"
