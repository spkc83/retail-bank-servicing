from __future__ import annotations

import json

import pytest

from hello_slm.banking_tool_wire import IGNORED_LABEL, ToolWireAdapter


class TemplateTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    chat_template = "unit-test-tool-template"

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        tokenize=False,
        add_generation_prompt=False,
        return_tensors=None,
    ):
        del tools, return_tensors
        parts = []
        for message in messages:
            role = message["role"]
            if role == "assistant" and message.get("tool_calls"):
                parts.append(
                    "assistant:" + json.dumps({"tool_calls": message["tool_calls"]}, sort_keys=True)
                )
            elif role == "tool":
                parts.append(
                    f"tool {message['name']}[{message['tool_call_id']}]:{message['content']}"
                )
            else:
                parts.append(f"{role}:{message.get('content', '')}")
        if add_generation_prompt:
            parts.append("assistant:")
        rendered = "\n".join(parts)
        if tokenize:
            return {"input_ids": [self._ids(rendered)]}
        return rendered

    def __call__(self, text, *, add_special_tokens=False, **kwargs):
        del add_special_tokens, kwargs
        return {"input_ids": self._ids(text)}

    def decode(self, tokens, *, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(int(token)) for token in tokens if int(token) > 2)

    @staticmethod
    def _ids(text):
        return [ord(char) for char in text]


class ToolAwareTokenizer(TemplateTokenizer):
    def __init__(self) -> None:
        self.tool_sets = []

    def apply_chat_template(self, messages, *, tools=None, **kwargs):
        self.tool_sets.append(
            tuple(tool["function"]["name"] for tool in tools) if tools is not None else None
        )
        return super().apply_chat_template(messages, tools=tools, **kwargs)


MANIFEST = [
    {
        "name": "list_cards",
        "description": "List cards.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "freeze_card",
        "description": "Freeze card.",
        "parameters": {
            "type": "object",
            "properties": {"last4": {"type": ["string", "null"]}},
            "additionalProperties": False,
        },
    },
    {
        "name": "list_transactions",
        "description": "List transactions.",
        "parameters": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 20}},
            "additionalProperties": False,
        },
    },
]


def tool_chain_messages():
    return [
        {"role": "user", "content": "Freeze my debit card ending in 4821.", "loss": False},
        {
            "role": "assistant",
            "content": None,
            "loss": True,
            "tool_calls": [
                {
                    "id": "call_freeze_001_0",
                    "index": 0,
                    "type": "function",
                    "function": {"name": "list_cards", "arguments": {}},
                },
                {
                    "id": "call_freeze_001_1",
                    "index": 1,
                    "type": "function",
                    "function": {"name": "freeze_card", "arguments": {"last4": "4821"}},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_freeze_001_0",
            "name": "list_cards",
            "content": {"ok": True, "result": [{"last4": "4821", "status": "active"}]},
            "loss": False,
        },
        {
            "role": "tool",
            "tool_call_id": "call_freeze_001_1",
            "name": "freeze_card",
            "content": {"ok": True, "result": {"card": {"last4": "4821", "status": "frozen"}}},
            "loss": False,
        },
        {
            "role": "assistant",
            "content": "Your debit card ending in 4821 is now frozen.",
            "loss": True,
        },
    ]


def test_assistant_only_labels_include_tool_calls_and_final_but_not_tool_results() -> None:
    adapter = ToolWireAdapter(TemplateTokenizer(), family="granite", public_tool_manifest=MANIFEST)

    rendered = adapter.render_training(tool_chain_messages(), max_seq_len=2048)
    labeled_text = "".join(
        chr(int(token))
        for token, label in zip(rendered.input_ids.tolist(), rendered.labels.tolist(), strict=True)
        if label != IGNORED_LABEL
    )

    assert '"name": "list_cards"' in labeled_text
    assert '"name": "freeze_card"' in labeled_text
    assert "Your debit card ending in 4821 is now frozen." in labeled_text
    assert '"status":"frozen"' not in labeled_text
    assert rendered.span_map[0].tool_call_ids == ("call_freeze_001_0", "call_freeze_001_1")
    assert all(span.role == "assistant" and span.label for span in rendered.span_map)


def test_granite_repeated_tool_calls_keep_stable_distinct_call_ids() -> None:
    messages = tool_chain_messages()
    messages[1]["tool_calls"] = [
        {
            "id": "call_repeat_0",
            "index": 0,
            "type": "function",
            "function": {"name": "freeze_card", "arguments": {"last4": "1111"}},
        },
        {
            "id": "call_repeat_1",
            "index": 1,
            "type": "function",
            "function": {"name": "freeze_card", "arguments": {"last4": "2222"}},
        },
    ]
    messages[2]["tool_call_id"] = "call_repeat_0"
    messages[2]["name"] = "freeze_card"
    messages[3]["tool_call_id"] = "call_repeat_1"
    messages[3]["name"] = "freeze_card"
    adapter = ToolWireAdapter(TemplateTokenizer(), family="granite", public_tool_manifest=MANIFEST)

    rendered = adapter.render_training(messages, max_seq_len=2048)

    assert rendered.span_map[0].tool_call_ids == ("call_repeat_0", "call_repeat_1")
    assert "call_repeat_0" in rendered.rendered_text
    assert "call_repeat_1" in rendered.rendered_text


def test_whole_chain_truncation_drops_older_chain_without_splitting_latest() -> None:
    system = [{"role": "system", "content": "Keep banking scope.", "loss": False}]
    old_chain = [
        {"role": "user", "content": "old " * 80, "loss": False},
        {"role": "assistant", "content": "old answer", "loss": True},
    ]
    latest = tool_chain_messages()
    adapter = ToolWireAdapter(TemplateTokenizer(), family="granite", public_tool_manifest=MANIFEST)
    latest_len = len(adapter.render_training(system + latest, max_seq_len=2048).rendered_text)

    rendered = adapter.render_training(system + old_chain + latest, max_seq_len=latest_len + 8)

    assert rendered.messages_used[0]["role"] == "system"
    assert rendered.messages_used[1]["content"] == latest[0]["content"]
    assert "old answer" not in rendered.rendered_text
    assert "Keep banking scope." in rendered.rendered_text
    assert "Freeze my debit card" in rendered.rendered_text


def test_whole_chain_truncation_rejects_split_latest_chain() -> None:
    adapter = ToolWireAdapter(TemplateTokenizer(), family="granite", public_tool_manifest=MANIFEST)

    with pytest.raises(ValueError, match="no complete user-to-final-assistant tool chain"):
        adapter.render_training(tool_chain_messages(), max_seq_len=40)


def test_parse_tool_call_blocks_validates_without_repair_or_argument_inference() -> None:
    adapter = ToolWireAdapter(TemplateTokenizer(), family="granite", public_tool_manifest=MANIFEST)

    parsed = adapter.parse_assistant(
        '<tool_call>{"id":"call_0","name":"list_cards","arguments":{}}</tool_call>'
        '<tool_call>{"id":"call_1","name":"freeze_card",'
        '"arguments":{"last4":"4821"}}</tool_call>'
    )

    assert parsed["tool_calls"][0]["id"] == "call_0"
    assert parsed["tool_calls"][1]["id"] == "call_1"
    assert [call["index"] for call in parsed["tool_calls"]] == [0, 1]
    with pytest.raises(ValueError, match="unknown tool name"):
        adapter.parse_assistant(
            '<tool_call>{"id":"call_bad","name":"close_account","arguments":{}}</tool_call>'
        )
    with pytest.raises(ValueError, match="unsupported arguments"):
        adapter.parse_assistant(
            '<tool_call>{"id":"call_2","name":"freeze_card",'
            '"arguments":{"account_number":"123"}}</tool_call>'
        )
    generated = adapter.parse_assistant(
        '<tool_call>{"name":"freeze_card","arguments":{"last4":"4821"}}</tool_call>'
    )
    assert generated["tool_calls"][0]["id"].startswith("call_generated_0_")
    assert generated == adapter.parse_assistant(
        '<tool_call>{"name":"freeze_card","arguments":{"last4":"4821"}}</tool_call>'
    )
    with pytest.raises(ValueError, match="must not contain prose"):
        adapter.parse_assistant(
            'Sure <tool_call>{"id":"call_3","name":"list_cards","arguments":{}}</tool_call>'
        )


def test_argument_type_and_range_validation_fail_closed() -> None:
    adapter = ToolWireAdapter(TemplateTokenizer(), family="granite", public_tool_manifest=MANIFEST)

    with pytest.raises(ValueError, match="invalid type"):
        adapter.parse_assistant(
            '<tool_call>{"id":"call_limit","name":"list_transactions",'
            '"arguments":{"limit":"5"}}</tool_call>'
        )
    with pytest.raises(ValueError, match="above maximum"):
        adapter.parse_assistant(
            '<tool_call>{"id":"call_limit","name":"list_transactions",'
            '"arguments":{"limit":21}}</tool_call>'
        )


def test_tool_result_correlation_is_validated_before_tokenization() -> None:
    messages = tool_chain_messages()
    messages[2]["tool_call_id"] = "call_missing"
    adapter = ToolWireAdapter(TemplateTokenizer(), family="granite", public_tool_manifest=MANIFEST)

    with pytest.raises(ValueError, match="tool result without preceding call"):
        adapter.render_training(messages, max_seq_len=2048)


def test_granite_renders_the_public_tool_contract() -> None:
    adapter = ToolWireAdapter(
        TemplateTokenizer(),
        family="granite",
        public_tool_manifest=MANIFEST,
    )

    rendered_tools = adapter.render_tools()

    assert rendered_tools[0]["type"] == "function"
    assert rendered_tools[0]["function"]["name"] == "list_cards"
    assert adapter.render_generation(tool_chain_messages()[:1])["tools"] == rendered_tools


def test_training_render_can_match_v6_single_tool_and_no_tool_contracts() -> None:
    tokenizer = ToolAwareTokenizer()
    adapter = ToolWireAdapter(tokenizer, family="granite", public_tool_manifest=MANIFEST)
    single_call = [
        {"role": "user", "content": "Freeze card 4821.", "loss": False},
        {
            "role": "assistant",
            "content": None,
            "loss": True,
            "tool_calls": [
                {
                    "id": "call_freeze_0",
                    "index": 0,
                    "type": "function",
                    "function": {"name": "freeze_card", "arguments": {"last4": "4821"}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_freeze_0",
            "name": "freeze_card",
            "content": {"ok": True, "result": {"card": {"last4": "4821"}}},
            "loss": False,
        },
        {"role": "assistant", "content": "Your card ending in 4821 is frozen.", "loss": True},
    ]

    adapter.render_training(single_call, max_seq_len=2048, tools=[MANIFEST[1]])

    assert tokenizer.tool_sets
    assert set(tokenizer.tool_sets) == {("freeze_card",)}

    tokenizer.tool_sets.clear()
    adapter.render_training(
        [
            {"role": "user", "content": "Good morning.", "loss": False},
            {"role": "assistant", "content": "Good morning! How can I help?", "loss": True},
        ],
        max_seq_len=2048,
        tools=[],
    )
    assert tokenizer.tool_sets
    assert set(tokenizer.tool_sets) == {()}


def test_non_granite_tool_wire_is_rejected() -> None:
    with pytest.raises(ValueError, match="only supports the Granite family"):
        ToolWireAdapter(
            TemplateTokenizer(),
            family="other",
            public_tool_manifest=MANIFEST,
        )
