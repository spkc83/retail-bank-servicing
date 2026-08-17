from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mock_bank import SessionBankRegistry
from model_service import (
    INPUT_TOKEN_BUDGET,
    MODEL_TOOLS,
    AgentExecutionError,
    AgentProtocolError,
    ConversationalBankingAgent,
    parse_tool_calls,
    router_diagnostic_fields,
    select_token_budgeted_context,
)

ROOT = Path(__file__).parents[1]


def bank() -> SessionBankRegistry:
    return SessionBankRegistry.from_json(ROOT / "synthetic_bank.json")


class RecordingModel:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_new_tokens: int,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "max_new_tokens": max_new_tokens,
            }
        )
        return self.outputs.pop(0)

    def count_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> int:
        return len(json.dumps({"messages": messages, "tools": tools}))


def router_guidance() -> dict[str, Any]:
    return {
        "route": "in_domain",
        "banking_probability": 0.99,
        "ood_probability": 0.01,
        "capability": "transfers",
        "capability_confidence": 0.81,
        "capability_candidates": [
            {"capability": "transfers", "probability": 0.81},
            {"capability": "accounts", "probability": 0.12},
            {"capability": "cards", "probability": 0.03},
        ],
        "relation_probabilities": {
            "context_dependent": 0.1,
            "agent_repair": 0.1,
            "topic_shift": 0.1,
            "clarification_answer": 0.1,
        },
    }


def v4_router_guidance(
    *,
    action: str,
    fine_intent: str = "replace_card",
    entity_resolution: str = "resolved",
) -> dict[str, Any]:
    return {
        **router_guidance(),
        "action": action,
        "fine_intent": fine_intent,
        "entity_resolution": entity_resolution,
    }


def test_public_tool_schemas_use_customer_facing_arguments() -> None:
    schemas = {item["function"]["name"]: item["function"]["parameters"] for item in MODEL_TOOLS}

    assert set(schemas) == {
        "list_accounts",
        "list_cards",
        "list_service_cases",
        "list_transactions",
        "list_transfers",
        "freeze_card",
        "replace_card",
        "dispute_transaction",
        "cancel_transfer",
    }
    assert set(schemas["cancel_transfer"]["properties"]) == {"recipient"}
    assert set(schemas["dispute_transaction"]["properties"]) == {"description"}
    assert "transfer_id" not in json.dumps(MODEL_TOOLS)
    assert "transaction_id" not in json.dumps(MODEL_TOOLS)


def test_tagged_json_parser_accepts_multiple_ordered_calls_and_ignores_prose() -> None:
    calls = parse_tool_calls(
        """I will check both.
<tool_call>
{"name": "list_transfers", "arguments": {}}
</tool_call>
<tool_call>
{"name": "list_transactions", "arguments": {"limit": 3}}
</tool_call>"""
    )

    assert [call.name for call in calls] == ["list_transfers", "list_transactions"]
    assert calls[0].id.startswith("call_")
    assert calls[0].id.endswith("_0_list_transfers")
    assert calls[1].id.endswith("_1_list_transactions")
    assert calls[0].id != calls[1].id
    assert [call.index for call in calls] == [0, 1]
    assert calls[1].arguments == {"limit": 3}


def test_malformed_first_tool_response_preserves_raw_model_trace() -> None:
    raw_output = '<tool_call>{"name":"list_accounts","arguments":{}}'
    agent = ConversationalBankingAgent(bank=bank(), model=RecordingModel([raw_output]))

    with pytest.raises(
        AgentExecutionError,
        match="malformed tool-call block",
    ) as failure:
        agent.run_turn(
            username="alex.demo",
            session_hash="session",
            message="Show my accounts.",
            conversation=[],
            router_result=v4_router_guidance(
                action="execute_tool",
                fine_intent="view_accounts",
            ),
        )

    assert failure.value.conversation == [{"role": "user", "content": "Show my accounts."}]
    assert failure.value.tool_calls == ()
    assert failure.value.tool_results == ()
    assert len(failure.value.model_passes) == 1
    assert failure.value.model_passes[0].label == "base"
    assert failure.value.model_passes[0].raw_output == raw_output


def test_invalid_first_tool_response_preserves_raw_model_trace() -> None:
    raw_output = '<tool_call>{"name":"list_cards","arguments":{}}</tool_call>'
    agent = ConversationalBankingAgent(bank=bank(), model=RecordingModel([raw_output]))

    with pytest.raises(AgentExecutionError, match="unexposed tool") as failure:
        agent.run_turn(
            username="alex.demo",
            session_hash="session",
            message="Show my accounts.",
            conversation=[],
            router_result=v4_router_guidance(
                action="execute_tool",
                fine_intent="view_accounts",
            ),
        )

    assert failure.value.conversation[-1] == {
        "role": "user",
        "content": "Show my accounts.",
    }
    assert failure.value.model_passes[0].raw_output == raw_output


@pytest.mark.parametrize(
    "output",
    [
        "<tool_call>not-json</tool_call>",
        '<tool_call>{"name": "", "arguments": {}}</tool_call>',
        '<tool_call>{"name": "list_accounts", "arguments": []}</tool_call>',
        "<tool_call>",
    ],
)
def test_tagged_json_parser_rejects_malformed_protocol(output: str) -> None:
    with pytest.raises(AgentProtocolError):
        parse_tool_calls(output)


def test_plain_first_pass_text_is_a_model_authored_conversational_answer() -> None:
    model = RecordingModel(["Hey! What can I help you with today?"])
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    result = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="yo, sup?",
        conversation=[],
        router_result=router_guidance(),
    )

    assert result.response == "Hey! What can I help you with today?"
    assert result.tool_calls == ()
    assert result.conversation == [
        {"role": "user", "content": "yo, sup?"},
        {"role": "assistant", "content": result.response},
    ]
    assert result.response_path == "direct_answer"
    assert [item.label for item in result.model_passes] == ["base"]
    assert result.model_passes[0].raw_output == result.response
    assert len(model.calls) == 1
    assert model.calls[0]["tools"] == MODEL_TOOLS
    assert model.calls[0]["messages"][0] == {
        "role": "system",
        "content": (
            "You are Harbor, the conversational customer-service assistant for "
            "Harborlight Bank. The customer is already authenticated. "
            "Use the supplied tools for customer-specific banking records or actions, "
            "use tool results for final answers, call dependent tools one at a time "
            "so each later call can use the earlier result, and never ask for account "
            "numbers, customer IDs, passwords, PINs, or private IDs. Respond warmly "
            "and concisely, acknowledge distress only when the customer explicitly "
            "expresses it, never infer distress from a neutral greeting or request, "
            "name banking products clearly, and never mention prototypes, demos, "
            "synthetic data, models, routers, tools, GPUs, CPUs, or implementation "
            "details."
        ),
    }


def test_v4_execute_exposes_only_predicted_intent_tool_across_followup() -> None:
    model = RecordingModel(
        [
            '<tool_call>{"name": "list_transactions", "arguments": {"limit": 2}}</tool_call>',
            "Here are your two most recent transactions.",
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    result = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="Show my two latest transactions.",
        conversation=[],
        router_result=v4_router_guidance(
            action="execute_tool",
            fine_intent="view_transactions",
            entity_resolution="not_required",
        ),
    )

    assert [call.name for call in result.tool_calls] == ["list_transactions"]
    assert len(model.calls) == 2
    assert [tool["function"]["name"] for tool in model.calls[0]["tools"]] == ["list_transactions"]
    assert model.calls[1]["tools"] is None
    system_prompt = model.calls[0]["messages"][0]["content"]
    assert "Use only list_transactions for this turn" in system_prompt
    assert "guidance supplies no tool arguments" in system_prompt


def _read_view_turn(final_output: str) -> Any:
    model = RecordingModel(
        [
            '<tool_call>{"name": "list_transactions", "arguments": {"limit": 2}}</tool_call>',
            final_output,
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)
    result = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="Show my two latest transactions.",
        conversation=[],
        router_result=v4_router_guidance(
            action="execute_tool",
            fine_intent="view_transactions",
            entity_resolution="not_required",
        ),
    )
    return model, result


def test_read_view_keeps_the_natural_lead_in_above_the_rendered_table() -> None:
    _model, result = _read_view_turn("Here are your two most recent transactions.")

    assert result.response.startswith("Here are your two most recent transactions.")
    assert "## Recent transactions" in result.response
    assert "| Date | Description | Amount | Status | Category | Disputed |" in result.response
    assert result.response_path == "base_tool_rendered"


def test_read_view_drops_a_model_authored_table_and_keeps_the_exact_rendered_one() -> None:
    _model, result = _read_view_turn(
        "Here are your recent transactions. | Date | Transaction |\n"
        "| --- | --- |\n| Recent | North Harbor Market |"
    )

    assert result.response.startswith("Here are your recent transactions.")
    assert "| Recent | North Harbor Market |" not in result.response
    assert "2026-07-25 15:42 UTC" in result.response


def test_read_view_falls_back_to_the_table_when_the_lead_in_leaks_a_private_id() -> None:
    _model, result = _read_view_turn("Here is txn_alex_001 from your records.")

    assert "txn_alex_001" not in result.response
    assert result.response.startswith("## Recent transactions")
    assert result.response_path == "base_tool_rendered"


def test_read_view_strips_canned_realizer_filler_from_the_lead_in() -> None:
    _model, result = _read_view_turn(
        "I found the following details: Here are your two most recent transactions. "
        "This reflects the information available in this session."
    )

    assert result.response.startswith("Here are your two most recent transactions.")
    assert "I found the following details:" not in result.response
    assert "information available in this session" not in result.response


def test_grounded_final_guidance_asks_for_a_natural_summary_of_rendered_records() -> None:
    model, _result = _read_view_turn("Here are your two most recent transactions.")

    followup_system = model.calls[1]["messages"][0]["content"]
    assert "table" in followup_system
    assert "do not repeat" in followup_system.casefold()


def test_v4_execute_retries_once_when_model_returns_prose_before_required_tool() -> None:
    model = RecordingModel(
        [
            "I found your transactions.",
            '<tool_call>{"name": "list_transactions", "arguments": {"limit": 5}}</tool_call>',
            "Here are your five most recent transactions.",
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    result = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="Show my five most recent transactions.",
        conversation=[],
        router_result=v4_router_guidance(
            action="execute_tool",
            fine_intent="view_transactions",
        ),
    )

    assert [call.name for call in result.tool_calls] == ["list_transactions"]
    assert result.tool_calls[0].arguments == {"limit": 5}
    assert [trace.label for trace in result.model_passes] == [
        "base",
        "required_tool_retry_1",
        "grounded_final",
    ]
    assert all(
        [tool["function"]["name"] for tool in call["tools"]] == ["list_transactions"]
        for call in model.calls[:2]
    )
    assert model.calls[2]["tools"] is None
    assert "A tool call is required before answering" in model.calls[1]["messages"][0]["content"]


def test_v4_execute_rejects_repeated_prose_without_executing_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = RecordingModel(
        [
            "I replaced that card.",
            "The replacement is complete.",
        ]
    )
    registry = bank()
    executed: list[str] = []

    def record_execution(*args: Any, **kwargs: Any) -> dict[str, Any]:
        executed.append(str(args[2]))
        return {"unexpected": True}

    monkeypatch.setattr(registry, "execute", record_execution)
    agent = ConversationalBankingAgent(bank=registry, model=model)

    with pytest.raises(
        AgentExecutionError,
        match="required tool call after one retry",
    ) as failure:
        agent.run_turn(
            username="alex.demo",
            session_hash="session",
            message="Replace the card ending in 4821.",
            conversation=[],
            router_result=v4_router_guidance(
                action="execute_tool",
                fine_intent="replace_card",
            ),
        )

    assert executed == []
    assert [trace.label for trace in failure.value.model_passes] == [
        "base",
        "required_tool_retry_1",
    ]
    assert [trace.raw_output for trace in failure.value.model_passes] == [
        "I replaced that card.",
        "The replacement is complete.",
    ]
    assert model.outputs == []


def test_v4_execute_rejects_multiple_routed_write_calls_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate_calls = (
        '<tool_call>{"name":"replace_card","arguments":{"last4":"4821"}}</tool_call>'
        '<tool_call>{"name":"replace_card","arguments":{"last4":"4821"}}</tool_call>'
    )
    model = RecordingModel(["I can replace it.", duplicate_calls])
    registry = bank()
    executed: list[str] = []

    def record_execution(*args: Any, **kwargs: Any) -> dict[str, Any]:
        executed.append(str(args[2]))
        return {"unexpected": True}

    monkeypatch.setattr(registry, "execute", record_execution)
    agent = ConversationalBankingAgent(bank=registry, model=model)

    with pytest.raises(
        AgentExecutionError,
        match="exactly one tool call",
    ):
        agent.run_turn(
            username="alex.demo",
            session_hash="session",
            message="Replace the card ending in 4821.",
            conversation=[],
            router_result=v4_router_guidance(
                action="execute_tool",
                fine_intent="replace_card",
            ),
        )

    assert executed == []


def test_v4_grounded_final_rejects_repeated_write_after_one_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repeated_call = '<tool_call>{"name":"replace_card","arguments":{"last4":"4821"}}</tool_call>'
    model = RecordingModel([repeated_call, repeated_call])
    registry = bank()
    executed: list[str] = []

    def record_execution(*args: Any, **kwargs: Any) -> dict[str, Any]:
        executed.append(str(args[2]))
        return {"status": "replacement_requested", "last4": "4821"}

    monkeypatch.setattr(registry, "execute", record_execution)
    agent = ConversationalBankingAgent(bank=registry, model=model)

    with pytest.raises(
        AgentExecutionError,
        match="grounded-final response attempted another routed tool call",
    ) as failure:
        agent.run_turn(
            username="alex.demo",
            session_hash="session",
            message="Replace the card ending in 4821.",
            conversation=[],
            router_result=v4_router_guidance(
                action="execute_tool",
                fine_intent="replace_card",
            ),
        )

    assert executed == ["replace_card"]
    assert [trace.label for trace in failure.value.model_passes] == [
        "base",
        "grounded_final",
    ]
    assert model.calls[1]["tools"] is None


@pytest.mark.parametrize("entity_resolution", ["missing", "ambiguous", "ineligible"])
def test_v4_unresolved_entity_asks_model_authored_clarification_without_tools(
    entity_resolution: str,
) -> None:
    clarification = "Which card would you like me to replace?"
    model = RecordingModel([clarification])
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    result = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="Replace that card.",
        conversation=[],
        router_result=v4_router_guidance(
            action="execute_tool",
            entity_resolution=entity_resolution,
        ),
    )

    assert result.response == clarification
    assert result.tool_calls == ()
    assert model.calls[0]["tools"] is None
    assert (
        "Ask exactly one concise, natural clarification question"
        in (model.calls[0]["messages"][0]["content"])
    )


def test_v4_clarify_never_executes_a_model_emitted_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    model = RecordingModel(
        ['<tool_call>{"name": "replace_card", "arguments": {"last4": "4821"}}</tool_call>']
    )
    registry = bank()
    executed: list[str] = []

    def record_execution(*args: Any, **kwargs: Any) -> dict[str, Any]:
        executed.append(str(args[2]))
        return {"unexpected": True}

    monkeypatch.setattr(registry, "execute", record_execution)
    agent = ConversationalBankingAgent(bank=registry, model=model)

    with pytest.raises(AgentProtocolError, match="unexposed tool"):
        agent.run_turn(
            username="alex.demo",
            session_hash="session",
            message="Replace that card.",
            conversation=[],
            router_result=v4_router_guidance(
                action="clarify",
                entity_resolution="ambiguous",
            ),
        )

    assert model.calls[0]["tools"] is None
    assert executed == []


def test_v4_converse_exposes_no_banking_tools() -> None:
    response = "Hello! How can I help you today?"
    model = RecordingModel([response])
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    result = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="Hello",
        conversation=[],
        router_result=v4_router_guidance(
            action="converse",
            fine_intent="conversation",
            entity_resolution="not_applicable",
        ),
    )

    assert result.response == response
    assert model.calls[0]["tools"] is None
    assert "Respond naturally and concisely" in model.calls[0]["messages"][0]["content"]
    assert "Never infer distress" in model.calls[0]["messages"][0]["content"]


def test_converse_turn_claiming_an_action_is_repaired_or_rejected() -> None:
    model = RecordingModel(
        [
            "I found your active card and froze it to stop unauthorized use.",
            "I can’t complete that from this conversation yet — would you like me to "
            "freeze the card ending in 4821?",
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    result = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="My card was stolen. Freeze it.",
        conversation=[],
        router_result=v4_router_guidance(
            action="converse",
            fine_intent="freeze_card",
            entity_resolution="not_required",
        ),
    )

    assert "froze it" not in result.response
    assert len(model.calls) == 2  # draft + customer-experience repair


def test_v4_uncertain_execution_prediction_does_not_expose_tool() -> None:
    model = RecordingModel(["Which card would you like help with?"])
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    result = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="Replace it",
        conversation=[],
        router_result={
            "route": "uncertain",
            "intent": "replace_card",
            "action": "execute_tool",
            "entity_resolution": "resolved",
        },
    )

    assert result.tool_calls == ()
    assert model.calls[0]["tools"] is None
    prompt = model.calls[0]["messages"][0]["content"]
    assert "clarification question" in prompt
    assert all(term not in prompt.lower() for term in ("classifier", "logit", "label"))


def test_v7_grounded_write_exposes_required_exact_selector_schema() -> None:
    model = RecordingModel(
        [
            '<tool_call>{"name":"replace_card","arguments":{"last4":"4821"}}</tool_call>',
            "Replacement is pending for your card ending in 4821.",
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    result = agent.run_turn(
        username="alex.demo",
        session_hash="v7-schema",
        message="Replace card 4821.",
        conversation=[],
        router_result={
            **v4_router_guidance(action="execute_tool"),
            "effective_decision_contract": "retail-bank-effective-turn-decision/v1",
            "decision_accepted": True,
            "argument_constraints": {"last4": "4821"},
        },
    )

    schema = model.calls[0]["tools"][0]["function"]["parameters"]
    assert schema["required"] == ["last4"]
    assert schema["properties"]["last4"] == {
        "type": "string",
        "pattern": "^4821$",
        "const": "4821",
        "enum": ["4821"],
    }
    assert result.tool_calls[0].arguments == {"last4": "4821"}


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({}, "required argument"),
        ({"last4": "7319"}, "const"),
        (
            {
                "last4": "4821 ",
            },
            "const",
        ),
    ],
)
def test_v7_grounded_write_rejects_missing_or_non_exact_model_argument(
    arguments: dict[str, Any],
    error: str,
) -> None:
    model = RecordingModel(
        [f'<tool_call>{{"name":"replace_card","arguments":{json.dumps(arguments)}}}</tool_call>']
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    with pytest.raises(AgentExecutionError, match=error):
        agent.run_turn(
            username="alex.demo",
            session_hash="v7-schema-reject",
            message="Replace card 4821.",
            conversation=[],
            router_result={
                **v4_router_guidance(action="execute_tool"),
                "effective_decision_contract": "retail-bank-effective-turn-decision/v1",
                "decision_accepted": True,
                "argument_constraints": {"last4": "4821"},
            },
        )


def test_v7_rejected_effective_decision_exposes_no_tool() -> None:
    model = RecordingModel(["Which card would you like me to replace?"])
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    result = agent.run_turn(
        username="alex.demo",
        session_hash="v7-rejected",
        message="Replace that card.",
        conversation=[],
        router_result={
            **v4_router_guidance(action="clarify", entity_resolution="ambiguous"),
            "effective_decision_contract": "retail-bank-effective-turn-decision/v1",
            "decision_accepted": False,
            "learned_action": "execute_tool",
            "learned_entity_resolution": "resolved",
            "argument_constraints": {},
            "entity_grounding_source": "live_candidate",
            "entity_candidate_count": 2,
        },
    )

    assert result.tool_calls == ()
    assert model.calls[0]["tools"] is None
    diagnostics = router_diagnostic_fields(
        {
            **v4_router_guidance(action="clarify", entity_resolution="ambiguous"),
            "effective_decision_contract": "retail-bank-effective-turn-decision/v1",
            "decision_accepted": False,
            "learned_action": "execute_tool",
            "learned_entity_resolution": "resolved",
            "argument_constraints": {},
            "entity_grounding_source": "live_candidate",
            "entity_candidate_count": 2,
        }
    )
    assert diagnostics["learned_action"] == "execute_tool"
    assert diagnostics["effective_action"] == "clarify"
    assert diagnostics["learned_entity_resolution"] == "resolved"
    assert diagnostics["effective_entity_resolution"] == "ambiguous"


def test_internal_language_leak_gets_one_customer_experience_repair() -> None:
    model = RecordingModel(
        [
            "Hi! I’m ready to help with the synthetic accounts in this demo.",
            "Hi, I’m Harbor. How can I help with your banking today?",
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    result = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="Hello",
        conversation=[],
        router_result=router_guidance(),
    )

    assert result.response == "Hi, I’m Harbor. How can I help with your banking today?"
    assert result.response_path == "direct_answer_customer_repaired"
    assert [item.label for item in result.model_passes] == [
        "base",
        "customer_experience_repair_1",
    ]
    assert model.calls[-1]["tools"] is None


def test_policy_turn_is_generated_from_evidence_without_banking_tools() -> None:
    model = RecordingModel(
        [
            "Harborlight Bank accepts mortgage applications for review; approval is "
            "not automatic. [Policy: mortgage.application.overview.us.v1]"
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)
    matches = (
        {
            "chunk_id": "mortgage.application.overview.us.v1",
            "title": "Mortgage application overview",
            "text": "Applications are reviewed before approval.",
            "effective_from": "2026-01-01",
        },
    )

    result = agent.run_policy_turn(
        username="alex.demo",
        session_hash="session",
        message="How do I start a mortgage application?",
        conversation=[],
        policy_matches=matches,
        corpus_revision="sha256:policy-v1",
    )

    assert result.response_path == "policy_grounded"
    assert result.policy_sources == ("mortgage.application.overview.us.v1",)
    assert result.tool_calls == ()
    assert len(model.calls) == 1
    assert model.calls[0]["tools"] is None
    assert "sha256:policy-v1" in model.calls[0]["messages"][0]["content"]


def test_policy_turn_repairs_an_invented_citation_once() -> None:
    model = RecordingModel(
        [
            "Apply online. [Policy: invented.policy]",
            "Applications are reviewed before approval. "
            "[Policy: mortgage.application.overview.us.v1]",
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)
    matches = (
        {
            "chunk_id": "mortgage.application.overview.us.v1",
            "title": "Mortgage application overview",
            "text": "Applications are reviewed before approval.",
        },
    )

    result = agent.run_policy_turn(
        username="alex.demo",
        session_hash="session",
        message="Can I get a mortgage?",
        conversation=[],
        policy_matches=matches,
        corpus_revision="sha256:policy-v1",
    )

    assert result.response_path == "policy_grounded_repaired"
    assert [item.label for item in result.model_passes] == [
        "policy_grounded",
        "policy_repair_1",
    ]
    assert all(call["tools"] is None for call in model.calls)


def test_tool_calls_execute_in_order_and_second_model_pass_writes_final_answer() -> None:
    model = RecordingModel(
        [
            """
<tool_call>
{"name": "list_transfers", "arguments": {}}
</tool_call>
<tool_call>
{"name": "list_transactions", "arguments": {"limit": 2}}
</tool_call>
""",
            "You have two transfers. River Consulting is pending, and Jamie Lee is completed.",
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    result = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="What transfers are there, and show my latest transactions too.",
        conversation=[],
        router_result=router_guidance(),
    )

    assert [call.name for call in result.tool_calls] == [
        "list_transfers",
        "list_transactions",
    ]
    assert len(result.tool_results) == 2
    assert all(set(item) == {"ok", "result"} for item in result.tool_results)
    assert all(item["ok"] is True for item in result.tool_results)
    assert len(model.calls) == 2
    assert model.calls[0]["tools"] == MODEL_TOOLS
    assert model.calls[1]["tools"] == MODEL_TOOLS
    assert result.tool_calls[0].id.endswith("_0_list_transfers")
    assert result.tool_calls[1].id.endswith("_1_list_transactions")
    assert model.calls[1]["messages"][-2]["tool_call_id"] == result.tool_calls[0].id
    assert model.calls[1]["messages"][-1]["tool_call_id"] == result.tool_calls[1].id
    transfer_result = model.calls[1]["messages"][-2]["content"]
    assert json.loads(transfer_result) == result.tool_results[0]
    assert '"amount_cents":45000' in transfer_result
    assert '"amount":' not in transfer_result
    system_prompt = model.calls[0]["messages"][0]["content"]
    assert "already authenticated" in system_prompt
    assert "Use the supplied tools" in system_prompt
    assert "never ask for account numbers" in system_prompt
    assert "CURRENT DUAL-HEAD CLASSIFIER GUIDANCE" not in system_prompt
    assert [item["role"] for item in result.conversation] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert "## Transfers" in result.response
    assert "## Recent transactions" in result.response
    assert "| Recipient | Amount | Status | Created |" in result.response
    assert "| Date | Description | Amount | Status | Category | Disputed |" in result.response
    assert result.conversation[-1]["content"] == result.response
    assert result.response_path == "base_tool_rendered"


def test_invalid_action_answer_gets_one_grounding_repair_without_tools() -> None:
    model = RecordingModel(
        [
            '<tool_call>{"name": "cancel_transfer", "arguments": '
            '{"recipient": "River Consulting"}}</tool_call>',
            "Done — I cancelled Jamie Lee's transfer.",
            "Done — I cancelled the transfer to River Consulting.",
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    result = agent.run_turn(
        username="alex.demo",
        session_hash="repair-session",
        message="Cancel the River Consulting transfer.",
        conversation=[],
        router_result=router_guidance(),
    )

    assert result.response == "Done — I cancelled the transfer to River Consulting."
    assert result.response_path == "base_tool_repaired"
    assert [item.label for item in result.model_passes] == [
        "base",
        "grounded_final",
        "final_repair_1",
    ]
    assert len(model.calls) == 3
    assert model.calls[-1]["tools"] is None
    assert "River Consulting" in model.calls[-1]["messages"][-1]["content"]


def test_model_selected_write_uses_friendly_argument_without_authorization_layer() -> None:
    model = RecordingModel(
        [
            """
<tool_call>
{"name": "cancel_transfer", "arguments": {"recipient": "River Consulting"}}
</tool_call>
""",
            "Done — I cancelled the River Consulting transfer.",
        ]
    )
    registry = bank()
    agent = ConversationalBankingAgent(bank=registry, model=model)

    result = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="Please take care of the River Consulting transfer.",
        conversation=[],
        router_result=router_guidance(),
    )

    assert result.tool_results[0]["ok"] is True
    assert set(result.tool_results[0]) == {"ok", "result"}
    assert registry.snapshot("alex.demo", "session")["transfers"][0]["status"] == "cancelled"


def test_backend_error_returns_safe_canonical_tool_result_to_model() -> None:
    model = RecordingModel(
        [
            """
<tool_call>
{"name": "cancel_transfer", "arguments": {"recipient": "Nobody"}}
</tool_call>
""",
            "I could not complete that operation because I could not find a matching transfer.",
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    result = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="Do those operations.",
        conversation=[],
        router_result=router_guidance(),
    )

    assert result.tool_results == (
        {
            "ok": False,
            "error": {
                "code": "record_match_count",
                "message": "The request did not match exactly one banking record.",
            },
        },
    )
    assert result.conversation[2]["tool_call_id"].endswith("_0_cancel_transfer")
    assert json.loads(model.calls[1]["messages"][-1]["content"]) == result.tool_results[0]
    assert len(model.calls) == 2


def test_invalid_model_arguments_remain_protocol_failures_without_repair() -> None:
    model = RecordingModel(
        [
            '<tool_call>{"name": "list_transactions", "arguments": {"limit": "two"}}</tool_call>',
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    with pytest.raises(AgentProtocolError, match="invalid type"):
        agent.run_turn(
            username="alex.demo",
            session_hash="session",
            message="Show my latest two transactions.",
            conversation=[],
            router_result=router_guidance(),
        )

    assert len(model.calls) == 1


def test_unknown_model_tool_remains_protocol_failure_without_fallback() -> None:
    model = RecordingModel(
        [
            '<tool_call>{"name": "close_account", "arguments": {}}</tool_call>',
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    with pytest.raises(AgentProtocolError, match="unsupported tool"):
        agent.run_turn(
            username="alex.demo",
            session_hash="session",
            message="Close my account.",
            conversation=[],
            router_result=router_guidance(),
        )

    assert len(model.calls) == 1


def test_repeated_same_name_calls_keep_distinct_tool_call_ids() -> None:
    model = RecordingModel(
        [
            """
<tool_call>
{"name": "list_transactions", "arguments": {"limit": 1}}
</tool_call>
<tool_call>
{"name": "list_transactions", "arguments": {"limit": 2}}
</tool_call>
""",
            "Here are the requested transaction views.",
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    result = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="Show my latest transaction, then show my latest two transactions.",
        conversation=[],
        router_result=router_guidance(),
    )

    assert [call.name for call in result.tool_calls] == [
        "list_transactions",
        "list_transactions",
    ]
    assert result.tool_calls[0].id.endswith("_0_list_transactions")
    assert result.tool_calls[1].id.endswith("_1_list_transactions")
    assert result.tool_calls[0].id != result.tool_calls[1].id
    assert [
        item["tool_call_id"] for item in model.calls[1]["messages"] if item["role"] == "tool"
    ] == [call.id for call in result.tool_calls]
    assert all(item["ok"] is True for item in result.tool_results)


def test_fallback_tool_call_ids_do_not_collide_across_retained_turns() -> None:
    tool_output = '<tool_call>{"name": "list_transfers", "arguments": {}}</tool_call>'
    model = RecordingModel(
        [
            tool_output,
            "You have a pending River Consulting transfer.",
            tool_output,
            "You still have a pending River Consulting transfer.",
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    first = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="Show my transfers.",
        conversation=[],
        router_result=router_guidance(),
    )
    second = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="Show my transfers again.",
        conversation=first.conversation,
        router_result=router_guidance(),
    )

    first_id = first.tool_calls[0].id
    second_id = second.tool_calls[0].id
    assert first_id.endswith("_0_list_transfers")
    assert second_id.endswith("_0_list_transfers")
    assert first_id != second_id
    retained_tool_ids = [
        item["tool_call_id"] for item in second.conversation if item["role"] == "tool"
    ]
    assert retained_tool_ids == [first_id, second_id]


def test_duplicate_model_tool_call_ids_are_protocol_failures() -> None:
    model = RecordingModel(
        [
            """
<tool_call>
{"id": "call_duplicate", "index": 0, "name": "list_accounts", "arguments": {}}
</tool_call>
<tool_call>
{"id": "call_duplicate", "index": 1, "name": "list_cards", "arguments": {}}
</tool_call>
""",
        ]
    )
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    with pytest.raises(AgentProtocolError, match="IDs must be unique"):
        agent.run_turn(
            username="alex.demo",
            session_hash="session",
            message="Show my accounts and cards.",
            conversation=[],
            router_result=router_guidance(),
        )

    assert len(model.calls) == 1


def test_model_can_chain_tool_calls_after_observing_tool_results() -> None:
    model = RecordingModel(
        [
            '<tool_call>{"name": "list_cards", "arguments": {}}</tool_call>',
            '<tool_call>{"name": "freeze_card", "arguments": {"last4": "4821"}}</tool_call>',
            "I found your active Everyday Visa Debit card ending in 4821 and froze it.",
        ]
    )
    registry = bank()
    agent = ConversationalBankingAgent(bank=registry, model=model)

    result = agent.run_turn(
        username="alex.demo",
        session_hash="session",
        message="Find my active card and freeze it.",
        conversation=[],
        router_result=router_guidance(),
    )

    assert result.response_path == "base_tool_chain"
    assert [call.name for call in result.tool_calls] == ["list_cards", "freeze_card"]
    assert result.tool_calls[1].arguments == {"last4": "4821"}
    assert [item.label for item in result.model_passes] == [
        "base",
        "grounded_final",
        "tool_followup_2",
    ]
    assert [item["role"] for item in result.conversation] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert model.calls[1]["messages"][-1]["name"] == "list_cards"
    assert model.calls[2]["messages"][-1]["name"] == "freeze_card"
    assert registry.snapshot("alex.demo", "session")["cards"][0]["status"] == "frozen"


def test_tool_chain_stops_at_total_call_limit() -> None:
    repeated_card_call = '<tool_call>{"name": "list_cards", "arguments": {}}</tool_call>'
    model = RecordingModel([repeated_card_call] * 9)
    agent = ConversationalBankingAgent(bank=bank(), model=model)

    with pytest.raises(
        AgentExecutionError,
        match="more than 8 total tool calls",
    ) as failure:
        agent.run_turn(
            username="alex.demo",
            session_hash="session",
            message="Keep checking my cards.",
            conversation=[],
            router_result=router_guidance(),
        )

    assert len(failure.value.tool_calls) == 8
    assert len(failure.value.tool_results) == 8


def test_token_budget_keeps_latest_complete_tool_chain_and_newest_fitting_turns() -> None:
    system: dict[str, Any] = {"role": "system", "content": "system"}
    old: list[dict[str, Any]] = [
        {"role": "user", "content": "old " * 30},
        {"role": "assistant", "content": "old answer " * 30},
    ]
    middle: list[dict[str, Any]] = [
        {"role": "user", "content": "middle"},
        {"role": "assistant", "content": "middle answer"},
    ]
    latest: list[dict[str, Any]] = [
        {"role": "user", "content": "show transfers"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_context_0_list_transfers",
                    "index": 0,
                    "type": "function",
                    "function": {"name": "list_transfers", "arguments": {}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_context_0_list_transfers",
            "name": "list_transfers",
            "content": '{"ok": true}',
        },
    ]

    selected = select_token_budgeted_context(
        system,
        [*old, *middle, *latest],
        tools=MODEL_TOOLS,
        token_counter=lambda messages, _tools: len(json.dumps(messages)),
        input_budget=540,
    )

    assert selected[0] == system
    assert latest == selected[-len(latest) :]
    assert middle[0] in selected
    assert old[0] not in selected
    latest_roles = [item["role"] for item in selected[-len(latest) :]]
    assert latest_roles == ["user", "assistant", "tool"]


def test_token_budget_rejects_oversized_latest_group_without_truncation() -> None:
    with pytest.raises(AgentProtocolError, match="latest conversation turn"):
        select_token_budgeted_context(
            {"role": "system", "content": "system"},
            [{"role": "user", "content": "x" * 100}],
            tools=MODEL_TOOLS,
            token_counter=lambda messages, _tools: len(json.dumps(messages)),
            input_budget=50,
        )


def test_token_budget_pins_pending_servicing_exchange_across_long_detour() -> None:
    anchor = [
        {"role": "user", "content": "I need to dispute a purchase."},
        {"role": "assistant", "content": "Which purchase should I review?"},
    ]
    detour = [
        {"role": "user", "content": f"policy question {index}"} for index in range(5) for _ in (0,)
    ]
    conversation: list[dict[str, Any]] = [*anchor]
    for item in detour:
        conversation.extend([item, {"role": "assistant", "content": "policy answer " * 15}])
    conversation.append({"role": "user", "content": "Okay, continue."})

    selected = select_token_budgeted_context(
        {"role": "system", "content": "system"},
        conversation,
        tools=MODEL_TOOLS,
        token_counter=lambda messages, _tools: len(json.dumps(messages)),
        input_budget=1000,
        pinned_exchange=anchor,
    )

    assert anchor[0] in selected
    assert anchor[1] in selected
    assert selected[-1]["content"] == "Okay, continue."


def test_default_input_budget_is_8192_tokens() -> None:
    assert INPUT_TOKEN_BUDGET == 8192
