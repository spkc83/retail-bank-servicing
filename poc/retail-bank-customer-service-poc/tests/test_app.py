from __future__ import annotations

import html
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def app_module(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv(
        "DEMO_AUTH_JSON",
        '{"alex.demo":"alex-test-password","maya.demo":"maya-test-password"}',
    )
    monkeypatch.setenv("POC_SKIP_MODEL_LOAD", "1")
    monkeypatch.setenv("POC_SKIP_ROUTER_LOAD", "1")
    monkeypatch.setenv("POC_SESSION_DB_DIR", str(tmp_path / "sessions"))
    for name in (
        "app",
        "model_service",
        "state",
        "zero_gpu_runtime",
    ):
        sys.modules.pop(name, None)
    return importlib.import_module("app")


def request(username: str = "alex.demo", session_hash: str = "browser-session"):
    return SimpleNamespace(username=username, session_hash=session_hash)


def route(
    route_name: str = "in_domain",
    *,
    banking_probability: float = 0.99,
    capability: str = "transfers",
    intent: str | None = None,
    relations: tuple[str, ...] = (),
) -> dict[str, object]:
    resolved_intent = intent or {
        "accounts": "view_accounts",
        "cards": "view_cards",
        "transfers": "view_transfers",
        "service_cases": "view_service_cases",
        "conversation": "conversation",
    }.get(capability, capability)
    return {
        "route": route_name,
        "banking_probability": banking_probability,
        "ood_probability": 1 - banking_probability,
        "confidence": max(banking_probability, 1 - banking_probability),
        "capability": capability,
        "capability_confidence": 0.8,
        "capability_candidates": [
            {"capability": capability, "probability": 0.8},
            {"capability": "accounts", "probability": 0.1},
            {"capability": "cards", "probability": 0.05},
        ],
        "intent": resolved_intent,
        "intent_confidence": 0.8,
        "intent_candidates": [
            {"intent": resolved_intent, "probability": 0.8},
        ],
        "lane": (
            "policy"
            if resolved_intent == "policy_knowledge"
            else "conversation"
            if resolved_intent == "conversation"
            else "servicing"
        ),
        "relation_probabilities": {
            "context_dependent": 0.1,
            "agent_repair": 0.1,
            "topic_shift": 0.1,
            "clarification_answer": 0.1,
            "resume_previous_service": (0.9 if "resume_previous_service" in relations else 0.1),
        },
        "active_relations": list(relations),
        "ood_banking_threshold": 0.2,
        "in_domain_threshold": 0.5,
        "relation_rescue_threshold": 0.5,
        "router_revision": "test-router",
    }


def test_app_constructs_expected_authenticated_api_surface(app_module) -> None:
    api_names = {
        dependency.get("api_name") for dependency in app_module.demo.config["dependencies"]
    }

    assert {
        "chat",
        "route",
        "customer_snapshot",
        "reset_demo",
    } <= api_names
    assert "zero_gpu_probe" not in api_names
    assert {username for username, _ in app_module.AUTH_CREDENTIALS} == {
        "alex.demo",
        "maya.demo",
    }
    assert "alex-test-password" in app_module.AUTH_MESSAGE
    assert "maya-test-password" in app_module.AUTH_MESSAGE


def test_gradio_presentation_uses_shared_harborlight_brand(app_module) -> None:
    source = (Path(app_module.__file__)).read_text(encoding="utf-8")

    assert app_module.BANK_NAME == "Harborlight Bank"
    assert app_module.ASSISTANT_NAME == "Harbor"
    assert app_module.demo.title == "Harbor | Harborlight Bank"
    assert "Retail Bank Customer Service POC" not in source
    assert "Talk naturally to the 9B synthetic bank agent" not in source
    assert "Ask the signed-in synthetic bank agent" not in source
    assert "Synthetic backend state" not in source
    assert 'gr.Accordion("Technical details", open=False)' in source
    assert 'elem_classes=["harbor-layout"]' in source
    assert 'elem_classes=["harbor-chat"]' in source
    assert "with gr.Column(scale=3, min_width=720" in source
    assert source.index('gr.Accordion("Technical details"') > source.index(
        "with gr.Column(scale=3, min_width=720"
    )
    assert "#082f49" in app_module.CSS
    assert "#0f766e" in app_module.CSS


def test_snapshot_uses_customer_friendly_product_labels(app_module) -> None:
    snapshot = app_module.BANK.snapshot("alex.demo", "brand-test")

    rendered = app_module.render_snapshot(snapshot)

    assert "Your products" in rendered
    assert "Checking account" in rendered
    assert "Savings account" in rendered
    assert "Debit cards" in rendered
    assert "Synthetic backend state" not in rendered


def test_diagnostics_show_v6_hierarchy_and_exact_exposed_tools(app_module) -> None:
    routed = {
        **route(capability="cards", intent="freeze_card"),
        "domain": "banking",
        "lane": "servicing",
        "family": "cards",
        "action": "execute_tool",
        "entity_resolution": "resolved",
    }

    diagnostics = app_module._render_diagnostics(routed, (), (), "test")

    assert "V6 domain: `banking`" in diagnostics
    assert "V6 lane: `servicing`" in diagnostics
    assert "V6 family: `cards`" in diagnostics
    assert "V6 intent: `freeze_card`" in diagnostics
    assert "V6 action: `execute_tool`" in diagnostics
    assert "V6 entity resolution: `resolved`" in diagnostics
    assert 'Exposed tools: `["freeze_card"]`' in diagnostics
    assert "list_accounts" not in diagnostics
    summary = diagnostics.split("### Full technical details", 1)[0]
    assert "Outcome: `test`" in summary
    assert "Granite passes: `0`" in summary
    assert "Relation probabilities" not in summary
    assert "SHA-256" not in summary


def test_diagnostics_retain_v3_route_compatibility(app_module) -> None:
    diagnostics = app_module._render_diagnostics(route(), (), (), "test")

    assert "V6 domain: `not available (V3 compatibility)`" in diagnostics
    assert "V6 intent: `view_transfers`" in diagnostics
    assert "Exposed tools:" in diagnostics
    assert "list_accounts" in diagnostics
    assert "cancel_transfer" in diagnostics


def test_response_provenance_uses_only_recorded_path_and_model_passes(app_module) -> None:
    passes = (
        SimpleNamespace(label="base"),
        SimpleNamespace(label="reflection"),
    )

    assert app_module.response_provenance("base_tool_repaired", passes) == (
        "Response path: base tool repaired · Recorded model passes: 2 (base, reflection)"
    )
    assert app_module.response_provenance("OOD stock response", ()) == (
        "Response path: OOD stock response · Recorded model passes: 0"
    )


def test_chat_turn_is_a_single_direct_zero_gpu_boundary(app_module) -> None:
    assert app_module.run_model_turn._zero_gpu_config == {
        "size": "large",
        "duration": 90,
    }
    assert not hasattr(app_module.generate_text, "_zero_gpu_config")
    chat_dependencies = [
        dependency
        for dependency in app_module.demo.config["dependencies"]
        if dependency.get("api_name") == "chat"
    ]
    assert len(chat_dependencies) == 1
    assert chat_dependencies[0]["targets"] != []
    assert all(
        dependency.get("trigger_after") is None
        for dependency in app_module.demo.config["dependencies"]
        if dependency.get("api_name") == "chat"
    )


def test_greeting_and_uncertain_turns_are_answered_by_9b(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app_module,
        "route_query",
        lambda *_args: route(
            "uncertain",
            banking_probability=0.52,
            capability="conversation",
        ),
    )
    monkeypatch.setattr(app_module, "count_tokens", lambda *_args: 50)
    monkeypatch.setattr(
        app_module,
        "generate_text",
        lambda *_args: "Hey! How can I help with your banking today?",
    )

    result = app_module.run_model_turn("yo, sup ?", [], [], {}, request())

    assert result[1] == result[2]
    assert result[1][-1]["role"] == "assistant"
    assert result[1][-1]["content"] == "Hey! How can I help with your banking today?"
    assert "model authored" in result[4]
    assert "conversation" in result[5]


def test_high_confidence_ood_uses_stock_response_inside_registered_gpu_turn(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app_module,
        "route_query",
        lambda *_args: route(
            "out_of_domain",
            banking_probability=0.001,
            capability="accounts",
        ),
    )

    result = app_module.run_model_turn("Write a Python web scraper.", [], [], {}, request())

    assert "Harborlight Bank" in result[1][-1]["content"]
    assert "synthetic" not in result[1][-1]["content"].casefold()
    assert result[1] == result[2]
    assert "out_of_domain" in result[5]


def test_test_mode_router_omission_routes_uncertain_to_9b(app_module) -> None:
    result = app_module.route_query("hello", [])

    assert result["route"] == "uncertain"
    assert result["capability"] is None
    assert result["capability_candidates"] == []


def test_classifier_failure_is_visible_and_blocks_9b(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingRouter:
        def classify(self, *_args, **_kwargs):
            raise RuntimeError("classifier unavailable")

    monkeypatch.setattr(app_module, "router", FailingRouter())

    def unexpected_generation(*_args):
        pytest.fail("9B generation must not run after classifier failure")

    monkeypatch.setattr(app_module, "generate_text", unexpected_generation)

    result = app_module.run_model_turn(
        "Show my balances.",
        [],
        [],
        {},
        request(),
    )

    assert result[1][-1]["content"] == app_module.MODEL_FAILURE_RESPONSE
    assert "classifier failed" in result[4]
    assert "classifier_error" in result[5]
    assert "RuntimeError" in result[5]
    assert "9B generator was not invoked" in result[5]


def test_model_selects_transfer_tool_and_receives_full_tool_history(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter(
        [
            '<tool_call>{"name": "list_transfers", "arguments": {}}</tool_call>',
            (
                "River Consulting has a pending USD 450 transfer, and Jamie Lee "
                "has a completed USD 125 transfer."
            ),
        ]
    )
    monkeypatch.setattr(app_module, "count_tokens", lambda *_args: 100)
    monkeypatch.setattr(app_module, "generate_text", lambda *_args: next(outputs))
    monkeypatch.setattr(app_module, "route_query", lambda *_args: route())

    result = app_module.run_model_turn(
        "What transfers are there on my account?",
        [],
        [],
        {},
        request(),
    )

    canonical = result[2]
    assert [item["role"] for item in canonical] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert canonical[1]["tool_calls"][0]["function"]["name"] == "list_transfers"
    assert "River Consulting" in result[1][-1]["content"]
    assert "list_transfers" in result[5]


def test_direct_answer_is_labeled_with_per_pass_provenance(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "count_tokens", lambda *_args: 100)
    monkeypatch.setattr(
        app_module,
        "generate_text",
        lambda *_args: "I can explain how savings interest works.",
    )
    monkeypatch.setattr(app_module, "route_query", lambda *_args: route())

    result = app_module.run_model_turn(
        "How does savings interest work?",
        [],
        [],
        {},
        request(),
    )

    assert result[1][-1]["content"] == "I can explain how savings interest works."
    assert "9B direct_answer" in result[5]
    assert "`base`" in result[5]
    assert "`reflection`" not in result[5]
    assert "Generation calls: `1`" in result[5]
    assert "prompt SHA-256" in result[5]
    assert "raw output SHA-256" in result[5]
    assert "I can explain how savings interest works." in result[5]
    assert "Runtime device:" in result[5]
    assert "CUDA device name:" in result[5]
    assert app_module.MODEL_ID in result[5]
    assert app_module.MODEL_REVISION in result[5]
    assert app_module.ROUTER_REVISION in result[5]


def test_gpu_failure_never_generates_cpu_servicing_answer(
    app_module,
) -> None:
    result = app_module.fail_model_turn(
        "What transfers are there on my account?",
        [],
        {},
        request(),
    )

    response = result[1][-1]["content"]
    assert response == app_module.MODEL_FAILURE_RESPONSE
    assert "River Consulting" not in response
    assert "try again" in response.casefold()
    assert result[1] == result[2]
    assert [item["role"] for item in result[2]] == ["user", "assistant"]
    assert "could not allocate" in result[4].lower()


def test_generic_model_error_preserves_safe_message_and_attached_trace(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = SimpleNamespace(
        label="base",
        input_tokens=10,
        prompt_sha256="prompt-hash",
        output_chars=14,
        output_sha256="output-hash",
        raw_output="partial model trace",
        runtime_device="cuda:0",
        cuda_device_name="test GPU",
    )

    class TracedRuntimeError(RuntimeError):
        model_passes = (trace,)

    monkeypatch.setattr(app_module, "count_tokens", lambda *_args: 100)
    monkeypatch.setattr(
        app_module,
        "generate_text",
        lambda *_args: (_ for _ in ()).throw(
            TracedRuntimeError("generation failed token=do-not-show")
        ),
    )
    monkeypatch.setattr(app_module, "route_query", lambda *_args: route())

    result = app_module.run_model_turn("Show my transfers.", [], [], {}, request())

    assert "Error: `generation failed token=&lt;redacted&gt;`" in result[5]
    assert "Granite passes: `1`" in result[5]
    assert "partial model trace" in result[5]
    assert "do-not-show" not in result[5]


def test_second_pass_failure_preserves_executed_write_in_history_and_diagnostics(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter(
        [
            '<tool_call>{"name": "freeze_card", "arguments": {"last4": "4821"}}</tool_call>',
            "",
        ]
    )
    monkeypatch.setattr(app_module, "count_tokens", lambda *_args: 100)
    monkeypatch.setattr(app_module, "generate_text", lambda *_args: next(outputs))
    monkeypatch.setattr(
        app_module,
        "route_query",
        lambda *_args: route(capability="cards", intent="freeze_card"),
    )

    result = app_module.run_model_turn("Freeze card 4821.", [], [], {}, request())

    assert [item["role"] for item in result[2]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert "freeze_card" in result[5]
    assert "success" in result[5]
    assert "`frozen`" in result[3]
    assert result[1][-1]["content"] == app_module.MODEL_FAILURE_RESPONSE
    assert result[-1]["pending_servicing"] is None


def test_hf_turn_uses_same_unique_card_grounding_contract(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_tools: list[object] = []
    outputs = iter(
        [
            '<tool_call>{"name":"freeze_card","arguments":{"last4":"4821"}}</tool_call>',
            "I froze your Everyday Visa Debit card ending in 4821.",
        ]
    )

    def generate(_messages, tools, _max_new_tokens):
        seen_tools.append(tools)
        return next(outputs)

    routed = {
        **route(capability="cards", intent="freeze_card"),
        "domain": "banking",
        "lane": "servicing",
        "family": "cards",
        "action": "execute_tool",
        "entity_resolution": "resolved",
        "joint_decision_contract": "hierarchical-router-joint-decision/v1",
        "joint_decision_accepted": True,
    }
    monkeypatch.setattr(app_module, "count_tokens", lambda *_args: 100)
    monkeypatch.setattr(app_module, "generate_text", generate)
    monkeypatch.setattr(app_module, "route_query", lambda *_args: routed)

    result = app_module.run_model_turn(
        "My debit card was taken. Lock it now.",
        [],
        [],
        {},
        request(session_hash="grounding-parity"),
    )

    schema = seen_tools[0][0]["function"]["parameters"]
    assert schema["properties"]["last4"]["const"] == "4821"
    assert "Grounding source: `live_candidate`" in result[5]
    assert "`frozen`" in result[3]


def test_first_pass_parse_failure_preserves_raw_output_in_diagnostics(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_output = '<tool_call>{"name":"list_accounts","arguments":{}}'
    monkeypatch.setattr(app_module, "count_tokens", lambda *_args: 100)
    monkeypatch.setattr(app_module, "generate_text", lambda *_args: raw_output)
    monkeypatch.setattr(
        app_module,
        "route_query",
        lambda *_args: {
            **route(capability="accounts", intent="view_accounts"),
            "domain": "banking",
            "family": "accounts",
            "action": "execute_tool",
            "entity_resolution": "not_required",
        },
    )

    result = app_module.run_model_turn("Show my accounts.", [], [], {}, request())

    assert result[1][-1]["content"] == app_module.MODEL_FAILURE_RESPONSE
    assert html.escape(raw_output) in result[5]
    assert 'Exposed tools: `["list_accounts"]`' in result[5]
    assert "Generation calls: `1`" in result[5]


def test_second_pass_failure_does_not_complete_undelivered_read(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter(
        [
            '<tool_call>{"name": "list_accounts", "arguments": {}}</tool_call>',
            "",
        ]
    )
    monkeypatch.setattr(app_module, "count_tokens", lambda *_args: 100)
    monkeypatch.setattr(app_module, "generate_text", lambda *_args: next(outputs))
    monkeypatch.setattr(
        app_module,
        "route_query",
        lambda *_args: route(capability="accounts"),
    )

    result = app_module.run_model_turn("Show my accounts.", [], [], {}, request())

    assert result[1][-1]["content"] == app_module.MODEL_FAILURE_RESPONSE
    assert result[-1]["pending_servicing"]["intent"] == "view_accounts"


def test_credential_like_text_reaches_router_and_model(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routed: list[str] = []
    generated: list[str] = []

    def record_route(message, _history, _dialogue_state):
        routed.append(message)
        return route()

    def record_generation(messages, *_args):
        generated.append(messages[-1]["content"])
        return "I can discuss banking support without using that credential."

    monkeypatch.setattr(
        app_module,
        "route_query",
        record_route,
    )
    monkeypatch.setattr(app_module, "count_tokens", lambda *_args: 100)
    monkeypatch.setattr(
        app_module,
        "generate_text",
        record_generation,
    )

    result = app_module.run_model_turn("My PIN is 1234", [], [], {}, request())

    assert routed == ["My PIN is 1234"]
    assert generated == ["My PIN is 1234"]
    assert "9B direct_answer" in result[5]
    assert result[1] == result[2]


def test_reset_clears_visible_and_canonical_history(
    app_module,
) -> None:
    result = app_module.reset_session(request())

    assert result[0] == []
    assert result[1] == []
    assert len(result) == 9
    assert result[-1] == app_module.DialogueState().as_dict()


def test_policy_detour_uses_grounded_granite_and_resumes_pending_service(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated: list[tuple[list[dict[str, object]], object]] = []
    outputs = iter(
        (
            "Which transaction would you like to dispute?",
            (
                "Card disputes are reviewed after submission and may require "
                "additional information. [Policy: card.dispute.us.v1]"
            ),
            "Certainly. Which purchase should we continue with?",
        )
    )

    def classify(message, *_args):
        if "policy" in message.casefold():
            return route(intent="policy_knowledge")
        if "continue" in message.casefold():
            return route(
                capability="conversation",
                intent="conversation",
                relations=("resume_previous_service",),
            )
        return route(intent="dispute_transaction")

    def generate(messages, tools, _max_new_tokens):
        generated.append((messages, tools))
        return next(outputs)

    monkeypatch.setattr(app_module, "route_query", classify)
    monkeypatch.setattr(app_module, "count_tokens", lambda *_args: 100)
    monkeypatch.setattr(app_module, "generate_text", generate)

    first = app_module.run_model_turn("I need to dispute a purchase.", [], [], {}, request())
    detour = app_module.run_model_turn(
        "What is the card dispute policy?",
        first[1],
        first[2],
        first[-1],
        request(),
    )
    resumed = app_module.run_model_turn(
        "Let's continue with that.",
        detour[1],
        detour[2],
        detour[-1],
        request(),
    )

    visible_answer = detour[1][-1]["content"]
    assert visible_answer.startswith(
        "Card disputes are reviewed after submission and may require additional information. [1]"
    )
    assert "[Policy: card.dispute.us.v1]" not in visible_answer
    assert "<details><summary>Policy references</summary>" in visible_answer
    assert "Disputing a card purchase" in visible_answer
    assert "2026-01-01" in visible_answer
    # The canonical conversation (validators/history) keeps the raw citation tag.
    assert detour[2][-1]["content"].endswith("[Policy: card.dispute.us.v1]")
    assert "card.dispute.us.v1" in detour[5]
    assert generated[1][1] is None
    assert detour[-1]["knowledge_detour_active"] is True
    assert resumed[-1]["knowledge_detour_active"] is False
    assert resumed[-1]["pending_servicing"]["intent"] == "dispute_transaction"
    assert "I need to dispute a purchase." in str(generated[2][0])


def test_policy_citation_falls_back_to_chunk_id_when_metadata_is_unavailable(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "POLICY_CHUNK_LOOKUP", {})
    monkeypatch.setattr(app_module, "count_tokens", lambda *_args: 100)
    monkeypatch.setattr(
        app_module,
        "generate_text",
        lambda *_args: (
            "Card disputes are reviewed after submission. [Policy: card.dispute.us.v1]"
        ),
    )
    monkeypatch.setattr(
        app_module,
        "route_query",
        lambda *_args: route(intent="policy_knowledge"),
    )

    result = app_module.run_model_turn("What is the card dispute policy?", [], [], {}, request())

    visible_answer = result[1][-1]["content"]
    assert "[1]" in visible_answer
    assert "card.dispute.us.v1" in visible_answer
    assert "<details><summary>Policy references</summary>" in visible_answer
    # The canonical conversation is unaffected by the missing metadata.
    assert result[2][-1]["content"].endswith("[Policy: card.dispute.us.v1]")


def test_non_policy_turn_visible_content_is_unaffected_by_citation_rendering(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "count_tokens", lambda *_args: 100)
    monkeypatch.setattr(
        app_module,
        "generate_text",
        lambda *_args: "Your available balance is USD 4,218.75.",
    )
    monkeypatch.setattr(app_module, "route_query", lambda *_args: route())

    result = app_module.run_model_turn("Show my account balances.", [], [], {}, request())

    assert result[1][-1]["content"] == "Your available balance is USD 4,218.75."


def test_policy_no_match_with_relevant_tool_evidence_uses_grounded_fallback(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated: list[tuple[list[dict[str, object]], object]] = []
    outputs = iter(
        (
            '<tool_call>{"name":"list_transactions","arguments":{}}</tool_call>',
            "Here are your five most recent transactions.",
            "North Harbor Market is the grocery merchant from your July 25 purchase.",
        )
    )

    def classify(message, *_args):
        if "north harbor market" in message.casefold() and "?" in message:
            return route(intent="policy_knowledge")
        return route(intent="view_transactions", capability="transactions")

    def generate(messages, tools, _max_new_tokens):
        generated.append((messages, tools))
        return next(outputs)

    monkeypatch.setattr(app_module, "route_query", classify)
    monkeypatch.setattr(app_module, "count_tokens", lambda *_args: 100)
    monkeypatch.setattr(app_module, "generate_text", generate)

    grounded = app_module.run_model_turn(
        "Show my five most recent transactions.", [], [], {}, request()
    )
    followup = app_module.run_model_turn(
        "what is north harbor market ?",
        grounded[1],
        grounded[2],
        grounded[-1],
        request(),
    )

    answer = followup[1][-1]["content"]
    assert "North Harbor Market" in answer
    assert "approved current policy" not in answer
    assert generated[-1][1] is None
    assert "fallback:policy-no-match-grounded-converse" in followup[5]


def test_policy_no_match_with_irrelevant_tool_evidence_keeps_stock_response(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter(
        (
            '<tool_call>{"name":"list_transactions","arguments":{}}</tool_call>',
            "Here are your five most recent transactions.",
        )
    )

    def classify(message, *_args):
        if "atm" in message.casefold():
            return route(intent="policy_knowledge")
        return route(intent="view_transactions", capability="transactions")

    def generate(messages, tools, _max_new_tokens):
        return next(outputs)

    monkeypatch.setattr(app_module, "route_query", classify)
    monkeypatch.setattr(app_module, "count_tokens", lambda *_args: 100)
    monkeypatch.setattr(app_module, "generate_text", generate)

    grounded = app_module.run_model_turn(
        "Show my five most recent transactions.", [], [], {}, request()
    )
    followup = app_module.run_model_turn(
        "What are your ATM withdrawal limits?",
        grounded[1],
        grounded[2],
        grounded[-1],
        request(),
    )

    answer = followup[1][-1]["content"]
    assert "approved current policy" in answer


def test_policy_for_x_question_keeps_stock_response_despite_account_evidence(
    app_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter(
        (
            '<tool_call>{"name":"list_accounts","arguments":{}}</tool_call>',
            "Here are your accounts.",
        )
    )

    def classify(message, *_args):
        if "policy" in message.casefold():
            return route(intent="policy_knowledge")
        return route(intent="view_accounts", capability="accounts")

    def generate(messages, tools, _max_new_tokens):
        return next(outputs)

    monkeypatch.setattr(app_module, "route_query", classify)
    monkeypatch.setattr(app_module, "count_tokens", lambda *_args: 100)
    monkeypatch.setattr(app_module, "generate_text", generate)

    grounded = app_module.run_model_turn("Show my account balances.", [], [], {}, request())
    followup = app_module.run_model_turn(
        "What is the policy for a savings account?",
        grounded[1],
        grounded[2],
        grounded[-1],
        request(),
    )

    answer = followup[1][-1]["content"]
    assert "approved current policy" in answer
