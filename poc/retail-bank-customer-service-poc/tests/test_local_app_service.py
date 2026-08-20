from __future__ import annotations

import json
from pathlib import Path

from local_app_service import LocalBankingController
from mock_bank import SessionBankRegistry
from responses import MODEL_FAILURE_RESPONSE


class StaticRouter:
    def __init__(self, route_name: str = "in_domain") -> None:
        self.route_name = route_name
        self.seen_history: list[dict[str, object]] | None = None

    def classify(self, message, history):
        self.seen_history = history
        return {
            "route": self.route_name,
            "banking_probability": 0.99 if self.route_name == "in_domain" else 0.01,
            "ood_probability": 0.01 if self.route_name == "in_domain" else 0.99,
            "confidence": 0.99,
            "capability": "accounts",
            "intent": "view_accounts",
            "intent_confidence": 0.9,
            "capability_confidence": 0.9,
            "capability_candidates": [{"capability": "accounts", "probability": 0.9}],
            "relation_probabilities": {
                "context_dependent": 0.05,
                "agent_repair": 0.05,
                "topic_shift": 0.05,
                "clarification_answer": 0.05,
            },
            "active_relations": [],
            "context_applied": bool(history),
            "router_revision": "test-router",
            "reason": "test route",
        }


class FakeRuntime:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, messages, tools, _max_new_tokens, *, sample: bool = False):
        self.calls.append({"messages": messages, "tools": tools, "sample": sample})
        return next(self.outputs)

    def count_tokens(self, messages, _tools):
        return len(messages) * 10

    def runtime_metadata(self):
        return {
            "runtime_device": "cuda:0",
            "cuda_device_name": "NVIDIA TITAN V",
            "weight_quantization": "bitsandbytes-nf4-double",
            "model_id": "spkc83/retail-bank-servicing-agent-9b",
            "model_revision": "1d56824995aa1adecfe20f62ca42fb1c0c443817",
        }


def bank(tmp_path: Path) -> SessionBankRegistry:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "synthetic_bank.json").read_text(encoding="utf-8"))
    return SessionBankRegistry(
        payload,
        database_dir=tmp_path / "sessions",
    )


def test_controller_runs_real_model_tool_loop_and_records_local_nf4_diagnostics(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(
        [
            '<tool_call>{"name":"list_accounts","arguments":{}}</tool_call>',
            "Your available balance is USD 4,218.75.",
        ]
    )
    controller = LocalBankingController(
        bank=bank(tmp_path),
        runtime=runtime,
        router=StaticRouter(),
    )

    result = controller.run_turn(
        username="alex.demo",
        session_hash="local-browser",
        message="Show my account balances.",
        conversation=[],
    )

    assert "## Accounts" in result.response
    assert "| Name | Type | Last 4 | Available | Current | Status |" in result.response
    assert "Everyday Checking" in result.response
    assert "USD 3,245.67" in result.response
    assert [message["role"] for message in result.conversation] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert result.tool_calls[0].name == "list_accounts"
    assert result.model_call_count == 2
    assert "NVIDIA TITAN V" in result.diagnostics
    assert "bitsandbytes-nf4-double" in result.diagnostics
    assert "Local CUDA / NF4" in result.diagnostics


def test_local_diagnostics_show_v6_hierarchy_and_exact_exposed_tools(
    tmp_path: Path,
) -> None:
    class V6Router:
        def classify(self, _message, _history):
            return {
                **StaticRouter().classify("", []),
                "domain": "banking",
                "lane": "servicing",
                "family": "accounts",
                "intent": "view_accounts",
                "action": "execute_tool",
                "entity_resolution": "not_required",
            }

    controller = LocalBankingController(
        bank=bank(tmp_path),
        runtime=FakeRuntime(
            [
                '<tool_call>{"name":"list_accounts","arguments":{}}</tool_call>',
                "Accounts returned.",
            ]
        ),
        router=V6Router(),
    )

    result = controller.run_turn(
        username="alex.demo",
        session_hash="local-browser",
        message="Show my accounts.",
        conversation=[],
    )

    assert "V6 domain: `banking`" in result.diagnostics
    assert "V6 lane: `servicing`" in result.diagnostics
    assert "V6 family: `accounts`" in result.diagnostics
    assert "V6 intent: `view_accounts`" in result.diagnostics
    assert "V6 action: `execute_tool`" in result.diagnostics
    assert "V6 entity resolution: `not_required`" in result.diagnostics
    assert 'Exposed tools: `["list_accounts"]`' in result.diagnostics
    summary = result.diagnostics.split("### Full technical details", 1)[0]
    assert "Outcome: `base tool rendered`" in summary
    assert "Granite passes: `2`" in summary
    assert "list_accounts: success" in summary
    assert "Effective grounding/source: `tool result: list_accounts`" in summary
    assert "Relation probabilities" not in summary
    assert "SHA-256" not in summary


def test_local_first_pass_parse_failure_preserves_raw_output_in_diagnostics(
    tmp_path: Path,
) -> None:
    raw_output = '<tool_call>{"name":"list_accounts","arguments":{}}'
    controller = LocalBankingController(
        bank=bank(tmp_path),
        runtime=FakeRuntime([raw_output]),
        router=StaticRouter(),
    )

    result = controller.run_turn(
        username="alex.demo",
        session_hash="local-browser",
        message="Show my accounts.",
        conversation=[],
    )

    assert result.response == MODEL_FAILURE_RESPONSE
    assert result.model_call_count == 1
    assert raw_output in result.diagnostics
    assert "Router failure type: `AgentProtocolError`" in result.diagnostics
    assert result.conversation[0] == {"role": "user", "content": "Show my accounts."}


def test_local_generic_error_preserves_safe_message_and_attached_trace(tmp_path: Path) -> None:
    trace = type(
        "Trace",
        (),
        {
            "label": "base",
            "input_tokens": 10,
            "prompt_sha256": "prompt-hash",
            "output_sha256": "output-hash",
            "raw_output": "partial local trace",
            "runtime_device": "cuda:0",
            "cuda_device_name": "test GPU",
        },
    )()

    class TracedRuntimeError(RuntimeError):
        model_passes = (trace,)

    class FailingRuntime(FakeRuntime):
        def generate(self, _messages, _tools, _max_new_tokens):
            raise TracedRuntimeError("generation failed password=do-not-show")

    controller = LocalBankingController(
        bank=bank(tmp_path),
        runtime=FailingRuntime([]),
        router=StaticRouter(),
    )

    result = controller.run_turn(
        username="alex.demo",
        session_hash="local-browser",
        message="Show my accounts.",
        conversation=[],
    )

    assert "Error: `generation failed password=&lt;redacted&gt;`" in result.diagnostics
    assert "Granite passes: `1`" in result.diagnostics
    assert "partial local trace" in result.diagnostics
    assert "do-not-show" not in result.diagnostics


def test_controller_uses_stock_response_for_high_confidence_ood(tmp_path: Path) -> None:
    controller = LocalBankingController(
        bank=bank(tmp_path),
        runtime=FakeRuntime([]),
        router=StaticRouter("out_of_domain"),
    )

    result = controller.run_turn(
        username="alex.demo",
        session_hash="local-browser",
        message="Explain telescope optics.",
        conversation=[],
    )

    assert "banking" in result.response.casefold()
    assert "synthetic" not in result.response.casefold()
    assert result.model_call_count == 0
    assert result.route["route"] == "out_of_domain"


def test_controller_supplies_visible_conversation_history_to_router(tmp_path: Path) -> None:
    router = StaticRouter("in_domain")
    controller = LocalBankingController(
        bank=bank(tmp_path),
        runtime=FakeRuntime(["The earlier transfer is complete."]),
        router=router,
    )
    history = [
        {"role": "user", "content": "Show my transfers."},
        {"role": "assistant", "content": "One transfer is complete."},
    ]

    controller.run_turn(
        username="alex.demo",
        session_hash="local-browser",
        message="When did that happen?",
        conversation=history,
    )

    assert router.seen_history == history


def test_controller_reset_restores_synthetic_backend_and_clears_history(
    tmp_path: Path,
) -> None:
    registry = bank(tmp_path)
    registry.execute("alex.demo", "local-browser", "freeze_card", {"last4": "4821"})
    controller = LocalBankingController(
        bank=registry,
        runtime=FakeRuntime([]),
        router=StaticRouter(),
    )

    snapshot = controller.reset("alex.demo", "local-browser")

    assert next(card for card in snapshot["cards"] if card["last4"] == "4821")["status"] == "active"


class SequenceRouter:
    def __init__(self, routes: list[dict[str, object]]) -> None:
        self.routes = iter(routes)

    def classify(self, _message, _history, **_kwargs):
        return next(self.routes)


class PolicyMatch:
    def as_dict(self):
        return {
            "chunk_id": "disputes.timeline.us.v1",
            "title": "Transaction dispute review",
            "text": "A submitted dispute is reviewed and status updates are provided.",
            "effective_from": "2026-01-01",
        }


class PolicyResult:
    matched = True
    matches = (PolicyMatch(),)
    corpus_revision = "sha256:policy-v1"


class FakePolicyKnowledge:
    def lookup(self, _query):
        return PolicyResult()


def routed(intent: str, *, relations: list[str] | None = None) -> dict[str, object]:
    return {
        "route": "in_domain",
        "banking_probability": 0.99,
        "ood_probability": 0.01,
        "confidence": 0.99,
        "intent": intent,
        "intent_confidence": 0.99,
        "intent_candidates": [{"intent": intent, "probability": 0.99}],
        "relation_probabilities": {},
        "active_relations": relations or [],
        "context_applied": True,
        "router_revision": "test-router-v5",
    }


def test_policy_detour_is_granite_grounded_and_resumes_pending_dispute(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(
        [
            "I’m sorry this happened. Which purchase would you like to dispute?",
            "A submitted dispute is reviewed and status updates are provided. "
            "[Policy: disputes.timeline.us.v1]",
            '<tool_call>{"name":"dispute_transaction","arguments":'
            '{"description":"Blue Line Coffee"}}</tool_call>',
            "I submitted a dispute for Blue Line Coffee.",
        ]
    )
    controller = LocalBankingController(
        bank=bank(tmp_path),
        runtime=runtime,
        router=SequenceRouter(
            [
                routed("dispute_transaction"),
                routed("policy_knowledge"),
                routed(
                    "dispute_transaction",
                    relations=["resume_previous_service"],
                ),
            ]
        ),
        policy_knowledge=FakePolicyKnowledge(),
    )

    first = controller.run_turn(
        username="alex.demo",
        session_hash="local-browser",
        message="I need to dispute a purchase.",
        conversation=[],
    )
    policy = controller.run_turn(
        username="alex.demo",
        session_hash="local-browser",
        message="How long does a dispute review take?",
        conversation=first.conversation,
    )
    resumed = controller.run_turn(
        username="alex.demo",
        session_hash="local-browser",
        message="Continue with Blue Line Coffee.",
        conversation=policy.conversation,
    )

    assert policy.response_path == "policy_grounded"
    assert policy.policy_sources == ("disputes.timeline.us.v1",)
    assert runtime.calls[1]["tools"] is None
    resume_prompt = runtime.calls[2]["messages"]
    assert {"role": "user", "content": "I need to dispute a purchase."} in resume_prompt
    assert {
        "role": "assistant",
        "content": "I’m sorry this happened. Which purchase would you like to dispute?",
    } in resume_prompt
    assert resumed.tool_calls[0].name == "dispute_transaction"
    assert resumed.dialogue_state["pending_servicing"] is None


def test_effective_policy_lane_dispatches_independently_of_legacy_intent_threshold(
    tmp_path: Path,
) -> None:
    policy_route = {
        **routed("policy_knowledge"),
        "lane": "policy",
        "action": "retrieve_policy",
        "entity_resolution": "not_required",
        "intent_confidence": 0.31,
        "effective_decision_contract": "retail-bank-effective-turn-decision/v1",
        "decision_accepted": True,
    }
    runtime = FakeRuntime(
        [
            "A submitted dispute is reviewed and status updates are provided. "
            "[Policy: disputes.timeline.us.v1]"
        ]
    )
    controller = LocalBankingController(
        bank=bank(tmp_path),
        runtime=runtime,
        router=SequenceRouter([policy_route]),
        policy_knowledge=FakePolicyKnowledge(),
    )

    result = controller.run_turn(
        username="alex.demo",
        session_hash="local-browser",
        message="What is the review policy?",
        conversation=[],
    )

    assert result.response_path == "policy_grounded"
    assert result.policy_sources == ("disputes.timeline.us.v1",)
    assert runtime.calls[0]["tools"] is None


def v7_route(intent: str, *, action: str, entity_resolution: str) -> dict[str, object]:
    return {
        **routed(intent),
        "domain": "banking",
        "lane": "servicing",
        "family": "cards",
        "action": action,
        "entity_resolution": entity_resolution,
        "joint_decision_contract": "hierarchical-router-joint-decision/v1",
        "joint_decision_accepted": True,
    }


def test_local_controller_grounds_unique_stolen_card_before_granite_call(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(
        [
            '<tool_call>{"name":"freeze_card","arguments":{"last4":"4821"}}</tool_call>',
            "I froze your Everyday Visa Debit card ending in 4821.",
        ]
    )
    controller = LocalBankingController(
        bank=bank(tmp_path),
        runtime=runtime,
        router=SequenceRouter(
            [v7_route("freeze_card", action="execute_tool", entity_resolution="resolved")]
        ),
    )

    result = controller.run_turn(
        username="alex.demo",
        session_hash="local-browser",
        message="My debit card was taken. Lock it now.",
        conversation=[],
    )

    schema = runtime.calls[0]["tools"][0]["function"]["parameters"]
    assert schema["required"] == ["last4"]
    assert schema["properties"]["last4"]["const"] == "4821"
    assert result.tool_calls[0].arguments == {"last4": "4821"}
    card = next(item for item in result.snapshot["cards"] if item["last4"] == "4821")
    assert card["status"] == "frozen"
    assert result.route["entity_grounding_source"] == "live_candidate"


def test_local_controller_rejects_account_suffix_before_card_mutation(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(
        ['<tool_call>{"name":"freeze_card","arguments":{"last4":"1042"}}</tool_call>']
    )
    controller = LocalBankingController(
        bank=bank(tmp_path),
        runtime=runtime,
        router=SequenceRouter(
            [v7_route("freeze_card", action="execute_tool", entity_resolution="resolved")]
        ),
    )

    result = controller.run_turn(
        username="alex.demo",
        session_hash="local-browser",
        message="My debit card was taken. Lock it now.",
        conversation=[],
    )

    card = next(item for item in result.snapshot["cards"] if item["last4"] == "4821")
    assert result.response == MODEL_FAILURE_RESPONSE
    assert result.tool_calls == ()
    assert result.model_passes[0].raw_output.endswith(
        '{"name":"freeze_card","arguments":{"last4":"1042"}}</tool_call>'
    )
    assert card["status"] == "active"


def test_reset_clears_dialogue_state(tmp_path: Path) -> None:
    controller = LocalBankingController(
        bank=bank(tmp_path),
        runtime=FakeRuntime(["Which card should I replace?"]),
        router=SequenceRouter([routed("replace_card")]),
    )
    result = controller.run_turn(
        username="alex.demo",
        session_hash="local-browser",
        message="Replace my card.",
        conversation=[],
    )
    assert result.dialogue_state["pending_servicing"]["intent"] == "replace_card"

    controller.reset("alex.demo", "local-browser")

    assert controller.dialogue_state("alex.demo", "local-browser")["pending_servicing"] is None


def test_failed_read_follow_up_keeps_pending_task_until_result_delivery(
    tmp_path: Path,
) -> None:
    controller = LocalBankingController(
        bank=bank(tmp_path),
        runtime=FakeRuntime(
            [
                '<tool_call>{"name":"list_accounts","arguments":{}}</tool_call>',
                "",
            ]
        ),
        router=SequenceRouter([routed("view_accounts")]),
        policy_knowledge=FakePolicyKnowledge(),
    )

    result = controller.run_turn(
        username="alex.demo",
        session_hash="local-browser",
        message="Show my accounts.",
        conversation=[],
    )

    assert result.response == MODEL_FAILURE_RESPONSE
    assert result.dialogue_state["pending_servicing"]["intent"] == "view_accounts"


def test_failed_mutation_follow_up_commits_action_without_pending_retry(
    tmp_path: Path,
) -> None:
    controller = LocalBankingController(
        bank=bank(tmp_path),
        runtime=FakeRuntime(
            [
                '<tool_call>{"name":"freeze_card","arguments":{"last4":"4821"}}</tool_call>',
                "",
            ]
        ),
        router=SequenceRouter([routed("freeze_card")]),
        policy_knowledge=FakePolicyKnowledge(),
    )

    result = controller.run_turn(
        username="alex.demo",
        session_hash="local-browser",
        message="Freeze card 4821.",
        conversation=[],
    )

    assert result.snapshot["cards"][0]["status"] == "frozen"
    assert result.dialogue_state["pending_servicing"] is None


class NoMatchPolicyResult:
    matched = False
    matches = ()
    corpus_revision = "sha256:policy-v1"


class NoMatchPolicyKnowledge:
    def lookup(self, _query):
        return NoMatchPolicyResult()


def test_policy_no_match_with_tool_evidence_falls_back_to_grounded_converse(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(
        [
            '<tool_call>{"name":"list_transactions","arguments":{}}</tool_call>',
            "Here are your five most recent transactions.",
            "North Harbor Market is the grocery merchant from your July 25 purchase of USD 86.24.",
        ]
    )
    controller = LocalBankingController(
        bank=bank(tmp_path),
        runtime=runtime,
        router=SequenceRouter(
            [
                routed("view_transactions"),
                routed("policy_knowledge"),
            ]
        ),
        policy_knowledge=NoMatchPolicyKnowledge(),
    )

    grounded = controller.run_turn(
        username="alex.demo",
        session_hash="local-browser",
        message="Show my five most recent transactions.",
        conversation=[],
    )
    followup = controller.run_turn(
        username="alex.demo",
        session_hash="local-browser",
        message="what is north harbor market ?",
        conversation=grounded.conversation,
    )

    assert followup.response_path != "policy_no_match"
    assert "North Harbor Market" in followup.response
    assert "couldn’t find an approved current policy" not in followup.response
    assert "fallback:policy-no-match-grounded-converse" in followup.diagnostics
    fallback_prompt = runtime.calls[-1]["messages"]
    assert runtime.calls[-1]["tools"] is None
    assert any(message.get("role") == "tool" for message in fallback_prompt)


def test_policy_no_match_with_irrelevant_tool_evidence_keeps_stock_response(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(
        [
            '<tool_call>{"name":"list_transactions","arguments":{}}</tool_call>',
            "Here are your five most recent transactions.",
        ]
    )
    controller = LocalBankingController(
        bank=bank(tmp_path),
        runtime=runtime,
        router=SequenceRouter(
            [
                routed("view_transactions"),
                routed("policy_knowledge"),
            ]
        ),
        policy_knowledge=NoMatchPolicyKnowledge(),
    )

    grounded = controller.run_turn(
        username="alex.demo",
        session_hash="local-browser",
        message="Show my five most recent transactions.",
        conversation=[],
    )
    followup = controller.run_turn(
        username="alex.demo",
        session_hash="local-browser",
        message="What are your ATM withdrawal limits?",
        conversation=grounded.conversation,
    )

    assert followup.response_path == "policy_no_match"
    assert "approved current policy" in followup.response
    assert len(runtime.calls) == 2


def test_policy_fallback_turn_rejects_fabricated_action_claims(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(
        [
            '<tool_call>{"name":"list_transactions","arguments":{}}</tool_call>',
            "Here are your five most recent transactions.",
            "I've frozen your card ending in 4821. North Harbor Market is a grocer.",
            "North Harbor Market is the grocery merchant from your July 25 purchase.",
        ]
    )
    controller = LocalBankingController(
        bank=bank(tmp_path),
        runtime=runtime,
        router=SequenceRouter(
            [
                routed("view_transactions"),
                routed("policy_knowledge"),
            ]
        ),
        policy_knowledge=NoMatchPolicyKnowledge(),
    )

    grounded = controller.run_turn(
        username="alex.demo",
        session_hash="local-browser",
        message="Show my five most recent transactions.",
        conversation=[],
    )
    followup = controller.run_turn(
        username="alex.demo",
        session_hash="local-browser",
        message="what is north harbor market ?",
        conversation=grounded.conversation,
    )

    assert "frozen" not in followup.response.lower()
    assert "North Harbor Market" in followup.response
    assert followup.response_path.endswith("_customer_repaired")


def test_policy_no_match_without_tool_evidence_keeps_stock_response(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime([])
    controller = LocalBankingController(
        bank=bank(tmp_path),
        runtime=runtime,
        router=SequenceRouter([routed("policy_knowledge")]),
        policy_knowledge=NoMatchPolicyKnowledge(),
    )

    result = controller.run_turn(
        username="alex.demo",
        session_hash="local-browser",
        message="What is the policy on gemstone loans?",
        conversation=[],
    )

    assert result.response_path == "policy_no_match"
    assert "approved current policy" in result.response
    assert runtime.calls == []
