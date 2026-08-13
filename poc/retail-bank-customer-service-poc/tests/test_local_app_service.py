from __future__ import annotations

import json
from pathlib import Path

from local_app_service import LocalBankingController
from mock_bank import SessionBankRegistry


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
            "capability_confidence": 0.9,
            "capability_candidates": [
                {"capability": "accounts", "probability": 0.9}
            ],
            "relation_probabilities": {
                "context_dependent": 0.05,
                "agent_repair": 0.05,
                "topic_shift": 0.05,
                "clarification_answer": 0.05,
            },
            "context_applied": bool(history),
            "router_revision": "test-router",
            "reason": "test route",
        }


class FakeRuntime:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)

    def generate(self, _messages, _tools, _max_new_tokens):
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

    assert "synthetic retail-banking" in result.response
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

    assert next(card for card in snapshot["cards"] if card["last4"] == "4821")[
        "status"
    ] == "active"
