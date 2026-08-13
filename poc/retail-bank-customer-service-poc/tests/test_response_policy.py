from __future__ import annotations

from model_service import ToolCall
from response_policy import (
    build_final_repair_messages,
    render_read_tool_results,
    validate_grounded_answer,
)


def test_read_results_render_exact_markdown_tables() -> None:
    calls = (
        ToolCall(id="call_transactions", index=0, name="list_transactions", arguments={}),
    )
    results = (
        {
            "ok": True,
            "result": {
                "transactions": [
                    {
                        "posted_at": "2026-08-11T09:30:00Z",
                        "description": "Market | Cafe",
                        "amount_cents": -1250,
                        "currency": "USD",
                        "status": "posted",
                        "category": "dining",
                        "disputed": False,
                    }
                ]
            },
        },
    )

    rendered = render_read_tool_results(calls, results)

    assert rendered is not None
    assert "| Date | Description | Amount | Status | Category | Disputed |" in rendered
    assert "Market \\| Cafe" in rendered
    assert "-USD 12.50" in rendered
    assert "2026-08-11 09:30 UTC" in rendered


def test_read_renderer_does_not_override_write_or_failed_results() -> None:
    write_call = ToolCall(
        id="call_freeze", index=0, name="freeze_card", arguments={"last4": "4821"}
    )
    failed_read = ToolCall(id="call_accounts", index=0, name="list_accounts", arguments={})

    assert render_read_tool_results((write_call,), ({"ok": True, "result": {}},)) is None
    assert render_read_tool_results(
        (failed_read,), ({"ok": False, "error": {"code": "backend_error"}},)
    ) is None


def test_grounding_validator_requires_action_outcome_and_selector() -> None:
    calls = (
        ToolCall(
            id="call_cancel",
            index=0,
            name="cancel_transfer",
            arguments={"recipient": "River Consulting"},
        ),
    )
    results = (
        {
            "ok": True,
            "result": {
                "transfer": {
                    "recipient": "River Consulting",
                    "status": "cancelled",
                    "amount_cents": 45000,
                    "currency": "USD",
                },
                "simulated": True,
            },
        },
    )

    invalid = validate_grounded_answer("Done. I cancelled Jamie Lee's transfer.", calls, results)
    valid = validate_grounded_answer(
        "Done — I cancelled the transfer to River Consulting.", calls, results
    )

    assert not invalid.valid
    assert any("River Consulting" in error for error in invalid.errors)
    assert valid.valid


def test_grounding_policy_rejects_and_redacts_private_backend_identifiers() -> None:
    calls = (
        ToolCall(
            id="call_cancel",
            index=0,
            name="cancel_transfer",
            arguments={"recipient": "River Consulting"},
        ),
    )
    results = (
        {
            "ok": True,
            "result": {
                "transfer": {
                    "transfer_id": "trf_internal_100",
                    "from_account_id": "acct_internal_200",
                    "recipient": "River Consulting",
                    "status": "cancelled",
                }
            },
        },
    )

    validation = validate_grounded_answer(
        "I cancelled River Consulting transfer trf_internal_100.", calls, results
    )
    repair = build_final_repair_messages(
        user_message="Cancel it",
        draft="wrong",
        calls=calls,
        results=results,
        errors=validation.errors,
    )

    assert not validation.valid
    assert "trf_internal_100" not in repair[-1]["content"]
    assert "acct_internal_200" not in repair[-1]["content"]
