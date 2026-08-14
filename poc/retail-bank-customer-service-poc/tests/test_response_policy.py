from __future__ import annotations

from model_service import ToolCall
from response_policy import (
    build_customer_experience_repair_messages,
    build_final_repair_messages,
    render_read_tool_results,
    validate_customer_facing_answer,
    validate_grounded_answer,
    validate_policy_answer,
)


def test_read_results_render_exact_markdown_tables() -> None:
    calls = (ToolCall(id="call_transactions", index=0, name="list_transactions", arguments={}),)
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
    assert (
        render_read_tool_results(
            (failed_read,), ({"ok": False, "error": {"code": "backend_error"}},)
        )
        is None
    )


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


def test_customer_facing_validator_rejects_internal_implementation_language() -> None:
    validation = validate_customer_facing_answer(
        "I can help with the synthetic accounts in this demo on the CPU backend."
    )

    assert not validation.valid
    assert "synthetic" in " ".join(validation.errors)
    assert "demo" in " ".join(validation.errors)
    assert "backend" in " ".join(validation.errors)
    assert validate_customer_facing_answer(
        "Hi, I’m Harbor. How can I help with your banking today?"
    ).valid


def test_policy_answer_requires_returned_citation_and_rejects_invented_citations() -> None:
    matches = (
        {
            "chunk_id": "mortgage.application.overview.us.v1",
            "title": "Mortgage application overview",
            "text": "A mortgage application is reviewed before approval.",
        },
    )

    missing = validate_policy_answer("Applications are reviewed.", matches)
    invented = validate_policy_answer(
        "Applications are reviewed. [Policy: mortgage.rates.us.v9]", matches
    )
    valid = validate_policy_answer(
        "Applications are reviewed before approval. [Policy: mortgage.application.overview.us.v1]",
        matches,
    )

    assert not missing.valid
    assert not invented.valid
    assert valid.valid


def test_policy_answer_rejects_unsupported_numeric_claim() -> None:
    validation = validate_policy_answer(
        "A replacement always arrives within 3 days. [Policy: card.replace]",
        (
            {
                "chunk_id": "card.replace",
                "title": "Replacement cards",
                "text": "Delivery estimates are provided when the request is submitted.",
            },
        ),
    )

    assert not validation.valid
    assert "unsupported numeric claims" in " ".join(validation.errors)


def test_policy_answer_accepts_number_present_in_evidence() -> None:
    validation = validate_policy_answer(
        "You must generally be at least 18 to apply. [Policy: mortgage.age]",
        (
            {
                "chunk_id": "mortgage.age",
                "title": "Mortgage eligibility",
                "text": "Applicants must generally be at least 18 years old.",
            },
        ),
    )

    assert validation.valid


def test_customer_experience_repair_receives_authoritative_evidence() -> None:
    repair = build_customer_experience_repair_messages(
        user_message="Can I apply for a mortgage?",
        draft="This demo can explain it.",
        errors=("answer contains internal term: demo",),
        authoritative_evidence=(
            {
                "chunk_id": "mortgage.application.overview.us.v1",
                "text": "Applications are reviewed before approval.",
            },
        ),
    )

    assert "Harborlight Bank" in repair[0]["content"]
    assert "authoritative_evidence" in repair[-1]["content"]
    assert "mortgage.application.overview.us.v1" in repair[-1]["content"]
