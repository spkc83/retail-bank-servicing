from __future__ import annotations

import json
from pathlib import Path

from model_service import ToolCall
from response_policy import (
    build_customer_experience_repair_messages,
    build_final_repair_messages,
    leading_prose,
    render_read_tool_results,
    strip_realizer_filler,
    validate_customer_facing_answer,
    validate_grounded_answer,
    validate_no_unsupported_action_claims,
    validate_policy_answer,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_leading_prose_keeps_the_sentence_before_an_inline_model_table() -> None:
    draft = "Here are your recent transactions. | Date | Description |\n| --- | --- |"

    assert leading_prose(draft) == "Here are your recent transactions."


def test_leading_prose_keeps_the_sentence_before_a_block_model_table() -> None:
    draft = "Here are your cards.\n\n| Name | Last 4 |\n| --- | --- |\n| Everyday | 4821 |"

    assert leading_prose(draft) == "Here are your cards."


def test_leading_prose_is_empty_when_the_draft_opens_with_a_table() -> None:
    assert leading_prose("| Name | Last 4 |\n| --- | --- |") == ""


def test_leading_prose_returns_a_table_free_draft_unchanged() -> None:
    assert leading_prose("  Here are your two most recent transactions.  ") == (
        "Here are your two most recent transactions."
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


def test_policy_answer_requires_exact_numeric_quantity_not_substring() -> None:
    matches = (
        {
            "chunk_id": "card.replace",
            "title": "Replacement cards",
            "text": "Replacement cards can take up to 30 days to arrive.",
        },
    )

    numeric = validate_policy_answer(
        "A replacement card can take up to 3 days. [Policy: card.replace]",
        matches,
    )
    word_form = validate_policy_answer(
        "A replacement card can take up to three days. [Policy: card.replace]",
        matches,
    )

    assert not numeric.valid
    assert "unsupported numeric claims" in " ".join(numeric.errors)
    assert not word_form.valid
    assert "unsupported numeric claims" in " ".join(word_form.errors)


def test_policy_answer_compares_date_components_independently() -> None:
    validation = validate_policy_answer(
        "This policy is effective January 1, 2026. [Policy: card.replace]",
        (
            {
                "chunk_id": "card.replace",
                "title": "Replacement cards",
                "text": "This policy applies to replacement cards.",
                "effective_from": "2026-01-01",
            },
        ),
    )

    assert validation.valid


def test_policy_answer_normalizes_number_words_and_formatted_quantities() -> None:
    matches = (
        {
            "chunk_id": "card.replace",
            "title": "Replacement cards",
            "text": "One replacement is allowed and the fee is $1,000.00.",
        },
    )

    validation = validate_policy_answer(
        "You may request 1 replacement, and the fee is $1000. [Policy: card.replace]",
        matches,
    )
    unsupported = validate_policy_answer(
        "You may request three replacements. [Policy: card.replace]",
        matches,
    )

    assert validation.valid
    assert not unsupported.valid
    assert "unsupported numeric claims" in " ".join(unsupported.errors)


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


def test_strip_realizer_filler_removes_a_canned_prefix_and_closer() -> None:
    text = (
        "I found the following details: Your card ending in 4821 is now frozen. "
        "This reflects the information available in this session."
    )

    assert strip_realizer_filler(text) == "Your card ending in 4821 is now frozen."


def test_strip_realizer_filler_leaves_a_natural_answer_untouched() -> None:
    text = "Your Everyday Visa Debit ending in 4821 is now frozen."

    assert strip_realizer_filler(text) == text


def test_strip_realizer_filler_empties_an_answer_that_is_only_filler() -> None:
    assert strip_realizer_filler("I checked the available information.") == ""


def test_strip_realizer_filler_drops_a_trailing_closer_after_a_table() -> None:
    text = (
        "Here are your cards.\n\n## Cards\n\n| Name |\n| --- |\n\n"
        "I can help with the next banking step."
    )

    assert strip_realizer_filler(text).endswith("| --- |")


def test_zero_tool_answers_must_not_claim_completed_actions() -> None:
    fabricated = validate_no_unsupported_action_claims(
        "I found your active card and froze it to stop unauthorized use.", ()
    )
    frozen_state = validate_no_unsupported_action_claims(
        "Your card ending in 4821 is now frozen.", ()
    )

    assert not fabricated.valid
    assert not frozen_state.valid
    assert any("without tool evidence" in error for error in fabricated.errors)


def test_zero_tool_answers_may_ask_and_describe_without_action_claims() -> None:
    question = validate_no_unsupported_action_claims(
        "Would you like me to freeze the card ending in 4821?", ()
    )
    neutral = validate_no_unsupported_action_claims(
        "Hi, I’m Harbor. How can I help with your banking today?", ()
    )

    assert question.valid
    assert neutral.valid


def test_zero_tool_answers_must_not_claim_retrieved_account_data() -> None:
    """The live v8 model produced the first string verbatim, twice, with no tool call."""

    pin = validate_no_unsupported_action_claims(
        "I found the current PIN in the account information. What new PIN should I use?", ()
    )
    balance = validate_no_unsupported_action_claims(
        "I checked your account and the available balance is 1,240.00.", ()
    )
    shows = validate_no_unsupported_action_claims("Your account shows two open service cases.", ())
    looked_up = validate_no_unsupported_action_claims(
        "I looked up the transfer to River Consulting for you.", ()
    )

    assert not pin.valid
    assert not balance.valid
    assert not shows.valid
    assert not looked_up.valid
    assert any("without tool evidence" in error for error in pin.errors)


def test_claims_attributed_to_the_conversation_are_not_retrieval_claims() -> None:
    """The shipped corpus clarifies with this exact sentence (history_ambiguous_card_train-r000)."""

    corpus = validate_no_unsupported_action_claims(
        "I found two cards in our conversation. Which should I replace? "
        "Please share the last four digits.",
        (),
    )
    mentioned = validate_no_unsupported_action_claims(
        "I found the card you mentioned earlier. Which one should I freeze?", ()
    )

    assert corpus.valid
    assert mentioned.valid


def test_retrieved_data_claims_are_allowed_when_tool_evidence_exists() -> None:
    evidence = ({"ok": True, "result": {"account": {"available": "1240.00"}}},)
    conversation = [
        {"role": "tool", "tool_call_id": "call_1", "name": "list_accounts", "content": "{}"},
    ]

    with_results = validate_no_unsupported_action_claims(
        "I checked your account and the available balance is 1,240.00.", evidence
    )
    with_history = validate_no_unsupported_action_claims(
        "I checked your account and the available balance is 1,240.00.",
        (),
        conversation=conversation,
    )

    assert with_results.valid
    assert with_history.valid


def test_zero_tool_answers_may_clarify_offer_and_cite_policy() -> None:
    clarify = validate_no_unsupported_action_claims(
        "Which card should I replace? Please share its last four digits.", ()
    )
    offer = validate_no_unsupported_action_claims(
        "I can help with accounts, cards, transactions, transfers and service cases. "
        "What would you like to do?",
        (),
    )
    policy = validate_no_unsupported_action_claims(
        "Harborlight Bank may decline or return a transaction when available funds are "
        "insufficient, and fees vary by account product.",
        (),
    )

    assert clarify.valid
    assert offer.valid
    assert policy.valid


def test_action_claims_are_allowed_when_any_tool_evidence_exists() -> None:
    evidence = ({"ok": True, "result": {"card": {"last4": "4821", "status": "frozen"}}},)

    validation = validate_no_unsupported_action_claims(
        "Your Everyday Visa Debit ending in 4821 is now frozen.", evidence
    )

    assert validation.valid


def test_action_claim_is_grounded_by_a_prior_tool_message_in_the_conversation() -> None:
    conversation = [
        {"role": "user", "content": "please freeze my card ending in 4821"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "freeze_card", "arguments": {}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "freeze_card",
            "content": '{"ok": true, "result": {"card": {"last4": "4821", "status": "frozen"}}}',
        },
        {"role": "assistant", "content": "I froze your Everyday Visa Debit ending in 4821."},
        {"role": "user", "content": "did you actually freeze the card"},
    ]

    validation = validate_no_unsupported_action_claims(
        "Yes, I froze it right away.", (), conversation=conversation
    )

    assert validation.valid


def test_action_claim_without_conversation_evidence_is_still_rejected() -> None:
    validation = validate_no_unsupported_action_claims("Yes, I froze it right away.", ())

    assert not validation.valid


def test_ineligibility_blocker_language_is_not_an_action_claim() -> None:
    validation = validate_no_unsupported_action_claims(
        "Your Cobble Everyday Debit ending in 7101 is frozen, so I can't do that. "
        "Which active card should I use instead?",
        (),
    )

    assert validation.valid


def test_explicit_denial_language_is_not_an_action_claim() -> None:
    validation = validate_no_unsupported_action_claims(
        "No card change was made. I only asked which one you wanted replaced.",
        (),
    )

    assert validation.valid


def test_guard_never_rejects_the_shipped_servicing_alignment_corpus() -> None:
    train_path = REPO_ROOT / "data" / "banking-servicing-alignment-v5" / "train.jsonl"
    rejections: list[tuple[str, tuple[str, ...]]] = []
    with train_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            messages = record["messages"]
            final = messages[-1]
            if final.get("role") != "assistant" or not isinstance(final.get("content"), str):
                continue
            validation = validate_no_unsupported_action_claims(
                final["content"], (), conversation=messages
            )
            if not validation.valid:
                rejections.append((record.get("record_id"), validation.errors))

    assert rejections == []


def test_tool_evidence_gate_requires_question_relevance() -> None:
    from response_policy import tool_evidence_answers_question

    grounded = [
        {"role": "user", "content": "Show my five most recent transactions."},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "list_transactions"}]},
        {
            "role": "tool",
            "name": "list_transactions",
            "content": '{"rows":[{"description":"North Harbor Market","amount":"-86.24"}]}',
        },
        {"role": "assistant", "content": "## Recent transactions"},
    ]
    assert tool_evidence_answers_question("what is north harbor market ?", grounded)
    assert not tool_evidence_answers_question(
        "What are your ATM withdrawal limits?",
        grounded,
    )
    accounts_evidence = [
        {
            "role": "tool",
            "name": "list_accounts",
            "content": (
                '{"accounts":[{"account_id":"acct-1","available_balance_cents":324567,'
                '"currency":"USD","customer_id":"cust-1","last4":"1042",'
                '"name":"Everyday Checking","status":"active","type":"checking"}]}'
            ),
        }
    ]
    for policy_question in (
        "What is the policy for a savings account?",
        "What is your policy on an active card?",
        "What is the policy for my checking account balance?",
        "Is there a fee?",
        "What are the transfer limits?",
    ):
        assert not tool_evidence_answers_question(policy_question, accounts_evidence), (
            policy_question
        )
    coffee_evidence = [
        {
            "role": "tool",
            "name": "list_transactions",
            "content": '{"rows":[{"description":"Blue Line Coffee","disputed":false}]}',
        }
    ]
    assert not tool_evidence_answers_question("Is there a fee?", coffee_evidence)
    assert not tool_evidence_answers_question("what is the status currency?", coffee_evidence)
    assert not tool_evidence_answers_question("what is north harbor market ?", [])
    assert not tool_evidence_answers_question(
        "what is north harbor market ?",
        [
            {"role": "user", "content": "north harbor market"},
            {"role": "assistant", "content": "North Harbor Market appears here."},
        ],
    )
    assert not tool_evidence_answers_question("what?", grounded)
    stale = [
        grounded[2],
        *[
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"}
            for i in range(10)
        ],
    ]
    assert not tool_evidence_answers_question("what is north harbor market ?", stale, window=8)


def test_policy_fallback_route_keeps_learned_decision_and_tags_diagnostics() -> None:
    from response_policy import POLICY_FALLBACK_NOTE, policy_context_fallback_route

    route = {
        "route": "in_domain",
        "lane": "policy",
        "intent": "policy_knowledge",
        "action": "retrieve_policy",
        "entity_resolution": "not_required",
        "constraint_diagnostics": ("constraint:example",),
    }
    amended = policy_context_fallback_route(route)
    assert amended["action"] == "converse"
    assert amended["entity_resolution"] == "not_required"
    assert amended["lane"] == "policy"
    assert amended["intent"] == "policy_knowledge"
    assert amended["learned_action"] == "retrieve_policy"
    assert amended["learned_entity_resolution"] == "not_required"
    assert amended["constraint_diagnostics"] == ("constraint:example", POLICY_FALLBACK_NOTE)
    assert route["action"] == "retrieve_policy"
    assert route["constraint_diagnostics"] == ("constraint:example",)
