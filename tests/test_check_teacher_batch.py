from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SYSTEM_PROMPT = (
    "You are Harbor, the conversational customer-service assistant for Harborlight Bank."
)
POLICY_CHUNK_ID = "card.dispute.us.v1"
POLICY_ORIGINAL = (
    "To dispute a posted card purchase, identify the transaction and explain the issue "
    "promptly. Harborlight Bank may request receipts or communications after submission. "
    f"[Policy: {POLICY_CHUNK_ID}]."
)
POLICY_CLAIMS = (
    "posted card purchase",
    "identify the transaction",
    "may request receipts or communications",
)
POLICY_FORBIDDEN = ("guaranteed provisional credit", "fixed investigation timeline")
POLICY_REWRITE = (
    "If you want to dispute a posted card purchase, identify the transaction and tell me "
    "what went wrong. Harborlight Bank may request receipts or communications after you "
    f"submit it, so keep anything related handy. [Policy: {POLICY_CHUNK_ID}]"
)
FREEZE_ORIGINAL = "Your debit card ending in 9609 is now frozen."
FREEZE_REWRITE = (
    "I've frozen your Everyday Visa Debit card ending in 9609, so nobody can spend on it. "
    "If you see a charge you don't recognise, say the word and I can open a dispute."
)
TABLE_ORIGINAL = (
    "| Account | Ending | Available | Current |\n"
    "|---|---:|---:|---:|\n"
    "| Main Checking | 1792 | USD 4,728.25 | USD 4,773.25 |"
)
TABLE_REWRITE = f"Here are both balances as they stand right now.\n\n{TABLE_ORIGINAL}"
ERROR_ORIGINAL = (
    "I could not open that dispute because no eligible transaction for "
    "Maple Street Electronics was found."
)
CLARIFY_ORIGINAL = (
    "Which card should I replace? Please share the last four digits shown in the app."
)


def _load_module() -> ModuleType:
    path = Path("scripts/retail_bank/check_teacher_batch.py")
    spec = importlib.util.spec_from_file_location("check_teacher_batch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _hash(record_id: str) -> str:
    return f"sha256:{hashlib.sha256(record_id.encode()).hexdigest()}"


def _record(
    record_id: str,
    *,
    user: str,
    final: str,
    path: str = "tool_success",
    mode: str = "execute_tool",
    family: str = "read_cards",
    template_id: str = "tool-template-v1",
    split: str = "train",
    tool: tuple[str, dict[str, Any]] | None = None,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT, "loss": False},
        {"role": "user", "content": user, "loss": False},
    ]
    if tool is not None:
        name, envelope = tool
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "loss": True,
                "tool_calls": [
                    {
                        "id": f"call_{record_id}_0",
                        "index": 0,
                        "type": "function",
                        "function": {"name": name, "arguments": {}},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "name": name,
                "tool_call_id": f"call_{record_id}_0",
                "loss": False,
                "content": envelope,
            }
        )
    messages.append({"role": "assistant", "content": final, "loss": True})
    return {
        "record_id": record_id,
        "metadata": {"split": split},
        "split_keys": {
            "scenario_family": family,
            "template_id": template_id,
            "customer_id": f"cust_{record_id}",
            "state_seed": "state-0000",
            "realization_seed": "realization-000",
        },
        "expected": {
            "path": path,
            "generation_contract": {"mode": mode, "version": "banking-v7-route-to-generation/v1"},
            "grounding_facts": [],
            "requires_tool": tool is not None,
            **(expected or {}),
        },
        "messages": messages,
    }


def _freeze_record(
    record_id: str = "freeze_row",
    *,
    final: str = FREEZE_ORIGINAL,
) -> dict[str, Any]:
    return _record(
        record_id,
        user="My debit card is missing, please freeze it",
        final=final,
        family="emergency_card_freeze",
        tool=(
            "freeze_card",
            {
                "ok": True,
                "result": {
                    "card": {
                        "card_id": "card_demo_debit",
                        "customer_id": "cust_demo",
                        "last4": "9609",
                        "name": "Everyday Visa Debit",
                        "status": "frozen",
                    }
                },
            },
        ),
    )


def _policy_record(record_id: str = "policy_row") -> dict[str, Any]:
    return _record(
        record_id,
        user="How do I challenge a purchase on my card that I did not make",
        final=POLICY_ORIGINAL,
        path="retrieval_grounded_policy",
        mode="retrieve_policy",
        family="faq_card_dispute",
        template_id="faq-card-dispute-v1",
        expected={
            "grounding_facts": list(POLICY_CLAIMS),
            "forbidden_facts": list(POLICY_FORBIDDEN),
            "policy_citations": [POLICY_CHUNK_ID],
        },
    )


def _table_record(record_id: str = "table_row") -> dict[str, Any]:
    return _record(
        record_id,
        user="Please show the accounts available to me and their balances",
        final=TABLE_ORIGINAL,
        family="read_accounts",
        tool=(
            "list_accounts",
            {"ok": True, "result": {"accounts": [{"last4": "1792", "name": "Main Checking"}]}},
        ),
    )


def _error_record(record_id: str = "error_row") -> dict[str, Any]:
    return _record(
        record_id,
        user="Please dispute the Maple Street Electronics charge",
        final=ERROR_ORIGINAL,
        path="tool_error",
        family="dispute_transaction",
        tool=(
            "dispute_transaction",
            {"ok": False, "error": {"code": "not_found", "message": "no eligible record"}},
        ),
    )


def _clarify_record(record_id: str = "clarify_row") -> dict[str, Any]:
    return _record(
        record_id,
        user="Please replace it for me",
        final=CLARIFY_ORIGINAL,
        path="clarification",
        mode="clarify",
        family="replace_card_ambiguous",
    )


def _converse_record(record_id: str = "converse_row") -> dict[str, Any]:
    return _record(
        record_id,
        user="Thanks, that is all clear now",
        final="You're welcome. I'm here whenever you need another banking hand.",
        path="conversation",
        mode="converse",
        family="conversational_thanks",
    )


def _final_of(record: dict[str, Any]) -> str:
    return str(record["messages"][-1]["content"])


def _user_of(record: dict[str, Any]) -> str:
    return str([m for m in record["messages"] if m["role"] == "user"][-1]["content"])


def _write_case(
    tmp_path: Path,
    records: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    records_dir = tmp_path / "records"
    records_dir.mkdir(exist_ok=True)
    (records_dir / "train.jsonl").write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )
    requests_path = tmp_path / "requests.jsonl"
    requests_path.write_text(
        "".join(
            json.dumps(
                {
                    "record_id": record["record_id"],
                    "immutable_hash": _hash(record["record_id"]),
                    "user_content": _user_of(record),
                    "final_response": _final_of(record),
                    "allowed_edits": ["user_content", "final_response"],
                }
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    responses_path = tmp_path / "responses.jsonl"
    responses_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in responses),
        encoding="utf-8",
    )
    return requests_path, responses_path, records_dir


def _response(
    record: dict[str, Any],
    *,
    final: str,
    user: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "record_id": record["record_id"],
        "immutable_hash": _hash(record["record_id"]),
        "user_content": _user_of(record) if user is None else user,
        "final_response": final,
        **(extra or {}),
    }


def _run(
    tmp_path: Path,
    records: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    *,
    extra_args: list[str] | None = None,
    capsys: pytest.CaptureFixture[str],
) -> tuple[int, str]:
    module = _load_module()
    requests_path, responses_path, records_dir = _write_case(tmp_path, records, responses)
    code = module.main(
        [
            "--requests",
            str(requests_path),
            "--responses",
            str(responses_path),
            "--records-dir",
            str(records_dir),
            *(extra_args or []),
        ]
    )
    return code, capsys.readouterr().out


def _rules(output: str) -> set[str]:
    return {
        line.split(": ", 2)[1]
        for line in output.splitlines()
        if len(line.split(": ", 2)) == 3
    }


def test_clean_batch_passes_and_reports_a_json_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    freeze = _freeze_record()
    policy = _policy_record()
    code, out = _run(
        tmp_path,
        [freeze, policy],
        [
            _response(freeze, final=FREEZE_REWRITE),
            _response(policy, final=POLICY_REWRITE),
        ],
        capsys=capsys,
    )

    assert code == 0, out
    summary = json.loads(out)
    assert summary["checked"] == 2
    assert summary["files"][0]["rows"] == 2
    assert sorted(summary["families"]) == ["emergency_card_freeze", "faq_card_dispute"]
    assert summary["sentence_histogram"] == {"2": 2}


def test_rule_a_flags_a_rewrite_that_drops_banking_facts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    freeze = _freeze_record()
    code, out = _run(
        tmp_path,
        [freeze],
        [
            _response(
                freeze,
                final=(
                    "I've locked your everyday debit card so nobody can spend on it. "
                    "Tell me if you want a replacement sent out."
                ),
            )
        ],
        capsys=capsys,
    )

    assert code == 2
    assert "a" in _rules(out)
    assert "freeze_row: a:" in out


def test_rule_b_flags_a_final_with_fewer_than_seven_words(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    freeze = _freeze_record()
    code, out = _run(
        tmp_path,
        [freeze],
        [_response(freeze, final="Debit card 9609 frozen.")],
        capsys=capsys,
    )

    assert code == 2
    assert "b" in _rules(out)


def test_rule_c_flags_internal_language(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    freeze = _freeze_record()
    code, out = _run(
        tmp_path,
        [freeze],
        [
            _response(
                freeze,
                final=(
                    "Our model froze the debit card ending in 9609 for you. "
                    "Nothing else on the account changed."
                ),
            )
        ],
        capsys=capsys,
    )

    assert code == 2
    assert "c" in _rules(out)


def test_rule_c_flags_the_word_session(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    freeze = _freeze_record()
    code, out = _run(
        tmp_path,
        [freeze],
        [
            _response(
                freeze,
                final=(
                    "Your debit card ending in 9609 is frozen for the rest of this session. "
                    "Tell me if you want a new one sent out."
                ),
            )
        ],
        capsys=capsys,
    )

    assert code == 2
    assert "c" in _rules(out)


def test_rule_d_flags_realizer_filler(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    freeze = _freeze_record()
    code, out = _run(
        tmp_path,
        [freeze],
        [
            _response(
                freeze,
                final=(
                    "For clarity, your debit card ending in 9609 is frozen now. "
                    "I can help with the next banking step."
                ),
            )
        ],
        capsys=capsys,
    )

    assert code == 2
    assert "d" in _rules(out)


def test_rule_e_flags_a_dropped_load_bearing_literal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    error = _error_record()
    code, out = _run(
        tmp_path,
        [error],
        [
            _response(
                error,
                final=(
                    "I wasn't able to open that dispute, so the Maple Street Electronics "
                    "transaction is unchanged. Check the date in the app and I'll try again."
                ),
            )
        ],
        capsys=capsys,
    )

    assert code == 2
    assert "e" in _rules(out)
    assert "could not" in out


def test_rule_f_flags_text_after_the_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    table = _table_record()
    code, out = _run(
        tmp_path,
        [table],
        [_response(table, final=f"{TABLE_REWRITE}\n\nTell me if anything looks off to you.")],
        capsys=capsys,
    )

    assert code == 2
    assert "f" in _rules(out)


def test_rule_f_flags_a_mangled_table_block(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    table = _table_record()
    mangled = TABLE_ORIGINAL.replace("Main Checking", "Main checking")
    code, out = _run(
        tmp_path,
        [table],
        [_response(table, final=f"Here are both balances right now.\n\n{mangled}")],
        capsys=capsys,
    )

    assert code == 2
    assert "f" in _rules(out)


def test_rule_g_flags_a_clarify_question_that_arrives_too_late(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clarify = _clarify_record()
    lead = " ".join(["happy"] * 20 + ["to help you with that and I want to get this right"])
    code, out = _run(
        tmp_path,
        [clarify],
        [
            _response(
                clarify,
                final=(
                    f"{lead} before I order anything, because more than one is open on your "
                    "profile here. So which card should I replace? Please share the last "
                    "four digits shown in the app."
                ),
            )
        ],
        extra_args=["--exempt-family", "replace_card_ambiguous"],
        capsys=capsys,
    )

    assert code == 2
    assert "g" in _rules(out)


def test_rule_h_flags_a_one_sentence_final(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    freeze = _freeze_record()
    code, out = _run(
        tmp_path,
        [freeze],
        [
            _response(
                freeze,
                final="I've frozen your Everyday Visa Debit card ending in 9609 for you now.",
            )
        ],
        capsys=capsys,
    )

    assert code == 2
    assert "h" in _rules(out)


def test_rule_h_accepts_a_one_sentence_final_for_an_exempt_family(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    freeze = _freeze_record()
    code, out = _run(
        tmp_path,
        [freeze],
        [
            _response(
                freeze,
                final="I've frozen your Everyday Visa Debit card ending in 9609 for you now.",
            )
        ],
        extra_args=["--exempt-family", "emergency_card_freeze"],
        capsys=capsys,
    )

    assert code == 0, out


def test_rule_h_flags_more_than_one_exclamation_mark(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    freeze = _freeze_record()
    code, out = _run(
        tmp_path,
        [freeze],
        [
            _response(
                freeze,
                final=(
                    "All done! Your debit card ending in 9609 is frozen! "
                    "Tell me if you want a replacement."
                ),
            )
        ],
        capsys=capsys,
    )

    assert code == 2
    assert "h" in _rules(out)


def test_rule_i_flags_a_changed_user_turn_under_finals_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    freeze = _freeze_record()
    code, out = _run(
        tmp_path,
        [freeze],
        [
            _response(
                freeze,
                final=FREEZE_REWRITE,
                user="My debit card has gone missing, can you freeze it",
            )
        ],
        extra_args=["--finals-only"],
        capsys=capsys,
    )

    assert code == 2
    assert "i" in _rules(out)


def test_rule_j_flags_a_duplicate_final(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = _freeze_record("freeze_one")
    second = _freeze_record("freeze_two")
    code, out = _run(
        tmp_path,
        [first, second],
        [
            _response(first, final=FREEZE_REWRITE),
            _response(second, final=FREEZE_REWRITE),
        ],
        capsys=capsys,
    )

    assert code == 2
    assert "j" in _rules(out)


def test_rule_j_flags_a_final_that_collides_with_an_untouched_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    freeze = _freeze_record()
    untouched = _record(
        "untouched_row",
        user="Is my debit card usable right now",
        final=FREEZE_REWRITE,
        family="read_cards",
        split="test",
    )
    module = _load_module()
    requests_path, responses_path, records_dir = _write_case(
        tmp_path,
        [freeze, untouched],
        [_response(freeze, final=FREEZE_REWRITE)],
    )
    code = module.main(
        [
            "--requests",
            str(requests_path),
            "--responses",
            str(responses_path),
            "--records-dir",
            str(records_dir),
        ]
    )
    out = capsys.readouterr().out

    assert code == 2
    assert "j" in _rules(out)


def test_rule_j_flags_an_overused_opening_trigram(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = [_freeze_record(f"freeze_{index}") for index in range(5)]
    responses = [
        _response(
            record,
            final=(
                f"Your debit card ending in 9609 is frozen as of now, item {index}. "
                "Tell me if you want a fresh one ordered."
            ),
        )
        for index, record in enumerate(records)
    ]
    code, out = _run(tmp_path, records, responses, capsys=capsys)

    assert code == 2
    assert "j" in _rules(out)
    assert "your debit card" in out


def test_rule_k_flags_a_completed_action_claim_without_tool_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    converse = _converse_record()
    code, out = _run(
        tmp_path,
        [converse],
        [
            _response(
                converse,
                final=(
                    "Glad that helped, and I've frozen the card for you already. "
                    "Ask me any time you need a hand with something else."
                ),
            )
        ],
        capsys=capsys,
    )

    assert code == 2
    assert "k" in _rules(out)


def test_rule_l_flags_an_invented_number_in_a_policy_final(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    policy = _policy_record()
    code, out = _run(
        tmp_path,
        [policy],
        [
            _response(
                policy,
                final=(
                    "If you want to dispute a posted card purchase, identify the transaction "
                    "within 99 days and tell me what went wrong. Harborlight Bank may request "
                    "receipts or communications after you submit it. "
                    f"[Policy: {POLICY_CHUNK_ID}]"
                ),
            )
        ],
        capsys=capsys,
    )

    assert code == 2
    assert "l" in _rules(out)


def test_rule_l_flags_a_dropped_required_policy_claim(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    policy = _policy_record()
    code, out = _run(
        tmp_path,
        [policy],
        [
            _response(
                policy,
                final=(
                    "If you want to dispute a posted card purchase, identify the transaction "
                    "and tell me what went wrong. Harborlight Bank reviews each one carefully "
                    f"after you submit it. [Policy: {POLICY_CHUNK_ID}]"
                ),
            )
        ],
        capsys=capsys,
    )

    assert code == 2
    assert "l" in _rules(out)


def test_rule_l_flags_a_forbidden_policy_claim(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    policy = _policy_record()
    code, out = _run(
        tmp_path,
        [policy],
        [
            _response(
                policy,
                final=(
                    "If you want to dispute a posted card purchase, identify the transaction "
                    "and you get guaranteed provisional credit. Harborlight Bank may request "
                    "receipts or communications after you submit it. "
                    f"[Policy: {POLICY_CHUNK_ID}]"
                ),
            )
        ],
        capsys=capsys,
    )

    assert code == 2
    assert "l" in _rules(out)


def test_rule_m_flags_a_user_turn_that_normalizes_to_a_poc_preset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    table = _table_record()
    code, out = _run(
        tmp_path,
        [table],
        [_response(table, final=TABLE_REWRITE, user="Show my account balances.")],
        capsys=capsys,
    )

    assert code == 2
    assert "m" in _rules(out)


def test_rule_m_flags_a_user_turn_that_matches_a_held_out_current(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    table = _table_record()
    code, out = _run(
        tmp_path,
        [table],
        [_response(table, final=TABLE_REWRITE, user="Show my five most recent transactions.")],
        capsys=capsys,
    )

    assert code == 2
    assert "m" in _rules(out)


def test_every_violation_is_reported_not_only_the_first(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    freeze = _freeze_record()
    code, out = _run(
        tmp_path,
        [freeze],
        [_response(freeze, final="For clarity, the frozen debit card is our model's doing.")],
        capsys=capsys,
    )

    assert code == 2
    assert {"c", "d", "h"} <= _rules(out)


def test_a_table_row_needs_one_lead_in_sentence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    table = _table_record()
    code, out = _run(tmp_path, [table], [_response(table, final=TABLE_ORIGINAL)], capsys=capsys)

    assert code == 2
    assert "h" in _rules(out)


def test_a_table_row_with_a_lead_in_sentence_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    table = _table_record()
    code, out = _run(tmp_path, [table], [_response(table, final=TABLE_REWRITE)], capsys=capsys)

    assert code == 0, out


def test_unknown_record_ids_are_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    freeze = _freeze_record()
    stray = _response(freeze, final=FREEZE_REWRITE)
    stray["record_id"] = "not_a_record"
    code, out = _run(tmp_path, [freeze], [stray], capsys=capsys)

    assert code == 2
    assert "not_a_record" in out


FRAUD_CHUNK_ID = "card.fraud.us.v1"
FRAUD_CLAIMS = (
    "lock the card",
    "report the transaction promptly",
    "do not share a password, security code, or one-time verification code",
)
FRAUD_ORIGINAL = (
    "If you see an unauthorized card purchase, lock the card and report the transaction "
    "promptly. Do not share a password, security code, or one-time verification code. "
    f"[Policy: {FRAUD_CHUNK_ID}]."
)
FRAUD_REWRITE = (
    "If you spot an unauthorized card purchase, lock the card and report the transaction "
    "promptly so we can look into it for you. Do not share a password, security code, or "
    f"one-time verification code with anyone who contacts you. [Policy: {FRAUD_CHUNK_ID}]"
)


def _fraud_record(record_id: str = "fraud_row") -> dict[str, Any]:
    return _record(
        record_id,
        user="Please what should I do if I see card fraud",
        final=FRAUD_ORIGINAL,
        path="retrieval_grounded_policy",
        mode="retrieve_policy",
        family="faq_card_fraud",
        template_id="faq-card-fraud-v1",
        expected={
            "grounding_facts": list(FRAUD_CLAIMS),
            "forbidden_facts": [
                "share your verification code",
                "wait for another fraudulent purchase",
            ],
            "policy_citations": [FRAUD_CHUNK_ID],
        },
    )


def test_rule_a_waives_a_credential_warning_the_released_final_already_carries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fraud = _fraud_record()
    code, out = _run(tmp_path, [fraud], [_response(fraud, final=FRAUD_REWRITE)], capsys=capsys)

    assert code == 0, out
    summary = json.loads(out)
    assert summary["waivers"] == [
        {
            "record_id": "fraud_row",
            "rule": "a",
            "reason": "fraud_row final_response requests private credentials",
        }
    ]


def test_rule_a_still_flags_a_rewrite_that_asks_for_a_credential(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fraud = _fraud_record()
    code, out = _run(
        tmp_path,
        [fraud],
        [
            _response(
                fraud,
                final=f"{FRAUD_REWRITE} Confirm your PIN here and I can check the account.",
            )
        ],
        capsys=capsys,
    )

    assert code == 2
    assert "a" in _rules(out)
