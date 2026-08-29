"""Every detector must prove it can fire.

The 2026-08-29 audit found that all thirteen leakage/PII assertions in this
suite were of the form ``== 0`` against clean fixtures: replacing every
detector body with ``return 0`` left all thirteen passing. A gate asserted
only against clean input certifies that nothing bad was *found*, which is a
much weaker claim than the release process treats it as making — and it is
how a generator with no PII gate at all, a report-only router PII check, and
two hashes that validate a value against themselves all survived review.

Each test here plants the violation and asserts the build rejects it. A test
that stops failing when its detector is deleted does not belong in this file.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

import pytest

from hello_slm import banking_conversation_router_data as router_data
from hello_slm import banking_servicing_alignment_data as alignment_data
from hello_slm import banking_tool_sft_data as tool_sft_data

#: One representative of each shape the PII patterns are meant to stop.
PLANTED_IDENTIFIERS = (
    pytest.param("Confirming at alex.morgan@example.com as requested.", id="email"),
    pytest.param("Your reference is 123-45-6789 for this case.", id="ssn"),
    pytest.param("The card on file is 4111 1111 1111 1111 as shown.", id="card-number"),
)


def _first_train_record() -> dict[str, Any]:
    records = [
        record
        for record in tool_sft_data.generate_records(pilot_count=60)
        if record.get("metadata", {}).get("split") == "train"
    ]
    assert records, "expected the tool-SFT generator to produce train records"
    return deepcopy(records[0])


@pytest.mark.parametrize("leak", PLANTED_IDENTIFIERS)
def test_tool_sft_rejects_a_planted_identifier_in_a_final(leak: str) -> None:
    record = _first_train_record()
    for message in reversed(record["messages"]):
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            message["content"] = leak
            break
    else:  # pragma: no cover - the corpus always has a final assistant message
        pytest.fail("record had no assistant message to plant into")

    with pytest.raises(tool_sft_data.BankingToolSftDataError, match="private identifier"):
        tool_sft_data.validate_records([record])


@pytest.mark.parametrize("leak", PLANTED_IDENTIFIERS)
def test_tool_sft_rejects_a_planted_identifier_in_a_tool_result(leak: str) -> None:
    """Tool and system content reaches the model's context exactly as a final does."""
    record = _first_train_record()
    record["messages"].append({"role": "tool", "name": "list_cards", "content": leak})

    with pytest.raises(tool_sft_data.BankingToolSftDataError, match="private identifier"):
        tool_sft_data.validate_records([record])


def test_tool_sft_accepts_the_same_record_untouched() -> None:
    """The mirror of the injections: the gate must not reject clean input."""
    tool_sft_data.validate_records([_first_train_record()])


@pytest.mark.parametrize("leak", PLANTED_IDENTIFIERS)
def test_router_build_raises_on_a_planted_identifier(leak: str) -> None:
    """The router counted PII into its report and published anyway."""
    counted = router_data._count_pii_matches([leak])
    assert counted >= 1, f"router PII patterns do not detect {leak!r}"


def test_router_pii_count_is_wired_to_a_raise() -> None:
    """A count that nothing reads is a statistic, not a gate."""
    source = router_data.__file__
    with open(source, encoding="utf-8") as handle:
        body = handle.read()
    report_block = body.split('"pii_matches": _count_pii_matches(', 1)[1].split("\ndef ", 1)[0]
    assert 'raise ValueError(' in report_block, (
        "router pii_matches must fail the build, not merely be reported"
    )


@pytest.mark.parametrize("leak", PLANTED_IDENTIFIERS)
def test_alignment_pii_patterns_detect_each_shape(leak: str) -> None:
    assert any(pattern.search(leak) for pattern in alignment_data.PII_PATTERNS)


def test_the_two_generators_agree_on_what_counts_as_a_leak() -> None:
    """Divergent PII definitions would let a row legal in one corpus into the other."""
    assert [pattern.pattern for pattern in tool_sft_data.PII_PATTERNS] == [
        pattern.pattern for pattern in alignment_data.PII_PATTERNS
    ]


def test_banned_wording_detector_fires_on_a_planted_term() -> None:
    """The banned-wording gate is the one detector the suite already exercised
    positively; pinned here so the injection set covers it too."""
    assert tool_sft_data.TRAINABLE_TEXT_BANNED_WORDS.search("this is a demo of the app")
    assert not tool_sft_data.TRAINABLE_TEXT_BANNED_WORDS.search(
        "I can freeze the card ending in that number"
    )


@pytest.mark.parametrize("digits", [12, 16, 19, 22, 30, 40])
def test_a_long_digit_run_cannot_outgrow_the_card_number_pattern(digits: int) -> None:
    """Padding a card number with extra digits used to walk through the gate.

    The pattern was ``(?:\\d[ -]?){12,19}\\b``, which cannot match a 22-digit
    string: after nineteen digits the next character is another digit, so the
    trailing boundary fails and the whole match is abandoned. Longer runs are
    more suspicious, not less, so the upper bound is gone.
    """
    assert any(pattern.search("4" * digits) for pattern in tool_sft_data.PII_PATTERNS), (
        f"a {digits}-digit run must be caught"
    )


def test_a_short_digit_run_is_still_ignored() -> None:
    """Last-four digits and reference numbers are legitimate corpus content."""
    for run in ("4821", "1" * 11):
        assert not any(pattern.search(run) for pattern in tool_sft_data.PII_PATTERNS), run


def test_normalized_text_helper_is_not_what_the_pii_gate_relies_on() -> None:
    """Normalization strips punctuation, which would erase an SSN's shape.

    The gate must scan raw content; this pins the reason it does.
    """
    normalized = tool_sft_data.normalized_user_text("123-45-6789")
    assert not re.search(r"\d{3}-\d{2}-\d{4}", normalized)
