from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from policy_retrieval import DEFAULT_POLICY_PATH, PolicyKnowledgeBase


@pytest.fixture(scope="module")
def knowledge_base() -> PolicyKnowledgeBase:
    return PolicyKnowledgeBase.from_json(DEFAULT_POLICY_PATH)


@pytest.mark.parametrize(
    ("query", "expected_chunk"),
    [
        ("Can I open a mortgage and what do I need to qualify?", "mortgage.opening.us.v1"),
        ("What do I need to open a checking or savings account?", "deposit.opening.us.v1"),
        ("Will you charge me if my account is overdrawn?", "deposit.overdraft.us.v1"),
        ("How is the interest on my savings calculated?", "savings.interest.us.v1"),
        ("How do I dispute a debit card purchase?", "card.dispute.us.v1"),
        ("My card was stolen; how do I replace it?", "card.replacement.us.v1"),
        ("I see card fraud I do not recognize. What should I do?", "card.fraud.us.v1"),
        ("What are the requirements for a home loan?", "mortgage.opening.us.v1"),
        ("Where can I find the APY for savings?", "savings.interest.us.v1"),
    ],
)
def test_lookup_returns_the_relevant_approved_policy_chunk(
    knowledge_base: PolicyKnowledgeBase,
    query: str,
    expected_chunk: str,
) -> None:
    result = knowledge_base.lookup(query)

    assert result.matched
    assert result.matches[0].chunk_id == expected_chunk
    assert result.selected_chunk_ids[0] == expected_chunk
    assert result.citations[0]["citation"] == f"[Policy: {expected_chunk}]"
    assert result.citations[0]["corpus_revision"] == result.corpus_revision
    assert result.matches[0].as_dict()["corpus_revision"] == result.corpus_revision


def test_lookup_is_deterministic_and_honors_limit(
    knowledge_base: PolicyKnowledgeBase,
) -> None:
    query = "What should I do about a fraudulent card purchase and replacing my card?"

    first = knowledge_base.lookup(query, limit=2)
    second = knowledge_base.lookup(query, limit=2)

    assert first == second
    assert len(first.matches) == 2
    assert first.selected_chunk_ids == (
        "card.fraud.us.v1",
        "card.replacement.us.v1",
    )


@pytest.mark.parametrize(
    "query",
    [
        "How often should I water a fiddle-leaf fig?",
        "What is my card's security code?",
    ],
)
def test_lookup_returns_explicit_no_match_for_unsupported_query(
    knowledge_base: PolicyKnowledgeBase,
    query: str,
) -> None:
    result = knowledge_base.lookup(query)

    assert not result.matched
    assert result.matches == ()
    assert result.selected_chunk_ids == ()
    assert result.citations == ()
    assert result.corpus_revision == knowledge_base.corpus_revision


def test_loaded_policy_schema_is_immutable_and_single_revision(
    knowledge_base: PolicyKnowledgeBase,
) -> None:
    assert knowledge_base.schema_version == 1
    assert knowledge_base.corpus_revision.startswith("sha256:")
    assert knowledge_base.chunks
    assert {chunk.corpus_revision for chunk in knowledge_base.chunks} == {
        knowledge_base.corpus_revision
    }

    with pytest.raises(FrozenInstanceError):
        knowledge_base.chunks[0].title = "Changed"  # type: ignore[misc]


def test_policy_chunks_expose_factual_grounding_contract(
    knowledge_base: PolicyKnowledgeBase,
) -> None:
    assert len(knowledge_base.chunks) == 7
    assert all(chunk.answer for chunk in knowledge_base.chunks)
    assert all(chunk.required_claims for chunk in knowledge_base.chunks)
    assert all(chunk.forbidden_claims for chunk in knowledge_base.chunks)
    assert {chunk.chunk_id for chunk in knowledge_base.chunks} >= {
        "card.dispute.us.v1",
        "card.replacement.us.v1",
        "card.fraud.us.v1",
    }
    assert all(
        claim.casefold() in chunk.answer.casefold()
        for chunk in knowledge_base.chunks
        for claim in chunk.required_claims
    )


def test_corpus_digest_rejects_tampered_policy_text(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_POLICY_PATH.read_text(encoding="utf-8"))
    payload["chunks"][0]["text"] += " Tampered."
    tampered_path = tmp_path / "policy_knowledge.json"
    tampered_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="corpus revision digest mismatch"):
        PolicyKnowledgeBase.from_json(tampered_path)


@pytest.mark.parametrize("limit", [0, -1])
def test_lookup_rejects_non_positive_limits(
    knowledge_base: PolicyKnowledgeBase,
    limit: int,
) -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        knowledge_base.lookup("mortgage", limit=limit)
