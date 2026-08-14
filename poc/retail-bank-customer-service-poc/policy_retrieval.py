from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

DEFAULT_POLICY_PATH = Path(__file__).with_name("policy_knowledge.json")
DEFAULT_MIN_SCORE = 0.30
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = frozenset(
    {
        "a",
        "about",
        "and",
        "are",
        "can",
        "do",
        "does",
        "for",
        "find",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "should",
        "the",
        "to",
        "what",
        "where",
        "will",
        "you",
    }
)
_TERM_ALIASES = {
    "apy": "interest",
    "challenge": "dispute",
    "charged": "charge",
    "charges": "charge",
    "disputed": "dispute",
    "disputing": "dispute",
    "eligible": "eligibility",
    "fraudulent": "fraud",
    "opening": "open",
    "overdrawn": "overdraft",
    "qualify": "eligibility",
    "qualifying": "eligibility",
    "recognized": "unauthorized",
    "recognize": "unauthorized",
    "replaced": "replace",
    "replacement": "replace",
    "replacing": "replace",
    "requirement": "eligibility",
    "requirements": "eligibility",
    "start": "open",
    "unrecognized": "unauthorized",
}
_INTENT_TERMS = frozenset(
    {
        "dispute",
        "eligibility",
        "fraud",
        "interest",
        "open",
        "overdraft",
        "replace",
        "stolen",
        "unauthorized",
    }
)
_CHUNK_REQUIRED_FIELDS = (
    "chunk_id",
    "title",
    "product",
    "jurisdiction",
    "effective_from",
    "effective_to",
    "text",
    "corpus_revision",
)
_CHUNK_OPTIONAL_FIELDS = ("answer", "required_claims", "forbidden_claims")
_ROOT_FIELDS = {"schema_version", "corpus_revision", "chunks"}


@dataclass(frozen=True)
class PolicyChunk:
    chunk_id: str
    title: str
    product: str
    jurisdiction: str
    effective_from: str
    effective_to: str | None
    text: str
    corpus_revision: str
    answer: str
    required_claims: tuple[str, ...]
    forbidden_claims: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyMatch:
    chunk_id: str
    title: str
    product: str
    jurisdiction: str
    effective_from: str
    effective_to: str | None
    text: str
    corpus_revision: str
    answer: str
    required_claims: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    score: float
    citation: str

    @classmethod
    def from_chunk(cls, chunk: PolicyChunk, score: float) -> PolicyMatch:
        return cls(
            **chunk.as_dict(),
            score=score,
            citation=f"[Policy: {chunk.chunk_id}]",
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyLookupResult:
    query: str
    corpus_revision: str
    matches: tuple[PolicyMatch, ...]

    @property
    def matched(self) -> bool:
        return bool(self.matches)

    @property
    def selected_chunk_ids(self) -> tuple[str, ...]:
        return tuple(match.chunk_id for match in self.matches)

    @property
    def citations(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "citation": match.citation,
                "chunk_id": match.chunk_id,
                "title": match.title,
                "product": match.product,
                "jurisdiction": match.jurisdiction,
                "effective_from": match.effective_from,
                "effective_to": match.effective_to,
                "corpus_revision": match.corpus_revision,
            }
            for match in self.matches
        )


@dataclass(frozen=True)
class PolicyKnowledgeBase:
    schema_version: int
    corpus_revision: str
    chunks: tuple[PolicyChunk, ...]
    min_score: float = DEFAULT_MIN_SCORE

    @classmethod
    def from_json(cls, path: str | Path) -> PolicyKnowledgeBase:
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not load policy corpus: {source}") from exc
        if not isinstance(payload, dict):
            raise ValueError("policy corpus must be a JSON object")
        if set(payload) != _ROOT_FIELDS:
            raise ValueError("policy corpus fields must be schema_version, corpus_revision, chunks")

        schema_version = payload.get("schema_version")
        corpus_revision = payload.get("corpus_revision")
        raw_chunks = payload.get("chunks")
        if type(schema_version) is not int or schema_version != 1:
            raise ValueError(f"unsupported policy schema version: {schema_version!r}")
        if not isinstance(corpus_revision, str) or not corpus_revision.startswith("sha256:"):
            raise ValueError("policy corpus_revision must be a sha256 revision")
        if not isinstance(raw_chunks, list) or not raw_chunks:
            raise ValueError("policy corpus must contain at least one chunk")

        expected_revision = _corpus_revision(schema_version, raw_chunks)
        if corpus_revision != expected_revision:
            raise ValueError(
                f"corpus revision digest mismatch: expected {expected_revision}, "
                f"found {corpus_revision}"
            )

        chunks = tuple(_parse_chunk(item, corpus_revision) for item in raw_chunks)
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("policy chunk_id values must be unique")
        return cls(schema_version, corpus_revision, chunks)

    def lookup(self, query: str, *, limit: int = 3) -> PolicyLookupResult:
        if limit <= 0:
            raise ValueError("limit must be positive")
        query_terms = frozenset(_terms(query))
        if not query_terms:
            return PolicyLookupResult(query, self.corpus_revision, ())

        document_terms = tuple(_terms(_search_text(chunk)) for chunk in self.chunks)
        document_frequency = Counter(term for terms in document_terms for term in frozenset(terms))
        idf = {
            term: math.log((len(self.chunks) + 1) / (frequency + 1)) + 1
            for term, frequency in document_frequency.items()
        }
        unknown_weight = math.log(len(self.chunks) + 1) + 1
        query_weight = sum(
            idf.get(term, unknown_weight) * _query_term_weight(term) for term in query_terms
        )
        ranked: list[tuple[float, str, PolicyChunk]] = []
        for chunk, terms in zip(self.chunks, document_terms, strict=True):
            overlap = query_terms.intersection(terms)
            if not overlap.intersection(_INTENT_TERMS):
                continue
            score = sum(idf[term] * _query_term_weight(term) for term in overlap) / query_weight
            if score >= self.min_score:
                ranked.append((score, chunk.chunk_id, chunk))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        matches = tuple(
            PolicyMatch.from_chunk(chunk, round(score, 6))
            for score, _chunk_id, chunk in ranked[:limit]
        )
        return PolicyLookupResult(query, self.corpus_revision, matches)


def retrieve_policy(query: str, *, limit: int = 3) -> PolicyLookupResult:
    return PolicyKnowledgeBase.from_json(DEFAULT_POLICY_PATH).lookup(query, limit=limit)


def _parse_chunk(value: object, corpus_revision: str) -> PolicyChunk:
    if not isinstance(value, dict):
        raise ValueError("each policy chunk must be a JSON object")
    missing = set(_CHUNK_REQUIRED_FIELDS) - set(value)
    extras = set(value) - set(_CHUNK_REQUIRED_FIELDS) - set(_CHUNK_OPTIONAL_FIELDS)
    if missing or extras:
        raise ValueError(
            f"policy chunk fields are invalid: missing={sorted(missing)}, extras={sorted(extras)}"
        )
    if value["corpus_revision"] != corpus_revision:
        raise ValueError("all policy chunks must use the corpus revision")
    for field in _CHUNK_REQUIRED_FIELDS:
        if field == "effective_to":
            continue
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"policy chunk {field} must be a non-empty string")
    if value["effective_to"] is not None and not isinstance(value["effective_to"], str):
        raise ValueError("policy chunk effective_to must be a date or null")
    try:
        effective_from = date.fromisoformat(value["effective_from"])
        effective_to = (
            date.fromisoformat(value["effective_to"]) if value["effective_to"] is not None else None
        )
    except ValueError as exc:
        raise ValueError("policy chunk effective dates must use ISO YYYY-MM-DD") from exc
    if effective_to is not None and effective_to < effective_from:
        raise ValueError("policy chunk effective_to cannot precede effective_from")
    answer = value.get("answer", value["text"])
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("policy chunk answer must be a non-empty string")
    claims: dict[str, tuple[str, ...]] = {}
    for field in ("required_claims", "forbidden_claims"):
        raw_claims = value.get(field, ())
        if not isinstance(raw_claims, list | tuple) or not all(
            isinstance(claim, str) and claim.strip() for claim in raw_claims
        ):
            raise ValueError(f"policy chunk {field} must be a list of non-empty strings")
        claims[field] = tuple(raw_claims)
    return PolicyChunk(
        chunk_id=cast(str, value["chunk_id"]),
        title=cast(str, value["title"]),
        product=cast(str, value["product"]),
        jurisdiction=cast(str, value["jurisdiction"]),
        effective_from=cast(str, value["effective_from"]),
        effective_to=cast(str | None, value["effective_to"]),
        text=cast(str, value["text"]),
        corpus_revision=cast(str, value["corpus_revision"]),
        answer=answer,
        required_claims=claims["required_claims"],
        forbidden_claims=claims["forbidden_claims"],
    )


def _corpus_revision(schema_version: int, chunks: list[object]) -> str:
    digest_chunks = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("each policy chunk must be a JSON object")
        digest_chunks.append(
            {key: value for key, value in chunk.items() if key != "corpus_revision"}
        )
    canonical = json.dumps(
        {"schema_version": schema_version, "chunks": digest_chunks},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _search_text(chunk: PolicyChunk) -> str:
    return " ".join(
        (chunk.chunk_id.replace(".", " "), chunk.title, chunk.product, chunk.text, chunk.answer)
    )


def _terms(text: str) -> tuple[str, ...]:
    return tuple(
        _TERM_ALIASES.get(token, token)
        for token in _TOKEN_PATTERN.findall(text.casefold())
        if token not in _STOP_WORDS
    )


def _query_term_weight(term: str) -> float:
    return 1.75 if term in _INTENT_TERMS else 1.0
